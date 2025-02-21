import glob
import logging
import warnings
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial
import gc

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from src.open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_input_dtype
from src.training.data import get_data
from src.training.distributed import is_master, init_distributed_device, broadcast_object
from src.training.logger import setup_logging
from src.training.params import parse_args
from src.training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from src.training.train import train_one_epoch, evaluate
from src.training.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from src.training.evaluate_clip_benchmark import evaluate_clip_benchmark
from src.training.pruning_manager import PruningManager

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

class WebDatasetWarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, text):
        if "Handling webdataset error" not in text:
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

    def isatty(self):
        return self.original_stderr.isatty()

    def fileno(self):
        return self.original_stderr.fileno()

    def readable(self):
        return self.original_stderr.readable()

    def writable(self):
        return self.original_stderr.writable()

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def ignore_warnings():
    """忽略warning信息"""
    # 设置日志级别
    sys.stderr = WebDatasetWarningFilter(sys.stderr)
    # logging.getLogger('webdataset').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    # 如果需要,可以为特定的库设置更详细的日志级别
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torchvision").setLevel(logging.ERROR)
    # 对于webdataset特定的警告
    

def cleanup_pruning_resources(data, pruning_manager):
    """在剪枝完成后清理资源，不影响训练过程
    Args:
        data: 数据字典
        pruning_manager: 剪枝管理器实例
    """
    try:
        # 1. 清理数据加载器
        if 'train' in data:
            loader = data['train'].dataloader
            if hasattr(loader, '_iterator'):
                loader._iterator = None
            
        # 2. 删除数据引用
        del data
        
        # 3. 基础的内存回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logging.info("Cleaned up resources after pruning")
            
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        # 确保基本的内存回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main(args):
    args = parse_args(args)
    # 忽略warning信息
    ignore_warnings()

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs_dir, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs_dir, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs_dir, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )

    # 执行一次性剪枝（在DDP和梯度检查点之前）
    pruning_manager = None
    if args.do_pruning:
        logging.info(f"Initializing pruning manager with mode: {args.pruning_mode}")
        try:
            # 创建示例输入
            image = torch.randn(1, 3, args.force_image_size, args.force_image_size, device=device)
            text = torch.randint(0, 100, (1, 77), device=device)
            
            # 确保模型处于训练模式并启用梯度
            model.eval()
            
            # 创建损失函数
            criterion = create_loss(args)
            
            # initialize datasets
            tokenizer = get_tokenizer(args.model)
            data = get_data(
                args,
                (preprocess_train, preprocess_val),
                epoch=0,
                tokenizer=tokenizer,
            )
            assert len(data), 'At least one train or eval dataset must be specified.'

            # 初始化剪枝管理器
            pruning_manager = PruningManager(
                model=model,
                config=vars(args),
                example_inputs=(image, text),
                data=data  # 传入数据集
            )
            
            # 如果是训练前剪枝模式，立即执行剪枝
            if args.pruning_mode == 'pre_training':
                logging.info("Performing one-time pruning before training...")
                pruning_info = pruning_manager.step(criterion=criterion)
                if pruning_info:
                    logging.info(f"Pruning completed successfully: {pruning_info}")
                    logging.info("Pruned Model:")
                    logging.info(f"{str(model)}")
                else:
                    logging.info("Model:")
                    logging.info(f"{str(model)}")
                    logging.warning("No pruning was performed")
                    
            # 完整清理资源
            cleanup_pruning_resources(data, pruning_manager)
                
        except Exception as e:
            logging.error(f"Failed to initialize/perform pruning: {e}")
            # 确保在发生错误时也清理资源
            cleanup_pruning_resources(data, pruning_manager)
            if args.distributed:
                torch.distributed.barrier()
            raise

    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        # logging.info("Model:")
        # logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs_dir, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        # tag: ddp+torch.compile相关配置
        if args.torchcompile:
            # if args.ddp_optimizer == "python_reducer":
            # 去掉训练速度会变慢 https://github.com/pytorch/pytorch/issues/109774#issuecomment-2046633776
            torch._dynamo.config.optimize_ddp = "python_reducer"
            # 忽略编译错误，回退到eager模式（主要是backward部分报错，暂未找到解决办法）
            # torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.compiled_autograd = True
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)
    else:
        if args.torchcompile:
            # 忽略编译错误，回退到eager模式（主要是backward部分报错，暂未找到解决办法）
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.compiled_autograd = True
            torch._dynamo.config.verbose = True

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cuda')
        
        # 根据checkpoint内容判断是否是剪枝后的模型
        is_pruned_checkpoint = 'model' in checkpoint and isinstance(checkpoint['model'], torch.nn.Module)
        
        if is_pruned_checkpoint:
            # 如果是剪枝后的模型，强制设置do_pruning为True
            if not args.do_pruning:
                logging.info("Detected pruned model checkpoint, automatically enabling pruning mode")
                args.do_pruning = True
            
            # 加载完整模型
            model = checkpoint['model']
            
            # 加载其他状态
            start_epoch = checkpoint["epoch"]
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if pruning_manager is not None and 'pruning_state' in checkpoint:
                pruning_manager.load_state_dict(checkpoint['pruning_state'])
            logging.info(f"=> resuming pruned model checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # 非剪枝模型的加载方式
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint["epoch"]
                    # 处理不同格式的state_dict保存方式
                    if 'state_dict' in checkpoint:
                        sd = checkpoint["state_dict"]
                    elif 'model' in checkpoint:
                        sd = checkpoint["model"].state_dict()
                    else:
                        sd = checkpoint
                    
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    
                    if optimizer is not None and 'optimizer' in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                    if pruning_manager is not None and 'pruning_state' in checkpoint:
                        pruning_manager.load_state_dict(checkpoint['pruning_state'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
                else:
                    # 处理只包含模型权重的checkpoint
                    if 'state_dict' in checkpoint:
                        sd = checkpoint["state_dict"]
                    elif 'model' in checkpoint:
                        sd = checkpoint["model"].state_dict()
                    else:
                        sd = checkpoint
                        
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # 处理直接保存的state_dict
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs_dir and args.logs_dir.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)
        
    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        
        if args.benchmark_data:
            #tag: evaluate with clip_benchmark
            evaluate_clip_benchmark(
                model=model, 
                transform=preprocess_val, 
                tokenizer=tokenizer, 
                epoch=start_epoch, 
                args=args,
                tb_writer=writer,
                evaluate_imagenet=args.evaluate_imagenet,
                evaluate_flickr=args.evaluate_flickr,
                evaluate_mscoco=args.evaluate_mscoco
            )
        else:
        # Evaluate.
            evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    # 开始训练循环
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        
        # 更新剪枝管理器的epoch信息
        if pruning_manager is not None:
            pruning_manager.config['current_epoch'] = epoch
            if is_master(args):
                if pruning_manager.config.get('pruning_done', False):
                    logging.info("Model has been fully pruned, continuing with fine-tuning...")
                else:
                    logging.info(f"Epoch {epoch}: Current pruning ratio: {pruning_manager.get_current_pruning_ratio():.4f}")
        
        train_one_epoch(
            model=model, 
            data=data, 
            loss=loss, 
            epoch=epoch, 
            optimizer=optimizer, 
            scaler=scaler, 
            scheduler=scheduler, 
            dist_model=dist_model, 
            args=args, 
            pruning_manager=pruning_manager,
            tb_writer=writer
        )
        
        completed_epoch = epoch + 1
        
        # 检查是否需要保存剪枝状态
        if args.save_logs and pruning_manager is not None:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "pruning_state": pruning_manager.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()
                
            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            
            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)
        
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

    # 训练完成后，使用 clip-benchmark 进行最终评估
    if args.benchmark_data:
        if is_master(args):
            logging.info("Training completed. Running final evaluation with clip-benchmark...")
            
            # 保存原始精度设置
            original_dtype = next(model.parameters()).dtype
            
            # 临时将模型转换为 FP32 进行评估
            model = model.cuda().to(dtype=torch.float32)
            
            evaluate_clip_benchmark(
                model=model, 
                transform=preprocess_val, 
                tokenizer=tokenizer, 
                epoch=completed_epoch, 
                args=args,
                tb_writer=writer, 
                train_loader=data['train'].dataloader,
                evaluate_imagenet=args.evaluate_imagenet,
                evaluate_flickr=args.evaluate_flickr,
                evaluate_mscoco=args.evaluate_mscoco
            )
            
            # 恢复原始精度
            model = model.to(dtype=original_dtype)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs_dir, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs_dir, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])

