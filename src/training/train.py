import json
import logging
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import copy

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ConvergenceTracker:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.loss_history = []
        self.step_history = []
        self.initial_loss = None
        
    def update(self, loss, step):
        if self.initial_loss is None:
            self.initial_loss = loss
            
        self.loss_history.append(loss)
        self.step_history.append(step)
        
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.step_history.pop(0)
    
    def get_metrics(self):
        if len(self.loss_history) < 2:  # 至少需要2个点才能计算
            return {
                "convergence/initial_descent_rate": 0.0,
                "convergence/recent_descent_rate": 0.0,
                "convergence/relative_improvement": 0.0
            }
            
        # 计算初始下降速率
        early_idx = min(len(self.loss_history) - 1, max(int(len(self.loss_history) * 0.2), 2))
        initial_descent = (self.loss_history[0] - self.loss_history[early_idx-1]) / (self.step_history[early_idx-1] - self.step_history[0])
        
        # 计算最近的收敛速度
        window = min(self.window_size, len(self.loss_history))
        if window < 2:
            recent_descent = 0.0
        else:
            recent_descent = (self.loss_history[-window] - self.loss_history[-1]) / (self.step_history[-1] - self.step_history[-window])
        
        # 计算相对改善程度
        relative_improvement = (self.initial_loss - self.loss_history[-1]) / self.initial_loss if self.initial_loss is not None else 0.0
            
        return {
            "convergence/initial_descent_rate": initial_descent,
            "convergence/recent_descent_rate": recent_descent,
            "convergence/relative_improvement": relative_improvement
        }

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()
    
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    
    # 初始化收敛跟踪器和梯度范数计量器
    if not hasattr(model, 'convergence_tracker'):
        model.convergence_tracker = ConvergenceTracker(window_size=50)
    grad_norm_m = AverageMeter()

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        
        if args.use_upop:
            # 搜索阶段
            if epoch < args.search_epochs:
                # 打印当前稀疏度
                if i % 100 == 0 and hasattr(model, 'get_sparsity_info'):
                    sparsity = model.get_sparsity_info()
                    print(f"Current sparsity: {sparsity}")
            
            # 搜索结束后压缩
            elif epoch == args.search_epochs and i == 0:
                print("Search completed. Compressing model...")
                # 创建新模型用于训练
                new_model = copy.deepcopy(model)
                new_model.compress(model)
                model = new_model
                print("Model compressed. Starting training phase...")
                
            # 压缩后的训练阶段
            else:
                pass  # 正常训练
                
        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch[:2]

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        if args.dataset_reinforcement and not args.dataset_reinforcement_mix_synthetic:
            syn_texts = batch[4].to(device=device, non_blocking=True)
            texts = torch.cat([texts, syn_texts[:, :texts.shape[-1]]], dim=0)

        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        
        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                if args.dataset_reinforcement:
                    batch_size = images.shape[0]
                    model_out.update({
                        'dist_image_features': batch[2].to(device=device, non_blocking=True),
                        'dist_text_features': batch[3].to(device=device, non_blocking=True),
                    })
                    if not args.dataset_reinforcement_mix_synthetic:
                        model_out.update({
                            "text_features": model_out["text_features"][:batch_size],
                            "syn_text_features": model_out["text_features"][batch_size:],
                            'dist_syn_text_features': batch[5].to(device=device, non_blocking=True)
                        })
                losses = loss(**model_out, output_dict=True)
                task_loss = sum(losses.values())
                
                # 在搜索阶段添加稀疏度损失
                if args.use_upop and epoch < args.search_epochs:
                    sparsity_loss = model.get_sparsity_loss() * args.w_sp_mlp
                    total_loss = task_loss + sparsity_loss
                    losses["sparsity_loss"] = sparsity_loss
                else:
                    total_loss = task_loss
                    
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    task_loss = sum(losses.values())
                    
                    # 在搜索阶段添加稀疏度损失
                    if args.use_upop and epoch < args.search_epochs:
                        sparsity_loss = model.get_sparsity_loss() * args.w_sp_mlp
                        total_loss = task_loss + sparsity_loss
                        losses["sparsity_loss"] = sparsity_loss
                    else:
                        total_loss = task_loss
                        
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        grad_norm_m.update(grad_norm)

        # 在优化器步骤之前更新 alpha 参数
        if args.use_upop and epoch < args.search_epochs:
            if hasattr(model, 'update_alpha_parameters'):
                model.update_alpha_parameters(step)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            
            convergence_metrics = model.convergence_tracker.get_metrics()
            
            # 添加稀疏度信息到日志
            if args.use_upop and epoch < args.search_epochs and hasattr(model, 'get_sparsity_info'):
                sparsity = model.get_sparsity_info()
                sparsity_info = f"Sparsity: {sparsity:.3f} "
            else:
                sparsity_info = ""
            
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:.5e} "
                f"Logit Scale: {logit_scale_scalar:.3f} " 
                f"Grad Norm: {grad_norm_m.val:.5f} "
                f"Descent Rate: {convergence_metrics.get('convergence/recent_descent_rate', 0):.5e} "
                f"{sparsity_info}"
                + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})
            
            # 添加稀疏度信息到tensorboard
            if args.use_upop and epoch < args.search_epochs and hasattr(model, 'get_sparsity_info'):
                log_data.update({
                    "sparsity": sparsity
                })
            
            log_data.update({
                "grad_norm": grad_norm_m.val,
                **convergence_metrics
            })

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        # 在optimizer.step()之后添加
        model.convergence_tracker.update(total_loss.item(), step)

    # end for

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    # # 添加debug信息
    # def print_shape_hook(module, input, output):
    #     print(f"Layer: {module.__class__.__name__}, Input shape: {input[0].shape}, Output shape: {output.shape}")
    
    # # 注册hook
    # hooks = []
    # for name, module in model.named_modules():
    #     if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
    #         hooks.append(module.register_forward_hook(print_shape_hook))

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # if i == 0:  # 只打印第一个batch的信息
                #     print("\n=== Batch 内容调试信息 ===")
                #     print(f"Batch type: {type(batch)}")
                #     print(f"Batch length: {len(batch)}")
                #     for idx, item in enumerate(batch):
                #         print(f"\nItem {idx}:")
                #         print(f"Type: {type(item)}")
                #         if torch.is_tensor(item):
                #             print(f"Shape: {item.shape}")
                #             print(f"Device: {item.device}")
                #         print(f"Content: {item}")
                #     print("=== 调试信息结束 ===\n")
                images, texts = batch[0], batch[1]
                logging.info(f"Loaded image batch shape: {images.shape}")
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)