import json
import logging
import os

try:
    import wandb
except ImportError:
    wandb = None

import torch
from open_clip_train.distributed import is_master
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_retrieval as zsr
from clip_benchmark.metrics import zeroshot_classification as zsc


def create_webdataset(
        task, 
        transform, 
        data_root=None, 
        dataset_len=None, 
        batch_size=64, 
        num_workers=4
):
    data_folder = f"wds_{task.replace('/', '-')}"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        task="zeroshot_retrieval",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset_zsc(
        task, 
        model, 
        transform, 
        tokenizer, 
        data_root=None, 
        dataset_len=None, 
        batch_size=64, 
        num_workers=4, 
        device='cpu'
):
    """Evaluate CLIP model on classification task."""

    # Load data
    dataset, dataloader = create_webdataset(
        task, 
        transform, 
        data_root, 
        dataset_len, 
        batch_size, 
        num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
            zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    metrics = zsc.evaluate(
        model,
        dataloader,
        tokenizer,
        classnames,
        zeroshot_templates,
        device,
        amp=False
    )
    metrics['mean_per_class_recall'] = float(metrics['mean_per_class_recall'])

    return metrics


def evaluate_webdataset_zsr(
        task, 
        model, 
        transform, 
        tokenizer, 
        data_root=None, 
        dataset_len=None, 
        batch_size=32, 
        num_workers=4, 
        device='cpu'
):
    """Evaluate CLIP model on retrieval task."""

    # Load data
    dataset, dataloader = create_webdataset(
        task, 
        transform, 
        data_root, 
        dataset_len, 
        batch_size, 
        num_workers
    )

    # Evaluate
    metrics = zsr.evaluate(
        model,
        dataloader,
        tokenizer,
        recall_k_list=[1, 5, 10],
        device=device,
        amp=False
    )

    return metrics


def evaluate_clip_benchmark(
        model, 
        transform, 
        tokenizer, 
        epoch, 
        args, 
        tb_writer=None, 
        train_loader=None,
        evaluate_imagenet=False,
        evaluate_flickr=False,
        evaluate_mscoco=False
):
    metrics = {}
    log_data = {}
    if not is_master(args):
        return metrics

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()

    torch.cuda.empty_cache()

    if evaluate_imagenet:
        # eval imagenet1k
        val_metrics = evaluate_webdataset_zsc(
            task="imagenet1k", 
            model=model, 
            transform=transform, 
            tokenizer=tokenizer,
            data_root=args.benchmark_data, 
            batch_size=args.batch_size, 
            device=args.device
        )
        logging.info(
            f"Eval Epoch: {epoch} [benchmark/imagenet1k]\t"
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in val_metrics.items()])
        )
        metrics.update({
            "key": "imagenet1k",
        "dataset": "ImageNet 1k",
            "metrics": val_metrics
        })
        if args.save_logs:
            with open(os.path.join(args.checkpoint_path, f"results_{epoch}.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")
        log_data.update({"benchmark/imagenet1k/" + name: val for name, val in val_metrics.items()})

    if evaluate_flickr:
        # evaluate flickr30k
        val_metrics = evaluate_webdataset_zsr(
            task="flickr30k", 
            model=model, 
            transform=transform, 
            tokenizer=tokenizer,
            data_root=args.benchmark_data, 
            batch_size=args.batch_size, 
            device=args.device
        )
        logging.info(
            f"Eval Epoch: {epoch} [benchmark/flickr30k]\t"
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in val_metrics.items()])
        )
        metrics.update({
            "key": "retrieval/flickr_1k_test_image_text_retrieval",
        "dataset": "Flickr",
            "metrics": val_metrics
        })
        if args.save_logs:
            with open(os.path.join(args.checkpoint_path, f"results_{epoch}.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")
        log_data.update({"benchmark/flickr30k/" + name: val for name, val in val_metrics.items()})

    if evaluate_mscoco:
        # eval mscoco_captions
        val_metrics = evaluate_webdataset_zsr(
            task="mscoco_captions", 
            model=model, 
            transform=transform, 
            tokenizer=tokenizer,
            data_root=args.benchmark_data, 
            batch_size=args.batch_size, 
            device=args.device
        )
        logging.info(
            f"Eval Epoch: {epoch} [benchmark/mscoco_captions]\t"
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in val_metrics.items()])
        )
        metrics.update({
            "key": "retrieval/mscoco_2014_5k_test_image_text_retrieval",
            "dataset": "MSCOCO",
            "metrics": val_metrics
        })
        if args.save_logs:
            with open(os.path.join(args.checkpoint_path, f"results_{epoch}.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")
        log_data.update({"benchmark/mscoco_captions/" + name: val for name, val in val_metrics.items()})

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if train_loader:
            num_batches_per_epoch = train_loader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)
