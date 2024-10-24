#!/bin/bash

# 定义训练参数数组
params=(
    # 主要配置
    -m src.training.main
    --name mobileclip_custom_datacompdr12m_50k_baseline_$(date +%Y-%m-%d_%H-%M-%S)
    --model MobileCLIP-Custom
    
    # 保存和日志配置
    --save-frequency 1
    --save-most-recent
    --log-every-n-steps 20
    --report-to wandb
    --wandb-project-name mobileclip
    
    # 训练策略
    --ddp-static-graph
    --local-loss
    --gather-with-grad
    --grad-checkpointing
    --grad-clip-norm 1.0
    
    # 数据集配置
    --train-data "/mnt/shared_39/data/DataCompDR-12M/train/{00000000..00000004}.tar"
    --dataset-type webdataset
    --train-num-samples 54267
    --dataset-resampled
    --dataset-reinforcement
    --dataset-reinforcement-config "configs/datacompdr12m.json"
    
    # --benchmark-data "/home/xuboyu/Data/datasets"
    # 评估参数
    # --evaluate-flickr
    # --evaluate-mscoco
    # --evaluate-imagenet

    # 验证数据（默认注释）
    --val-data "/mnt/shared_39/data/DataCompDR-12M/train/00000005.tar"
    --val-num-samples 1000

    # 图像处理参数
    --image-mean 0.0 0.0 0.0
    --image-std 1.0 1.0 1.0
    --image-interpolation bilinear
    --force-image-size 256
    
    # 优化器参数
    --lr 5.0e-4
    --beta1 0.9
    --beta2 0.95
    --warmup 1000
    --wd 0.1
    --batch-size 336
    --epochs 20
    --lr-scheduler cosine
    --accum-freq 1
    
    # 蒸馏参数
    --distill-logit-scale 100
    --distill-loss-weights 0.0 1.0
    --distill-teacher-dimension 768 768
    --distill-average-after-softmax
    
    # 其他配置
    --precision amp_bf16
    --workers 4
    --pin-memory
    --seed 0
    --zeroshot-frequency 1
    
    # 数据增强（默认注释）
    # --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 use_timm=True
)

# 执行训练命令
python "${params[@]}"