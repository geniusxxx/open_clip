# num_gpus=1
# num_nodes=1
# ROLE_RANK=0

# torchrun --nproc_per_node $num_gpus --nnodes $num_nodes --node_rank $ROLE_RANK \
python \
    -m src.training.main \
    --name mobileclips1_datacompdr12m_50k_baseline_$(date +%Y-%m-%d_%H-%M-%S) \
    --model MobileCLIP-S1 \
    --save-frequency 1 \
    --save-most-recent \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --zeroshot-frequency 1 \
    --image-mean 0.0 0.0 0.0 \
    --image-std 1.0 1.0 1.0 \
    --image-interpolation bilinear \
    --train-data "/mnt/shared_39/data/DataCompDR-12M/train/{00000000..00000004}.tar" \
    --dataset-type webdataset \
    --train-num-samples 54267 \
    --lr 1.0e-5 \
    --beta1 0.99 \
    --beta2 0.998 \
    --warmup 1000 \
    --wd 0.05 \
    --batch-size 336\
    --epochs 20 \
    --precision amp_bf16 \
    --workers 4 \
    --lr-scheduler cosine \
    --accum-freq 1 \
    --force-image-size 256 \
    --log-every-n-steps 20 \
    --seed 0 \
    --dataset-resampled \
    --save-most-recent \
    --grad-clip-norm 1.0 \
    --report-to wandb \
    --wandb-project-name mobileclip \
    --dataset-reinforcement \
    --dataset-reinforcement-config configs/datacompdr12m.json \
    --distill-logit-scale 100 \
    --distill-loss-weights 0.0 1.0 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax \
    --benchmark-data "/home/xuboyu/Data/datasets" \
    --pin-memory \
    --evaluate-flickr \
    # --evaluate-mscoco \ 
    # --evaluate-imagenet \
    # --max_restarts=0 \
    # --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 use_timm=True \
    # --val-data '/mnt/shared_39/data/DataCompDR-12M/train/4.tar' \
