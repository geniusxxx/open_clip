import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import numpy as np
import torch
import torch.nn.functional as F
import torch_pruning as tp
import timm
import argparse
import open_clip

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help='model name')
    parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'taylor', 'l2', 'l1', 'hessian'])
    parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--prune_num_heads', default=False, action='store_true', help='global pruning')
    parser.add_argument('--head_pruning_ratio', default=0.0, type=float, help='head pruning ratio')

    args = parser.parse_args()
    return args

# Here we re-implement the forward function of timm.models.vision_transformer.Attention
# as the original forward function requires the input and output channels to be identical.
def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_inputs = torch.randn(1,3,256,256)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == 'l2':
        imp = tp.importance.GroupNormImportance(p=2)
    elif args.pruning_type == 'l1':
        imp = tp.importance.GroupNormImportance(p=1)
    elif args.pruning_type == 'hessian':
        imp = tp.importance.GroupHessianImportance()
    else: raise NotImplementedError


    # Load the model
    model_name = "MobileCLIP-S1"
    pretrained = "/home/xuboyu/Projects/CLIP/test_mobileclip/ml-mobileclip/outputs/checkpoints/mobileclip_s1/open_clip_pytorch_model.bin"
            
    print("\n=== 创建模型 ===")
    model, _, _ = open_clip.create_model_and_transforms(    
        model_name,
        pretrained=pretrained,
        precision="amp",
        device="cuda",
        output_dict=True,
    )
    # model = timm.create_model(model_name, pretrained=pretrained)
    
    # 创建示例输入
    image = torch.randn(1, 3, 256, 256).cuda()
    text = torch.randint(0, 100, (1, 77)).cuda()
    example_inputs = (image, text)
    # base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    print("Pruning %s..."%args.model_name)
    print(model)
    num_heads = {}
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, timm.models.fastvit.Attention):
            m.forward = forward.__get__(m, timm.models.fastvit.Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            num_heads[m.qkv] = m.num_heads 
        if args.bottleneck and isinstance(m, timm.models.fastvit.ConvMlp): 
            ignored_layers.append(m.fc2) # only prune the internal layers of FFN & Attention

    pruner = tp.pruner.MetaPruner(
        model, 
        example_inputs, 
        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        pruning_ratio=args.pruning_ratio, # target pruning ratio
        ignored_layers=ignored_layers,
        num_heads=num_heads, # number of heads in self attention
        prune_num_heads=args.prune_num_heads, # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=8
    )


    for i, g in enumerate(pruner.step(interactive=True)):
        g.prune()

    # Modify the attention head size and all head size aftering pruning
    head_id = 0
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            print("Head #%d"%head_id)
            print("[Before Pruning] Num Heads: %d, Head Dim: %d =>"%(m.num_heads, m.head_dim))
            m.num_heads = pruner.num_heads[m.qkv]
            m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            print("[After Pruning] Num Heads: %d, Head Dim: %d"%(m.num_heads, m.head_dim))
            print()
            head_id+=1

    print(model)

    print("----------------------------------------")
    print("Summary:")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    # print("Base MACs: %.2f G, Pruned MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9))
    # print("Base Params: %.2f M, Pruned Params: %.2f M"%(base_params/1e6, pruned_params/1e6))

    latency_mean, latency_std = tp.utils.benchmark.measure_latency(model, example_inputs=torch.randn(16,3,224,224).to(device), repeat=300)
    print("Latency: %.4f ms, Std: %.4f ms"%(latency_mean, latency_std))

    if args.save_as is not None:
        print("Saving the pruned model to %s..."%args.save_as)
        os.makedirs(os.path.dirname(args.save_as), exist_ok=True)
        model.zero_grad()
        torch.save(model, args.save_as)

if __name__=='__main__':
    main()