import os
import numpy as np
import torch
import logging
import torch_pruning as tp
from src.open_clip import create_model_and_transforms
from src.training.pruning_siglip import PruningManager
import timm
import torch.nn as nn

def get_model_channels(model):
    """获取模型当前的通道数
    Args:
        model: 模型
    Returns:
        current_channels: 记录当前通道数的字典
    """
    current_channels = {}
    for name, module in model.named_modules():
        # 只处理视觉编码器的层
        if not 'visual' in name:
            continue
            
        if isinstance(module, torch.nn.Linear):
            current_channels[name] = module.out_features
                
        elif hasattr(module, 'num_heads') and hasattr(module, 'qkv'):
            current_channels[name] = module.num_heads * module.head_dim
            current_channels[name+'_heads'] = module.num_heads
                
    return current_channels

def print_model_structure(model, title="", original_channels=None):
    """打印模型结构，包括通道数变化"""
    print(f"\n=== {title} ===")
     
    for name, module in model.named_modules():
        if 'visual.trunk' in name:
            if isinstance(module, timm.models.vision_transformer.Attention):
                print(f"{name}:")
                # 对于attention层，显示输入通道数到输出通道数的变化
                orig_channels = original_channels.get(name, module.qkv.in_features) if original_channels else module.qkv.in_features
                print(f"  Channels: ({orig_channels},{orig_channels}) -> ({module.qkv.in_features},{module.qkv.out_features//3})")
                print(f"  Heads: {module.num_heads} -> {module.num_heads}")
            elif isinstance(module, timm.models.vision_transformer.Mlp):
                if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                    print(f"{name}.fc1:")
                    orig_out = original_channels.get(f"{name}.fc1", module.fc1.out_features) if original_channels else module.fc1.out_features
                    print(f"  Channels: ({module.fc1.in_features},{orig_out}) -> ({module.fc1.in_features},{module.fc1.out_features})")
                    print(f"{name}.fc2:")
                    orig_out = original_channels.get(f"{name}.fc2", module.fc2.out_features) if original_channels else module.fc2.out_features
                    print(f"  Channels: ({module.fc2.in_features},{orig_out}) -> ({module.fc2.in_features},{module.fc2.out_features})")


def print_pruning_statistics(pruning_manager: PruningManager):
    """打印剪枝前后的计算量和参数量变化"""
    print("\n=== 剪枝效果 ===")
    # 获取总体统计
    base_macs, base_params = pruning_manager.base_macs, pruning_manager.base_params
    current_macs, current_params = pruning_manager.current_macs, pruning_manager.current_params
    print(f"Total - MACs: {base_macs/1e9:.2f}G -> {current_macs/1e9:.2f}G ({current_macs/base_macs*100:.1f}%)")
    print(f"Total - Params: {base_params/1e6:.2f}M -> {current_params/1e6:.2f}M ({current_params/base_params*100:.1f}%)")
    
    # 获取视觉编码器统计
    vision_macs, vision_params = pruning_manager.get_vision_encoder_stats()
    # 计算视觉编码器原始参数量（通过总参数量减去当前总参数量与视觉编码器参数量的差值）
    original_vision_params = vision_params + (base_params - current_params)
    # 计算视觉编码器原始MACs（因为MACs主要来自视觉部分）
    original_vision_macs = base_macs
    
    print(f"Vision - MACs: {original_vision_macs/1e9:.2f}G -> {vision_macs/1e9:.2f}G ({vision_macs/original_vision_macs*100:.1f}%)")
    print(f"Vision - Params: {original_vision_params/1e6:.2f}M -> {vision_params/1e6:.2f}M ({vision_params/original_vision_params*100:.1f}%)")

def main():
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model_name = "ViT-B-16-SigLIP-256"
    pretrained = "/home/xuboyu/Projects/CLIP/test_mobileclip/ml-mobileclip/outputs/checkpoints/vit_b_16_siglip_256/webli/open_clip_pytorch_model.bin"
    model_kwargs = {}
    model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
    model_kwargs['init_logit_bias'] = -10               
    print("\n=== 创建模型 ===")
    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="amp",
        device="cuda",
        output_dict=True,
        **model_kwargs, 
    )
    
    # 创建示例输入
    image = torch.randn(1, 3, 256, 256).cuda()
    text = torch.randint(0, 100, (1, 64)).cuda()
    example_inputs = (image, text)
    
    # 记录初始通道数（不打印）
    original_channels = get_model_channels(model)
    
    # 初始化剪枝管理器
    print("\n=== 初始化剪枝管理器 ===")
    config = {
        'do_pruning': True,
        'current_epoch': 0,
        'current_step': 0,
    }
    
    pruning_manager = PruningManager(
        model=model,
        config=config,
        example_inputs=example_inputs
    )
    
    # 执行剪枝
    print("\n=== 执行剪枝 ===")
    pruning_info = pruning_manager.step()
    if pruning_info:
        print(f"Pruning completed: {pruning_info}")
    
    print(model)

    # 打印剪枝后的模型结构（包含通道数变化）
    print_model_structure(model, "剪枝后的模型结构", original_channels)
    
    # 打印剪枝统计信息
    print_pruning_statistics(pruning_manager)
    
    # 验证模型前向传播
    print("\n=== 验证模型前向传播 ===")
    try:
        with torch.no_grad():
            output = model(*example_inputs)
            print("模型前向传播成功")
    except Exception as e:
        print(f"模型前向传播失败: {e}")

if __name__ == "__main__":
    main() 