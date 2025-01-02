import os
import numpy as np
import torch
import logging
import torch_pruning as tp
from src.open_clip import create_model_and_transforms
from src.training.pruning_mobileclip import PruningManager
import timm
import torch.nn as nn
import torch.nn.functional as F

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
            
        if isinstance(module, torch.nn.Conv2d):
            current_channels[name] = module.out_channels
                
        elif isinstance(module, torch.nn.Linear):
            current_channels[name] = module.out_features
                
    return current_channels

def print_model_structure(model, title="", original_channels=None):
    """打印模型结构，包括通道数变化"""
    print(f"\n=== {title} ===")
     
    for name, module in model.named_modules():
        if 'visual.trunk' in name:
            # 处理ConvMlp (MLP层)
            if isinstance(module, timm.models.fastvit.ConvMlp):
                if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                    print(f"{name}.fc1:")
                    orig_out = original_channels.get(f"{name}.fc1", module.fc1.out_channels) if original_channels else module.fc1.out_channels
                    print(f"  Channels: ({module.fc1.in_channels},{orig_out}) -> ({module.fc1.in_channels},{module.fc1.out_channels})")
                    print(f"{name}.fc2:")
                    orig_out = original_channels.get(f"{name}.fc2", module.fc2.out_channels) if original_channels else module.fc2.out_channels
                    print(f"  Channels: ({module.fc2.in_channels},{orig_out}) -> ({module.fc2.in_channels},{module.fc2.out_channels})")
            
            # 处理RepMixer (token mixer)
            elif isinstance(module, timm.models.fastvit.RepMixer):
                if hasattr(module, 'mixer') and module.mixer is not None:
                    print(f"{name}:")
                    mixer = module.mixer
                    if hasattr(mixer, 'conv_kxk') and mixer.conv_kxk is not None and len(mixer.conv_kxk) > 0:
                        conv = mixer.conv_kxk[0].conv
                        orig_channels = original_channels.get(f"{name}.mixer", conv.out_channels) if original_channels else conv.out_channels
                        print(f"  Channels: ({conv.in_channels},{orig_channels}) -> ({conv.in_channels},{conv.out_channels})")
            
            # 处理Attention (最后一个stage的attention层)
            elif isinstance(module, timm.models.fastvit.Attention):
                print(f"{name}:")
                if hasattr(module, 'qkv') and module.qkv is not None:
                    orig_features = original_channels.get(name, module.qkv.in_features) if original_channels else module.qkv.in_features
                    print(f"  Features: ({orig_features},{orig_features}) -> ({module.qkv.in_features},{module.qkv.out_features//3})")
            
            # 处理MobileOneBlock (stem和final_conv)
            elif isinstance(module, timm.models.fastvit.MobileOneBlock):
                if hasattr(module, 'conv_kxk') and module.conv_kxk is not None and len(module.conv_kxk) > 0:
                    if hasattr(module.conv_kxk[0], 'conv'):
                        conv = module.conv_kxk[0].conv
                        print(f"{name}:")
                        orig_channels = original_channels.get(name, conv.out_channels) if original_channels else conv.out_channels
                        print(f"  Channels: ({conv.in_channels},{orig_channels}) -> ({conv.in_channels},{conv.out_channels})")
            
            # 处理downsample层
            elif isinstance(module, timm.models.fastvit.PatchEmbed):
                if hasattr(module, 'proj') and module.proj is not None and len(module.proj) > 0:
                    if isinstance(module.proj[0], timm.models.fastvit.ReparamLargeKernelConv):
                        if hasattr(module.proj[0], 'large_conv') and hasattr(module.proj[0].large_conv, 'conv'):
                            conv = module.proj[0].large_conv.conv
                            print(f"{name}:")
                            orig_channels = original_channels.get(name, conv.out_channels) if original_channels else conv.out_channels
                            print(f"  Channels: ({conv.in_channels},{orig_channels}) -> ({conv.in_channels},{conv.out_channels})")

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

def print_pruning_details(model, title="模块剪枝详情"):
    """打印特定模块的剪枝详情"""
    print(f"\n=== {title} ===")
    
    for name, module in model.named_modules():
        # 监控ConvMlp结构
        if isinstance(module, timm.models.fastvit.ConvMlp):
            print(f"\nConvMlp {name}:")
            # 打印conv信息
            if hasattr(module, 'conv'):
                conv_module = module.conv.conv  # 获取实际的Conv2d层
                print(f"conv: in={conv_module.in_channels}, out={conv_module.out_channels}, groups={conv_module.groups}")
            
            # 打印fc1和fc2信息
            if hasattr(module, 'fc1'):
                print(f"fc1: in={module.fc1.in_channels}, out={module.fc1.out_channels}")
            if hasattr(module, 'fc2'):
                print(f"fc2: in={module.fc2.in_channels}, out={module.fc2.out_channels}")
        
        # 监控Attention结构
        if isinstance(module, timm.models.fastvit.Attention):
            print(f"\nAttention {name}:")
            if hasattr(module, 'qkv'):
                print(f"qkv: in={module.qkv.in_features}, out={module.qkv.out_features}")
                print(f"num_heads={module.num_heads}, head_dim={module.head_dim}")
            if hasattr(module, 'proj'):
                print(f"proj: in={module.proj.in_features}, out={module.proj.out_features}")

def main():
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model_name = "MobileCLIP-S1"
    pretrained = "/home/xuboyu/Projects/CLIP/test_mobileclip/ml-mobileclip/outputs/checkpoints/mobileclip_s1/open_clip_pytorch_model.bin"
    print("\n=== 创建模型 ===")
    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="amp_bf16",
        device="cuda",
        output_dict=True,
    )
    
    # 创建示例输入
    image = torch.randn(1, 3, 256, 256).cuda()
    text = torch.randint(0, 100, (1, 77)).cuda()
    example_inputs = (image, text)
    
    # 记录初始通道数（不打印）
    original_channels = get_model_channels(model)
    
    # 打印剪枝前的模块信息
    print_pruning_details(model, "剪枝前的模块信息")
    
    # 准备配置
    config = {
        'do_pruning': True,
        'pruning_type': 'hessian',  # 在这里指定pruning_type: 'l1', 'l2', 'random', 'taylor', 'hessian'
        'pruning_ratio': 0.5,
        'progressive_pruning': False,
        'pruning_start_epoch': 0,
        'pruning_end_epoch': None,
        'global_pruning': False,
        'round_to': 8,
        'bottleneck': True,
        'pruning_done': False,
        'isomorphic': False,
        'prune_num_heads': False,
        'prune_head_dims': True,
        'head_pruning_ratio': 0.5,
        'current_epoch': 0,
        'current_step': 0
    }
    
    # 初始化剪枝管理器
    pruning_manager = PruningManager(
        model=model,
        config=config,
        example_inputs=example_inputs
    )
    
    print("初始模型统计信息:")
    print(pruning_manager.get_model_stats())
    
    # 根据pruning_type执行不同的剪枝流程
    if config['pruning_type'] in ['taylor', 'hessian']:
        print(f"\n=== 测试 {config['pruning_type']} 剪枝方法 ===")
        # 执行带梯度累积的剪枝
        num_batches = 5  # 可以根据需要调整
        for batch_idx in range(num_batches):
            # 模拟一个batch的前向传播
            batch_size = 4
            images = torch.randn(batch_size, 3, 256, 256).cuda()
            texts = torch.randint(0, 100, (batch_size, 77)).cuda()
            
            # 模拟CLIP的输出
            with torch.enable_grad():
                output = model(images, texts)
                
                # 在最后一个batch执行剪枝
                if batch_idx == num_batches - 1:
                    pruning_info = pruning_manager.step(output)
                    print(f"\n剪枝后的模型统计信息:")
                    print(pruning_info)
    else:
        # 直接执行剪枝
        pruning_info = pruning_manager.step()
        if pruning_info:
            print(f"Pruning completed: {pruning_info}")
    
    print(model)
    
    # 打印剪枝后的模块信息
    print_pruning_details(model, "剪枝后的模块信息")
    
    # 打印剪枝后的模型结构（包含通道数变化）
    print_model_structure(model, "剪枝后的模型结构", original_channels)
    
    # 打印剪枝统计信息
    print_pruning_statistics(pruning_manager)
    
    # 验证模型前向传播
    print("\n=== 验证剪枝后的模型前向传播 ===")
    try:
        with torch.no_grad():
            output = model(*example_inputs)
            print("模型前向传播成功")
                
    except Exception as e:
        import traceback
        print(f"模型前向传播失败: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 