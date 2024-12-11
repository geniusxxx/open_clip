import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from open_clip import create_model_and_transforms
import os
from typing import List, Dict, Tuple
import torch.nn.functional as F
from collections import defaultdict
import argparse
from timm.models.fastvit import RepMixerBlock, AttentionBlock, ConvMlp
import webdataset as wds
import braceexpand
import math


def get_block_type(block: nn.Module) -> str:
    """判断block的类型
    
    Args:
        block: 需要判断类型的block
        
    Returns:
        block类型的字符串标识
    """
    block_type = type(block).__name__
    if "RepMixer" in block_type:
        return "repmixer"
    elif "Attention" in block_type:  # 使用更宽松的匹配来处理不同的attention block变体
        return "attention"
    else:
        return "unknown"


class FastVitRedundancyAnalyzer:
    """用于分析FastViT模型冗余度的分析器。

    该类实现了对FastViT模型的各种冗余度分析，包括：
    - 通道相关性分析
    - 空间相关性分析
    - 激活模式分析
    - BN层权重分析

    Attributes:
        model: 要分析的模型实例
        preprocess: 模型的预处理函数
        device: 运行设备（CPU或GPU）
        features: 存储中间特征的字典
        attention_maps: 存储注意力图的字典
        mlp_features: 存储MLP特征的字典
        accumulated_features: 累积的特征字典，用于跨batch分析
        batch_count: 已处理的batch数量
        log_file: 日志文件句柄
    """

    def __init__(self, model, preprocess, device='cuda'):
        """初始化FastViT冗余分析器。

        Args:
            model: 模型实例
            preprocess: 模型的预处理函数
            device: 运行设备，默认为'cuda'
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 使用模型原有的预处理
        self.preprocess = preprocess
        
        # 存储中间特征的字典
        self.features = defaultdict(dict)
        self.attention_maps = defaultdict(dict)
        self.mlp_features = defaultdict(dict)
        
        # 添加累积特征的字典
        self.accumulated_features = defaultdict(list)
        self.batch_count = 0
        
        # 日志文件句柄
        self.log_file = None
        
        # 注册hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook来获取中间层特征。

        为模型的各个层注册前向传播hook，用于收集：
        - RepMixer层的输入输出特征
        - Token mixer的输出特征
        - MLP的输出特征
        - Attention层的注意力图
        """
        def get_repmixer_feature(name):
            def hook(module, input, output):
                self.features[name]["input"] = input[0].detach()
                self.features[name]["output"] = output.detach()
                
                # 获取token_mixer的输出
                if hasattr(module, "token_mixer"):
                    token_mixer_out = module.token_mixer(input[0])
                    self.features[name]["token_mixer"] = token_mixer_out.detach()
                
                # 获取mlp的输出
                if hasattr(module, "mlp"):
                    mlp_out = module.mlp(output)  # 使用token_mixer的输出作为mlp的输入
                    self.mlp_features[name] = mlp_out.detach()
            return hook
            
        def get_attention_feature(name):
            def hook(module, input, output):
                self.features[name]["input"] = input[0].detach()
                self.features[name]["output"] = output.detach()
                
                if hasattr(module, "token_mixer"):
                    # 获取注意力图
                    x = input[0]  # [B, C, H, W]
                    B, C, H, W = x.shape
                    x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
                    
                    qkv = module.token_mixer.qkv(x)
                    qkv = qkv.reshape(B, -1, 3, module.token_mixer.num_heads, C // module.token_mixer.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, HW, head_dim]
                    q, k, v = qkv.unbind(0)
                    
                    # 计算注意力分数
                    attn = (q @ k.transpose(-2, -1)) * module.token_mixer.scale
                    attn = attn.softmax(dim=-1)
                    self.attention_maps[name] = attn.detach()
                
                if hasattr(module, "mlp"):
                    mlp_out = module.mlp(output)
                    self.mlp_features[name] = mlp_out.detach()
            return hook
        
        # 注册hooks
        for i, stage in enumerate(self.model.visual.trunk.stages):
            for j, block in enumerate(stage.blocks):
                block_type = get_block_type(block)
                if block_type == "repmixer":
                    block.register_forward_hook(get_repmixer_feature(f"stage_{i}_block_{j}_repmixer"))
                elif block_type == "attention":
                    block.register_forward_hook(get_attention_feature(f"stage_{i}_block_{j}_attention"))
                    self.log(f"注册attention hook: stage_{i}_block_{j}")
    
    def log(self, message: str):
        """记录日志信息到文件。

        Args:
            message: 要记录的日志信息
        """
        if self.log_file is not None:
            self.log_file.write(f"{message}\n")
            self.log_file.flush()
    
    def visualize_redundancy(self, results: Dict[str, Dict[str, torch.Tensor]], 
                           save_dir: str):
        """可视化冗余分析结果。

        为不同类型的分析结果生成可视化图表，包括相关性矩阵、激活分布等。

        Args:
            results: 包含各层分析结果的嵌套字典
            save_dir: 可视化结果的保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建一个文本文件来保存所有标量值
        scalar_file = os.path.join(save_dir, 'scalar_metrics.txt')
        with open(scalar_file, 'w') as f:
            f.write("=== Scalar Metrics ===\n\n")
            
            # 首先输出BN层的统计信息
            f.write("=== BN Layer Statistics ===\n\n")
            for layer_name, layer_results in results.items():
                if "_pre_bn" in layer_name:
                    f.write(f"\n{layer_name}:\n")
                    if 'zero_ratio' in layer_results:
                        f.write(f"  Zero ratio: {layer_results['zero_ratio'].item():.4f}\n")
                    if 'weight_mean' in layer_results:
                        f.write(f"  Mean: {layer_results['weight_mean'].item():.4f}\n")
                    if 'weight_std' in layer_results:
                        f.write(f"  Std: {layer_results['weight_std'].item():.4f}\n")
                    if 'weight_min' in layer_results:
                        f.write(f"  Min: {layer_results['weight_min'].item():.4f}\n")
                    if 'weight_max' in layer_results:
                        f.write(f"  Max: {layer_results['weight_max'].item():.4f}\n")
            
            # 然后输出其他标量值
            f.write("\n=== Other Metrics ===\n\n")
            for layer_name, layer_results in results.items():
                if "_pre_bn" not in layer_name:
                    for metric_name, matrix in layer_results.items():
                        if any(key in metric_name for key in ["effective_rank", "activation_pattern_diversity"]):
                            value = matrix.item()
                            f.write(f"{layer_name} - {metric_name}: {value:.4f}\n")
        
        # 批量处理可视化
        plot_batches = []
        for layer_name, layer_results in results.items():
            for metric_name, matrix in layer_results.items():
                # 跳过BN层的单值统计信息
                if "_pre_bn" in layer_name and metric_name in ['zero_ratio', 'weight_mean', 'weight_std', 'weight_min', 'weight_max']:
                    continue
                
                # 跳过已经写入文件的标量值
                if any(key in metric_name for key in ["effective_rank", "activation_pattern_diversity"]):
                    continue
                
                # 跳过没有意义的_std统计变量，但保留correlation的std
                if metric_name.endswith('_std') and not any(key in metric_name for key in ["channel_correlation", "spatial_correlation"]):
                    continue
                
                plot_batches.append((layer_name, metric_name, matrix))
        
        # 并行处理可视化（每批16个）
        batch_size = 16
        for i in range(0, len(plot_batches), batch_size):
            batch = plot_batches[i:i+batch_size]
            for layer_name, metric_name, matrix in batch:
                plt.close('all')
                plt.figure(figsize=(12, 10))
                
                # 将数据一次性移到CPU
                if torch.is_tensor(matrix):
                    matrix = matrix.detach().cpu()
                
                if metric_name == "attention_entropy":
                    plt.bar(range(len(matrix)), matrix.numpy())
                    plt.title(f"{layer_name} - {metric_name}")
                    plt.xlabel("Head Index")
                    plt.ylabel("Entropy")
                    
                elif "correlation" in metric_name:
                    matrix_np = matrix.numpy()
                    mask = np.zeros_like(matrix_np, dtype=bool)
                    np.fill_diagonal(mask, True)
                    
                    sns.heatmap(matrix_np, 
                              cmap='RdBu_r',
                              vmin=-1,
                              vmax=1,
                              center=0,
                              mask=mask,
                              square=True,
                              xticklabels=False,
                              yticklabels=False,
                              cbar_kws={'label': 'Correlation'})
                    plt.title(f"{layer_name} - {metric_name}")
                    
                elif metric_name in ["mean_activation", "std_activation", "channel_activation_frequency"]:
                    values = matrix.numpy()
                    plt.plot(range(len(values)), values, 'b-', linewidth=2)
                    plt.fill_between(range(len(values)), values, alpha=0.2)
                    plt.title(f"{layer_name} - {metric_name}")
                    plt.xlabel("Channel Index")
                    plt.ylabel("Value")
                    plt.grid(True, alpha=0.3)
                    
                elif "singular_values" in metric_name:
                    values = matrix.numpy()
                    plt.semilogy(range(len(values)), values, 'r-', linewidth=2)
                    plt.title(f"{layer_name} - {metric_name}")
                    plt.xlabel("Index")
                    plt.ylabel("Singular Value")
                    plt.grid(True)
                    
                elif "spatial_pattern" in metric_name:
                    pattern = matrix.numpy()
                    sns.heatmap(pattern, cmap='viridis', square=True)
                    plt.title(f"{layer_name} - {metric_name}")
                    
                elif "kernel_energy" in metric_name:
                    energy = matrix.numpy()
                    plt.imshow(energy, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Energy')
                    plt.title(f"{layer_name} - {metric_name}")
                    plt.xlabel("Output Channel")
                    plt.ylabel("Input Channel")
                    
                elif metric_name == "weight_histogram":
                    if isinstance(matrix, dict):
                        edges = matrix['edges'].cpu() if torch.is_tensor(matrix['edges']) else matrix['edges']
                        values = matrix['values'].cpu() if torch.is_tensor(matrix['values']) else matrix['values']
                        
                        # 创建更美观的权重分布直方图
                        plt.figure(figsize=(15, 10))
                        plt.bar(edges[:-1], values, 
                               width=edges[1]-edges[0],
                               align='edge',
                               alpha=0.7,
                               color='blue')
                        plt.title(f"{layer_name} - Weight Distribution")
                        plt.xlabel("Weight Value")
                        plt.ylabel("Count")
                        plt.grid(True, alpha=0.3)
                        
                        # 添加统计信息
                        if 'weight_mean' in layer_results and 'weight_std' in layer_results:
                            mean = layer_results['weight_mean'].item()
                            std = layer_results['weight_std'].item()
                            plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
                            plt.axvline(mean + std, color='green', linestyle=':', label=f'Mean ± Std')
                            plt.axvline(mean - std, color='green', linestyle=':')
                            plt.legend()
                        
                        # 设置更宽松的x轴范围
                        if 'weight_min' in layer_results and 'weight_max' in layer_results:
                            min_val = layer_results['weight_min'].item()
                            max_val = layer_results['weight_max'].item()
                            # 使用更大的边距，确保分布不会太紧凑
                            margin = (max_val - min_val) * 0.2  # 使用20%的数据范围作为边距
                            plt.xlim(min_val - margin, max_val + margin)
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"{layer_name}_{metric_name}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        self.log("可视化结果生成完成")
    
    def __del__(self):
        """析构函数，确保日志文件被正确关闭"""
        if self.log_file is not None:
            self.log_file.close()
    
    def process_batch(self, batch_idx: int, total_batches: int):
        """处理一个batch的数据
        
        Args:
            batch_idx: batch索引
            total_batches: 总batch数
        """
        # 只在终端显示进度信息
        print(f"\rProcessing batch {batch_idx+1}/{total_batches}", end="")
        if batch_idx == total_batches - 1:
            print()  # 最后一个batch后换行

    def analyze_redundancy_batch(self, images: torch.Tensor, save_dir: str, num_samples: int = None, processed_samples: int = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """分析一批图像的冗余度。

        对输入的一批图像进行前向传播，收集特征并进行冗余度分析。

        Args:
            images: 输入图像张量，形状为[B, C, H, W]
            save_dir: 结果保存目录
            num_samples: 总样本数（可选）
            processed_samples: 当前已处理的样本数（可选）

        Returns:
            包含各层分析结果的嵌套字典
        """
        # 初始化保存目录和日志文件
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        if self.log_file is None:
            log_path = os.path.join(save_dir, 'analysis.log')
            self.log_file = open(log_path, 'w')
        
        # 记录分析开始
        self.log("开始分析模型冗余度...")
        self.log(f"模型结构: {type(self.model).__name__}")
        
        # 确保图像在正确的设备上
        images = images.to(self.device)
        self.log(f"输入图像形状: {images.shape}")
        
        # 存储所有层的特征和分析结果
        features = {}
        mlp_layers = {}
        
        def hook_fn(name):
            def _hook(module, input, output):
                features[name] = output
                if isinstance(module, ConvMlp):
                    mlp_layers[name] = module
                # 累积特征
                if self.batch_count == 0:
                    self.accumulated_features[name] = []
                features_flat = output.view(output.size(0), output.size(1), -1)
                self.accumulated_features[name].append(features_flat.detach())
                self.log(f"提取特征: {name}, 形状: {output.shape}")
            return _hook
        
        # 注册钩子
        hooks = []
        
        # 注册hooks
        for i, stage in enumerate(self.model.visual.trunk.stages):
            for j, block in enumerate(stage.blocks):
                block_type = get_block_type(block)
                if block_type == "repmixer":
                    hooks.append(block.register_forward_hook(hook_fn(f"stage_{i}_block_{j}_repmixer")))
                elif block_type == "attention":
                    hooks.append(block.register_forward_hook(hook_fn(f"stage_{i}_block_{j}_attention")))
                    self.log(f"注册attention hook: stage_{i}_block_{j}")
        
        # 前向传播
        self.log("开始前向传播...")
        with torch.no_grad():
            _ = self.model(images)
        self.log("前向传播完成")
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        self.batch_count += 1
        
        # 如果是最后一个batch，计算累积结果
        if num_samples is not None and processed_samples is not None and processed_samples >= num_samples:  # 最后一个batch
            results = self._compute_accumulated_results(mlp_layers)
        else:  # 其他batch，返回初步结果
            results = self._compute_preliminary_results(features, mlp_layers)
        
        return results

    def _compute_preliminary_results(self, features: Dict[str, torch.Tensor], mlp_layers: Dict[str, nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
        """计算初步的分析结果"""
        results = {}
        for name, feat in features.items():
            self.log(f"分析层: {name}")
            
            # 基础特征分析
            layer_results = self.analyze_repmixer_redundancy(feat)
            
            # 激活频率分析
            activation_results = self.analyze_activation_frequency(feat)
            layer_results.update(activation_results)
            
            # 如果是MLP层，进行额外分析
            if name in mlp_layers:
                mlp = mlp_layers[name]
                # 权重矩阵分析
                weight_results = self.analyze_mlp_weights(mlp)
                layer_results.update(weight_results)
                # Kernel pattern分析
                kernel_results = self.analyze_kernel_pattern(mlp)
                layer_results.update(kernel_results)
            
            results[name] = layer_results
        
        return results

    def _compute_accumulated_results(self, mlp_layers: Dict[str, nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
        """计算累积的分析结果"""
        results = {}
        
        # 获取所有需要处理的层名称
        layer_names = list(self.accumulated_features.keys())
        
        for name in layer_names:
            self.log(f"计算累积结果: {name}")
            
            # 合并所有batch的特征
            accumulated_features = self.accumulated_features[name]
            all_features = torch.cat(accumulated_features, dim=0)
            
            # 基础特征分析
            layer_results = self.analyze_repmixer_redundancy(all_features)
            
            # 激活频率分析
            activation_results = self.analyze_activation_frequency(all_features)
            layer_results.update(activation_results)
            
            # 如果是MLP层，进行额外分析
            if name in mlp_layers:
                mlp = mlp_layers[name]
                # 权重矩阵分析
                weight_results = self.analyze_mlp_weights(mlp)
                layer_results.update(weight_results)
                # Kernel pattern分析
                kernel_results = self.analyze_kernel_pattern(mlp)
                layer_results.update(kernel_results)
            
            results[name] = layer_results
            
            # 清理当前层的累积特征以节省内存
            del self.accumulated_features[name]
            del accumulated_features
            del all_features
            torch.cuda.empty_cache()
        
        return results

    def analyze_repmixer_redundancy(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """分析RepMixer层的冗余度。

        计算特征图的通道相关性和空间相关性，以及其他统计指标。

        Args:
            features: 输入特征张量，形状为[B, C, H, W]或[B, C, HW]

        Returns:
            包含各种冗余度指标的字典，包括：
            - channel_correlation: 通道间相关性矩阵
            - spatial_correlation: 空间位置间相关性矩阵
            - mean_activation: 平均激活值
            - std_activation: 激活值标准差
        """
        results = {}
        
        with torch.amp.autocast('cuda', enabled=True):
            # 1. Channel维度的冗余分析
            if len(features.shape) == 4:
                features_flat = features.view(features.size(0), features.size(1), -1)  # [B, C, HW]
            else:
                features_flat = features  # 如果已经是展平的特征
            
            # 先在空间维度上平均，得到每个样本每个通道的平均激活
            features_mean = features_flat.mean(dim=2)  # [B, C]
            
            # 对每个通道的激活进行标准化
            features_std = features_flat.std(dim=2)  # [B, C]
            features_normalized = (features_mean - features_mean.mean(dim=0, keepdim=True)) / (features_std.mean(dim=0, keepdim=True) + 1e-5)  # [B, C]
            
            # 计算通道间的相关性
            if features_normalized.size(0) > 1:
                # 先计算每个样本对每个通道的贡献
                channel_activations = features_normalized  # [B, C]
                # 然后计算通道间的相关性
                channel_corr = torch.corrcoef(channel_activations.T)  # [C, C]
            else:
                # 如果只有一个样本，使用空间位置作为样本
                spatial_features = features_flat[0]  # [C, HW]
                spatial_features = (spatial_features - spatial_features.mean(dim=1, keepdim=True)) / (spatial_features.std(dim=1, keepdim=True) + 1e-5)
                channel_corr = torch.corrcoef(spatial_features)
            
            results["channel_correlation"] = channel_corr
            
            # 2. 空间维度的冗余分析（仅在需要时计算）
            if features_flat.size(1) > 1:
                spatial_features = features_flat.permute(0, 2, 1)  # [B, HW, C]
                spatial_features = (spatial_features - spatial_features.mean(dim=2, keepdim=True)) / (spatial_features.std(dim=2, keepdim=True) + 1e-5)
                spatial_corr = torch.corrcoef(spatial_features[0])  # 使用第一个样本的空间相关性
                results["spatial_correlation"] = spatial_corr
            
            # 3. 计算特征响应的统计信息
            results["mean_activation"] = features_mean.mean(dim=0)  # 在batch维度上平均
            results["std_activation"] = features_flat.reshape(-1, features_flat.size(1)).std(dim=0)
            
            # 确保所有结果都是有效的tensor
            for key in results:
                if torch.isnan(results[key]).any():
                    self.log(f"Warning: NaN values detected in {key}")
                    results[key] = torch.zeros_like(results[key])
                if torch.isinf(results[key]).any():
                    self.log(f"Warning: Inf values detected in {key}")
                    results[key] = torch.zeros_like(results[key])
        
        return results

    def analyze_mlp_weights(self, layer: ConvMlp) -> Dict[str, torch.Tensor]:
        """分析MLP层权重矩阵的特性。

        计算权重矩阵的奇异值分解和有效秩。

        Args:
            layer: 要分析的ConvMlp层实例

        Returns:
            包含权重分析结果的字典，包括：
            - fc1_singular_values: fc1层的奇异值
            - fc1_effective_rank: fc1层的有效秩
            - fc2_singular_values: fc2层的奇异值
            - fc2_effective_rank: fc2层的有效秩
        """
        results = {}
        
        with torch.amp.autocast('cuda', enabled=True):
            # 分析fc1权重
            fc1_weight = layer.fc1.weight.reshape(layer.fc1.weight.size(0), -1)
            # 使用更快的SVD算法
            if fc1_weight.size(0) > fc1_weight.size(1):
                fc1_weight = fc1_weight.T
            fc1_s = torch.linalg.svdvals(fc1_weight)
            
            # 计算有效秩（使用香农熵方法）
            # 1. 用最大奇异值归一化
            fc1_s_norm = fc1_s / fc1_s[0]
            # 2. 计算能量分布
            fc1_p = fc1_s_norm / fc1_s_norm.sum()
            # 3. 计算香农熵
            fc1_entropy = -(fc1_p * torch.log(fc1_p + 1e-10)).sum()
            # 4. 计算有效秩
            fc1_effective_rank = torch.exp(fc1_entropy)
            
            results['fc1_singular_values'] = fc1_s
            results['fc1_effective_rank'] = fc1_effective_rank
            
            # 分析fc2权重
            fc2_weight = layer.fc2.weight.reshape(layer.fc2.weight.size(0), -1)
            if fc2_weight.size(0) > fc2_weight.size(1):
                fc2_weight = fc2_weight.T
            fc2_s = torch.linalg.svdvals(fc2_weight)
            
            # 同样计��fc2的有效秩
            # 1. 用最大奇异值归一化
            fc2_s_norm = fc2_s / fc2_s[0]
            # 2. 计算能量分布
            fc2_p = fc2_s_norm / fc2_s_norm.sum()
            # 3. 计算香农熵
            fc2_entropy = -(fc2_p * torch.log(fc2_p + 1e-10)).sum()
            # 4. 计算有效秩
            fc2_effective_rank = torch.exp(fc2_entropy)
            
            results['fc2_singular_values'] = fc2_s
            results['fc2_effective_rank'] = fc2_effective_rank
        
        return results

    def analyze_kernel_pattern(self, layer: ConvMlp) -> Dict[str, torch.Tensor]:
        """分析深度卷积的kernel pattern。

        计算卷积核的空间模式和能量分布。

        Args:
            layer: 要分析的ConvMlp层实例

        Returns:
            包含kernel分析结果的字典，包括：
            - spatial_pattern: 空间模式
            - kernel_energy: kernel能量分布
        """
        results = {}
        
        with torch.amp.autocast('cuda', enabled=True):
            # 分析conv权重
            conv_weight = layer.conv.conv.weight
            # 计算空间pattern（保持在GPU上）
            spatial_pattern = conv_weight.mean(dim=(0,1))
            results['spatial_pattern'] = spatial_pattern
            
            # 计算kernel的能量分布
            kernel_energy = torch.norm(conv_weight, dim=(2,3))
            results['kernel_energy'] = kernel_energy
        
        return results

    def analyze_activation_frequency(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """分析特征图的激活频率。

        计算各通道的激活频率和激活模式多样性。

        Args:
            features: 输入特征张量，形状为[B, C, H, W]或[B, C, HW]

        Returns:
            包含激活分析结果的字典，包括：
            - channel_activation_frequency: 通道激活频率
            - activation_pattern_diversity: 激活模式多样性
        """
        results = {}
        
        with torch.amp.autocast('cuda', enabled=True):
            # 避免创建新的tensor，直接在原tensor上操作
            features_flat = features.view(features.size(0), features.size(1), -1)
            activation_freq = (features_flat > 0).float().mean(dim=(0,2))
            results['channel_activation_frequency'] = activation_freq
            
            # 优化激活模式多样性计算
            if features_flat.size(0) * features_flat.size(2) > 10000:
                # 如果样本太多，随机采样
                idx = torch.randperm(features_flat.size(0) * features_flat.size(2))[:10000]
                activation_patterns = (features_flat.view(-1, features_flat.size(1))[idx] > 0).float()
            else:
                activation_patterns = (features_flat.view(-1, features_flat.size(1)) > 0).float()
            
            pattern_diversity = torch.unique(activation_patterns, dim=0).size(0)
            results['activation_pattern_diversity'] = torch.tensor(pattern_diversity, device=features.device)
        
        return results

    def save_feature_analysis(self, features: torch.Tensor, save_path: str):
        """将特征值分布分析结果保存到文件。

        计算并保存特征值的统计信息和分布特征。

        Args:
            features: 要分析的特征张量
            save_path: 保存路径
        """
        with open(save_path, 'a') as f:
            f.write(f"\n=== 特征值分布分析 ===\n")
            f.write(f"特征形状: {features.shape}\n")
            f.write(f"最小值: {features.min().item():.6f}\n")
            f.write(f"最大值: {features.max().item():.6f}\n")
            f.write(f"平均值: {features.mean().item():.6f}\n")
            f.write(f"标准差: {features.std().item():.6f}\n")
            
            f.write("\n不同阈值下的稀疏度：\n")
            features_flat = features.contiguous().view(-1)
            thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            for threshold in thresholds:
                sparsity_t = (features_flat.abs() < threshold).float().mean()
                f.write(f"阈值 {threshold:.0e}: {sparsity_t.item():.6f}\n")
            
            f.write("\n数值分布区间：\n")
            percentiles = torch.tensor([0, 25, 50, 75, 100], device=features.device)
            values = torch.quantile(features_flat.abs(), percentiles.float() / 100)
            for p, v in zip(percentiles, values):
                f.write(f"{p}%: {v.item():.6f}\n")
            f.write("\n")

    def save_results(self, results: Dict[str, Dict[str, torch.Tensor]], 
                    save_dir: str):
        """保存分析结果。

        将所有分析结果保存到指定目录。

        Args:
            results: 包含各层分析结果的嵌套字典
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细的数值结果
        with open(os.path.join(save_dir, "detailed_results.txt"), "w") as f:
            for layer_name, layer_results in results.items():
                f.write(f"\n{layer_name} Statistics:\n")
                
                for metric_name, values in layer_results.items():
                    if metric_name == "channel_correlation" or metric_name == "feature_correlation":
                        values_np = values.cpu().numpy()
                        mean_corr = np.mean(values_np)
                        max_corr = np.max(values_np)
                        num_redundant = np.sum(values_np > 0.5)
                        
                        f.write(f"  {metric_name}:\n")
                        f.write(f"    Mean correlation: {mean_corr:.4f}\n")
                        f.write(f"    Max correlation: {max_corr:.4f}\n")
                        f.write(f"    Number of redundant pairs (>0.5): {num_redundant}\n")
                        
                    elif metric_name == "spatial_correlation":
                        if isinstance(values, torch.Tensor):
                            values_np = values.cpu().numpy()
                            mean_corr = np.mean(values_np)
                            max_corr = np.max(values_np)
                            num_redundant = np.sum(values_np > 0.5)
                            
                            f.write(f"  {metric_name}:\n")
                            f.write(f"    Mean correlation: {mean_corr:.4f}\n")
                            f.write(f"    Max correlation: {max_corr:.4f}\n")
                            f.write(f"    Number of redundant pairs (>0.5): {num_redundant}\n")
                            
                    elif metric_name == "activation_sparsity":
                        f.write(f"  {metric_name}: {values.item():.4f}\n")
                    
                    # 添加对BN层前权重分析结果的处理
                    elif metric_name in ["zero_ratio", "weight_mean", "weight_std", "weight_abs_mean"]:
                        value = values.item() if torch.is_tensor(values) else values
                        f.write(f"  {metric_name}: {value:.4f}\n")
                    
                f.write("\n")
        
        self.log(f"分析结果已保存到: {save_dir}")

    def save_consistency_check(self, features_dict: Dict[str, torch.Tensor], save_path: str):
        """保存数据一致性检��结果。

        检查并记录特征数据的维度和数值范围。

        Args:
            features_dict: 包含各层特征的字典
            save_path: 保存路径
        """
        with open(save_path, 'w') as f:
            f.write("=== 数据一致性检查 ===\n\n")
            
            # 1. 特征维度检查
            f.write("1. 特征维度检查:\n")
            for name, feat in features_dict.items():
                f.write(f"{name}: {feat.shape}\n")
            f.write("\n")
            
            # 2. 数值范围检查
            f.write("2. 数值范围检查:\n")
            for name, feat in features_dict.items():
                f.write(f"{name}:\n")
                f.write(f"  Min: {feat.min().item():.4f}\n")
                f.write(f"  Max: {feat.max().item():.4f}\n")
                f.write(f"  Mean: {feat.mean().item():.4f}\n")
                f.write(f"  Std: {feat.std().item():.4f}\n")
                f.write("\n")
            
            # 记录到日志
            self.log(f"数据一致性检查结果已保存到: {save_path}")

    def analyze_attention_redundancy(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """分析Attention层的冗余度。

        计算注意力特征的相关性和稀疏度。

        Args:
            features: 输入特征张量，形状为[B, C, H, W]

        Returns:
            包含注意力分析结果的字典，包括：
            - feature_correlation: 特征相关性矩阵
            - activation_sparsity: 激活稀疏度
        """
        self.log(f"分析Attention特征，形状: {features.shape}")
        results = {}
        
        # 1. 特征激活的相关性
        features_flat = features.contiguous().view(features.size(0), features.size(1), -1)  # [B, C, HW]
        features_mean = features_flat.mean(dim=2)  # [B, C]
        
        # 保存特征值分布分析到文件
        if hasattr(self, 'save_dir'):
            analysis_file = os.path.join(self.save_dir, 'feature_analysis.txt')
            self.save_feature_analysis(features, analysis_file)
        
        # 添加一个小的扰动以避免数值问题
        features_mean = features_mean + torch.randn_like(features_mean) * 1e-8
        
        # 计算特征间的相关性
        if features_mean.size(0) > 1:  # 如果batch size > 1
            feature_corr = torch.corrcoef(features_mean.t())  # [C, C]
        else:
            # 如果batch size = 1，使用空间维度作为样本
            feature_corr = torch.corrcoef(features_flat[0])  # [C, C]
        results["feature_correlation"] = feature_corr
        
        # 2. 激活值的稀疏度
        sparsity = (features_flat.abs() < 1e-6).float().mean()
        results["activation_sparsity"] = sparsity
        
        self.log(f"Attention分析完成，���征相关性形状: {feature_corr.shape}, 稀疏度: {sparsity.item():.4f}")
        
        return results

    def analyze_pre_bn_weights(self, layer: nn.Module) -> Dict[str, torch.Tensor]:
        """分析BN层之前的权重矩阵。

        计算权重的基本统计信息和分布特征。

        Args:
            layer: 要分析的层实例

        Returns:
            包含权重分析结果的字典，包括：
            - zero_ratio: 零值比例
            - weight_mean: 权重平均值
            - weight_std: 权重标准差
            - weight_histogram: 权重直方图数据
        """
        results = {}
        
        with torch.amp.autocast('cuda', enabled=True):
            # 获取权重
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight
            elif isinstance(layer, nn.Linear):
                weight = layer.weight
            elif hasattr(layer, 'conv') and isinstance(layer.conv, nn.Conv2d):  # 处理ConvNormAct
                weight = layer.conv.weight
            else:
                self.log(f"Unsupported layer type: {type(layer)}")
                return {}
            
            # 计算0值的比例
            zero_ratio = (weight.abs() < 1e-6).float().mean()
            results['zero_ratio'] = zero_ratio
            
            # 计算权重的基本统计信息
            weight_mean = weight.mean()
            weight_std = weight.std()
            weight_min = weight.min()
            weight_max = weight.max()
            
            # 将权重移到CPU上计算直方图
            weight_cpu = weight.detach().cpu()
            hist_values = torch.histogram(weight_cpu.flatten(), bins=50)
            results['weight_histogram'] = {
                'edges': hist_values[1].to(weight.device),  # 将结果移回原设备
                'values': hist_values[0].to(weight.device)
            }
            
            # 添加统计信息到结果中
            results.update({
                'weight_mean': weight_mean,
                'weight_std': weight_std,
                'weight_min': weight_min,
                'weight_max': weight_max
            })
            
            self.log(f"Weight analysis results:")
            self.log(f"- Zero ratio: {zero_ratio.item():.4f}")
            self.log(f"- Mean: {weight_mean.item():.4f}")
            self.log(f"- Std: {weight_std.item():.4f}")
            self.log(f"- Min: {weight_min.item():.4f}")
            self.log(f"- Max: {weight_max.item():.4f}")
        
        return results


def get_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", type=str, required=True, help="模型架构名称")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--train-data", type=str, help="训练数据路径")
    parser.add_argument("--save-dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--num-samples", type=int, default=4000, help="要处理样本数量")
    parser.add_argument("--batch-size", type=int, default=32, help="batch大小")
    parser.add_argument("--workers", type=int, default=2, help="数据加载的worker数量")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--no-shuffle", action="store_true", help="不打乱数据顺序")
    
    # 添加分析类型控制参数
    parser.add_argument("--analysis-type", type=str, default="both", choices=["static", "dynamic", "both"],
                      help="分析类型: static(只做静态分析), dynamic(只做动态分析), both(都做)")
    
    return parser


def main():
    """主函数"""
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    # 如果提供了随机种子，设置所有随机源
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        
        def seed_worker(worker_id):
            worker_seed = args.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
    else:
        seed_worker = None
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print("正在加载模型...")
    model, _, preprocess = create_model_and_transforms(
        model_name=args.model_arch,
        pretrained=args.model_path,
        image_mean=(0, 0, 0),
        image_std=(1, 1, 1),
        image_interpolation="bilinear",
    )
    
    # 确保模型处于评估模式
    # model.eval()
    
    # 初始化分析器
    analyzer = FastVitRedundancyAnalyzer(
        model=model,
        preprocess=preprocess,
        device=device
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 第一阶段：静态分析（BN层权重分析）
    if args.analysis_type in ["static", "both"]:
        print("\n=== 第一阶段：分析模型权重 ===")
        bn_analysis_results = {}
        static_dir = os.path.join(args.save_dir, 'static_features')  # 创建static_features子目录
        os.makedirs(static_dir, exist_ok=True)
        log_path = os.path.join(static_dir, 'bn_analysis.log')
        with open(log_path, 'w') as f:
            f.write("=== 查找BN层和分析权重 ===\n\n")
            bn_count = 0
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    bn_count += 1
                    print(f"分析BN层 {bn_count}: {name}")
                    f.write(f"\nFound BN layer: {name}\n")
                    parent_name = '.'.join(name.split('.')[:-1])
                    f.write(f"Looking for parent layer: {parent_name}\n")
                    
                    parent = dict(model.named_modules()).get(parent_name)
                    if parent is None:
                        f.write(f"Warning: Could not find parent layer for {name}\n")
                        continue
                    
                    f.write(f"Parent layer type: {type(parent).__name__}\n")
                    
                    if isinstance(parent, (nn.Conv2d, nn.Linear)) or (hasattr(parent, 'conv') and isinstance(parent.conv, nn.Conv2d)):
                        f.write(f"Analyzing weights for parent layer: {parent_name}\n")
                        pre_bn_results = analyzer.analyze_pre_bn_weights(parent)
                        bn_analysis_results[f"{parent_name}_pre_bn"] = pre_bn_results
                        f.write(f"Analysis complete for {parent_name}\n")
                        f.write(f"Results: {pre_bn_results}\n")
                    else:
                        f.write(f"Parent layer is not Conv2d, Linear or ConvNormAct, skipping analysis\n")
            
            f.write(f"\nTotal BN layers found: {bn_count}\n")
        
        # 保存和可视化BN层分析结果
        print("\n保存BN层分析结果...")
        analyzer.save_results(bn_analysis_results, static_dir)
        analyzer.visualize_redundancy(bn_analysis_results, static_dir)
    
    # 第二阶段：动态特征分析
    if args.analysis_type in ["dynamic", "both"]:
        if args.train_data:
            print("\n=== 第二阶段：分析模型动态特征 ===")
            # 创建数据加载器
            print("准备数据加载器...")
            shards = wds.SimpleShardList(args.train_data)
            
            # 创建图片信息记录文件
            image_log_path = os.path.join(args.save_dir, 'processed_images.log')
            
            def safe_decode(text):
                """安全解码文本,处理编码错误"""
                if isinstance(text, bytes):
                    try:
                        return text.decode('utf-8')
                    except UnicodeDecodeError:
                        return text.decode('utf-8', errors='replace')
                return str(text)
                
            def safe_sample_handler(exn):
                """安全的样本处理函数,处理所有可能的错误"""
                if isinstance(exn, UnicodeDecodeError):
                    return None  # 跳过有编码问题的样本
                return wds.warn_and_continue(exn)
                
            # 根据no-shuffle参数决定是否添加shuffle操作
            pipeline = [
                shards,
                wds.tarfile_to_samples(handler=safe_sample_handler),  # 使用自定义的错误处理
            ]
            
            # 增加缓存大小
            SHUFFLE_BUFFER_SIZE = 20000  # 增大shuffle buffer
            INITIAL_SHUFFLE_BUFFER_SIZE = 10000
            
            if not args.no_shuffle:
                pipeline.append(wds.shuffle(SHUFFLE_BUFFER_SIZE, initial=INITIAL_SHUFFLE_BUFFER_SIZE))
            
            pipeline.extend([
                wds.decode("pilrgba", handler=safe_sample_handler),
                wds.rename(image="jpg;png;jpeg;webp"),  # 移除text
                wds.map_dict(image=preprocess),
                wds.to_tuple("image", "__url__", "__key__"),  # 添加URL和key信息
                wds.batched(args.batch_size),  # 在pipeline中进行batch
            ])
            
            # 设置足够大的epoch长度
            total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
            dataset = wds.DataPipeline(*pipeline).with_epoch(total_batches * 2)  # 乘2是为了确保有足够的数据
            
            print(f"配置信息:")
            print(f"- 总样本数: {args.num_samples}")
            print(f"- Worker数量: {args.workers}")
            print(f"- Batch大小: {args.batch_size}")
            print(f"- 目标batch数: {total_batches}")
            print(f"- Shuffle buffer大小: {SHUFFLE_BUFFER_SIZE}")
            print(f"- 初始Shuffle buffer大小: {INITIAL_SHUFFLE_BUFFER_SIZE}")
            
            # 设置worker_init_fn和generator
            g = None
            if args.seed is not None:
                g = torch.Generator()
                g.manual_seed(args.seed)
            
            # 修改dataloader配置,减少worker数量,增加预取
            actual_workers = min(2, args.workers)  # 限制worker数量为2
            dataloader = wds.WebLoader(
                dataset,
                batch_size=None,  # 因为已经在pipeline中做了batch
                num_workers=actual_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=20,  # 大幅增加预取因子
                worker_init_fn=seed_worker,
                generator=g
            )
            
            print(f"- 实际使用的worker数量: {actual_workers}")
            print(f"- 预取因子: 20")
            
            # 分析动态特征
            print(f"\n开始分析动态特征 (共{args.num_samples}张图片，分{args.num_samples // args.batch_size}个batch处理)...")
            total_batches = args.num_samples // args.batch_size
            processed_samples = 0
            
            # 使用字典来存储累积的结果
            accumulated_results = {}
            
            # 添加内存监控
            def print_memory_stats():
                if torch.cuda.is_available():
                    print(f"\nGPU内存使用情况:")
                    print(f"- 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                    print(f"- 缓存: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
                    print(f"- 最大分配: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            
            with open(image_log_path, 'w', encoding='utf-8') as log_file:
                log_file.write("=== 处理图片记录 ===\n\n")
                
                for i, (images, urls, keys) in enumerate(dataloader):
                    if processed_samples >= args.num_samples:
                        break
                        
                    current_batch_size = images.size(0)
                    processed_samples += current_batch_size
                    
                    # 每50个batch打印一次内存使用情况
                    if i % 50 == 0:
                        print_memory_stats()
                        # 清理缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 记录当前batch的图片信息
                    log_file.write(f"\nBatch {i+1}/{total_batches}:\n")
                    for j, (url, key) in enumerate(zip(urls, keys)):
                        log_file.write(f"  {j+1}. URL: {safe_decode(url)}\n     Key: {safe_decode(key)}\n")
                    log_file.flush()
                    
                    print(f"\r处理batch {i+1}/{total_batches} (每个batch包含{current_batch_size}张图片，已处理{processed_samples}/{args.num_samples}张图片)", 
                          end="", flush=True)
                    
                    # 对当前batch的图片进行分析
                    results = analyzer.analyze_redundancy_batch(images, args.save_dir, args.num_samples, processed_samples)
                    
                    # 在线更新累积结果
                    for layer_name, layer_results in results.items():
                        if layer_name not in accumulated_results:
                            accumulated_results[layer_name] = {}
                        
                        for metric_name, values in layer_results.items():
                            if metric_name not in accumulated_results[layer_name]:
                                if torch.is_tensor(values):
                                    # 对tensor类型，保持在GPU上进行累积
                                    accumulated_results[layer_name][metric_name] = {
                                        'sum': torch.zeros_like(values, dtype=torch.float64),  # 使用float64提高精度
                                        'sum_sq': torch.zeros_like(values, dtype=torch.float64),  # 添加平方和以计算方差
                                        'count': 0
                                    }
                                else:
                                    accumulated_results[layer_name][metric_name] = {
                                        'sum': 0.0,
                                        'sum_sq': 0.0,
                                        'count': 0
                                    }
                                
                            # 更新累积和
                            if torch.is_tensor(values):
                                values = values.to(dtype=torch.float64)  # 转换为float64
                                accumulated_results[layer_name][metric_name]['sum'] += values
                                accumulated_results[layer_name][metric_name]['sum_sq'] += values * values
                            else:
                                accumulated_results[layer_name][metric_name]['sum'] += float(values)
                                accumulated_results[layer_name][metric_name]['sum_sq'] += float(values) * float(values)
                            accumulated_results[layer_name][metric_name]['count'] += 1
                    
                    # 释放当前batch的结果
                    del results
                    torch.cuda.empty_cache()
            
            print(f"\n\n实际处理了{processed_samples}张图片，共{i+1}个batch")
            print(f"图片信息已保存到: {image_log_path}")
            
            if processed_samples < args.num_samples:
                print(f"警告：要求处理{args.num_samples}张图片，但实际只处理了{processed_samples}张图片")
                print("这可能是因为数据集中的图片数量不足")
            
            print("\n计算最终结果...")
            
            # 计算平均值和标准差
            final_results = {}
            for layer_name, layer_results in accumulated_results.items():
                final_results[layer_name] = {}
                for metric_name, values in layer_results.items():
                    count = values['count']
                    if torch.is_tensor(values['sum']):
                        mean = values['sum'] / count
                        # 计算标准差
                        variance = (values['sum_sq'] / count) - (mean * mean)
                        std = torch.sqrt(torch.clamp(variance, min=0))  # 使用clamp避免数值误差导致的负值
                        
                        # 将结果转回float32以节省内存
                        final_results[layer_name][metric_name] = mean.to(dtype=torch.float32)
                        final_results[layer_name][f"{metric_name}_std"] = std.to(dtype=torch.float32)
                    else:
                        mean = values['sum'] / count
                        variance = (values['sum_sq'] / count) - (mean * mean)
                        std = math.sqrt(max(0, variance))
                        final_results[layer_name][metric_name] = mean
                        final_results[layer_name][f"{metric_name}_std"] = std
            
            # 清理累积结果
            del accumulated_results
            torch.cuda.empty_cache()
            
            # 保存和可视化最终结果
            dynamic_dir = os.path.join(args.save_dir, 'dynamic_features')
            os.makedirs(dynamic_dir, exist_ok=True)
            print("保存动态特征分析结果...")
            analyzer.save_results(final_results, dynamic_dir)
            analyzer.visualize_redundancy(final_results, dynamic_dir)
            
            print(f"\n分析完成！所有结果已保存到: {args.save_dir}")
            if args.analysis_type in ["static", "both"]:
                print(f"- 静态特征分析结果: {os.path.join(args.save_dir, 'static_features')}")
            if args.analysis_type in ["dynamic", "both"]:
                if args.train_data:
                    print(f"- 动态特征分析结果: {os.path.join(args.save_dir, 'dynamic_features')}")
        else:
            print("\n未提供训练数据，跳过动态特征分析")
    
    # 打印完成信息
    print(f"\n分析完成！")
    if args.analysis_type in ["static", "both"]:
        print(f"- 静态特征分析结果: {os.path.join(args.save_dir, 'static_features')}")
    if args.analysis_type in ["dynamic", "both"]:
        if args.train_data:
            print(f"- 动态特征分析结果: {os.path.join(args.save_dir, 'dynamic_features')}")


if __name__ == "__main__":
    main()