# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified for OpenCLIP and MobileCLIP

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union, Dict, Any, List

def bipartite_soft_matching_static(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
    distill_token: bool = False,
) -> torch.Tensor:
    """计算token合并的静态匹配矩阵 (batch_size=1)
    
    Args:
        metric: [1, N, D] 特征矩阵
        r: 要合并的token数量
        class_token: 是否有class token
        distill_token: 是否有distillation token
    
    Returns:
        合并矩阵 [N-r, N]
    """
    # 确保metric的形状正确 [1, N, D]
    N, D = metric.shape[1:]
    
    # 计算相似度矩阵 [1, N, N]
    similarity = torch.matmul(metric, metric.transpose(1, 2))
    similarity = similarity.squeeze(0)  # [N, N]
    
    # 处理特殊token
    n_tokens = 1 if class_token else 0
    n_tokens += 1 if distill_token else 0
    
    if n_tokens > 0:
        # 排除特殊token的相似度计算
        similarity_sub = similarity[n_tokens:, n_tokens:]
    else:
        similarity_sub = similarity
    
    # 展平并找到最相似的token对
    similarity_flat = similarity_sub.reshape(-1)
    remaining_tokens = N - n_tokens
    r = min(r, remaining_tokens // 2)  # 确保r不超过可用token数量的一半
    
    # 添加数值稳定性
    similarity_flat = similarity_flat / (similarity_flat.max() + 1e-6)
    
    # 获取top-k相似度和索引
    _, topk_indices = torch.topk(similarity_flat, k=r, dim=0)
    
    # 创建合并矩阵 [N-r, N]
    merge_matrix = torch.zeros(N-r, N, device=metric.device, dtype=metric.dtype)
    
    # 设置特殊token
    if class_token:
        merge_matrix[0, 0] = 1.0
    if distill_token and n_tokens > 1:
        merge_matrix[1, 1] = 1.0
    
    # 计算行列索引
    row_indices = (topk_indices // remaining_tokens) + n_tokens
    col_indices = (topk_indices % remaining_tokens) + n_tokens
    
    # 创建已合并token的跟踪集合
    merged_tokens = set()
    
    # 填充合并矩阵
    out_idx = n_tokens
    for i in range(r):
        row, col = row_indices[i].item(), col_indices[i].item()
        
        # 跳过已经合并的token
        if row in merged_tokens or col in merged_tokens:
            continue
            
        # 标记这些token为已合并
        merged_tokens.add(row)
        merged_tokens.add(col)
        
        # 合并这对token (平均权重)
        if out_idx < N-r:
            merge_matrix[out_idx, row] = 0.5
            merge_matrix[out_idx, col] = 0.5
            out_idx += 1
    
    # 处理剩余未合并的token
    remaining_indices = [i for i in range(n_tokens, N) if i not in merged_tokens]
    for idx in remaining_indices:
        if out_idx < N-r:
            merge_matrix[out_idx, idx] = 1.0
            out_idx += 1
    
    return merge_matrix

def create_simple_merge_matrix(
    N: int,
    r: int,
    device: torch.device,
    dtype: torch.dtype,
    class_token: bool = True,
    distill_token: bool = False
) -> torch.Tensor:
    """创建一个基于相似度的静态合并矩阵
    
    Args:
        N: token数量
        r: 要合并的token数量
        device: 设备
        dtype: 数据类型
        class_token: 是否有class token
        distill_token: 是否有distillation token
        
    Returns:
        合并矩阵 [N-r, N]
    """
    # 创建合并矩阵
    merge_matrix = torch.zeros(N-r, N, device=device, dtype=dtype)
    
    # 处理特殊token
    n_tokens = 1 if class_token else 0
    n_tokens += 1 if distill_token else 0
    
    # 设置特殊token
    if class_token:
        merge_matrix[0, 0] = 1.0
    if distill_token and n_tokens > 1:
        merge_matrix[1, 1] = 1.0
    
    # 计算可合并的token数量
    mergeable_tokens = N - n_tokens
    r = min(r, mergeable_tokens // 2)  # 确保不会合并过多token
    
    if r <= 0:
        # 如果不需要合并，则每个token独立保留
        for i in range(n_tokens, N):
            if i - r < N - r:
                merge_matrix[i - r, i] = 1.0
        return merge_matrix
    
    # 使用改进的合并策略
    out_idx = n_tokens
    remaining_tokens = set(range(n_tokens, N))
    
    # 1. 基于位置的权重
    position_weights = torch.ones(N, device=device, dtype=dtype)
    for i in range(n_tokens, N):
        # 赋予中心位置较高的权重
        relative_pos = (i - n_tokens) / (N - n_tokens)
        position_weights[i] = 1.0 - abs(relative_pos - 0.5)
    
    # 2. 合并相邻token，考虑位置权重
    while len(remaining_tokens) >= 2 and out_idx < N - r:
        tokens = sorted(list(remaining_tokens))
        max_weight = -1
        best_pair = None
        
        # 寻找最佳合并对
        for i in range(len(tokens)-1):
            t1, t2 = tokens[i], tokens[i+1]
            # 计算合并权重（考虑位置和相邻性）
            weight = (position_weights[t1] + position_weights[t2]) * 0.5
            if weight > max_weight:
                max_weight = weight
                best_pair = (t1, t2)
        
        if best_pair is None:
            break
            
        t1, t2 = best_pair
        # 使用加权平均进行合并
        w1 = position_weights[t1] / (position_weights[t1] + position_weights[t2])
        w2 = position_weights[t2] / (position_weights[t1] + position_weights[t2])
        
        merge_matrix[out_idx, t1] = w1
        merge_matrix[out_idx, t2] = w2
        
        remaining_tokens.remove(t1)
        remaining_tokens.remove(t2)
        out_idx += 1
    
    # 3. 处理剩余的token
    for idx in remaining_tokens:
        if out_idx < N - r:
            merge_matrix[out_idx, idx] = 1.0
            out_idx += 1
    
    # 4. 归一化每一行
    row_sums = merge_matrix.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # 避免除零
    merge_matrix = merge_matrix / row_sums
    
    return merge_matrix

def merge_wavg(x: torch.Tensor, merge_matrix: torch.Tensor) -> torch.Tensor:
    """使用预计算的合并矩阵进行token合并 (batch_size=1)"""
    # [1, N, D] @ [N-r, N].T -> [1, N-r, D]
    return torch.matmul(merge_matrix, x.squeeze(0)).unsqueeze(0)

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """处理r值，支持多种配置方式:
    - int: 每层使用相同的r值
    - Tuple[int, float]: (r, inflection) 对，用于渐进式调整
    - List[int]: 为每层指定具体的r值
    """
    if isinstance(r, list):
        if len(r) < num_layers:
            return r + [0] * (num_layers - len(r))
        return r[:num_layers]
    
    if isinstance(r, tuple):
        r_val, inflect = r
        min_val = int(r_val * (1.0 - inflect))
        max_val = 2 * r_val - min_val
        step = (max_val - min_val) / (num_layers - 1)
        return [int(min_val + step * i) for i in range(num_layers)]
    
    return [r] * num_layers

class StaticToMeBlock(nn.Module):
    """静态版本的ToMe transformer block，专门用于batch_size=1的ONNX导出"""
    def __init__(self, block, r_value):
        super().__init__()
        self.block = block
        self.r = r_value
        self.class_token = True
        self.distill_token = False
        
        # 在初始化时就创建合并矩阵的模板
        self.register_buffer('merge_matrix', None, persistent=False)
        self.register_buffer('token_importance', None, persistent=False)
        
    def _compute_token_importance(self, x):
        """计算token的重要性分数"""
        # 使用注意力权重的方差作为重要性指标
        if hasattr(self.block, 'attn'):
            with torch.no_grad():
                # 获取注意力权重
                attn = self.block.attn.qkv(x)
                B, N, C = x.shape
                num_heads = self.block.attn.num_heads
                attn = attn.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = attn[0], attn[1], attn[2]
                
                # 计算注意力分数
                attn_weights = (q @ k.transpose(-2, -1)) * self.block.attn.scale
                attn_weights = attn_weights.softmax(dim=-1)  # [B, H, N, N]
                
                # 计算token的重要性分数
                importance = attn_weights.var(dim=(1, 2)).mean(dim=0)  # [N]
                
                # 确保class token的重要性最高
                if self.class_token:
                    importance[0] = importance.max() * 1.2
                
                return importance
        return None
        
    def forward(self, x):
        # 1. 应用自注意力
        if hasattr(self.block, 'attn'):
            norm1 = getattr(self.block, 'ln_1', getattr(self.block, 'norm1', None))
            norm2 = getattr(self.block, 'ln_2', getattr(self.block, 'norm2', None))
            
            if norm1 is not None and norm2 is not None:
                # 计算注意力
                x_norm = norm1(x)
                attn_output = self.block.attn(x_norm)
                if hasattr(self.block, 'drop_path1'):
                    attn_output = self.block.drop_path1(attn_output)
                x = x + attn_output
                
                # 2. 应用token合并
                if self.r > 0:
                    # 首次运行时初始化merge_matrix
                    if self.merge_matrix is None:
                        with torch.no_grad():
                            N = x.shape[1]
                            # 计算token重要性
                            importance = self._compute_token_importance(x_norm)
                            self.token_importance = importance if importance is not None else torch.ones(N, device=x.device)
                            
                            # 创建合并矩阵
                            self.merge_matrix = create_simple_merge_matrix(
                                N=N,
                                r=self.r,
                                device=x.device,
                                dtype=x.dtype,
                                class_token=self.class_token,
                                distill_token=self.distill_token
                            )
                            print(f"Created merge matrix with shape {self.merge_matrix.shape}")
                    
                    # 应用合并
                    x_before = x
                    x = merge_wavg(x, self.merge_matrix)
                    
                    # 调试信息
                    with torch.no_grad():
                        print(f"Token count: before={x_before.shape[1]}, after={x.shape[1]}")
                
                # 3. 应用MLP
                x_norm = norm2(x)
                mlp_output = self.block.mlp(x_norm)
                if hasattr(self.block, 'drop_path2'):
                    mlp_output = self.block.drop_path2(mlp_output)
                x = x + mlp_output
                
        else:
            # 对于其他类型的block，直接应用
            x = self.block(x)
        
        return x

def apply_tome(model, r: Union[int, List[int], Tuple[int, float]], 
              trace_source: bool = False, prop_attn: bool = True):
    """应用静态版本的ToMe到模型"""
    print(f"\n[StaticToMe] Applying ToMe with r={r}")
    
    # 1. 获取vision transformer
    if hasattr(model, 'trunk'):
        vision_transformer = model.trunk
        print("[StaticToMe] Found trunk in model")
    elif hasattr(model, 'blocks'):
        vision_transformer = model
        print("[StaticToMe] Using model directly as vision transformer")
    else:
        raise ValueError("Model must have trunk or blocks attribute")
    
    # 2. 计算每层的r值
    num_layers = len(vision_transformer.blocks)
    r_values = parse_r(num_layers, r)
    print(f"[StaticToMe] Computed r values for {num_layers} layers: {r_values}")
    
    # 3. 替换每个block
    for i, block in enumerate(vision_transformer.blocks):
        r_val = r_values[i]
        if r_val > 0:
            print(f"[StaticToMe] Converting block {i} with r={r_val}")
            new_block = StaticToMeBlock(
                block=block,
                r_value=r_val
            )
            # 替换block
            vision_transformer.blocks[i] = new_block
        else:
            print(f"[StaticToMe] Skipping block {i} (r=0)")
    
    print("[StaticToMe] Successfully converted all blocks")
    return vision_transformer

def remove_tome(model):
    """移除ToMe修改"""
    # 1. 获取vision transformer
    if hasattr(model, 'trunk'):  # 如果是TimmModel
        vision_transformer = model.trunk
    elif hasattr(model, 'blocks'):  # 如果直接是VisionTransformer
        vision_transformer = model
    else:
        return
    
    # 2. 恢复原始blocks
    for i, block in enumerate(vision_transformer.blocks):
        if isinstance(block, StaticToMeBlock):
            vision_transformer.blocks[i] = block.block

# 使用示例
"""
# 1. 导入
from open_clip.src.training.tome_token_merging import apply_tome, remove_tome

# 2. 应用ToMe (多种方式)
model = create_model(...)  # 你的CLIP或MobileCLIP模型

# 方式1: 固定r值
apply_tome(model, r=8)  # 每层合并8个tokens

# 方式2: 渐进式r值
apply_tome(model, r=(8, 0.4))  # 从小到大渐进调整r值

# 方式3: 自定义每层r值
apply_tome(model, r=[4, 8, 8, 4])  # 为每层指定具体的r值

# 3. 导出ONNX
torch.onnx.export(...)

# 4. 如果需要，移除ToMe
remove_tome(model)
"""
