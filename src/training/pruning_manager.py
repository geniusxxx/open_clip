import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
import timm
import traceback
import math
import gc

logger = logging.getLogger("train")

def clean_memory():
    """清理内存的辅助函数"""
    # 强制进行垃圾回收
    gc.collect()
    # 如果可用，清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class PruningManager:
    """剪枝管理器：负责MobileCLIP模型剪枝和与训练流程的集成"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        data: Optional[Dict] = None
    ):
        self.model = model
        self.config = config
        self.data = data
        
        # 确保example_inputs正确
        if isinstance(example_inputs, (list, tuple)):
            self.example_inputs = tuple(x if isinstance(x, torch.Tensor) else torch.tensor(x) 
                                     for x in example_inputs)
        else:
            self.example_inputs = example_inputs if isinstance(example_inputs, torch.Tensor) \
                                else torch.tensor(example_inputs)
        
        # 初始化
        self.pruner = None
        self.base_macs, self.base_params = tp.utils.count_ops_and_params(model, self.example_inputs)
        self.current_macs, self.current_params = self.base_macs, self.base_params
        
        # 准备模型并初始化pruner
        self._prepare_model()
        self.initialize_pruner()
        
    def _prepare_model(self):
        """准备模型，包括替换forward函数等"""
        # 1. 定义新的forward函数
        def forward(self_module, x):
            # logging.info(f"Using custom forward! num_heads={self_module.num_heads}, qkv shape={self_module.qkv.out_features}")
            B, C, H, W = x.shape
            N = H * W
            x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
            
            qkv = self_module.qkv(x)
            qkv = qkv.reshape(B, N, 3, self_module.num_heads, self_module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            if hasattr(self_module, 'fused_attn') and self_module.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self_module.attn_drop.p if self_module.training else 0.
                )
            else:
                q = q * self_module.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self_module.attn_drop(attn)
                x = attn @ v
            
            x = x.transpose(1, 2).reshape(B, N, -1)
            x = self_module.proj(x)
            x = self_module.proj_drop(x)
            x = x.transpose(-2, -1).reshape(B, -1, H, W)  # 恢复空间维度
            return x

        # 2. 初始化
        self.num_heads = {}
        ignored_layers = []
        
        # 3. 保护text encoder和全局参数
        if hasattr(self.model, 'text'):
            ignored_layers.append(self.model.text)
        
        # 4. 保护visual encoder中的关键层
        if hasattr(self.model, 'visual'):
            if hasattr(self.model.visual.trunk, 'head'):
                ignored_layers.append(self.model.visual.trunk.head)

        # 5. 处理attention和mlp层
        for m in self.model.modules():
            # 5.1 处理Attention层
            if isinstance(m, timm.models.fastvit.Attention):
                if hasattr(m, 'qkv'):
                    # 替换forward函数
                    # original_forward = m.forward
                    m.forward = forward.__get__(m, timm.models.fastvit.Attention)
                    # logger.info(f"Replaced forward function: original={original_forward}, new={m.forward}")
                    # print(m.qkv)
                    self.num_heads[m.qkv] = m.num_heads

            if isinstance(m, timm.models.fastvit.ConvMlp):
                if self.config.get('bottleneck', False):
                    if hasattr(m, 'fc2'):
                        ignored_layers.append(m.fc2)

        self.ignored_layers = ignored_layers
        
    def _get_importance_criterion(self):
        """获取重要性评估准则"""
        imp_types = {
            'random': tp.importance.RandomImportance(),
            'l1': tp.importance.GroupNormImportance(p=1),
            'l2': tp.importance.GroupNormImportance(p=2),
            'taylor': tp.importance.GroupTaylorImportance(),
            'hessian': tp.importance.GroupHessianImportance(),
        }
        imp = imp_types.get(self.config.get('pruning_type', 'l1'))
        if imp is None:
            raise ValueError(f"Unsupported pruning type: {self.config.get('pruning_type')}")
        return imp
        
    def initialize_pruner(self):
        """初始化剪枝器"""
        imp = self._get_importance_criterion()
        logger.info(f"初始化剪枝器，目标剪枝率: {self.config.get('pruning_ratio', 0.5)}")
            
        # 收集所有不应该参与剪枝的参数
        unwrapped_parameters = []
        
        # 特殊处理logit_scale参数
        if hasattr(self.model, 'logit_scale'):
            logit_scale_value = self.model.logit_scale.data.reshape(1, 1)
            logit_scale_param = nn.Parameter(logit_scale_value, requires_grad=self.model.logit_scale.requires_grad)
            unwrapped_parameters.append((logit_scale_param, 1))
            # logger.info(f"Excluding logit_scale parameter with shape {logit_scale_param.shape}, pruning_dim=1")
        
        # 处理所有LayerScale2d的gamma参数
        for m in self.model.modules():
            if isinstance(m, timm.models.fastvit.LayerScale2d):
                unwrapped_parameters.append((m.gamma, 0))
                # logger.info(f"Adding LayerScale2d gamma parameter with shape {m.gamma.shape}, pruning_dim=0")
        
        # 根据剪枝模式设置参数
        pruning_mode = self.config.get('pruning_mode', 'pre_training')
        
        # 基础参数
        pruner_kwargs = {
            'model': self.model,
            'example_inputs': self.example_inputs,
            'importance': imp,
            'global_pruning': self.config.get('global_pruning', False),
            'pruning_ratio': self.config.get('pruning_ratio', 0.5),
            'ignored_layers': self.ignored_layers,
            'num_heads': self.num_heads,
            'round_to': self.config.get('round_to', 8),
            'isomorphic': self.config.get('isomorphic', False),
            'prune_num_heads': self.config.get('prune_num_heads', False),
            'prune_head_dims': self.config.get('prune_head_dims', False),
            'head_pruning_ratio': self.config.get('head_pruning_ratio', 0.0),
            'unwrapped_parameters': unwrapped_parameters,
        }
        
        # 根据模式添加额外参数
        if pruning_mode != 'pre_training':
            pruner_kwargs.update({
                'iterative_steps': self.config.get('iterative_steps', 1),
                'iterative_pruning_ratio_scheduler': self.config.get('iterative_pruning_ratio_scheduler', 'linear')
            })
            
        self.pruner = tp.pruner.MetaPruner(**pruner_kwargs)
        
        # 如果是Hessian方法，需要初始化梯度累积
        if self.config.get('pruning_type') == 'hessian':
            self.pruner.importance.zero_grad()

    def get_current_pruning_ratio(self) -> float:
        """获取当前的剪枝比例"""
        if self.pruner is None:
            return 0.0
        return self.pruner.current_pruning_ratio if hasattr(self.pruner, 'current_pruning_ratio') else self.config.get('pruning_ratio', 0.5)

    def should_prune(self) -> bool:
        """判断是否应该执行剪枝"""
        if not self.config.get('do_pruning', False) or self.config.get('pruning_done', False):
            return False
            
        current_epoch = self.config.get('current_epoch', 0)
        current_step = self.config.get('current_step', 0)
        
        # 训练前剪枝模式
        if self.config.get('pruning_mode') == 'pre_training':
            return current_epoch == self.config.get('pruning_start_epoch', 0) and current_step == 0
            
        # 训练中剪枝模式
        return True

    def step(self, model_out=None, criterion=None) -> Optional[Dict[str, Any]]:
        """执行一步剪枝
        Args:
            model_out: 模型输出字典，包含所有特征
            criterion: DRClipLoss实例
        Returns:
            pruning_info: 剪枝信息字典
        """
        if self.config.get('pruning_done', False):
            return None
            
        try:
            # 获取当前剪枝比例
            current_ratio = self.get_current_pruning_ratio()
            
            # 检查是否需要执行剪枝
            if not self.should_prune():
                return None

            # 处理Taylor和Hessian剪枝
            if self.config.get('pruning_type') in ['taylor', 'hessian']:
                if criterion is None:
                    logger.warning("No criterion provided for Taylor/Hessian pruning")
                    return None
                    
                imp = self.pruner.importance
                if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
                    self.model.train()  # 确保模型处于训练模式
                    self.model.zero_grad()
                    
                    if isinstance(imp, tp.importance.GroupHessianImportance):
                        imp.zero_grad()
                        
                    logger.info(f"Processing {'Hessian' if isinstance(imp, tp.importance.GroupHessianImportance) else 'Taylor'} pruning")
                    
                    if self.data is None or 'train' not in self.data:
                        logger.error("No training data provided for importance calculation")
                        return None
                        
                    train_loader = self.data['train'].dataloader
                    device = next(self.model.parameters()).device
                    input_dtype = torch.float32
                    
                    processed_batches = 0
                    max_batches = self.config.get('taylor_batches', 10)
                    
                    for batch in train_loader:
                        if processed_batches >= max_batches:
                            break
                            
                        if isinstance(imp, tp.importance.GroupHessianImportance):
                            self.model.zero_grad()
                            
                        images, texts = batch[:2]
                        batch_size = min(32, images.shape[0])  # 限制batch size
                        images = images[:batch_size]
                        texts = texts[:batch_size]
                        
                        with torch.cuda.amp.autocast(enabled=True):
                            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                            texts = texts.to(device=device, non_blocking=True)
                            
                            if self.config.get('dataset_reinforcement', False) and not self.config.get('dataset_reinforcement_mix_synthetic', False):
                                syn_texts = batch[4][:batch_size].to(device=device, non_blocking=True)
                                texts = torch.cat([texts, syn_texts[:, :texts.shape[-1]]], dim=0)
                            
                            model_out = self.model(images, texts)
                            
                            if self.config.get('dataset_reinforcement', False):
                                model_out.update({
                                    'dist_image_features': batch[2][:batch_size].to(device=device, non_blocking=True),
                                    'dist_text_features': batch[3][:batch_size].to(device=device, non_blocking=True),
                                })
                                if not self.config.get('dataset_reinforcement_mix_synthetic', False):
                                    model_out.update({
                                        "text_features": model_out["text_features"][:batch_size],
                                        "syn_text_features": model_out["text_features"][batch_size:],
                                        'dist_syn_text_features': batch[5][:batch_size].to(device=device, non_blocking=True)
                                    })
                            
                            losses = criterion(**model_out, output_dict=True)
                            total_loss = sum(losses.values())
                            
                            if isinstance(imp, tp.importance.GroupHessianImportance):
                                for loss in losses.values():
                                    self.model.zero_grad()
                                    loss.backward(retain_graph=True)
                                    imp.accumulate_grad(self.model)
                            else:  # Taylor方法
                                total_loss.backward()
                            
                        processed_batches += 1
                        logger.info(f"Processed batch {processed_batches}/{max_batches}")
                    
                    logger.info(f"Completed importance calculation with {processed_batches} batches")
                            
            # 执行剪枝
            logger.info(f"Executing pruning step with ratio {current_ratio}")
            pruning_groups = list(self.pruner.step(interactive=True))
            if not pruning_groups:  # 如果没有可剪枝的组
                logger.warning("No prunable groups found")
                return None
                
            # 执行实际的剪枝操作
            logger.info(f"Found {len(pruning_groups)} pruning groups")
            for i, group in enumerate(pruning_groups):
                group.prune()
                logger.info(f"Pruned group {i+1}/{len(pruning_groups)}: {group}")
            
            # 更新attention heads
            head_id = 0
            for m in self.model.modules():
                if hasattr(m, 'num_heads') and hasattr(m, 'qkv'):
                    if not hasattr(m, 'latent_len'):
                        old_heads = m.num_heads
                        old_dim = m.head_dim
                        m.num_heads = self.pruner.num_heads[m.qkv]
                        m.head_dim = m.qkv.out_features // (3 * m.num_heads)
                        logger.info(f"Head #{head_id}: {old_heads}=>{m.num_heads} heads, {old_dim}=>{m.head_dim} dims")
                        head_id += 1
                        
            # 更新模型统计信息
            self.current_macs, self.current_params = tp.utils.count_ops_and_params(
                self.model, self.example_inputs)
            macs_reduction = (self.base_macs - self.current_macs) / self.base_macs
            logger.info(f"MACs: {self.base_macs:,} => {self.current_macs:,} ({macs_reduction*100:.2f}% reduction)")
            
            # 只有在成功执行剪枝后才设置标志
            self.config['pruning_done'] = True
            logger.info("Pruning completed successfully")
            
            # 剪枝完成后清理内存
            clean_memory()
            
            # 返回剪枝信息
            return {
                'pruning_type': self.config.get('pruning_type'),
                'pruning_ratio': current_ratio,
                'macs_reduction': macs_reduction,
                'current_macs': self.current_macs,
                'base_macs': self.base_macs,
                'current_step': 1,
                'total_steps': 1
            }
                
        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
            logger.error(traceback.format_exc())
            if self.config.get('distributed', False):
                torch.distributed.barrier()
            raise

    def state_dict(self) -> Dict[str, Any]:
        """返回需要保存的剪枝状态
        Returns:
            state_dict: 包含剪枝状态的字典
        """
        try:
            state = {
                'config': self.config,
                'base_macs': self.base_macs,
                'base_params': self.base_params,
                'current_macs': self.current_macs,
                'current_params': self.current_params,
                'pruning_done': self.config.get('pruning_done', False),
                'num_heads': self.num_heads
            }
            
            # 如果pruner存在且有state_dict方法，也保存pruner状态
            if self.pruner is not None and hasattr(self.pruner, 'state_dict'):
                state['pruner_state'] = self.pruner.state_dict()
                
            logger.info("Successfully created pruning state_dict")
            return state
            
        except Exception as e:
            logger.error(f"Error in creating state_dict: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从state_dict加载剪枝状态
        Args:
            state_dict: 包含剪枝状态的字典
        """
        try:
            # 加载基本状态
            self.config.update(state_dict['config'])
            self.base_macs = state_dict['base_macs']
            self.base_params = state_dict['base_params']
            self.current_macs = state_dict['current_macs']
            self.current_params = state_dict['current_params']
            self.num_heads = state_dict['num_heads']
            
            # 如果有pruner状态且pruner存在，加载pruner状态
            if 'pruner_state' in state_dict and self.pruner is not None and hasattr(self.pruner, 'load_state_dict'):
                self.pruner.load_state_dict(state_dict['pruner_state'])
                
            logger.info("Successfully loaded pruning state")
            
        except Exception as e:
            logger.error(f"Error in loading state_dict: {str(e)}")
            logger.error(traceback.format_exc())
            raise
