import torch
import torch_pruning as tp
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
import torch.nn as nn
import torch.nn.functional as F
import timm

class PruningManager:
    """剪枝管理器：负责MobileCLIP模型剪枝和与训练流程的集成"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        self.model = model
        self.config = {
            'current_epoch': 0,
            'current_step': 0,
            'pruning_start_epoch': 0,
            'pruning_end_epoch': None,
            'progressive_pruning': False,
            'do_pruning': True,
            'pruning_interval': 1,
            'pruning_ratio': 0.8,
            'pruning_type': 'taylor',
            'global_pruning': False,
            'round_to': 8,
            'bottleneck': True,
            'pruning_done': False,
            'isomorphic': False,
            'prune_num_heads': False,
            'prune_head_dims': True,
            'head_pruning_ratio': 0.8,
        }
        self.config.update(config)
        
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
        
        # 直接将logit_scale参数添加到ignored_layers
        # if hasattr(self.model, 'logit_scale'):
        #     print(f"logit_scale: {self.model.logit_scale}")
        #     ignored_layers.append(self.model.logit_scale)
        
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
                    m.forward = forward.__get__(m, timm.models.fastvit.Attention)
                    self.num_heads[m.qkv] = m.num_heads

            if isinstance(m, timm.models.fastvit.ConvMlp):
                if self.config['bottleneck']:
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
        imp = imp_types.get(self.config['pruning_type'])
        if imp is None:
            raise ValueError(f"Unsupported pruning type: {self.config['pruning_type']}")
        return imp
        
    def initialize_pruner(self):
        """初始化剪枝器"""
        imp = self._get_importance_criterion()
        current_ratio = self.get_current_pruning_ratio()
        print(f"\n当前剪枝率: {current_ratio}")
            
        # 收集所有不应该参与剪枝的参数
        unwrapped_parameters = []
        
        # 特殊处理logit_scale参数
        if hasattr(self.model, 'logit_scale'):
            # 将logit_scale参数包装成一个2维张量，这样可以有明确的输入输出维度
            logit_scale_value = self.model.logit_scale.data.reshape(1, 1)
            logit_scale_param = nn.Parameter(logit_scale_value, requires_grad=self.model.logit_scale.requires_grad)
            unwrapped_parameters.append((logit_scale_param, 1))  # 使用1作为pruning_dim，表示输出维度
            print(f"Excluding logit_scale parameter with shape {logit_scale_param.shape}, pruning_dim=1")
        
        # 处理所有LayerScale2d的gamma参数
        for m in self.model.modules():
            if isinstance(m, timm.models.fastvit.LayerScale2d):
                unwrapped_parameters.append((m.gamma, 0))  # 在通道维度(dim=0)上跟随剪枝
                print(f"Adding LayerScale2d gamma parameter with shape {m.gamma.shape}, pruning_dim=0")
            
        self.pruner = tp.pruner.MetaPruner(
            model=self.model,
            example_inputs=self.example_inputs,
            importance=imp,
            global_pruning=self.config['global_pruning'],
            pruning_ratio=current_ratio,
            ignored_layers=self.ignored_layers,
            num_heads=self.num_heads,
            round_to=self.config['round_to'],
            isomorphic=self.config['isomorphic'],
            prune_num_heads=self.config['prune_num_heads'],
            prune_head_dims=not self.config['prune_num_heads'], 
            head_pruning_ratio=self.config['head_pruning_ratio'],
            unwrapped_parameters=unwrapped_parameters,
        )
        
        # 如果是Hessian方法，需要初始化梯度累积
        if self.config['pruning_type'] == 'hessian':
            self.pruner.importance.zero_grad()

    def get_current_pruning_ratio(self) -> float:
        """获取当前的剪枝比例（支持渐进式剪枝）"""
        if not self.config.get('progressive_pruning', False):
            return self.config['pruning_ratio']
            
        current_epoch = self.config.get('current_epoch', 0)
        start_epoch = self.config.get('pruning_start_epoch', 0)
        end_epoch = self.config.get('pruning_end_epoch')
        
        if end_epoch is None:
            end_epoch = self.config.get('epochs', 10)
            
        if current_epoch < start_epoch:
            return 0.0
        if current_epoch >= end_epoch:
            return self.config['pruning_ratio']
            
        progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
        return self.config['pruning_ratio'] * progress

    def should_prune(self) -> bool:
        """判断是否应该执行剪枝"""
        if not self.config['do_pruning'] or self.config.get('pruning_done', False):
            return False
            
        current_epoch = self.config['current_epoch']
        current_step = self.config['current_step']
        
        # 只在训练开始时执行一次剪枝
        if current_epoch == self.config['pruning_start_epoch'] and current_step == 0:
            return True
            
        return False

    def step(self, model_out=None):
        """执行一步剪枝"""
        if not self.should_prune():
            return None
            
        try:
            # 处理Taylor和Hessian剪枝的梯度累积
            if self.config['pruning_type'] in ['taylor', 'hessian']:
                if model_out is None:
                    logging.warning("No model output provided for Taylor/Hessian pruning")
                    return None
                    
                # 获取损失
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                
                batch_size = image_features.shape[0]
                labels = torch.arange(batch_size, device=image_features.device).long()
                loss = (
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                ) / 2
                
                # 对于Hessian方法，需要分别处理每个样本的梯度
                if self.config['pruning_type'] == 'hessian':
                    loss_per_sample = (
                        F.cross_entropy(logits_per_image, labels, reduction='none') +
                        F.cross_entropy(logits_per_text, labels, reduction='none')
                    ) / 2
                    
                    for l in loss_per_sample:
                        self.model.zero_grad()
                        l.backward(retain_graph=True)
                        self.pruner.importance.accumulate_grad(self.model)
                else:  # Taylor方法
                    loss.backward(retain_graph=True)
            
            # 1. 执行剪枝
            pruning_groups = list(self.pruner.step(interactive=True))
            for group in pruning_groups:
                group.prune()
                print(f"Pruning groups: {group}")
            
            # 2. 更新attention heads
            head_id = 0
            for m in self.model.modules():
                if hasattr(m, 'num_heads') and hasattr(m, 'qkv'):
                    if not hasattr(m, 'latent_len'):
                        print(f"Head #{head_id}")
                        print(f"[Before Pruning] Num Heads: {m.num_heads}, Head Dim: {m.head_dim} =>")
                        # 更新头数和维度
                        m.num_heads = self.pruner.num_heads[m.qkv]
                        m.head_dim = m.qkv.out_features // (3 * m.num_heads)
                        print(f"[After Pruning] Num Heads: {m.num_heads}, Head Dim: {m.head_dim}")
                        print()
                        head_id += 1
                        
            # 3. 更新模型统计信息
            self.current_macs, self.current_params = tp.utils.count_ops_and_params(
                self.model, self.example_inputs)
            
            # 4. 标记剪枝完成
            self.config['pruning_done'] = True
            
            return self.get_model_stats()
            
        except Exception as e:
            logging.error(f"Error during pruning: {e}")
            return None

    def get_vision_encoder_stats(self) -> Tuple[float, float]:
        """获仅视觉编码器的参数量和MACs"""
        # 计算视觉编码器参数量
        vision_params = sum(p.numel() for name, p in self.model.named_parameters() 
                          if 'visual' in name)
        
        # 计算视觉编码器MACs（由于example_inputs只包含图像，所以current_macs已经是视觉部分）
        vision_macs = self.current_macs
        
        return vision_macs, vision_params

    def get_model_stats(self) -> str:
        """获取模型统计信息，包括总体和视觉编码器部分"""
        # 获取总体统计
        macs_ratio = self.current_macs / self.base_macs
        params_ratio = self.current_params / self.base_params
        
        # 获取视觉编码器统计
        vision_macs, vision_params = self.get_vision_encoder_stats()
        vision_macs_G = vision_macs / 1e9
        vision_params_M = vision_params / 1e6
        
        return (f"Total - MACs: {self.current_macs/1e9:.2f}G ({macs_ratio*100:.1f}%), "
                f"Params: {self.current_params/1e6:.2f}M ({params_ratio*100:.1f}%)\n"
                f"Vision - MACs: {vision_macs_G:.2f}G, "
                f"Params: {vision_params_M:.2f}M")

    def state_dict(self) -> Dict[str, Any]:
        """获取剪枝状态，用于保存checkpoint"""
        state = {
            'config': self.config,
            'base_macs': self.base_macs,
            'base_params': self.base_params,
            'current_macs': self.current_macs,
            'current_params': self.current_params,
        }
        
        if self.pruner is not None:
            state.update({
                'pruning_ratio': self.pruner.pruning_ratio,
                'num_heads': self.pruner.num_heads,
            })
            
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载剪枝状态"""
        self.config.update(state_dict['config'])
        self.base_macs = state_dict['base_macs']
        self.base_params = state_dict['base_params']
        self.current_macs = state_dict['current_macs']
        self.current_params = state_dict['current_params']
        
        if 'pruning_ratio' in state_dict:
            self.initialize_pruner()
            if self.pruner is not None:
                self.pruner.pruning_ratio = state_dict['pruning_ratio']
                self.pruner.num_heads = state_dict['num_heads'] 