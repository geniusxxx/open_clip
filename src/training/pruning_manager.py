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
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler

logger = logging.getLogger("train")

def clean_memory():
    """清理GPU和CPU内存的辅助函数。

    执行强制性的垃圾回收并清理GPU缓存（如果可用）。
    这个函数通常在大量内存操作（如剪枝）后调用，以防止内存泄漏。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class PruningManager:
    """MobileCLIP模型的剪枝管理器。

    这个类负责管理模型剪枝的整个生命周期，包括初始化、执行剪枝操作、
    维护剪枝状态，以及与训练流程的集成。支持多种剪枝模式和重要性评估方法。

    Attributes:
        model: 要剪枝的PyTorch模型。
        config: 包含剪枝配置的字典。
        data: 可选的数据加载器字典，用于某些重要性评估方法。
        current_prune_count: 当前已执行的剪枝次数。
        total_pruning_steps: 计划执行的总剪枝次数。
        recovery_steps: 剪枝周期中的恢复步数。
        gradient_collect_steps: 剪枝周期中的梯度收集步数。
        warmup: 剪枝周期中的预热步数。
        steps_between_warmup_and_first_gradient_collection: 剪枝周期中的预收集训练步数。
        current_ratio: 当前的剪枝比例。
        target_ratio: 目标剪枝比例。
        gradient_history: 存储历史梯度信息的列表。
        current_step: 当前训练步数。
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        data: Optional[Dict] = None
    ):
        """初始化剪枝管理器。

        Args:
            model: 要剪枝的PyTorch模型。
            config: 包含剪枝配置的字典，必须包含以下键：
                - do_pruning: 是否启用剪枝
                - pruning_mode: 剪枝模式 ('pre_training' 或 'during_training')
                - pruning_ratio: 目标剪枝比例
                - iterative_steps: 剪枝步数
                - recovery_steps: 剪枝周期中的恢复步数
                - gradient_collect_steps: 剪枝周期中的梯度收集步数
                - warmup: 剪枝周期中的预热步数
            example_inputs: 用于追踪模型结构的示例输入。
            data: 可选的数据加载器字典，用于某些重要性评估方法。

        Raises:
            ValueError: 如果配置参数无效或不完整。
        """
        self.model = model
        self.config = config
        self.data = data
        
        # 初始化剪枝状态
        self.total_pruning_steps = self.config.get('iterative_steps', 1)
        self.recovery_steps = self.config.get('recovery_steps', 1500)
        self.gradient_collect_steps = self.config.get('gradient_collect_steps', 10)
        self.warmup = self.config.get('warmup', 100)  # 添加warmup步数
        self.steps_between_warmup_and_first_gradient_collection = self.config.get('steps_between_warmup_and_first_gradient_collection', 0)  # 添加预收集训练步数
        self.current_ratio = 0.0
        self.target_ratio = self.config.get('pruning_ratio', 0.5)
        
        # 处理示例输入
        if isinstance(example_inputs, (list, tuple)):
            self.example_inputs = tuple(x if isinstance(x, torch.Tensor) else torch.tensor(x) 
                                     for x in example_inputs)
        else:
            self.example_inputs = example_inputs if isinstance(example_inputs, torch.Tensor) \
                                else torch.tensor(example_inputs)
        
        # 初始化性能统计
        self.pruner = None
        self.base_macs, self.base_params = tp.utils.count_ops_and_params(model, self.example_inputs)
        self.current_macs, self.current_params = self.base_macs, self.base_params
        
        # 初始化梯度累积
        self.gradient_history = []
        self.current_step = 0
        
        # 准备模型并初始化pruner
        self._prepare_model()
        self.initialize_pruner()
        
    def _prepare_model(self):
        """准备模型以进行剪枝。

        这个方法执行以下操作：
        1. 重新定义Attention层的forward函数以支持动态head维度
        2. 收集需要保护的层（不参与剪枝）
        3. 记录attention heads的信息

        Note:
            这是一个内部方法，不应直接调用。
        """
        # 1. 定义新的forward函数
        def forward(self_module, x):
            # logging.info(f"Using custom forward! num_heads={self_module.num_heads}, qkv shape={self_module.qkv.out_features}")
            # B, C, H, W = x.shape
            # N = H * W
            # x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
            B, N, C = x.shape
            
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
            # x = x.transpose(-2, -1).reshape(B, -1, H, W)  # 恢复空间维度
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
            if isinstance(m, timm.models.vision_transformer.Attention):
                if hasattr(m, 'qkv'):
                    # 替换forward函数
                    # original_forward = m.forward
                    m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)
                    # logger.info(f"Replaced forward function: original={original_forward}, new={m.forward}")
                    # print(m.qkv)
                    self.num_heads[m.qkv] = m.num_heads

            if isinstance(m, timm.models.vision_transformer.Mlp):
                if self.config.get('bottleneck', False):
                    if hasattr(m, 'fc2'):
                        ignored_layers.append(m.fc2)

        self.ignored_layers = ignored_layers
        
    def _get_importance_criterion(self):
        """获取重要性评估准则。

        Returns:
            tp.importance.Importance: 重要性评估器实例，可以是以下之一：
                - RandomImportance: 随机选择
                - GroupNormImportance: 基于L1/L2范数
                - GroupTaylorImportance: 基于Taylor展开
                - GroupHessianImportance: 基于Hessian矩阵

        Raises:
            ValueError: 如果指定了不支持的剪枝类型。
        """
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
        """初始化剪枝器。

        这个方法执行以下操作：
        1. 设置重要性评估方法
        2. 配置剪枝参数
        3. 设置剪枝比例调度器
        4. 创建MetaPruner实例

        Note:
            这个方法会在构造函数中自动调用，通常不需要手动调用。
        """
        imp = self._get_importance_criterion()
        logger.info(f"初始化剪枝器，目标剪枝率: {self.config.get('pruning_ratio', 0.5)}")
            
        # 收集所有不应该参与剪枝的参数
        unwrapped_parameters = []
        
        # 特殊处理logit_scale参数
        # if hasattr(self.model, 'logit_scale'):
        #     logit_scale_value = self.model.logit_scale.data.reshape(1, 1)
        #     logit_scale_param = nn.Parameter(logit_scale_value, requires_grad=self.model.logit_scale.requires_grad)
        #     unwrapped_parameters.append((logit_scale_param, 1))
        
        # 处理所有LayerScale2d的gamma参数
        # for m in self.model.modules():
        #     if isinstance(m, timm.models.fastvit.LayerScale2d):
        #         unwrapped_parameters.append((m.gamma, 0))
        
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
            'prune_head_dims': self.config.get('prune_head_dims', True),
            'head_pruning_ratio': self.config.get('head_pruning_ratio', 0.0),
            'unwrapped_parameters': unwrapped_parameters,
        }
        
        # 根据模式添加额外参数
        if pruning_mode != 'pre_training':
            scheduler_type = self.config.get('iterative_pruning_ratio_scheduler', 'linear')
            
            if scheduler_type == 'linear':
                # 只需要传递调度器函数，不需要提前计算比例
                pruner_kwargs.update({
                    'iterative_steps': self.total_pruning_steps,
                    'iterative_pruning_ratio_scheduler': linear_scheduler
                })
            
        self.pruner = tp.pruner.MetaPruner(**pruner_kwargs)
        
        # 如果是Hessian方法，需要初始化梯度累积
        if self.config.get('pruning_type') == 'hessian':
            self.pruner.importance.zero_grad()

    def get_current_pruning_ratio(self) -> float:
        """获取当前的剪枝比例。

        根据剪枝模式和当前状态计算应该使用的剪枝比例。
        对于渐进式剪枝，比例会随着剪枝次数逐步增加。

        Returns:
            float: 当前应该使用的剪枝比例，范围在[0, 1]之间。
        """
        if self.pruner is None:
            return 0.0
            
        # 训练前剪枝模式
        if self.config.get('pruning_mode') == 'pre_training':
            return self.config.get('pruning_ratio', 0.5)
            
        # 训练时迭代剪枝模式
        if self.config.get('pruning_mode') == 'during_training':
            if hasattr(self.pruner, 'current_step'):
                if hasattr(self.pruner, 'per_step_pruning_ratio'):
                    if 0 <= self.pruner.current_step < len(self.pruner.per_step_pruning_ratio):
                        return self.pruner.per_step_pruning_ratio[self.pruner.current_step]
                    
        return self.target_ratio

    def accumulate_gradient(self, model: nn.Module) -> bool:
        """记录当前batch的梯度信息（已经裁剪过）。
        
        只在warmup完成后开始积累梯度。
        
        Args:
            model: 当前的PyTorch模型
            
        Returns:
            bool: 如果当前在梯度收集阶段返回True，否则返回False
        """
        
        # 防止在同一个step重复调用
        if hasattr(self, '_last_accumulate_step') and self._last_accumulate_step == self.current_step:
            return self._last_accumulate_result
            
        # 打印当前梯度状态
        def log_gradient_stats():
            total_grad_norm = 0.0
            max_grad = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_norm = p.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    max_grad = max(max_grad, grad_norm)
            total_grad_norm = total_grad_norm ** 0.5
            return total_grad_norm, max_grad
            
        # 如果剪枝已完成，直接返回False
        if self.config.get('pruning_done', False):
            # logger.info(f"Step {self.current_step}: 剪枝已完成，不再收集梯度")
            self._last_accumulate_step = self.current_step
            self._last_accumulate_result = False
            return False
            
        # 在warmup阶段不积累梯度，返回False
        if self.current_step < self.warmup:
            if not hasattr(self, '_last_warmup_step') or self._last_warmup_step != self.current_step:
                current_warmup_step = self.current_step + 1
                grad_norm, max_grad = log_gradient_stats()
                # logger.info(f"Step {self.current_step}: Warmup {current_warmup_step}/{self.warmup}，"
                        #   f"跳过梯度收集 [总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
                self._last_warmup_step = self.current_step
            self._last_accumulate_step = self.current_step
            self._last_accumulate_result = False
            return False
            
        # 在pre_collect阶段不积累梯度，返回False
        if self.current_step < self.warmup + self.steps_between_warmup_and_first_gradient_collection:
            if not hasattr(self, '_last_training_step') or self._last_training_step != self.current_step:
                current_training_step = self.current_step - self.warmup + 1
                total_training_steps = self.steps_between_warmup_and_first_gradient_collection
                grad_norm, max_grad = log_gradient_stats()
                # logger.info(f"Step {self.current_step}: 正常训练 {current_training_step}/{total_training_steps}，"
                        #   f"跳过梯度收集 [总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
                self._last_training_step = self.current_step
            self._last_accumulate_step = self.current_step
            self._last_accumulate_result = False
            return False
            
        # 计算在当前周期内的位置
        steps_after_pre_collect = self.current_step - (self.warmup + self.steps_between_warmup_and_first_gradient_collection)
        pruning_cycle = self.recovery_steps + self.gradient_collect_steps
        cycle_position = steps_after_pre_collect % pruning_cycle
        
        # 检查是否刚完成最后一次剪枝，返回False
        if hasattr(self.pruner, 'current_step') and self.pruner.current_step >= self.total_pruning_steps:
            if not self.config.get('pruning_done', False):
                grad_norm, max_grad = log_gradient_stats()
                logger.info(f"Step {self.current_step}: 所有剪枝步骤已完成 " 
                          f"[总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
                self.config['pruning_done'] = True
            self._last_accumulate_step = self.current_step
            self._last_accumulate_result = False
            return False
        
        # 在恢复训练阶段不积累梯度，返回False
        if cycle_position >= self.gradient_collect_steps:
            current_recovery_step = cycle_position - self.gradient_collect_steps + 1
            grad_norm, max_grad = log_gradient_stats()
            # logger.info(f"Step {self.current_step}: 恢复训练中，第{current_recovery_step}/{self.recovery_steps}步 "
                    #   f"[总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
            self._last_accumulate_step = self.current_step
            self._last_accumulate_result = False
            return False
        
        # 在梯度收集阶段
        if cycle_position == 0:
            # logger.info(f"Step {self.current_step}: 开始收集梯度 (计划收集{self.gradient_collect_steps}步)")
            # 开始新的梯度收集周期，但不清空梯度（因为已经有了当前step的梯度）
            # model.zero_grad()  # 注释掉这行，不清空已有的梯度
            grad_norm, max_grad = log_gradient_stats()
            # logger.info(f"Step {self.current_step}: 清零后梯度状态 [总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
            # 初始化计数器
            if not hasattr(self, '_collected_steps'):
                self._collected_steps = 0
            self._collected_steps = 1  # 从1开始，因为当前step已经有了梯度
            # logger.info(f"Step {self.current_step}: 已收集{self._collected_steps}/{self.gradient_collect_steps}个裁剪后的梯度 "
                    #   f"[总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
            
        # 更新收集的步数
        else:
            self._collected_steps = getattr(self, '_collected_steps', 0) + 1
            grad_norm, max_grad = log_gradient_stats()
            # logger.info(f"Step {self.current_step}: 已收集{self._collected_steps}/{self.gradient_collect_steps}个裁剪后的梯度 "
                    #   f"[总梯度范数: {grad_norm:.4f}, 最大梯度: {max_grad:.4f}]")
        
        # 记录本次调用结果
        self._last_accumulate_step = self.current_step
        self._last_accumulate_result = True
        
        # 返回True表示在梯度收集阶段
        return True

    def calculate_importance_from_history(self) -> Optional[Dict[str, torch.Tensor]]:
        """基于累积的裁剪后梯度计算重要性分数。

        Returns:
            Optional[Dict[str, torch.Tensor]]: 包含每个参数重要性分数的字典。
                如果没有足够的梯度，返回None。

        Note:
            重要性分数基于参数值和其平均梯度（已裁剪）的乘积计算。
            使用已经裁剪的梯度计算平均值，确保数值稳定性。
        """
        if not hasattr(self, '_collected_steps') or self._collected_steps < self.gradient_collect_steps:
            return None
            
        # 计算重要性（使用已经累积的裁剪后梯度）
        importance = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 计算平均梯度（通过除以收集的步数）
                param.grad = param.grad / self._collected_steps
                # 计算Taylor重要性（使用平均梯度）
                importance[name] = torch.abs(param * param.grad)
                
        return importance

    def should_prune(self) -> bool:
        """判断是否应该执行剪枝。

        检查以下条件：
        1. 剪枝功能是否启用
        2. 是否在指定的剪枝epoch范围内
        3. 是否有足够的梯度
        4. 是否达到剪枝时机

        Returns:
            bool: 如果应该执行剪枝返回True，否则返回False。
        """
        if not self.config.get('do_pruning', False) or self.config.get('pruning_done', False):
            return False
            
        current_epoch = self.config.get('current_epoch', 0)
        
        # 训练前剪枝模式
        if self.config.get('pruning_mode') == 'pre_training':
            # 只在初始化时执行一次，之后直接返回 False
            if hasattr(self, '_pre_training_pruning_done'):
                return False
            # 第一次执行时设置标记
            self._pre_training_pruning_done = True
            return current_epoch == self.config.get('pruning_start_epoch', 0) and self.current_step == 0
            
        # 训练时迭代剪枝模式
        if self.config.get('pruning_mode') == 'during_training':
            # 检查是否在剪枝区间内
            if current_epoch < self.config.get('pruning_start_epoch', 0):
                return False
            if self.config.get('pruning_end_epoch') is not None and current_epoch > self.config.get('pruning_end_epoch'):
                return False
                
            # 1. 检查是否完成warmup和pre_collect
            if self.current_step < self.warmup + self.steps_between_warmup_and_first_gradient_collection:
                return False
                
            # 2. 计算在当前周期内的位置
            steps_after_pre_collect = self.current_step - (self.warmup + self.steps_between_warmup_and_first_gradient_collection)
            pruning_cycle = self.recovery_steps + self.gradient_collect_steps
            cycle_position = steps_after_pre_collect % pruning_cycle
                
            # 3. 检查是否收集了足够的梯度
            if not hasattr(self, '_collected_steps') or self._collected_steps < self.gradient_collect_steps:
                return False

            # 4. 检查是否已经完成所有剪枝步骤
            if hasattr(self.pruner, 'current_step') and self.pruner.current_step >= self.total_pruning_steps:
                return False

            # 5. 检查是否是剪枝步骤
            should_prune = cycle_position == self.gradient_collect_steps
            
            if should_prune:
                # 如果这是最后一次剪枝，提前设置pruning_done标志
                if hasattr(self.pruner, 'current_step') and self.pruner.current_step == self.total_pruning_steps - 1:
                    logger.info(f"Step {self.current_step}: 准备执行最后一次剪枝")
                
                logger.info(f"Step {self.current_step}: 准备执行第{self.pruner.current_step + 1}/{self.total_pruning_steps}次剪枝 "
                           f"(warmup后第{steps_after_pre_collect+self.steps_between_warmup_and_first_gradient_collection}步)，已收集{self._collected_steps}/{self.gradient_collect_steps}个梯度")
            return should_prune
            
        return False

    def step(self, model_out=None, criterion=None) -> Optional[Dict[str, Any]]:
        """执行一步剪枝操作。

        Args:
            model_out: 模型输出字典，包含特征等信息。
            criterion: 损失函数实例，用于计算重要性。

        Returns:
            Optional[Dict[str, Any]]: 包含剪枝信息的字典，如果不需要剪枝返回None。
        """
        if self.config.get('pruning_done', False):
            return None
            
        try:
            # 检查是否需要执行剪枝
            if not self.should_prune():
                return None

            # 添加调试信息
            # logger.info(f"Current pruning state:")
            # logger.info(f"- Current step: {self.pruner.current_step if hasattr(self.pruner, 'current_step') else 0}")
            # logger.info(f"- Total steps: {self.total_pruning_steps}")
            
            # 根据剪枝模式选择不同的实现
            if self.config.get('pruning_mode') == 'during_training':
                # 使用累积的梯度信息计算重要性
                importance_scores = self.calculate_importance_from_history()
                if importance_scores is None:
                    logger.warning("No importance scores available")
                    return None
                    
                # 设置重要性分数
                if hasattr(self.pruner.importance, 'set_importance'):
                    self.pruner.importance.set_importance(importance_scores)
            else:
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
            current_step = self.pruner.current_step
            if current_step >= len(self.pruner.per_step_pruning_ratio):
                logger.warning(f"当前步骤({current_step})超出剪枝比例数组长度({len(self.pruner.per_step_pruning_ratio)})，跳过剪枝")
                return None
                
            current_ratio = self.pruner.per_step_pruning_ratio[current_step]
            logger.info(f"Executing pruning step {current_step+1}/{self.total_pruning_steps} with ratio {current_ratio}")
            logger.info(f"Per step pruning ratios: {self.pruner.per_step_pruning_ratio}, length: {len(self.pruner.per_step_pruning_ratio)}")
            
            # 执行剪枝操作
            pruning_groups = list(self.pruner.step(interactive=True))
            if not pruning_groups:  # 如果没有可剪枝的组
                logger.warning("No prunable groups found")
                return None
                
            # 执行实际的剪枝操作
            for i, group in enumerate(pruning_groups):
                group.prune()
                
            logger.info(f"Successfully completed pruning step {current_step+1}/{self.total_pruning_steps}")
            
            # 更新attention heads
            head_id = 0
            for m in self.model.modules():
                if hasattr(m, 'num_heads') and hasattr(m, 'qkv'):
                    if not hasattr(m, 'latent_len'):
                        old_heads = m.num_heads
                        old_dim = m.head_dim
                        m.num_heads = self.pruner.num_heads[m.qkv]
                        m.head_dim = m.qkv.out_features // (3 * m.num_heads)
                        head_id += 1
                        
            # 更新模型统计信息
            self.current_macs, self.current_params = tp.utils.count_ops_and_params(
                self.model, self.example_inputs)
            macs_reduction = (self.base_macs - self.current_macs) / self.base_macs
            params_reduction = (self.base_params - self.current_params) / self.base_params
            logger.info(f"MACs: {self.base_macs:,} => {self.current_macs:,} ({macs_reduction*100:.2f}% reduction)")
            logger.info(f"Params: {self.base_params:,} => {self.current_params:,} ({params_reduction*100:.2f}% reduction)")
            
            # 更新迭代剪枝状态
            self.current_ratio = current_ratio
            
            # 如果是训练前剪枝模式，执行完就标记为完成
            if self.config.get('pruning_mode') == 'pre_training':
                self.config['pruning_done'] = True
                logger.info("Pre-training pruning completed, no more pruning will be performed.")
            
            # 在最后一次剪枝完成时重新初始化优化器
            if hasattr(self.pruner, 'current_step') and self.pruner.current_step == self.total_pruning_steps:
                logger.info("最后一次剪枝完成，准备重新初始化优化器...")
                
                # 获取当前优化器和调度器
                if hasattr(self.model, 'optimizer'):
                    new_optimizer = self.reinitialize_optimizer(
                        self.model, 
                        self.model.optimizer
                    )
                    self.model.optimizer = new_optimizer
                    logger.info("优化器重新初始化完成")
                    
                    # 验证梯度清零是否正常
                    self.model.zero_grad()
                    total_grad_norm = 0.0
                    max_grad = 0.0
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_norm = p.grad.data.norm(2).item()
                            total_grad_norm += grad_norm ** 2
                            max_grad = max(max_grad, grad_norm)
                    total_grad_norm = total_grad_norm ** 0.5
                    logger.info(f"重新初始化后的梯度清零测试 - 总梯度范数: {total_grad_norm:.4f}, 最大梯度: {max_grad:.4f}")
            
            # 剪枝完成后清理内存
            clean_memory()
            
            # 返回剪枝信息
            return {
                'pruning_type': self.config.get('pruning_type'),
                'pruning_ratio': current_ratio,
                'macs_reduction': macs_reduction,
                'current_macs': self.current_macs,
                'base_macs': self.base_macs,
                'current_step': self.pruner.current_step,
                'total_steps': self.total_pruning_steps
            }
                
        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
            logger.error(traceback.format_exc())
            if self.config.get('distributed', False):
                torch.distributed.barrier()
            raise

    def state_dict(self) -> Dict[str, Any]:
        """返回需要保存的剪枝状态。

        Returns:
            Dict[str, Any]: 包含完整剪枝状态的字典，可用于恢复剪枝状态。

        Note:
            这个方法通常用于保存检查点。
        """
        try:
            state = {
                'config': self.config,
                'base_macs': self.base_macs,
                'base_params': self.base_params,
                'current_macs': self.current_macs,
                'current_params': self.current_params,
                'pruning_done': self.config.get('pruning_done', False),
                'num_heads': self.num_heads,
                # 迭代剪枝状态
                'current_ratio': self.current_ratio,
                'total_pruning_steps': self.total_pruning_steps,
                'recovery_steps': self.recovery_steps,
                'gradient_collect_steps': self.gradient_collect_steps,
                'warmup': self.warmup,
                'target_ratio': self.target_ratio,
                'steps_between_warmup_and_first_gradient_collection': self.steps_between_warmup_and_first_gradient_collection
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
        """从保存的状态字典加载剪枝状态。

        Args:
            state_dict: 包含剪枝状态的字典，应该是由state_dict()方法生成的。

        Raises:
            Exception: 如果加载过程中发生错误。
        """
        try:
            # 加载基本状态
            self.config.update(state_dict['config'])
            self.base_macs = state_dict['base_macs']
            self.base_params = state_dict['base_params']
            self.current_macs = state_dict['current_macs']
            self.current_params = state_dict['current_params']
            self.num_heads = state_dict['num_heads']
            
            # 加载迭代剪枝状态
            self.current_ratio = state_dict.get('current_ratio', 0.0)
            self.total_pruning_steps = state_dict.get('total_pruning_steps', self.config.get('iterative_steps', 1))
            self.recovery_steps = state_dict.get('recovery_steps', 1500)
            self.gradient_collect_steps = state_dict.get('gradient_collect_steps', 10)
            self.warmup = state_dict.get('warmup', 100)
            self.target_ratio = state_dict.get('target_ratio', self.config.get('pruning_ratio', 0.5))
            self.steps_between_warmup_and_first_gradient_collection = state_dict.get('steps_between_warmup_and_first_gradient_collection', 0)
            
            # 如果有pruner状态且pruner存在，加载pruner状态
            if 'pruner_state' in state_dict and self.pruner is not None and hasattr(self.pruner, 'load_state_dict'):
                self.pruner.load_state_dict(state_dict['pruner_state'])
                
            logger.info("Successfully loaded pruning state")
            
        except Exception as e:
            logger.error(f"Error in loading state_dict: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_step(self, step: int) -> None:
        """更新当前训练步数。

        Args:
            step: 当前的训练步数。

        Note:
            这个方法通常在每个训练步骤后调用。
            同时更新config中的current_step以保持同步。
        """
        self.current_step = step
        self.config['current_step'] = step  # 确保config中的step也同步更新

    def reinitialize_optimizer(self, model, optimizer, scheduler=None):
        """剪枝完成后重新初始化优化器"""
        logger.info("重新初始化优化器...")
        
        # 1. 收集当前优化器的状态
        old_state = optimizer.state_dict()
        old_lr = optimizer.param_groups[0]['lr']
        
        # 2. 创建参数组
        params_groups = []
        # 2.1 找出所有需要梯度的参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # 2.2 区分bias和非bias参数
        bias_params = []
        non_bias_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bias' in name:
                    bias_params.append(param)
                else:
                    non_bias_params.append(param)
        
        # 2.3 构建参数组
        params_groups = [
            {'params': non_bias_params, 'weight_decay': old_state['param_groups'][0].get('weight_decay', 0.0)},
            {'params': bias_params, 'weight_decay': 0.0}  # bias通常不使用weight decay
        ]
        
        # 3. 创建新优化器
        new_optimizer = type(optimizer)(
            params_groups,
            lr=old_lr,
            **{k: v for k, v in optimizer.defaults.items() if k != 'lr'}
        )
        
        # 4. 验证新优化器状态
        self._verify_optimizer_state(model, new_optimizer)
        
        return new_optimizer

    def _verify_optimizer_state(self, model, optimizer):
        """验证优化器状态"""
        managed_params = set()
        for group in optimizer.param_groups:
            managed_params.update(id(p) for p in group['params'])
        
        model_params = set(id(p) for p in model.parameters() if p.requires_grad)
        
        if managed_params != model_params:
            missing_params = len(model_params - managed_params)
            extra_params = len(managed_params - model_params)
            logger.warning(f"优化器状态不匹配！缺少{missing_params}个参数，多余{extra_params}个参数")
            return False
        
        logger.info("优化器状态验证通过")
        return True
    
    def monitor_gradient_state(self, model, step):
        """监控梯度状态"""
        total_params = 0
        params_with_grad = 0
        params_in_backward = 0
        
        for name, p in model.named_parameters():
            total_params += 1
            if p.requires_grad:
                params_with_grad += 1
            if p.grad is not None:
                params_in_backward += 1
            
        logger.info(f"Step {step} 梯度状态统计：")
        logger.info(f"- 总参数数量: {total_params}")
        logger.info(f"- 需要梯度的参数: {params_with_grad}")
        logger.info(f"- 有梯度的参数: {params_in_backward}")
