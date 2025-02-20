import torch
import sys
import copy
import types
import torch.nn as nn
import onnx
import argparse
import open_clip
from src.open_clip import create_model_and_transforms
import torch.nn.functional as F
from typing import Optional

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def _expand_token(token, batch_size: int):
    """将token扩展到指定的batch size"""
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    """全局池化文本特征
    
    Args:
        x: 输入特征
        text: 原始文本tokens
        pool_type: 池化类型，可选 'first', 'last', 'argmax'
    """
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

def replace_gelu_with_quick_gelu(model: nn.Module) -> nn.Module:
    """将模型中的GELU替换为QuickGELU"""
    class QuickGELU(nn.Module):
        def forward(self, x: torch.Tensor):
            return x * torch.sigmoid(1.702 * x)
            
    model = copy.deepcopy(model)
    for name, module in model.named_modules():  # 使用named_modules而不是named_children
        if isinstance(module, nn.GELU):
            parent = model
            name_parts = name.split('.')
            # 遍历到倒数第二层
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            # 替换最后一层的GELU
            setattr(parent, name_parts[-1], QuickGELU())
    return model

class VisualEncoder:
    def __init__(self, model, preprocess, framework, reparam=True, model_arch=None, normalize=True):
        self.framework = framework
        self.reparam = reparam
        self.model_arch = model_arch
        self.normalize = normalize
        self.encoder = self._get_visual_encoder(model)
        self.encoder.eval()
        self.output_path = None
        self.preprocess = preprocess  # 保存预处理函数
    
    def _get_visual_encoder(self, model):
        if self.framework == 'open_clip':
            visual_encoder = model.visual
            # print(f"\nVisual Encoder: {visual_encoder}")
            if self.reparam:
                if self.model_arch and 'repvit' in self.model_arch.lower():
                    print("Detected RepVit model, performing fusion...")
                    visual_encoder = self._fuse_model(visual_encoder)
                    print("Model fusion completed.")
                    print("\nModel Parameters:")
                    for name, module in visual_encoder.named_modules():
                        if isinstance(module, nn.Conv2d):
                            print(f"{name}: in={module.in_channels}, out={module.out_channels}, "
                                f"kernel={module.kernel_size}, stride={module.stride}")
                else:
                    print("Detected FastVit model, performing reparameterization...")
                    visual_encoder = self._reparameterize_model(visual_encoder)
                    print("Model reparameterization completed.")
                    # print(f"\nVisual Encoder after reparameterization: {visual_encoder}")

            # 包装带可选归一化的编码器
            class NormalizedEncoder(nn.Module):
                def __init__(self, base_encoder, normalize):
                    super().__init__()
                    self.base_encoder = base_encoder
                    self.normalize = normalize

                def forward(self, x):
                    features = self.base_encoder(x)
                    if self.normalize:
                        # # 手动实现 L2 归一化，使其与计算图一致
                        # square = features * features  # Mul 操作
                        # sum_square = torch.sum(square, dim=-1, keepdim=True)  # ReduceSum 操作
                        # sqrt = torch.sqrt(sum_square)  # Sqrt 操作
                        # features = features / sqrt  # Div 操作
                        features = F.normalize(features, dim=-1)
                    return features

            return NormalizedEncoder(visual_encoder, self.normalize)
        elif self.framework == 'mobileclip':
            return NormalizedEncoder(model.image_encoder, self.normalize)
        raise ValueError(f"Unsupported framework: {self.framework}")
    
    @staticmethod
    def _reparameterize_model(model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model
    @staticmethod
    def _fuse_model(model: nn.Module) -> nn.Module:
        """reparameterize model for repvit."""
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "fuse"):
                module.fuse()
        return model
    
    def verify_outputs(self, test_image):
        """验证PyTorch和ONNX输出是否一致"""
        import numpy as np
        import onnxruntime
        
        # PyTorch预处理和推理
        self.encoder.eval()
        with torch.no_grad():
            # 使用相同的预处理
            processed_input = self.preprocess(test_image).unsqueeze(0)
            pytorch_output = self.encoder(processed_input)
        
        # ONNX预处理和推理
        ort_session = onnxruntime.InferenceSession(self.output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: processed_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 比较输出
        pytorch_output = pytorch_output.cpu().numpy()
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        print(f"\nOutput Verification Results:")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # 打印更多信息以帮助调试
        print("\nInput tensor info:")
        print(f"Shape: {processed_input.shape}")
        print(f"Range: [{processed_input.min():.3f}, {processed_input.max():.3f}]")
        print(f"Mean: {processed_input.mean():.3f}")
        print(f"Std: {processed_input.std():.3f}")
        
        print("\nOutput tensor info:")
        print(f"PyTorch shape: {pytorch_output.shape}")
        print(f"ONNX shape: {onnx_output.shape}")
        print(f"PyTorch range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
        print(f"ONNX range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
        
        return max_diff < 1e-5
    
    def export_onnx(self, output_path, resolution=256, verbose=False, verify=False, test_image=None, dynamic_axes=False):
        # print(f"\nVisual Encoder: {self.encoder}")
        
        dummy_input = torch.randn(1, 3, resolution, resolution)
        
        def get_shape_hook(name):
            def hook(model, input, output):
                # 只打印主要层的输出
                if name in ['trunk.stem.0', 'trunk.stem.1', 'trunk.stem.2', 
                           'trunk.stages.0', 'trunk.stages.1', 'trunk.stages.2', 
                           'trunk.stages.3', 'trunk.final_conv', 'head']:
                    print(f"{name} output shape: {output.shape}")
            return hook
        
        hook_handles = []
        def register_hooks(model, prefix=''):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                handle = module.register_forward_hook(get_shape_hook(full_name))
                hook_handles.append(handle)
                register_hooks(module, full_name)
        
        print("\n=== Visual Encoder Feature Shapes ===")
        print(f"dummy_input shape: {dummy_input.shape}\n")
        
        register_hooks(self.encoder)
        
        print("Running forward pass to get feature shapes...")
        with torch.no_grad():
            _ = self.encoder(dummy_input)
        print("=== End of Visual Encoder Feature Shapes ===\n")
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
            
        # 保存输出路径
        if '_visual' not in output_path:
            self.output_path = output_path.replace('.onnx', '_visual.onnx')
        else:
            self.output_path = output_path
            
        # 导出ONNX
        export_args = {
            'model': self.encoder,
            'args': dummy_input,
            'f': self.output_path,
            'opset_version': 18,
            'verbose': verbose,
            'export_params': True,
            'do_constant_folding': False,
            'input_names': ['input'],
            'output_names': ['image_features'],
        }
        
        if dynamic_axes:
            export_args['dynamic_axes'] = {
                'input': {0: 'batch_size'},
                'image_features': {0: 'batch_size'}
            }
            
        torch.onnx.export(**export_args)
        
        print(f"Visual Encoder has been exported to {self.output_path}")
        
        # 使用onnxsim简化模型
        print("\n使用onnxsim简化模型...")
        import onnxsim
        onnx_model = onnx.load(self.output_path)
        try:
            # 简化模型
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                print("模型简化成功，保存简化后的模型...")
                onnx.save(model_simp, self.output_path)
            else:
                print("警告: 模型简化失败，将使用原始模型")
                onnx.save(onnx_model, self.output_path)
        except Exception as e:
            print(f"警告: 模型简化过程中出错: {str(e)}")
            print("将使用原始模型")
            onnx.save(onnx_model, self.output_path)
        
        # 验证ONNX模型
        onnx_model = onnx.load(self.output_path)
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
            print("Visual encoder ONNX model is valid.")
            
            # 仅在指定验证时执行
            if verify:
                print("\nVerifying outputs...")
                verification_result = self.verify_outputs(test_image)
                if verification_result:
                    print("Output verification passed! PyTorch and ONNX outputs are consistent.")
                else:
                    print("Warning: Output verification failed! PyTorch and ONNX outputs have significant differences.")
                return verification_result
            return True
        except onnx.checker.ValidationError as e:
            print("Visual encoder ONNX model is invalid.")
            print(f"Error: {e}")
            return False

class TextEncoder:
    def __init__(self, model, tokenizer, framework, normalize=True, export_mode="full", use_quick_gelu=False):
        self.framework = framework
        self.normalize = normalize
        self.export_mode = export_mode
        self.encoder = self._get_text_encoder(model)
        if use_quick_gelu:
            self.encoder = replace_gelu_with_quick_gelu(self.encoder)
        self.encoder.eval()
        self.output_path = None
        self.tokenizer = tokenizer
    
    def _get_text_encoder(self, model):
        if self.framework == 'open_clip':
            text_encoder = model.text
            
            if self.export_mode == "adapter":
                # 包装adapter以支持normalize
                class NormalizedAdapter(nn.Module):
                    def __init__(self, adapter, normalize):
                        super().__init__()
                        self.adapter = adapter
                        self.normalize = normalize

                    def forward(self, x):
                        features = self.adapter(x)
                        if self.normalize:
                            # square = features * features
                            # sum_square = torch.sum(square, dim=-1, keepdim=True)
                            # sqrt = torch.sqrt(sum_square)
                            # features = features / sqrt
                            features = F.normalize(features, dim=-1)
                        return features
                
                return NormalizedAdapter(text_encoder.adapter, self.normalize)
            elif self.export_mode == "base":
                # 创建一个不包含adapter的text encoder副本
                text_encoder = copy.deepcopy(text_encoder)
                
                # 创建一个新的forward方法，与原始TextTransformer完全一致，但不使用adapter
                def new_forward(self, text):
                    cast_dtype = self.transformer.get_cast_dtype()
                    seq_len = text.shape[1]

                    x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
                    attn_mask = self.attn_mask
                    if self.cls_emb is not None:
                        seq_len += 1
                        x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
                        cls_mask = self.build_cls_mask(text, cast_dtype)
                        if attn_mask is not None:
                            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

                    x = x + self.positional_embedding[:seq_len].to(cast_dtype)
                    x = self.transformer(x, attn_mask=attn_mask)

                    # x.shape = [batch_size, n_ctx, transformer.width]
                    if self.cls_emb is not None:
                        # presence of appended cls embed (CoCa) overrides pool_type, always take last token
                        pooled, tokens = text_global_pool(x, pool_type='last')
                        pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
                    else:
                        x = self.ln_final(x)
                        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

                    if self.text_projection is not None:
                        if isinstance(self.text_projection, nn.Linear):
                            pooled = self.text_projection(pooled)
                        else:
                            pooled = pooled @ self.text_projection

                    if self.output_tokens:
                        return pooled, tokens
                    return pooled

                def build_cls_mask(self, text, cast_dtype: torch.dtype):
                    """构建cls token的attention mask"""
                    cls_mask = (text != self.pad_id).unsqueeze(1)
                    cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
                    additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
                    additive_mask.fill_(0)
                    additive_mask.masked_fill_(~cls_mask, float("-inf"))
                    additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
                    return additive_mask

                # 添加build_cls_mask方法和替换forward方法
                text_encoder.build_cls_mask = types.MethodType(build_cls_mask, text_encoder)
                text_encoder.forward = types.MethodType(new_forward, text_encoder)
                
                # 移除adapter属性
                if hasattr(text_encoder, 'adapter'):
                    delattr(text_encoder, 'adapter')
            
            # 包装带可选归一化的编码器
            class NormalizedTextEncoder(nn.Module):
                def __init__(self, base_encoder, normalize):
                    super().__init__()
                    self.base_encoder = base_encoder
                    self.normalize = normalize

                def forward(self, x):
                    features = self.base_encoder(x)
                    
                    # 应用归一化（如果需要）
                    if self.normalize:
                        # square = features * features
                        # sum_square = torch.sum(square, dim=-1, keepdim=True)
                        # sqrt = torch.sqrt(sum_square)
                        # features = features / sqrt
                        features = F.normalize(features, dim=-1)
                            
                    return features

            return NormalizedTextEncoder(text_encoder, self.normalize)
            
        elif self.framework == 'mobileclip':
            if self.export_mode == "adapter":
                return NormalizedAdapter(model.text_encoder.adapter, self.normalize)
            elif self.export_mode == "base":
                text_encoder = copy.deepcopy(model.text_encoder)
                
                # 创建一个新的forward方法，与原始TextTransformer完全一致，但不使用adapter
                def new_forward(self, text):
                    cast_dtype = self.transformer.get_cast_dtype()
                    seq_len = text.shape[1]

                    x = self.token_embedding(text).to(cast_dtype)
                    attn_mask = self.attn_mask
                    if self.cls_emb is not None:
                        seq_len += 1
                        x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
                        cls_mask = self.build_cls_mask(text, cast_dtype)
                        if attn_mask is not None:
                            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

                    x = x + self.positional_embedding[:seq_len].to(cast_dtype)
                    x = self.transformer(x, attn_mask=attn_mask)

                    if self.cls_emb is not None:
                        pooled, tokens = text_global_pool(x, pool_type='last')
                        pooled = self.ln_final(pooled)
                    else:
                        x = self.ln_final(x)
                        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

                    if self.text_projection is not None:
                        if isinstance(self.text_projection, nn.Linear):
                            pooled = self.text_projection(pooled)
                        else:
                            pooled = pooled @ self.text_projection

                    if self.output_tokens:
                        return pooled, tokens
                    return pooled

                def build_cls_mask(self, text, cast_dtype: torch.dtype):
                    """构建cls token的attention mask"""
                    cls_mask = (text != self.pad_id).unsqueeze(1)
                    cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
                    additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
                    additive_mask.fill_(0)
                    additive_mask.masked_fill_(~cls_mask, float("-inf"))
                    additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
                    return additive_mask

                # 添加build_cls_mask方法和替换forward方法
                text_encoder.build_cls_mask = types.MethodType(build_cls_mask, text_encoder)
                text_encoder.forward = types.MethodType(new_forward, text_encoder)
                
                # 移除adapter属性
                if hasattr(text_encoder, 'adapter'):
                    delattr(text_encoder, 'adapter')
                    
                return NormalizedTextEncoder(text_encoder, self.normalize)
            else:
                return NormalizedTextEncoder(model.text_encoder, self.normalize)
                
        raise ValueError(f"Unsupported framework: {self.framework}")
    
    def verify_outputs(self, test_texts):
        """验证PyTorch和ONNX输出是否一致"""
        import numpy as np
        import onnxruntime
        
        # 存储所有文本特征
        all_features = []
        
        # 逐个处理文本，确保batch size为1
        for single_text in test_texts:
            if self.export_mode == "adapter":
                # adapter模式下，使用随机特征作为输入
                with torch.no_grad():  # 添加 no_grad 上下文
                    input_features = torch.randn(1, 512)  # 假设维度是512
                    pytorch_output = self.encoder(input_features)
                
                # ONNX推理
                ort_session = onnxruntime.InferenceSession(self.output_path)
                ort_inputs = {
                    ort_session.get_inputs()[0].name: input_features.cpu().numpy()
                }
            else:
                # 1. Tokenize单个文本
                text_tokens = self.tokenizer([single_text])
                
                # 2. PyTorch推理
                with torch.no_grad():
                    pytorch_output = self.encoder(text_tokens)
                
                # 3. ONNX推理
                ort_session = onnxruntime.InferenceSession(self.output_path)
                ort_inputs = {
                    ort_session.get_inputs()[0].name: text_tokens.cpu().numpy().astype(np.int32)
                }
            
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # 4. 比较输出
            pytorch_output = pytorch_output.cpu().numpy()  # 现在不会有梯度问题
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            
            if self.export_mode == "adapter":
                print(f"\nOutput Verification Results for adapter:")
            else:
                print(f"\nOutput Verification Results for text: {single_text}")
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            
            if max_diff >= 1e-5:
                return False
        
        return True
    
    def export_onnx(self, output_path, verbose=False, verify=False, test_texts=None, dynamic_axes=False):
        print(f"\nText Encoder: {self.encoder}")
        
        if self.export_mode == "adapter":
            # adapter的输入尺寸应该是text encoder的输出尺寸
            dummy_input = torch.randn(1, 512, dtype=torch.float32)  # 使用float32类型
            input_name = 'text_features'
            output_name = 'adapter_features'
            if '_text' not in output_path:
                self.output_path = output_path.replace('.onnx', '_text_adapter.onnx')
            else:
                self.output_path = output_path.replace('_text.onnx', '_text_adapter.onnx')
        else:
            dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.int32)
            input_name = 'input'
            output_name = 'text_features'
            if '_text' not in output_path:
                suffix = '_text_base.onnx' if self.export_mode == "base" else '_text.onnx'
                self.output_path = output_path.replace('.onnx', suffix)
            else:
                self.output_path = output_path

        def get_shape_hook(name):
            def hook(model, input, output):
                # 处理tuple类型的输出
                if isinstance(output, tuple):
                    shapes = [o.shape if hasattr(o, 'shape') else type(o) for o in output]
                    print(f"{name} output shapes: {shapes}")
                else:
                    print(f"{name} output shape: {output.shape}")
            return hook
        
        hook_handles = []
        def register_hooks(model, prefix=''):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                handle = module.register_forward_hook(get_shape_hook(full_name))
                hook_handles.append(handle)
                register_hooks(module, full_name)
        
        print("\n=== Text Encoder Feature Shapes ===")
        print(f"dummy_input shape: {dummy_input.shape}\n")
        
        register_hooks(self.encoder)
        
        print("Running forward pass to get feature shapes...")
        with torch.no_grad():
            _ = self.encoder(dummy_input)
        print("=== End of Text Encoder Feature Shapes ===\n")
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
            
        # Export to ONNX
        export_args = {
            'model': self.encoder,
            'args': dummy_input,
            'f': self.output_path,
            'opset_version': 18,
            'verbose': verbose,
            'export_params': True,
            'do_constant_folding': False,
            'input_names': [input_name], 
            'output_names': [output_name], 
        }
        
        if dynamic_axes:
            export_args['dynamic_axes'] = {
                input_name: {0: 'batch_size'}, 
                output_name: {0: 'batch_size'}
            }
            
        torch.onnx.export(**export_args)
        
        print(f"Text Encoder has been exported to {self.output_path}")
        
        # 使用onnxsim简化模型
        print("\n使用onnxsim简化模型...")
        import onnxsim
        onnx_model = onnx.load(self.output_path)
        try:
            # 简化模型
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                print("模型简化成功，保存简化后的模型...")
                onnx.save(model_simp, self.output_path)
            else:
                print("警告: 模型简化失败，将使用原始模型")
                onnx.save(onnx_model, self.output_path)
        except Exception as e:
            print(f"警告: 模型简化过程中出错: {str(e)}")
            print("将使用原始模型")
            onnx.save(onnx_model, self.output_path)
        
        # Verify model
        onnx_model = onnx.load(self.output_path)
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
            print("Text encoder ONNX model is valid.")
            
            # 仅在指定验证时执行
            if verify:
                print("\nVerifying outputs...")
                verification_result = self.verify_outputs(test_texts)
                if verification_result:
                    print("Output verification passed! PyTorch and ONNX outputs are consistent.")
                else:
                    print("Warning: Output verification failed! PyTorch and ONNX outputs have significant differences.")
                return verification_result
            return True
            
        except onnx.checker.ValidationError as e:
            print("Text encoder ONNX model is invalid.")
            print(f"Error: {e}")
            return False

def parsers(args):
    parser = argparse.ArgumentParser(description='Export CLIP model to ONNX')
    parser.add_argument('--framework', type=str, required=True,
                       choices=['open_clip', 'mobileclip'])
    parser.add_argument('--model-arch', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--verbose-onnx', action='store_true')
    parser.add_argument('--output-path', type=str, default="./")
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--reparam', type=lambda x: (str(x).lower() == 'true'),
                       choices=[True, False], default=True)
    parser.add_argument('--export-image', action='store_true')
    parser.add_argument('--export-text', action='store_true')
    parser.add_argument('--export-all', action='store_true')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX output against PyTorch output')
    parser.add_argument('--dynamic-axes', action='store_true',
                       help='Export ONNX with dynamic axes for batch dimension')
    parser.add_argument('--normalize', type=lambda x: (str(x).lower() == 'true'),
                       choices=[True, False], default=True,
                       help='Whether to add normalization in the exported model')
    parser.add_argument('--export-mode', type=str, default="full",
                       choices=["full", "base", "adapter"],
                       help='Export mode for text encoder')
    parser.add_argument('--use-quick-gelu', type=lambda x: (str(x).lower() == 'true'),
                       choices=[True, False], default=False,
                       help='Whether to use quickgelu in the exported model')
    return parser.parse_args(args)

def main(args):
    args = parsers(args)
    
    # 只在需要验证时才准备测试数据
    test_image = None
    test_texts = None
    if args.verify:
        from PIL import Image
        test_image = Image.open("../assets/CLIP.png").convert('RGB')
        test_texts = ["a diagram", "a dog", "a cat"]
    
    # Load model and transforms
    model, _, preprocess_val = create_model_and_transforms(
        model_name=args.model_arch,
        pretrained=args.model_path,
        image_mean=(0, 0, 0),
        image_std=(1, 1, 1),
        image_interpolation="bilinear",
        force_image_size=(args.resolution, args.resolution)
    )
    
    export_results = []
    
    # Export visual encoder
    if args.export_all or args.export_image:
        print("\nExporting visual encoder...")
        visual_encoder = VisualEncoder(
            model=model,
            preprocess=preprocess_val,
            framework=args.framework, 
            reparam=args.reparam, 
            model_arch=args.model_arch,
            normalize=args.normalize  # 添加归一化参数
        )
        visual_result = visual_encoder.export_onnx(
            output_path=args.output_path,
            resolution=args.resolution,
            verbose=args.verbose_onnx,
            verify=args.verify,
            test_image=test_image if args.verify else None,
            dynamic_axes=args.dynamic_axes
        )
        export_results.append(('Visual Encoder', visual_result))
    
    # Export text encoder
    if args.export_all or args.export_text:
        print("\nExporting text encoder...")
        tokenizer = open_clip.get_tokenizer(args.model_arch)
        
        if args.export_mode == "full":
            # 导出完整的text encoder
            text_encoder = TextEncoder(
                model=model,
                tokenizer=tokenizer,
                framework=args.framework,
                normalize=args.normalize,
                export_mode="full",
                use_quick_gelu=args.use_quick_gelu
            )
            text_result = text_encoder.export_onnx(
                output_path=args.output_path,
                verbose=args.verbose_onnx,
                verify=args.verify,
                test_texts=test_texts if args.verify else None,
                dynamic_axes=args.dynamic_axes
            )
            export_results.append(('Text Encoder (Full)', text_result))
        elif args.export_mode == "base":
            # 只导出基础text encoder
            base_encoder = TextEncoder(
                model=model,
                tokenizer=tokenizer,
                framework=args.framework,
                normalize=args.normalize,
                export_mode="base",
                use_quick_gelu=args.use_quick_gelu
            )
            base_result = base_encoder.export_onnx(
                output_path=args.output_path,
                verbose=args.verbose_onnx,
                verify=args.verify,
                test_texts=test_texts if args.verify else None,
                dynamic_axes=args.dynamic_axes
            )
            export_results.append(('Text Encoder (Base)', base_result))
        else:  # args.export_mode == "adapter"
            # 只导出adapter
            adapter_encoder = TextEncoder(
                model=model,
                tokenizer=tokenizer,
                framework=args.framework,
                normalize=args.normalize,  # 使用命令行参数的normalize值
                export_mode="adapter",
                use_quick_gelu=args.use_quick_gelu
            )
            adapter_result = adapter_encoder.export_onnx(
                output_path=args.output_path,
                verbose=args.verbose_onnx,
                verify=args.verify,
                test_texts=test_texts if args.verify else None,
                dynamic_axes=args.dynamic_axes
            )
            export_results.append(('Text Adapter', adapter_result))
    
    # Print summary
    print("\nExport Summary:")
    for name, success in export_results:
        status = "Success" if success else "Failed"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main(sys.argv[1:])