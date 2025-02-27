import torch
import sys
import copy
import torch.nn as nn
import onnx
import argparse
import open_clip
from src.open_clip import create_model_and_transforms
from .tome_token_merging import apply_tome, remove_tome

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

class VisualEncoder:
    def __init__(self, model, preprocess, framework, reparam=True, model_arch=None, normalize=True, use_tome=False, tome_r=8):
        self.framework = framework
        self.reparam = reparam
        self.model_arch = model_arch
        self.normalize = normalize
        self.use_tome = use_tome  # 是否使用ToMe
        self.tome_r = tome_r      # ToMe的token合并数量
        self.encoder = self._get_visual_encoder(model)
        self.encoder.eval()
        self.output_path = None
        self.preprocess = preprocess
    
    def _get_visual_encoder(self, model):
        if self.framework == 'open_clip':
            visual_encoder = model.visual
            
            # 1. 如果使用ToMe，先应用ToMe
            if self.use_tome:
                print(f"\nApplying ToMe with r={self.tome_r}...")
                apply_tome(visual_encoder, r=self.tome_r)
                print("ToMe applied successfully.")
            
            # 2. 如果需要重参数化
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

            # 3. 包装带手动L2归一化的编码器
            class NormalizedEncoder(nn.Module):
                def __init__(self, base_encoder, normalize):
                    super().__init__()
                    self.base_encoder = base_encoder
                    self.normalize = normalize
                    self.final_token_count = None

                def forward(self, x):
                    # 1. 应用base_encoder（包含ToMe）
                    features = self.base_encoder(x)
                    
                    # 2. 记录当前token数量
                    if len(features.shape) == 3:  # [B, N, D]
                        B, N, D = features.shape
                        self.final_token_count = N
                        print(f"Current token count: {N}")
                    
                    # 3. 应用归一化
                    if self.normalize:
                        if len(features.shape) == 3:  # [B, N, D]
                            # 对每个token进行归一化
                            square = features * features  # [B, N, D]
                            sum_square = torch.sum(square, dim=-1, keepdim=True)  # [B, N, 1]
                            sqrt = torch.sqrt(sum_square)  # [B, N, 1]
                            features = features / (sqrt + 1e-6)  # [B, N, D]
                        else:
                            # 处理其他维度情况
                            norm = torch.norm(features, dim=-1, keepdim=True)
                            features = features / (norm + 1e-6)
                    
                    return features

            # 4. 创建并初始化归一化编码器
            normalized_encoder = NormalizedEncoder(visual_encoder, self.normalize)
            
            # 5. 运行一次前向传播以确定最终token数量
            if self.use_tome:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    _ = normalized_encoder(dummy_input)
                    print(f"Final token count after ToMe: {normalized_encoder.final_token_count}")
            
            return normalized_encoder
            
        elif self.framework == 'mobileclip':
            visual_encoder = model.image_encoder
            # 同样应用ToMe
            if self.use_tome:
                print(f"\nApplying ToMe with r={self.tome_r}...")
                apply_tome(visual_encoder, r=self.tome_r)
                print("ToMe applied successfully.")
            return NormalizedEncoder(visual_encoder, self.normalize)
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
        
        # 导出完成后，如果使用了ToMe，可以选择移除它
        if self.use_tome:
            print("\nRemoving ToMe modifications...")
            remove_tome(self.encoder.base_encoder)
            print("ToMe removed successfully.")

class TextEncoder:
    def __init__(self, model, tokenizer, framework, normalize=True):
        self.framework = framework
        self.normalize = normalize  # 添加normalize参数
        self.encoder = self._get_text_encoder(model)
        self.encoder.eval()
        self.output_path = None
        self.tokenizer = tokenizer  # 保存tokenizer
    
    def _get_text_encoder(self, model):
        if self.framework == 'open_clip':
            text_encoder = model.text

            # 包装带手动L2归一化的编码器
            class NormalizedTextEncoder(nn.Module):
                def __init__(self, base_encoder, normalize):
                    super().__init__()
                    self.base_encoder = base_encoder
                    self.normalize = normalize

                def forward(self, x):
                    features = self.base_encoder(x)
                    if self.normalize:
                        # 手动实现L2归一化
                        square = features * features  # Mul 操作
                        sum_square = torch.sum(square, dim=-1, keepdim=True)  # ReduceSum 操作
                        sqrt = torch.sqrt(sum_square)  # Sqrt 操作
                        features = features / sqrt  # Div 操作
                    return features

            return NormalizedTextEncoder(text_encoder, self.normalize)
        elif self.framework == 'mobileclip':
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
            pytorch_output = pytorch_output.cpu().numpy()
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            print(f"\nOutput Verification Results for text: {single_text}")
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            
            if max_diff >= 1e-5:
                return False
        
        return True
    
    def export_onnx(self, output_path, verbose=False, verify=False, test_texts=None, dynamic_axes=False):
        print(f"\nText Encoder: {self.encoder}")
        
        dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.int32)
        
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
            
        # 保存输出路径
        if '_text' not in output_path:
            self.output_path = output_path.replace('.onnx', '_text.onnx')
        else:
            self.output_path = output_path
            
        # Export to ONNX
        export_args = {
            'model': self.encoder,
            'args': dummy_input,
            'f': self.output_path,
            'opset_version': 18,
            'verbose': verbose,
            'export_params': True,
            'do_constant_folding': False,
            'input_names': ['input'],
            'output_names': ['text_features'],
        }
        
        if dynamic_axes:
            export_args['dynamic_axes'] = {
                'input': {0: 'batch_size'},
                'text_features': {0: 'batch_size'}
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
    parser.add_argument('--use-tome', type=lambda x: (str(x).lower() == 'true'),
                       choices=[True, False], default=True, 
                       help='Whether to use ToMe token merging')
    parser.add_argument('--tome-r', type=int, default=8,
                       help='Number of tokens to merge in each layer for ToMe')
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
            normalize=args.normalize,
            use_tome=args.use_tome,
            tome_r=args.tome_r
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
        text_encoder = TextEncoder(
            model=model,
            tokenizer=tokenizer,
            framework=args.framework,
            normalize=args.normalize
        )
        text_result = text_encoder.export_onnx(
            output_path=args.output_path,
            verbose=args.verbose_onnx,
            verify=args.verify,
            test_texts=test_texts if args.verify else None,
            dynamic_axes=args.dynamic_axes
        )
        export_results.append(('Text Encoder', text_result))
    
    # Print summary
    print("\nExport Summary:")
    for name, success in export_results:
        status = "Success" if success else "Failed"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main(sys.argv[1:])