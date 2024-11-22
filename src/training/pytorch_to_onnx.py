import torch
import sys
import copy
import torch.nn as nn
import onnx
import argparse
from src.open_clip import create_model_and_transforms

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

class VisualEncoder:
    def __init__(self, model, framework, reparam=True):
        self.framework = framework
        self.reparam = reparam
        self.encoder = self._get_visual_encoder(model)
        self.encoder.eval()
    
    def _get_visual_encoder(self, model):
        if self.framework == 'open_clip':
            visual_encoder = model.visual
            if self.reparam:
                visual_encoder = self._reparameterize_model(visual_encoder)
            return visual_encoder
        elif self.framework == 'mobileclip':
            return model.image_encoder
        raise ValueError(f"Unsupported framework: {self.framework}")
    
    @staticmethod
    def _reparameterize_model(model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model
    
    def export_onnx(self, output_path, resolution=256, verbose=False):
        print(f"\nVisual Encoder: {self.encoder}")
        
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
        
        # Export to ONNX
        if '_visual' not in output_path:
            visual_output_path = output_path.replace('.onnx', '_visual.onnx')
        else:
            visual_output_path = output_path
        torch.onnx.export(
            model=self.encoder,
            args=dummy_input,
            f=visual_output_path,
            opset_version=18,
            verbose=verbose,
            export_params=True,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Visual Encoder has been exported to {visual_output_path}")
        
        # Verify model
        onnx_model = onnx.load(visual_output_path)
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
            print("Visual encoder ONNX model is valid.")
            return True
        except onnx.checker.ValidationError as e:
            print("Visual encoder ONNX model is invalid.")
            print(f"Error: {e}")
            return False

class TextEncoder:
    def __init__(self, model, framework):
        self.framework = framework
        self.encoder = self._get_text_encoder(model)
        self.encoder.eval()
    
    def _get_text_encoder(self, model):
        if self.framework == 'open_clip':
            return model.text
        elif self.framework == 'mobileclip':
            return model.text_encoder
        raise ValueError(f"Unsupported framework: {self.framework}")
    
    def export_onnx(self, output_path, verbose=False):
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
        
        # Export to ONNX
        if '_text' not in output_path:
            text_output_path = output_path.replace('.onnx', '_text.onnx')
        else:
            text_output_path = output_path
        torch.onnx.export(
            model=self.encoder,
            args=dummy_input,
            f=text_output_path,
            opset_version=18,
            verbose=verbose,
            export_params=True,
            do_constant_folding=False,
            input_names=['input_ids'],
            output_names=['text_features'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'text_features': {0: 'batch_size'}
            }
        )
        
        print(f"Text Encoder has been exported to {text_output_path}")
        
        # Verify model
        onnx_model = onnx.load(text_output_path)
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
            print("Text encoder ONNX model is valid.")
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
    return parser.parse_args(args)

def main(args):
    args = parsers(args)
    
    # Load model
    if args.framework == 'open_clip':
        model, _, _ = create_model_and_transforms(
            model_name=args.model_arch,
            pretrained=args.model_path,
            image_mean=(0, 0, 0),
            image_std=(1, 1, 1),
            image_interpolation="bilinear",
        )
    elif args.framework == 'mobileclip':
        # Add mobileclip model loading here
        pass
    
    # 检查是否指定了导出选项
    if not (args.export_all or args.export_image or args.export_text):
        print("Error: Please specify what to export using --export-all, --export-image, or --export-text")
        sys.exit(1)
    
    export_results = []
    
    # Export visual encoder
    if args.export_all or args.export_image:
        print("\nExporting visual encoder...")
        visual_encoder = VisualEncoder(model, args.framework, args.reparam)
        visual_result = visual_encoder.export_onnx(
            output_path=args.output_path,
            resolution=args.resolution,
            verbose=args.verbose_onnx
        )
        export_results.append(('Visual Encoder', visual_result))
    
    # Export text encoder
    if args.export_all or args.export_text:
        print("\nExporting text encoder...")
        text_encoder = TextEncoder(model, args.framework)
        text_result = text_encoder.export_onnx(
            output_path=args.output_path,
            verbose=args.verbose_onnx
        )
        export_results.append(('Text Encoder', text_result))
    
    # Print summary
    print("\nExport Summary:")
    for name, success in export_results:
        status = "Success" if success else "Failed"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main(sys.argv[1:])