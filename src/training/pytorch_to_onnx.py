import torch
# import os
import sys
import copy
import torch.nn as nn

from src.open_clip import create_model_and_transforms
# from mobileclip.modules.common.mobileone import reparameterize_model
import onnx
import argparse

# print("Using openclip from:", open_clip.__file__)
# framework = 'open_clip'
# model_arch = 'MobileCLIP-S2'
# model_path = '/home/xuboyu/Projects/CLIP/open_clip/checkpoints/mobileclip_s2_256/mobileclip_s2.pt'
# output_path = 'custom_clip_model1.onnx' 

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def parsers(args):
    parser = argparse.ArgumentParser(description='Export Clip model XXX to ONNX file XXX.onnx')
    parser.add_argument(
        '--framework',
        type=str,
        required=True,
        help="Specify framework from the available choices.",
        choices=['open_clip', 'mobileclip']
    )
    parser.add_argument(
        '--model-arch', 
        type=str, 
        required=True,
        help="Specify model arch from the available choices.",
        choices=['MobileCLIP-S0', 'MobileCLIP-S1', 'MobileCLIP-S2', 'MobileCLIP-Custom', 'mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'ViT-B-32']
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Specify location of model checkpoint.",
    )
    parser.add_argument(
        '--verbose_onnx', 
        action='store_true', 
        help='Print detailed ONNX operations'
    )
    parser.add_argument('--output-path', type=str, default="./")
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--reparam', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=True)
    return parser.parse_args(args)


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            # print(f"Reparameterizing module: {module}")
            module.reparameterize()
            # print(f"Reparameterized module: {module}")
    return model

def get_visual_encoder(model, framework, reparam):
    if framework == 'open_clip':
        visual_encoder = model.visual
        if reparam:
            visual_encoder = reparameterize_model(visual_encoder)
    elif framework == 'mobileclip':
        visual_encoder = model.image_encoder
        # if reparam:
            # visual_encoder = reparameterize_model(visual_encoder)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    return visual_encoder

def main(args):
    args = parsers(args)
    print(f"Reparameterize argument: {args.reparam}")
    if args.framework == 'open_clip':
        # model, _, _ = open_clip.create_model_and_transforms(
        model, _, _ = create_model_and_transforms(
            model_name=args.model_arch, 
            pretrained=args.model_path,
            image_mean=(0, 0, 0),
            image_std=(1, 1, 1),
            image_interpolation="bilinear",
        )
    elif args.framework == 'mobileclip':
        # print(f"Before calling create_model_and_transforms: args.reparam={args.reparam}, type={type(args.reparam)}")
        pass
        # model, _, _ = mobileclip.create_model_and_transforms(
        #     model_name=args.model_arch, 
        #     pretrained=args.model_path,
        #     reparameterize=args.reparam

        # )

    visual_encoder = get_visual_encoder(model, args.framework, args.reparam)
    visual_encoder.eval()
    print(f"Visual Encoder: {visual_encoder}")

    def get_shape_hook(name):
        def hook(model, input, output):
            # 只打印主要层的最终输出
            if name in ['trunk.stem.0', 'trunk.stem.1', 'trunk.stem.2', 'trunk.stages.0', 'trunk.stages.1', 
                    'trunk.stages.2', 'trunk.stages.3', 'trunk.final_conv', 'head']:
                print(f"{name} output shape: {output.shape}")
        return hook

    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)

    hook_handles = []
    
    def register_hooks(model, prefix=''):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # 保存hook handle
            handle = module.register_forward_hook(get_shape_hook(full_name))
            hook_handles.append(handle)
            register_hooks(module, full_name)

    print("\n=== Feature Map Shapes ===")
    print(f"dummy_input shape: {dummy_input.shape}\n")
    
    register_hooks(visual_encoder)
    
    print("Running forward pass to get feature map shapes...")
    with torch.no_grad():
        _ = visual_encoder(dummy_input)
    print("=== End of Feature Map Shapes ===\n")

    # 移除所有hooks
    for handle in hook_handles:
        handle.remove()

    torch.onnx.export(
        model=visual_encoder, 
        args=dummy_input,
        f=args.output_path,
        opset_version=18,
        verbose=args.verbose_onnx,
        export_params=True,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
    )
    print(f"Visual Encoder has been exported to {args.output_path}")

    onnx_model = onnx.load(args.output_path)

    try:
        onnx.checker.check_model(onnx_model, full_check=True, check_custom_domain=True)
        print("Model check passed. The ONNX model is valid.")
    except onnx.checker.ValidationError as e:
        print("Model check failed. The ONNX model is invalid.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main(sys.argv[1:])