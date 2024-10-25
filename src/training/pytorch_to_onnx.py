import torch
import os
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

parser = argparse.ArgumentParser(description='Export Clip model XXX to ONNX file XXX.onnx')
parser.add_argument(
    "--framework",
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
    "--model-path",
    type=str,
    default=None,
    help="Specify location of model checkpoint.",
)
parser.add_argument('--output-path', type=str, default="./")
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--reparam', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=True)
args = parser.parse_args()


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
            print(f"Reparameterizing module: {module}")
            module.reparameterize()
    return model

def get_visual_encoder(model, framework, reparam):
    if framework == 'open_clip':
        visual_encoder = model.visual
        if reparam:
            visual_encoder = reparameterize_model(visual_encoder)
    elif framework == 'mobileclip':
        # print("Has 'visual' attribute:", hasattr(model, 'visual'))
        # print("Has 'vision' attribute:", hasattr(model, 'vision'))
        # print("Has 'image_encoder' attribute:", hasattr(model, 'image_encoder'))
        visual_encoder = model.image_encoder
        # if reparam:
            # visual_encoder = reparameterize_model(visual_encoder)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    return visual_encoder

def main():
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

    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)

    torch.onnx.export(
        model=visual_encoder, 
        args=dummy_input,
        f=args.output_path,
        opset_version=18,
        verbose=True,
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
    main()