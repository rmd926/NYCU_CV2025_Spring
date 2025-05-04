import argparse

import torch
import torch.nn as nn

from train_resnet import get_resnet_instance_segmentation
from train_effnet_v2 import get_eff_instance_segmentation
from attention_modules import FPNTransformer, SpatialChannelTransformer


def count_params(backbone: str, num_classes: int, use_attn: bool):
    """
    Build the specified Mask R-CNN model, optionally attach
    the Spatial-Channel Transformer to the mask head, and
    print total vs. trainable parameter counts.
    """
    # 1) Instantiate the model based on backbone choice
    if backbone == "resnet":
        model = get_resnet_instance_segmentation(num_classes, use_attn)
    else:  # backbone == "effnet"
        model = get_eff_instance_segmentation(num_classes, use_attn)

    # 2) If attention is enabled, prepend the Spatial-Channel
    #    Transformer to the mask head
    if use_attn:
        mask_tr = SpatialChannelTransformer(in_channels=256)
        model.roi_heads.mask_head = nn.Sequential(
            mask_tr,
            model.roi_heads.mask_head
        )

    # 3) Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Backbone:             {backbone}")
    print(f"Using attention:      {use_attn}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Mask R-CNN parameter counts"
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet", "effnet"],
        default="resnet",
        help="Backbone model to use: 'resnet' or 'effnet'"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes (including background)"
    )
    parser.add_argument(
        "--use_attn",
        action="store_true",
        help="Include FPN→Transformer and Spatial-Channel attention"
    )
    args = parser.parse_args()

    count_params(args.backbone, args.num_classes, args.use_attn)
