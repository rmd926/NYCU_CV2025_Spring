import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pycocotools import mask as mask_utils

from dataloader import get_loaders, collate_fn
from train_resnet import get_resnet_instance_segmentation
from train_effnet_v2 import get_eff_instance_segmentation
from attention_modules import SpatialChannelTransformer


def load_model(checkpoint_path: str,
               num_classes: int,
               backbone: str,
               use_attn: bool,
               device: torch.device) -> nn.Module:
    """
    Build and load a Mask R-CNN model.

    1) Construct the model with the specified backbone.
    2) Optionally prepend the Spatial-Channel Transformer.
    3) Load weights from checkpoint.
    4) Switch to eval mode.
    """
    if backbone == 'resnet':
        model = get_resnet_instance_segmentation(num_classes, use_attn)
    elif backbone == 'effnet':
        model = get_eff_instance_segmentation(num_classes, use_attn)
    else:
        raise ValueError(f'Unsupported backbone: {backbone}')

    if use_attn:
        # Insert the transformer into the mask head
        mask_tr = SpatialChannelTransformer(in_channels=256)
        model.roi_heads.mask_head = nn.Sequential(
            mask_tr,
            model.roi_heads.mask_head
        )

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    return model


def run_inference(model: nn.Module,
                  loader: DataLoader,
                  device: torch.device,
                  score_thresh: float = 0.5) -> list:
    """
    Perform inference, filter by score threshold, encode masks in RLE,
    and return a list of result dicts.
    """
    results = []
    with torch.no_grad():
        for images, image_ids, _ in loader:
            img = images[0].to(device)
            image_id = int(image_ids[0].item())
            output = model([img])[0]

            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            masks = output['masks'].cpu().numpy()

            for box, score, label, mask in zip(
                    boxes, scores, labels, masks):
                if score < score_thresh:
                    continue

                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                # Binarize mask and encode as RLE
                bin_mask = (mask[0] > 0.5).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(bin_mask))
                rle['counts'] = rle['counts'].decode('ascii')

                results.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [x1, y1, w, h],
                    'score': float(score),
                    'segmentation': {
                        'size': rle['size'],
                        'counts': rle['counts']
                    }
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN inference and export RLE JSON'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to trained checkpoint'
    )
    parser.add_argument(
        '--backbone',
        choices=['resnet', 'effnet'],
        default='resnet',
        help='Which backbone to use'
    )
    parser.add_argument(
        '--train_dir',
        default='dataset/train',
        help='Training directory (for loader config)'
    )
    parser.add_argument(
        '--test_dir',
        default='dataset/test_release',
        help='Directory of test images'
    )
    parser.add_argument(
        '--id_map_json',
        default='dataset/test_image_name_to_ids.json',
        help='JSON file mapping filenames to IDs'
    )
    parser.add_argument(
        '--output_dir',
        default='inference_output',
        help='Directory to save output JSON'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=5,
        help='Number of classes (including background)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--score_thresh',
        type=float,
        default=0.5,
        help='Score threshold for detections'
    )
    parser.add_argument(
        '--use_attn',
        action='store_true',
        help='Enable Spatial-Channel attention in mask head'
    )
    args = parser.parse_args()

    device = (
        torch.device('cuda')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Load the model
    model = load_model(
        args.checkpoint,
        args.num_classes,
        args.backbone,
        args.use_attn,
        device
    )

    # Prepare the test DataLoader
    _, _, test_loader = get_loaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        id_map_json=args.id_map_json,
        batch_size=1,
        val_split=0.0,
        num_workers=args.num_workers,
        seed=args.seed
    )

    print('Running inference on test set...')
    results = run_inference(
        model,
        test_loader,
        device,
        score_thresh=args.score_thresh
    )
    print(f'→ Collected {len(results)} results')

    # Save to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'test-results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Saved predictions to {out_path}')
    print('To package for submission:')
    print(
        f'  cd {args.output_dir} && '
        'zip submission.zip test-results.json'
    )


if __name__ == '__main__':
    main()
