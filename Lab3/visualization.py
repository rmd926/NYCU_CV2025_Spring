# import os, argparse, random, colorsys
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# from dataloader import (
#     MedicalInstanceDataset,
#     MedicalTestDataset,
#     collate_fn
# )
# from augment import get_val_transform          # val 模式需要
# from inference import load_model               # 直接重用
# # ------------------------------------------------------------

# # ----------------------- 0. utils ----------------------------
# def _random_color(idx):
#     """為每個 instance 產生可重現的隨機顏色 (含透明度)."""
#     random.seed(idx)
#     h, s, v = random.random(), 0.6 + random.random() * 0.4, 1.0
#     r, g, b = colorsys.hsv_to_rgb(h, s, v)
#     return r, g, b, 0.4                         # α = 0.4


# # ----------------------- 1. Pred 圖 --------------------------
# def visualize_single(img_np, preds, score_thresh=0.5, save_path=None):
#     """
#     img_np : (H,W,3) RGB, range 0~1
#     preds  : dict {boxes, labels, scores, masks}
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     ax.imshow(img_np); ax.axis('off')

#     boxes  = preds['boxes'].cpu().numpy()
#     labels = preds['labels'].cpu().numpy()
#     scores = preds['scores'].cpu().numpy()
#     masks  = preds['masks'].cpu().numpy()  # (N,1,H,W)

#     for i, (box, lab, sc, msk) in enumerate(zip(boxes, labels, scores, masks)):
#         if sc < score_thresh:
#             continue
#         color = _random_color(i)

#         # --- mask ---
#         mask_bin = msk[0] > 0.5
#         ax.imshow(np.dstack([mask_bin * c for c in color[:3]] +
#                             [mask_bin * color[3]]))

#         # --- bbox ---
#         x1, y1, x2, y2 = box
#         rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                   linewidth=2, edgecolor=color[:3],
#                                   facecolor='none')
#         ax.add_patch(rect)

#         # --- label ---
#         ax.text(x1, y1 - 2, f"{lab}:{sc:.2f}", color='white', fontsize=8,
#                 bbox=dict(facecolor=color[:3], alpha=.7,
#                           pad=1, edgecolor='none'))

#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=150)
#         plt.close(fig)
#     else:
#         plt.show()


# # ----------------------- 2. GT 圖 ----------------------------
# def visualize_gt(img_np, target, save_path=None, draw_mask=False):
#     """
#     只畫 Ground‑Truth. 若 draw_mask=True 亦疊上 GT mask(半透明綠色)
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     ax.imshow(img_np); ax.axis('off')

#     boxes  = target['boxes'].numpy()
#     labels = target['labels'].numpy()
#     masks  = target['masks'].numpy() if draw_mask else None

#     for i, (bx, lb) in enumerate(zip(boxes, labels)):
#         x1, y1, x2, y2 = bx
#         rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                   linewidth=2, linestyle='--',
#                                   edgecolor='lime', facecolor='none')
#         ax.add_patch(rect)

#         if masks is not None and masks.shape[0] > i:
#             mask_bin = masks[i].astype(bool)
#             ax.imshow(np.dstack([mask_bin * 0,
#                                  mask_bin * 1.0,
#                                  mask_bin * 0,
#                                  mask_bin * 0.25]))   # α=0.25

#         ax.text(x1, y1 - 2, f'{lb}', color='white', fontsize=8,
#                 bbox=dict(facecolor='lime', alpha=.7, pad=1,
#                           edgecolor='none'))

#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=150)
#         plt.close(fig)
#     else:
#         plt.show()


# # ----------------------- 3. main -----------------------------
# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(args.vis_dir, exist_ok=True)

#     # 3‑1 dataset
#     if args.mode == 'test':
#         dataset = MedicalTestDataset(args.test_dir, args.id_map_json,
#                                      transforms=None)
#     else:  # val
#         dataset = MedicalInstanceDataset(args.train_dir,
#                                          transforms=get_val_transform())

#     # 3‑2 model
#     model = load_model(args.checkpoint, args.num_classes,
#                        args.use_attn, device)

#     # 3‑3 inference & save
#     model.eval()
#     with torch.no_grad():
#         total = min(args.num_images, len(dataset))
#         for idx in range(total):
#             if args.mode == 'test':
#                 img, image_id, fname = dataset[idx]
#                 img_id_str = f"{image_id.item()}"
#                 target = None
#             else:
#                 img, target = dataset[idx]
#                 img_id_str = f"val_{idx}"

#             preds = model([img.to(device)])[0]

#             img_np = img.permute(1, 2, 0).cpu().numpy()

#             # (a) GT only
#             if target is not None:
#                 gt_path = os.path.join(args.vis_dir,
#                                        f"{img_id_str}_gt.png")
#                 visualize_gt(img_np, target, gt_path, args.draw_gt_mask)

#             # (b) Prediction overlaid
#             pred_path = os.path.join(args.vis_dir,
#                                      f"{img_id_str}_pred.png")
#             visualize_single(img_np, preds, args.score_thresh, pred_path)

#             print(f"[{idx+1}/{total}]  "
#                   f"{'GT→'+gt_path if target is not None else ''} "
#                   f"| Pred→{pred_path}")


# # ----------------------- 4. CLI ------------------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("Mask‑R‑CNN visualizer")
#     parser.add_argument('--checkpoint', required=True,
#                         help='trained .pth weight')
#     parser.add_argument('--train_dir', default='dataset/train')
#     parser.add_argument('--test_dir',  default='dataset/test_release')
#     parser.add_argument('--id_map_json',
#                         default='dataset/test_image_name_to_ids.json')
#     parser.add_argument('--num_classes', type=int, default=5)
#     parser.add_argument('--vis_dir', default='vis_output')
#     parser.add_argument('--mode', choices=['val', 'test'], default='val')
#     parser.add_argument('--num_images', type=int, default=2)
#     parser.add_argument('--score_thresh', type=float, default=0.5)
#     parser.add_argument('--use_attn', action='store_true')
#     parser.add_argument('--draw_gt_mask', action='store_true',
#                         help='overlay GT mask in GT image')
#     args = parser.parse_args()

#     main(args)
# visualization.py
"""
Visualize Mask R-CNN predictions and optionally ground truth overlays.
Supports ResNet and EfficientNet-V2 backbones with optional attention modules.
"""

import os
import argparse
import random
import colorsys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataloader import MedicalInstanceDataset, MedicalTestDataset, collate_fn
from augment import get_val_transform
from inference import load_model  # Updated to accept backbone parameter


def _random_color(idx: int):
    """Generate a reproducible RGBA color for a given instance index."""
    random.seed(idx)
    h = random.random()
    s = 0.6 + random.random() * 0.4
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r, g, b, 0.4  # alpha=0.4


def visualize_predictions(img_np: np.ndarray,
                          preds: dict,
                          score_thresh: float = 0.5,
                          save_path: str = None):
    """
    Overlay predicted masks, boxes, and labels on an image.

    Args:
        img_np: HxWx3 numpy array with values in [0,1].
        preds: Dictionary with keys 'boxes', 'labels', 'scores', 'masks'.
        score_thresh: Minimum score to display a prediction.
        save_path: If provided, save visualization to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.axis('off')

    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    masks = preds['masks'].cpu().numpy()

    for i, (box, label, score, mask) in enumerate(
            zip(boxes, labels, scores, masks)):
        if score < score_thresh:
            continue

        color = _random_color(i)
        mask_bin = mask[0] > 0.5
        ax.imshow(np.dstack([
            mask_bin * c for c in color[:3]
        ] + [mask_bin * color[3]]))

        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color[:3], facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 2, f"{label}:{score:.2f}",
            color='white', fontsize=8,
            bbox=dict(facecolor=color[:3], alpha=0.7, pad=1,
                      edgecolor='none')
        )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()


def visualize_ground_truth(img_np: np.ndarray,
                           target: dict,
                           save_path: str = None,
                           draw_mask: bool = False):
    """
    Draw ground truth boxes and labels. Optionally overlay masks.

    Args:
        img_np: HxWx3 numpy array with values in [0,1].
        target: Dictionary with keys 'boxes', 'labels', 'masks'.
        save_path: If provided, save visualization to this path.
        draw_mask: Whether to overlay the ground truth masks.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.axis('off')

    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    masks = target['masks'].numpy() if draw_mask else None

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, linestyle='--',
            edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)

        if masks is not None and i < masks.shape[0]:
            mask_bin = masks[i].astype(bool)
            ax.imshow(np.dstack([
                mask_bin * 0,
                mask_bin * 1.0,
                mask_bin * 0,
                mask_bin * 0.25
            ]))

        ax.text(
            x1, y1 - 2, str(label),
            color='white', fontsize=8,
            bbox=dict(facecolor='lime', alpha=0.7, pad=1,
                      edgecolor='none')
        )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Mask R-CNN predictions and GT'
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--backbone', choices=['resnet', 'effnet'], default='resnet',
        help='Select backbone architecture'
    )
    parser.add_argument(
        '--train_dir', default='dataset/train',
        help='Directory for training/validation data'
    )
    parser.add_argument(
        '--test_dir', default='dataset/test_release',
        help='Directory for test images'
    )
    parser.add_argument(
        '--id_map_json', default='dataset/test_image_name_to_ids.json',
        help='JSON mapping test filenames to IDs'
    )
    parser.add_argument(
        '--num_classes', type=int, default=5,
        help='Number of classes including background'
    )
    parser.add_argument(
        '--vis_dir', default='vis_output',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--mode', choices=['val', 'test'], default='val',
        help='Visualization mode: validation or test'
    )
    parser.add_argument(
        '--num_images', type=int, default=2,
        help='Number of images to visualize'
    )
    parser.add_argument(
        '--score_thresh', type=float, default=0.5,
        help='Score threshold for predictions'
    )
    parser.add_argument(
        '--use_attn', action='store_true',
        help='Enable Spatial–Channel attention in mask head'
    )
    parser.add_argument(
        '--draw_gt_mask', action='store_true',
        help='Overlay GT masks on ground truth visualization'
    )
    args = parser.parse_args()

    device = (
        torch.device('cuda')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    os.makedirs(args.vis_dir, exist_ok=True)

    if args.mode == 'test':
        dataset = MedicalTestDataset(
            args.test_dir, args.id_map_json, transforms=None
        )
    else:
        dataset = MedicalInstanceDataset(
            args.train_dir, transforms=get_val_transform()
        )

    model = load_model(
        args.checkpoint, args.num_classes,
        args.backbone, args.use_attn, device
    )

    model.eval()
    with torch.no_grad():
        total = min(args.num_images, len(dataset))

        for idx in range(total):
            if args.mode == 'test':
                img, image_id, _ = dataset[idx]
                prefix = f"{int(image_id.item())}"
                target = None
            else:
                img, target = dataset[idx]
                prefix = f"val_{idx}"

            preds = model([img.to(device)])[0]
            img_np = img.permute(1, 2, 0).cpu().numpy()

            if target is not None:
                gt_path = os.path.join(args.vis_dir, f"{prefix}_gt.png")
                visualize_ground_truth(
                    img_np, target,
                    save_path=gt_path,
                    draw_mask=args.draw_gt_mask
                )

            pred_path = os.path.join(args.vis_dir, f"{prefix}_pred.png")
            visualize_predictions(
                img_np, preds,
                score_thresh=args.score_thresh,
                save_path=pred_path
            )

            print(
                f"[{idx+1}/{total}] "
                f"{('GT→' + gt_path) if target is not None else ''} "
                f"| Pred→{pred_path}"
            )


if __name__ == '__main__':
    main()
