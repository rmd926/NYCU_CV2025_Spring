import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models.detection.backbone_utils import (
    BackboneWithFPN,
    LastLevelMaxPool,
)
from torchvision.models.detection import MaskRCNN

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from dataloader import (
    MedicalInstanceDataset,
    MedicalTestDataset,
    collate_fn,
)
from augment import get_train_transform, get_val_transform
from plot_curve import plot_loss_curve, plot_map_curve

# Attention modules
from attention_modules import FPNTransformer, SpatialChannelTransformer
# Automatic Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# Workaround for OpenMP conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LOSS_KEYS = [
    "loss_objectness",
    "loss_rpn_box_reg",
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
]


def set_seed(seed: int):
    """Set random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    """Load model weights from the given checkpoint path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from '{path}'")


class AttnBackbone(nn.Module):
    """Wraps a BackboneWithFPN and applies FPNTransformer."""

    def __init__(self, backbone_with_fpn, fpn_transformer):
        super().__init__()
        self.body = backbone_with_fpn.body
        self.fpn = backbone_with_fpn.fpn
        self.out_channels = backbone_with_fpn.out_channels
        self.fpn_transformer = fpn_transformer

    def forward(self, x):
        feats = self.body(x)
        fpn_feats = self.fpn(feats)
        keys = sorted(fpn_feats.keys())
        seq = [fpn_feats[k] for k in keys]
        enhanced = self.fpn_transformer(seq)
        return {k: enhanced[i] for i, k in enumerate(keys)}


def get_eff_instance_segmentation(num_classes: int, use_attn: bool = False):
    """
    Build Mask R-CNN with an EfficientNet-V2-M backbone and optional
    Transformer attention in the FPN.
    """
    # 1) Load pretrained EfficientNet-V2-M feature extractor
    effnet = efficientnet_v2_m(
        weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1
    ).features

    # 2) Freeze the first two blocks to retain low-level representations
    for idx in [0, 1]:
        for param in effnet[idx].parameters():
            param.requires_grad = False

    # 3) Keep all subsequent blocks trainable
    for idx in range(2, len(effnet)):
        for param in effnet[idx].parameters():
            param.requires_grad = True

    # 4) Specify which backbone layers (by index) to feed into the FPN
    return_layers = {"2": "0", "3": "1", "4": "2", "6": "3"}

    # 5) Infer channel dimensions for each selected layer
    in_channels_list = []
    for src_idx in return_layers.keys():
        block = effnet[int(src_idx)]
        if hasattr(block, "out_channels"):
            c = block.out_channels
        elif hasattr(block, "conv") and hasattr(block.conv, "out_channels"):
            c = block.conv.out_channels
        else:
            # Fallback: inspect submodules for an out_channels attr
            for m in block.modules():
                if hasattr(m, "out_channels"):
                    c = m.out_channels
                    break
        in_channels_list.append(c)

    # 6) Wrap with a Feature Pyramid Network and add P6
    backbone = BackboneWithFPN(
        effnet,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )

    # 7) Optionally insert a Transformer into the FPN outputs
    if use_attn:
        fpn_tr = FPNTransformer(in_channels=backbone.out_channels)
        backbone = AttnBackbone(backbone, fpn_tr)

    # 8) Build the Mask R-CNN model
    return MaskRCNN(backbone, num_classes=num_classes)


def train_one_epoch(model, loader, optimizer, device, epoch,
                    use_amp=False, scaler=None):
    """Run one training epoch with optional AMP."""
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    total = 0.0
    comps = {k: 0.0 for k in LOSS_KEYS}
    n_batches = len(loader)

    for imgs, targets in loop:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        if use_amp:
            with autocast():
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        total += loss.item()
        for k in LOSS_KEYS:
            comps[k] += loss_dict.get(
                k, torch.tensor(0.0, device=device)
            ).item()

        loop.set_postfix(train_loss=total / (loop.n + 1))

    avg = total / n_batches
    avg_comps = {k: comps[k] / n_batches for k in LOSS_KEYS}
    return avg, avg_comps


def validate(model, loader, device, epoch):
    """Run one validation epoch (loss only)."""
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)

    total = 0.0
    comps = {k: 0.0 for k in LOSS_KEYS}
    n_batches = len(loader)

    with torch.no_grad():
        for imgs, targets in loop:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            total += loss.item()
            for k in LOSS_KEYS:
                comps[k] += loss_dict.get(
                    k, torch.tensor(0.0, device=device)
                ).item()

            loop.set_postfix(val_loss=total / (loop.n + 1))

    avg = total / n_batches
    avg_comps = {k: comps[k] / n_batches for k in LOSS_KEYS}
    return avg, avg_comps


def build_coco_ground_truth(dataset):
    """Convert dataset annotations into COCO-format ground truth."""
    coco = COCO()
    images, annotations, categories = [], [], []
    ann_id = 1

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        _, H, W = img.shape
        images.append({"id": idx, "width": W, "height": H})

        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        masks = target["masks"].numpy()
        areas = target["area"].numpy()
        iscrowd = target["iscrowd"].numpy()

        for box, label, m, area, ic in zip(
            boxes, labels, masks, areas, iscrowd
        ):
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(label),
                "segmentation": rle,
                "area": float(area),
                "bbox": [
                    float(box[0]), float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1])
                ],
                "iscrowd": int(ic),
            })
            ann_id += 1

    for cid in sorted({ann["category_id"] for ann in annotations}):
        categories.append({"id": cid, "name": str(cid)})

    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco.createIndex()
    return coco


def evaluate_coco(model, dataset, device):
    """Evaluate model on dataset using COCO metrics."""
    model.eval()
    coco_gt = build_coco_ground_truth(dataset)
    results = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    for idx, (imgs, _) in enumerate(
            tqdm(loader, desc="COCOeval", leave=False)):
        img = imgs[0].to(device)
        with torch.no_grad():
            out = model([img])[0]

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks = out["masks"].cpu().numpy()

        for box, score, label, m in zip(boxes, scores, labels, masks):
            rle = mask_utils.encode(
                np.asfortranarray((m[0] > 0.5).astype(np.uint8))
            )
            rle["counts"] = rle["counts"].decode("ascii")
            results.append({
                "image_id": idx,
                "category_id": int(label),
                "segmentation": rle,
                "score": float(score),
            })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[:]


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare train/val splits
    full_ds = MedicalInstanceDataset(args.train_dir, transforms=None)
    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(
        list(range(n_total)), [n_train, n_val], generator=generator
    )

    train_ds = Subset(
        MedicalInstanceDataset(args.train_dir,
                                transforms=get_train_transform()),
        train_idx.indices
    )
    val_ds = Subset(
        MedicalInstanceDataset(args.train_dir,
                                transforms=get_val_transform()),
        val_idx.indices
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    # Build and load model
    model = get_eff_instance_segmentation(
        args.num_classes, use_attn=args.use_attn
    )
    if args.use_attn:
        mask_tr = SpatialChannelTransformer(in_channels=256)
        model.roi_heads.mask_head = nn.Sequential(
            mask_tr, model.roi_heads.mask_head
        )
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    # Set up AMP scaler if needed
    scaler = GradScaler() if args.use_amp else None

    # Configure optimizer and scheduler
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    if args.use_cosine:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        T_max = max(1, args.epochs - args.warmup_epochs)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=args.eta_min
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.patience
        )

    train_losses, val_losses, map50s = [], [], []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # Warm-up or step scheduler
        if args.use_cosine:
            if epoch <= args.warmup_epochs:
                scale = epoch / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * scale
            else:
                scheduler.step()

        print(f"[Epoch {epoch}] LR: {[pg['lr'] for pg in optimizer.param_groups]}")

        start_time = time.time()

        # Train, validate, evaluate
        t_loss, t_comps = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, args.use_amp, scaler
        )
        torch.cuda.empty_cache()

        v_loss, v_comps = validate(model, val_loader, device, epoch)
        torch.cuda.empty_cache()

        stats = evaluate_coco(model, val_ds, device)
        torch.cuda.empty_cache()
        mAP50 = stats[0]

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        map50s.append(mAP50)

        if not args.use_cosine:
            scheduler.step(mAP50)

        print(
            f"[Epoch {epoch}] Train Loss={t_loss:.4f}  "
            f"Val Loss={v_loss:.4f}  mAP50={mAP50:.4f}  "
            f"Time={time.time() - start_time:.1f}s"
        )

        if mAP50 > 0:
            no_improve = 0
            ckpt = os.path.join(
                args.output_dir,
                f"epoch{epoch}_loss_{v_loss:.4f}_map{mAP50:.4f}.pth"
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  ▶ Saved new best model: {ckpt}")
        else:
            no_improve += 1
            if not args.use_cosine and no_improve >= args.early_stop:
                print("Early stopping due to no improvement.")
                break

    # Plot metrics and save final checkpoint
    plot_loss_curve(train_losses, val_losses)
    plot_map_curve(map50s)
    final_ckpt = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete; final model saved to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN + EfficientNetV2-M training"
    )
    parser.add_argument("--train_dir", default="dataset/train")
    parser.add_argument("--test_dir", default="dataset/test_release")
    parser.add_argument(
        "--id_map_json", default="dataset/test_image_name_to_ids.json"
    )
    parser.add_argument("--output_dir", default="output_aug_eff")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4
    )
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducibility"
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument(
        "--num_classes", type=int, default=5,
        help="Number of classes including background"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint for fine-tuning"
    )
    parser.add_argument(
        "--use_attn", action="store_true",
        help="Enable FPN transformer & spatial-channel attention"
    )
    parser.add_argument(
        "--use_amp", action="store_true",
        help="Enable mixed precision training (FP16 via AMP)"
    )
    parser.add_argument(
        "--use_cosine", action="store_true",
        help="Use CosineAnnealingLR scheduler"
    )
    parser.add_argument(
        "--eta_min", type=float, default=0.0,
        help="Minimum LR for cosine annealing"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0,
        help="Number of warm-up epochs before cosine decay"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
