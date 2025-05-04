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

from torchvision.models import ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from dataloader import MedicalInstanceDataset, collate_fn
from augment import get_train_transform, get_val_transform
from plot_curve import plot_loss_curve, plot_map_curve

# Attention modules
from attention_modules import FPNTransformer, SpatialChannelTransformer
# Automatic Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# Allow OpenMP on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LOSS_KEYS = [
    "loss_objectness",
    "loss_rpn_box_reg",
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    """Load model weights from a checkpoint file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from '{path}'")


class AttnBackbone(nn.Module):
    """Backbone wrapper that applies FPNTransformer on FPN outputs."""

    def __init__(self, backbone_with_fpn, fpn_transformer):
        super().__init__()
        self.body = backbone_with_fpn.body
        self.fpn = backbone_with_fpn.fpn
        self.out_channels = backbone_with_fpn.out_channels
        self.fpn_transformer = fpn_transformer

    def forward(self, x):
        features = self.body(x)
        fpn_feats = self.fpn(features)
        keys = sorted(fpn_feats.keys())
        seq = [fpn_feats[k] for k in keys]
        enhanced = self.fpn_transformer(seq)
        return {k: enhanced[i] for i, k in enumerate(keys)}


def get_resnet_instance_segmentation(num_classes: int, use_attn: bool = False):
    """
    Create Mask R-CNN with ResNet-101+FPN backbone.
    If use_attn is True, insert Transformer into the FPN.
    """
    backbone0 = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=3,
    )

    if use_attn:
        c_out = backbone0.out_channels
        fpn_tr = FPNTransformer(in_channels=c_out)
        backbone = AttnBackbone(backbone0, fpn_tr)
    else:
        backbone = backbone0

    return MaskRCNN(backbone, num_classes=num_classes)


def train_one_epoch(model, loader, optimizer, device, epoch,
                    use_amp=False, scaler=None):
    """Train the model for one epoch."""
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    total_loss = 0.0
    comp_sums = {k: 0.0 for k in LOSS_KEYS}
    n_batches = len(loader)

    for imgs, targets in loop:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        if use_amp:
            with autocast():
                losses = model(imgs, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model(imgs, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        for key in LOSS_KEYS:
            comp_sums[key] += losses.get(key, torch.tensor(0.0, device=device)).item()

        loop.set_postfix(train_loss=total_loss / (loop.n + 1))

    avg_loss = total_loss / n_batches
    avg_comps = {k: comp_sums[k] / n_batches for k in LOSS_KEYS}
    return avg_loss, avg_comps


def validate(model, loader, device, epoch):
    """Validate the model for one epoch."""
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)

    total_loss = 0.0
    comp_sums = {k: 0.0 for k in LOSS_KEYS}
    n_batches = len(loader)

    with torch.no_grad():
        for imgs, targets in loop:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(imgs, targets)
            loss = sum(losses.values())

            total_loss += loss.item()
            for key in LOSS_KEYS:
                comp_sums[key] += losses.get(key, torch.tensor(0.0, device=device)).item()

            loop.set_postfix(val_loss=total_loss / (loop.n + 1))

    avg_loss = total_loss / n_batches
    avg_comps = {k: comp_sums[k] / n_batches for k in LOSS_KEYS}
    return avg_loss, avg_comps


def build_coco_ground_truth(dataset):
    """
    Convert dataset annotations to COCO format for evaluation.
    """
    coco = COCO()
    images, annotations, categories = [], [], []
    ann_id = 1

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        _, h, w = img.shape
        images.append({"id": idx, "width": w, "height": h})

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
                    float(box[2] - box[0]), float(box[3] - box[1])
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
    """
    Run COCO evaluation on the dataset and model.
    """
    model.eval()
    coco_gt = build_coco_ground_truth(dataset)
    results = []
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

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

    # Prepare datasets and loaders
    full_ds = MedicalInstanceDataset(args.train_dir, transforms=None)
    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(
        list(range(n_total)), [n_train, n_val], generator=gen
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
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Build model
    model = get_resnet_instance_segmentation(
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

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    if args.use_cosine:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        t_cos = max(1, args.epochs - args.warmup_epochs)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=t_cos, eta_min=args.eta_min
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=args.patience
        )

    train_losses, val_losses, map50s = [], [], []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # Warm-up or cosine step
        if args.use_cosine:
            if epoch <= args.warmup_epochs:
                scale = epoch / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * scale
            else:
                scheduler.step()

        print(
            f"[Epoch {epoch}] Learning rates: "
            f"{[pg['lr'] for pg in optimizer.param_groups]}"
        )

        start_t = time.time()

        # 1) Train
        tloss, tcomps = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, args.use_amp, scaler
        )
        torch.cuda.empty_cache()

        # 2) Validate
        vloss, vcomps = validate(model, val_loader, device, epoch)
        torch.cuda.empty_cache()

        # 3) COCO evaluation
        stats = evaluate_coco(model, val_ds, device)
        torch.cuda.empty_cache()
        map50 = stats[0]

        train_losses.append(tloss)
        val_losses.append(vloss)
        map50s.append(map50)

        if not args.use_cosine:
            scheduler.step(map50)

        print(
            f"[Epoch {epoch}] "
            f"Train Loss={tloss:.4f}, Val Loss={vloss:.4f}, "
            f"mAP50={map50:.4f}, Time={time.time() - start_t:.1f}s"
        )

        if map50 > 0.30:
            no_improve = 0
            ckpt = os.path.join(
                args.output_dir,
                f"epoch{epoch}_loss_{vloss:.4f}_map{map50:.4f}.pth"
            )
            torch.save(model.state_dict(), ckpt)
            print(f"→ Saved new best model: {ckpt}")
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print("Early stopping due to no improvement.")
                break

    # Plot and save final model
    plot_loss_curve(train_losses, val_losses)
    plot_map_curve(map50s)
    final_ckpt = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Final model saved at {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN training with optional attention"
    )
    parser.add_argument("--train_dir", default="dataset/train")
    parser.add_argument("--test_dir", default="dataset/test_release")
    parser.add_argument(
        "--id_map_json", default="dataset/test_image_name_to_ids.json"
    )
    parser.add_argument("--output_dir", default="output_aug_resnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument(
        "--num_classes", type=int, default=5,
        help="Number of classes including background"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--use_attn", action="store_true",
        help="Enable FPNTransformer and SpatialChannelTransformer"
    )
    parser.add_argument(
        "--use_amp", action="store_true",
        help="Enable mixed precision (FP16 with AMP)"
    )
    parser.add_argument(
        "--use_cosine", action="store_true",
        help="Use CosineAnnealingLR scheduler"
    )
    parser.add_argument(
        "--eta_min", type=float, default=0.0,
        help="Minimum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0,
        help="Number of warm-up epochs for learning rate"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
