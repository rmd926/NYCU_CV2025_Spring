import os
import json
import csv
import torch
import torchvision
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import DataLoader

# Fix random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Data augmentation, loading and training modules
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from augment import TransformConfig, CustomAugmentation, SVHNColorPolicy, \
    TransformWrapper, ComposeTransforms
from dataloader import CocoDigitsDataset, collate_fn

# Build model
def get_model(num_classes, use_res_fpn=False):
    """
    Build a Faster R-CNN model.
    If use_res_fpn is True, use the custom StrongFPN from ResFPN.py to enhance the built-in FPN.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT")
    if use_res_fpn:
        # Only import and replace if the custom ResidualFPN is requested.
        from ResFPN import ResFPN
        model.backbone.fpn = ResFPN(model.backbone.fpn, num_res_blocks=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Training and validation procedures
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model for one epoch and return the average losses.
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_anc_loss = 0.0
    count = 0
    progress_bar = tqdm(data_loader, total=len(data_loader),
                        desc=f"Epoch {epoch}", ncols=100, leave=True)
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss_classifier = loss_dict.get("loss_classifier", torch.tensor(0.0))
        loss_box_reg = loss_dict.get("loss_box_reg", torch.tensor(0.0))
        loss_objectness = loss_dict.get("loss_objectness", torch.tensor(0.0))
        loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0))
        anchor_loss = loss_objectness + loss_rpn_box_reg
        batch_loss = loss_classifier + loss_box_reg + anchor_loss

        total_loss += batch_loss.item()
        total_cls_loss += loss_classifier.item()
        total_bbox_loss += loss_box_reg.item()
        total_anc_loss += anchor_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        progress_bar.set_postfix({
            "Cls": f"{loss_classifier.item():.4f}",
            "BBox": f"{loss_box_reg.item():.4f}",
            "Anc": f"{anchor_loss.item():.4f}"
        })
        count += 1

    avg_total_loss = total_loss / count
    avg_cls_loss = total_cls_loss / count
    avg_bbox_loss = total_bbox_loss / count
    avg_anc_loss = total_anc_loss / count
    print(f"[Epoch {epoch}] Train Loss: Total {avg_total_loss:.4f}, "
          f"Cls: {avg_cls_loss:.4f}, BBox: {avg_bbox_loss:.4f}, "
          f"Anc: {avg_anc_loss:.4f}")
    return avg_total_loss, avg_cls_loss, avg_bbox_loss, avg_anc_loss


def validate_loss(model, data_loader, device):
    """
    Compute loss on the validation set without updating parameters.
    """
    model.train()  # Keep model in train mode to get loss values
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_anc_loss = 0.0
    count = 0
    progress_bar = tqdm(data_loader, total=len(data_loader),
                        desc="Validation Loss", ncols=100, leave=True)
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss_classifier = loss_dict.get("loss_classifier",
                                            torch.tensor(0.0, device=device))
            loss_box_reg = loss_dict.get("loss_box_reg",
                                         torch.tensor(0.0, device=device))
            loss_objectness = loss_dict.get("loss_objectness",
                                            torch.tensor(0.0, device=device))
            loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg",
                                             torch.tensor(0.0, device=device))
            anchor_loss = loss_objectness + loss_rpn_box_reg
            batch_loss = loss_classifier + loss_box_reg + anchor_loss

            total_loss += batch_loss.item()
            total_cls_loss += loss_classifier.item()
            total_bbox_loss += loss_box_reg.item()
            total_anc_loss += anchor_loss.item()

            progress_bar.set_postfix({
                "Cls": f"{loss_classifier.item():.4f}",
                "BBox": f"{loss_box_reg.item():.4f}",
                "Anc": f"{anchor_loss.item():.4f}"
            })
            count += 1

    avg_total_loss = total_loss / count if count > 0 else 0.0
    avg_cls_loss = total_cls_loss / count if count > 0 else 0.0
    avg_bbox_loss = total_bbox_loss / count if count > 0 else 0.0
    avg_anc_loss = total_anc_loss / count if count > 0 else 0.0
    print(f"[Validation] Avg Loss: Total {avg_total_loss:.4f}, "
          f"Cls: {avg_cls_loss:.4f}, BBox: {avg_bbox_loss:.4f}, "
          f"Anc: {avg_anc_loss:.4f}")
    return avg_total_loss, avg_cls_loss, avg_bbox_loss, avg_anc_loss


def evaluate(model, data_loader, device):
    """
    Evaluate the model and return the prediction results.
    """
    model.eval()
    results = []
    progress_bar = tqdm(data_loader, total=len(data_loader),
                        desc="Evaluation", ncols=100, leave=True)
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, tgt in zip(outputs, targets):
                image_id = tgt["image_id"].item()
                boxes = output["boxes"].cpu().numpy().tolist()
                labels = output["labels"].cpu().numpy().tolist()
                scores = output["scores"].cpu().numpy().tolist()
                results.append((image_id, boxes, labels, scores))
    return results


# Prediction and post-processing
def generate_coco_predictions(results, out_file="pred.json", score_threshold=0.0):
    """
    Save the prediction results in COCO JSON format.
    the default score_threshold will be set to 0.5
    but we will use the Find_threshold.py to find the
    most appropriate threshold, and finally used in the inference.py.
    """
    coco_results = []
    for image_id, boxes, labels, scores in results:
        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            x_min, y_min, x_max, y_max = box
            w, h = x_max - x_min, y_max - y_min
            coco_results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "score": float(score)
            })
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco_results, f, ensure_ascii=False, indent=2)
    print(f"[Task1] COCO predictions saved to {out_file}, total = {len(coco_results)}")
    return len(coco_results)


def generate_task2_predictions(results, out_file="task2_pred.csv"):
    """
    Generate digit strings from predictions and save them in CSV format.
    """
    flattened = []
    for image_id, boxes, labels, scores in results:
        for box, label, score in zip(boxes, labels, scores):
            flattened.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": box,
                "score": score
            })
    grouped = defaultdict(list)
    for item in flattened:
        grouped[item["image_id"]].append(item)
    final_list = []
    for image_id, ann_list in grouped.items():
        ann_list.sort(key=lambda x: x["bbox"][0])
        digits = [str((ann["category_id"] - 1) % 10) for ann in ann_list]
        pred_label = "".join(digits)
        final_list.append((image_id, pred_label))
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        for row in final_list:
            writer.writerow(row)
    print(f"[Task2] Task2 predictions saved to {out_file}, total = {len(final_list)}")


def check_digit_categories(ann_file):
    """
    Check whether the categories in the annotation file are set as expected (ID 1~10 -> '0'~'9').
    """
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "categories" not in data:
        print(f"[Warning] No 'categories' found in {ann_file}")
        return
    categories = data["categories"]
    id_name_map = {cat["id"]: cat["name"] for cat in categories}
    for cid in range(1, 11):
        expected_name = str(cid - 1)
        actual_name = id_name_map.get(cid)
        if actual_name != expected_name:
            raise ValueError(f"[ERROR] Category ID {cid} should be named "
                             f"'{expected_name}' but found '{actual_name}'")
    print(f"[Check] Categories in '{ann_file}' are valid: ID 1~10 -> name '0'~'9'")


def compute_map(gt_ann_file, pred_ann_file):
    """
    Compute mAP using the COCO API.
    """
    coco_gt = COCO(gt_ann_file)
    coco_dt = coco_gt.loadRes(pred_ann_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]


def parse_arguments():
    """
    Parse and return command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN with StrongFPN")
    parser.add_argument("--train_root", type=str, default="dataset/train",
                        help="Training dataset root directory")
    parser.add_argument("--train_ann", type=str, default="dataset/train.json",
                        help="Training annotation file")
    parser.add_argument("--valid_root", type=str, default="dataset/valid",
                        help="Validation dataset root directory")
    parser.add_argument("--valid_ann", type=str, default="dataset/valid.json",
                        help="Validation annotation file")
    parser.add_argument("--train_bs", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--valid_bs", type=int, default=4,
                        help="Validation batch size")
    parser.add_argument("--num_epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--eta_min", type=float, default=5e-6,
                        help="Minimum LR for cosine annealing")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers")
    parser.add_argument("--use_res_fpn", action="store_true",
                        help="Use the custom ResidualFPN for enhancement")
    parser.add_argument("--finetune_weights", type=str, default=None,
                        help="Pre-trained weight file for fine-tuning (not checkpoint resume)")
    return parser.parse_args()


def run_training(args, device):
    """
    Run the training process and return training curves, the model, and the validation DataLoader.
    """
    # Build datasets and DataLoaders
    svhn_policy = SVHNColorPolicy()
    config = TransformConfig()
    train_transforms = ComposeTransforms([
        TransformWrapper(svhn_policy),
        CustomAugmentation(config, apply_affine=True),
        TransformWrapper(T.ToTensor())
    ])
    valid_transforms = TransformWrapper(T.ToTensor())

    train_dataset = CocoDigitsDataset(
        root=args.train_root,
        ann_file=args.train_ann,
        transforms=train_transforms
    )
    valid_dataset = CocoDigitsDataset(
        root=args.valid_root,
        ann_file=args.valid_ann,
        transforms=valid_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Build model, optimizer, and learning rate scheduler
    num_classes = 11  # Digits 0-9 plus background
    model = get_model(num_classes, use_res_fpn=args.use_res_fpn)
    model.to(device)

    # If fine-tuning weights are provided, load them
    if args.finetune_weights is not None:
        print(f"Loading pre-trained weights from {args.finetune_weights} for fine-tuning.")
        state_dict = torch.load(args.finetune_weights, map_location=device)
        model.load_state_dict(state_dict)

    initial_lr = args.lr
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    # group
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": initial_lr * 0.5},
        {"params": head_params, "lr": initial_lr}
    ], weight_decay=0.0005)

    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = initial_lr

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - args.warmup_epochs,
        eta_min=args.eta_min
    )

    train_loss_list = []
    val_loss_list = []
    map_list = []

    print("Start Training...")
    for epoch in range(1, args.num_epochs + 1):
        # Disable data augmentation in the last 15 epochs
        if epoch > args.num_epochs - 15:
            train_dataset.transforms = TransformWrapper(T.ToTensor())
            print(f"[Epoch {epoch}] No augmentation applied.")
        else:
            print(f"[Epoch {epoch}] Using augmentation for training.")


        # Warmup and LR adjustment
        if epoch <= args.warmup_epochs:
            lr_scale = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lr_scale
            print(f"[Epoch {epoch}] Warmup lr set to: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            scheduler.step()
            print(f"[Epoch {epoch}] CosineAnnealing lr: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_cls, train_bbox, train_anchor = train_one_epoch(
            model, optimizer, train_loader, device, epoch)
        val_loss, val_cls, val_bbox, val_anchor = validate_loss(
            model, valid_loader, device)
        results = evaluate(model, valid_loader, device)
        num_preds = generate_coco_predictions(results, out_file="pred.json", score_threshold=0.5)

        if num_preds == 0:
            print(f"[Epoch {epoch}] Warning: No predictions generated (empty JSON). Setting mAP to 0.")
            epoch_map = 0.0
        else:
            try:
                epoch_map = compute_map(args.valid_ann, "pred.json")
            except Exception as e:
                print(f"[Epoch {epoch}] Error computing mAP: {e}")
                epoch_map = 0.0

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} "
              f"(Cls: {train_cls:.4f}, BBox: {train_bbox:.4f}, Anc: {train_anchor:.4f})")
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} "
              f"(Cls: {val_cls:.4f}, BBox: {val_bbox:.4f}, Anc: {val_anchor:.4f})")
        print(f"[Epoch {epoch}] mAP: {epoch_map:.4f}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        map_list.append(epoch_map)

        if epoch_map > 0:
            weight_filename = (
                f"fasterrcnn_res50_digits_loss_{val_loss:.4f}_"
                f"mAP_{epoch_map:.4f}_resfpn_ft.pth"
            )
            torch.save(model.state_dict(), weight_filename)
            print(f"New best model saved to '{weight_filename}'")

    return train_loss_list, val_loss_list, map_list, model, valid_loader


def run_final_evaluation(model, valid_loader, device, args,
                         train_loss_list, val_loss_list, map_list):
    """
    Run final evaluation, generate predictions, and plot training loss and mAP curves.
    """
    print("Final model saved to 'fasterrcnn_res50_digits_final_resfpn.pth'")
    print("Generating predictions on validation set...")
    results = evaluate(model, valid_loader, device)
    generate_coco_predictions(results, out_file="pred.json", score_threshold=0.5)
    generate_task2_predictions(results, out_file="task2_pred.csv")
    # Plot the loss and mAP curves (requires plot_loss_curve and plot_map_curve in utils module)
    from utils import plot_loss_curve, plot_map_curve
    plot_loss_curve(train_loss_list, val_loss_list)
    plot_map_curve(map_list)
    print("Final evaluation complete.")


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if args.use_res_fpn:
        print("Using the proposed method: resnet50 + residual_fpn + roi")
    else:
        print("Using the baseline model: resnet50 + fpn + roi")
    check_digit_categories(args.train_ann)
    check_digit_categories(args.valid_ann)

    train_loss_list, val_loss_list, map_list, model, valid_loader = run_training(
        args, device)

    torch.save(model.state_dict(), "fasterrcnn_res50_digits_final.pth")
    print("Final model saved to 'fasterrcnn_res50_digits_final.pth'")

    run_final_evaluation(model, valid_loader, device, args,
                         train_loss_list, val_loss_list, map_list)


if __name__ == "__main__":
    main()
