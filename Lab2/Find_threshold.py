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

# Data augmentation, loading and evaluation modules
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from augment import TransformWrapper
from dataloader import CocoDigitsDataset, collate_fn


def get_model(num_classes, use_res_fpn=False):
    """
    Build Faster R-CNN model.
    If use_res_fpn is True, the custom ResidualFPN module (in ResFPN.py) is used
    to enhance the built-in FPN.
    
    Args:
        num_classes (int): Number of target classes.
        use_res_fpn (bool): Whether to use custom ResidualFPN enhancement.
    
    Returns:
        nn.Module: The constructed Faster R-CNN model.
    """
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )
    if use_res_fpn:
        from ResFPN import ResFPN  # load Residual FPN
        model.backbone.fpn = ResFPN(model.backbone.fpn, num_res_blocks=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate(model, data_loader, device, id_to_filename=None):
    """
    Evaluate the model and return a list of predictions.
    
    Args:
        model (nn.Module): The detection model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device on which to perform inference.
        id_to_filename (dict, optional): Mapping from image_id to filename.
    
    Returns:
        list: List of prediction tuples (filename, boxes, labels, scores).
    """
    model.eval()
    results = []
    progress_bar = tqdm(
        data_loader, total=len(data_loader),
        desc="Inference", ncols=100, leave=True
    )
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, tgt in zip(outputs, targets):
                if "file_name" in tgt:
                    file_name = tgt["file_name"]
                elif id_to_filename is not None:
                    file_name = id_to_filename.get(
                        tgt["image_id"].item(), str(tgt["image_id"].item())
                    )
                else:
                    file_name = str(tgt["image_id"].item())
                boxes = output["boxes"].cpu().numpy().tolist()
                labels = output["labels"].cpu().numpy().tolist()
                scores = output["scores"].cpu().numpy().tolist()
                results.append((file_name, boxes, labels, scores))
    return results


def generate_coco_predictions(results, out_file="pred_.json",
                              score_threshold=0.5):
    """
    Save prediction results in COCO JSON format.
    
    Args:
        results (list): List of predictions.
        out_file (str): Output file path.
        score_threshold (float): Threshold to filter predictions.
    
    Returns:
        int: Total number of valid predictions saved.
    """
    coco_results = []
    # Sort results by filename numeric order
    sorted_results = sorted(
        results, key=lambda x: int(os.path.splitext(x[0])[0])
    )
    for file_name, boxes, labels, scores in sorted_results:
        image_id = int(os.path.splitext(file_name)[0])
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
    print(
        f"[Task1] COCO predictions saved to {out_file}, "
        f"total = {len(coco_results)}"
    )
    return len(coco_results)


def generate_task2_predictions(results, out_file="pred_.csv",
                               score_threshold=0.5):
    """
    Generate digit strings from predictions and save them in CSV format.
    
    Args:
        results (list): List of predictions.
        out_file (str): Output CSV file path.
        score_threshold (float): Threshold to filter predictions.
    """
    final_list = []
    sorted_results = sorted(
        results, key=lambda x: int(os.path.splitext(x[0])[0])
    )
    for file_name, boxes, labels, scores in sorted_results:
        image_id = int(os.path.splitext(file_name)[0])
        valid_items = [
            (label, box)
            for label, box, score in zip(labels, boxes, scores)
            if score >= score_threshold
        ]
        if not valid_items:
            pred_label = "-1"
        else:
            valid_items.sort(key=lambda x: (x[1][0] + x[1][2]) / 2)
            digits = [str((label - 1) % 10) for label, _ in valid_items]
            pred_label = "".join(digits)
        final_list.append((image_id, pred_label))
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        for row in final_list:
            writer.writerow(row)
    print(
        f"[Task2] Task2 predictions saved to {out_file}, "
        f"total = {len(final_list)}"
    )


def check_digit_categories(ann_file):
    """
    Check whether categories in the annotation file are set as expected
    (ID 1~10 -> '0'~'9').
    
    Args:
        ann_file (str): Path to the annotation JSON file.
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
            raise ValueError(
                f"[ERROR] Category ID {cid} should be named "
                f"'{expected_name}' but found '{actual_name}'"
            )
    print(
        f"[Check] Categories in '{ann_file}' are valid: "
        f"ID 1~10 -> name '0'~'9'"
    )


def compute_map(gt_ann_file, pred_ann_file):
    """
    Compute mAP using the COCO API.
    
    Args:
        gt_ann_file (str): Ground truth annotation file.
        pred_ann_file (str): Prediction annotation file.
    
    Returns:
        float: Computed mAP score.
    """
    coco_gt = COCO(gt_ann_file)
    coco_dt = coco_gt.loadRes(pred_ann_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]


def evaluate_thresholds(model, data_loader, device, gt_ann_file,
                        id_to_filename, thresholds):
    """
    Evaluate model at different score thresholds, printing mAP and CSV
    accuracy for each threshold.
    
    Args:
        model (nn.Module): Detection model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device for inference.
        gt_ann_file (str): Ground truth annotation file.
        id_to_filename (dict): Mapping from image_id to filename.
        thresholds (list): List of score thresholds.
    """
    # Get inference results
    results = evaluate(model, data_loader, device, id_to_filename)
    # Generate Ground Truth CSV only once
    gt_csv = "gt.csv"
    generate_gt_csv(gt_ann_file, out_file=gt_csv)
    for thr in thresholds:
        temp_json = "pred_temp.json"
        generate_coco_predictions(results, out_file=temp_json,
                                  score_threshold=thr)
        mAP = compute_map(gt_ann_file, temp_json)
        temp_csv = "pred_temp.csv"
        generate_task2_predictions(results, out_file=temp_csv,
                                   score_threshold=thr)
        acc = compute_csv_accuracy(gt_csv, temp_csv)
        print(
            f"Score threshold {thr:.2f}: mAP = {mAP:.4f}, "
            f"CSV Accuracy = {acc:.4f}"
        )


def generate_gt_csv(gt_ann_file, out_file="gt.csv"):
    """
    Generate a CSV file with the ground truth labels.
    
    Args:
        gt_ann_file (str): Ground truth annotation file.
        out_file (str): Output CSV file path.
    """
    with open(gt_ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = data["images"]
    annotations = data["annotations"]
    gt_dict = defaultdict(list)
    for ann in annotations:
        image_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        x_center = x + w / 2
        digit = str((ann["category_id"] - 1) % 10)
        gt_dict[image_id].append((x_center, digit))
    final_list = []
    for img in images:
        img_id = img["id"]
        if img_id not in gt_dict:
            final_list.append((img_id, "-1"))
        else:
            sorted_items = sorted(gt_dict[img_id], key=lambda x: x[0])
            digit_str = "".join([item[1] for item in sorted_items])
            final_list.append((img_id, digit_str))
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "gt_label"])
        for row in final_list:
            writer.writerow(row)
    print(f"Ground truth CSV saved to {out_file}")


def compute_csv_accuracy(gt_csv, pred_csv):
    """
    Compute the accuracy between the ground truth CSV and prediction CSV.
    
    Args:
        gt_csv (str): Ground truth CSV file path.
        pred_csv (str): Prediction CSV file path.
    
    Returns:
        float: Accuracy between the CSV files.
    """
    gt = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = int(row["image_id"])
            gt[image_id] = row["gt_label"]
    pred = {}
    with open(pred_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = int(row["image_id"])
            pred[image_id] = row["pred_label"]
    total = 0
    correct = 0
    for image_id, gt_str in gt.items():
        total += 1
        pred_str = pred.get(image_id, "-1")
        if gt_str == pred_str:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def parse_arguments():
    """
    Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference for Faster R-CNN with StrongFPN"
    )
    parser.add_argument(
        "--train_root", type=str, default="dataset/train",
        help="Training dataset root directory"
    )
    parser.add_argument(
        "--train_ann", type=str, default="dataset/train.json",
        help="Training annotation file"
    )
    parser.add_argument(
        "--valid_root", type=str, default="dataset/valid",
        help="Validation dataset root directory"
    )
    parser.add_argument(
        "--valid_ann", type=str, default="dataset/valid.json",
        help="Validation annotation file"
    )
    parser.add_argument(
        "--use_res_fpn", action="store_true",
        help="Use the custom ResidualFPN enhancement"
    )
    # Add batch size argument for flexibility
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for the DataLoader"
    )
    return parser.parse_args()


def main():
    """
    Main inference flow.
    """
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Check annotation files for category information
    for ann_file in [args.train_ann, args.valid_ann]:
        if os.path.exists(ann_file):
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "categories" in data:
                print(f"[Check] Categories in '{ann_file}' are present.")
            else:
                print(f"[Warning] No 'categories' found in {ann_file}")
        else:
            print(f"[Warning] {ann_file} does not exist.")

    # Build validation dataset and DataLoader for inference
    valid_transforms = TransformWrapper(T.ToTensor())
    valid_dataset = CocoDigitsDataset(
        root=args.valid_root,
        ann_file=args.valid_ann,
        transforms=valid_transforms
    )
    id_to_filename = valid_dataset.id_to_filename
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    num_classes = 11
    model = get_model(num_classes, use_res_fpn=args.use_res_fpn)
    model.to(device)

    # Load pre-trained weights (for fine-tuning, not resume training)
    checkpoint_path = (
        "best_resfpn_v2.pth"
    )
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Evaluate different score thresholds for mAP and CSV accuracy
    thresholds = [round(x, 2) for x in np.arange(0.45, 0.91, 0.01)]
    evaluate_thresholds(
        model, valid_loader, device,
        gt_ann_file=args.valid_ann,
        id_to_filename=id_to_filename,
        thresholds=thresholds
    )

    # Generate final predictions
    results = evaluate(model, valid_loader, device, id_to_filename)
    generate_coco_predictions(results, out_file="pred_.json", score_threshold=0.5)
    generate_task2_predictions(results, out_file="pred_.csv", score_threshold=0.5)

    # Compute final CSV accuracy
    gt_csv = "gt.csv"
    accuracy = compute_csv_accuracy(gt_csv, "pred_.csv")
    print(f"Final CSV String accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
