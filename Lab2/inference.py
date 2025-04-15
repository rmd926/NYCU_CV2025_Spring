import os
import json
import csv
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np
import argparse

# Fix random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# A simple wrapper that applies a transformation to an image and passes through the target.
class TransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        return self.transform(image), target


# CocoDigitsDataset for Test Data.
class CocoDigitsDataset(Dataset):
    
    def __init__(self, root, ann_file=None, transforms=None):
        self.root = root
        self.transforms = transforms

        if ann_file is not None and os.path.exists(ann_file):
            with open(ann_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
        else:
            valid_extensions = ('.jpg', '.jpeg', '.png')
            files = sorted([f for f in os.listdir(root)
                            if f.lower().endswith(valid_extensions)])
            images = [{"id": i + 1, "file_name": file_name}
                      for i, file_name in enumerate(files)]
            coco_data = {"images": images}

        self.image_info = {img['id']: img for img in coco_data['images']}
        self.ids = list(self.image_info.keys())
        self.id_to_filename = {
            img['id']: img['file_name'] for img in coco_data['images']
        }

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        info = self.image_info[image_id]
        file_name = info['file_name']
        img_path = os.path.join(self.root, file_name)
        img = Image.open(img_path).convert("RGB")
        target = {
            "file_name": file_name,
            "image_id": torch.tensor([image_id], dtype=torch.int64)
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

# Custom collate_fn to be used in DataLoader.
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes, use_res_fpn=False):
    """
    Build a Faster R-CNN model.

    If use_res_fpn is True, use the custom ResidualFPN (ResFPN.py) to enhance the built-in FPN.

    Args:
        num_classes (int): Number of classes including background.
        use_res_fpn (bool): Whether to use the custom ResidualFPN.

    Returns:
        model (nn.Module): Faster R-CNN model.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )
    if use_res_fpn:
        from ResFPN import ResFPN  # Load Residual FPN from ResFPN.py
        model.backbone.fpn = ResFPN(model.backbone.fpn, num_res_blocks=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Override NMS to use a dummy function (useful for debugging / custom behavior)
    def dummy_nms(boxes, scores, idxs, iou_threshold):
        return torch.arange(boxes.size(0), device=boxes.device)

    torchvision.ops.batched_nms = dummy_nms

    return model


def inference(model, data_loader, device):
    """
    Run inference on the test dataset and return prediction results.

    Args:
        model (nn.Module): The detection model.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device used for inference.

    Returns:
        list: A list of tuples (file_name, boxes, labels, scores).
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
                file_name = tgt["file_name"]
                boxes = output["boxes"].cpu().numpy().tolist()
                labels = output["labels"].cpu().numpy().tolist()
                scores = output["scores"].cpu().numpy().tolist()
                results.append((file_name, boxes, labels, scores))
    return results


def generate_coco_predictions(results, out_file="pred.json", score_threshold=0.5):
    """
    Save the prediction results in COCO JSON format.

    Args:
        results (list): List of prediction results.
        out_file (str): Output JSON file name.
        score_threshold (float): Threshold to filter weak predictions.

    Returns:
        int: Total number of valid predictions saved.
    """
    coco_results = []
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
    print(f"[Task1] COCO predictions saved to {out_file}, total = {len(coco_results)}")
    return len(coco_results)


def generate_task2_predictions(results, out_file="pred.csv", score_threshold=0.5):
    """
    Generate digit strings from predictions and save them in CSV format.

    Args:
        results (list): List of prediction results.
        out_file (str): Output CSV file name.
        score_threshold (float): Threshold to filter weak predictions.
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
    print(f"[Task2] Task2 predictions saved to {out_file}, total = {len(final_list)}")


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
        "--test_root", type=str, default="dataset/test",
        help="Test dataset root directory"
    )
    parser.add_argument(
        "--test_ann", type=str, default=None,
        help="Test annotation file (if available)"
    )
    parser.add_argument(
        "--use_res_fpn", action="store_true",
        help="Use the custom ResidualFPN for enhancement"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="best_resfpn_v2.pth",
        help="Checkpoint file for inference"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if args.use_res_fpn:
        print("Using the proposed method: resnet50 + residual_fpn + roi")
    else:
        print("Using the baseline model: resnet50 + fpn + roi")

    # Build test dataset and DataLoader
    test_transforms = TransformWrapper(T.ToTensor())
    test_dataset = CocoDigitsDataset(
        root=args.test_root,
        ann_file=args.test_ann,
        transforms=test_transforms
    )
    # id_to_filename can be used if needed for further evaluation
    id_to_filename = test_dataset.id_to_filename
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    num_classes = 11  # 10 digits + background
    model = get_model(num_classes, use_res_fpn=args.use_res_fpn)
    model.to(device)

    print(f"Loading checkpoint from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results = inference(model, test_loader, device)
    generate_coco_predictions(results, out_file="pred.json", score_threshold=0.73)
    generate_task2_predictions(results, out_file="pred.csv", score_threshold=0.73)


if __name__ == "__main__":
    main()
