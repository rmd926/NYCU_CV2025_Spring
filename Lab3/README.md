# NYCU_CV 2025 Spring Lab3

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction

This repository provides the code and instructions for **training** and **evaluating** an Instance Segmentation model (**Mask R-CNN**). Initially, **ResNet-101** and **EfficientNet_V2** serve as the backbone of baseline models. Once experiments with these baseline models are completed, we enhance them with two compact attention modules which we proposed: an **FPNTransformer**—placed between FPN and RPN for global, cross-scale feature fusion—and a **Spatial–Channel Transformer** in the Mask Head for adaptive channel recalibration and boundary-aware sampling. The pipeline supports mixed-precision training, cosine-annealed learning-rate schedules with warm-up, and extensive augmentations. Utility scripts cover data loading, augmentation, model building, COCO-style evaluation, inference, parameter profiling, dataset statistics, and visualization.

---

## Environment Setup

We recommend **Python 3.12.x**. Install required dependencies via:

```bash
pip install -r requirements.txt
```

---

## Command-Line Options

All scripts share common arguments:

| Command Option    | Explanations                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `--train_dir`     | Directory containing the training data (default: `dataset/train`)                                   |
| `--test_dir`      | Directory containing the test images (default: `dataset/test_release`)                              |
| `--id_map_json`   | JSON file mapping test filenames to COCO image IDs (default: `dataset/test_image_name_to_ids.json`) |
| `--output_dir`    | Directory in which to save outputs (default: `inference_output`)                                    |
| `--epochs`        | Total number of training epochs (default: 50)                                                       |
| `--batch_size`    | Number of samples per batch (default: 2)                                                            |
| `--lr`            | Initial learning rate (default: 1e-4)                                                               |
| `--weight_decay`  | Weight decay factor (default: 1e-4)                                                                 |
| `--val_split`     | Fraction of data for validation (default: 0.2)                                                      |
| `--num_workers`   | Number of DataLoader workers (default: 0)                                                           |
| `--seed`          | Random seed for reproducibility (default: 42)                                                       |
| `--patience`      | Epochs with no mAP improvement before LR reduction                                                  |
| `--early_stop`    | Epochs without mAP improvement to trigger early stop                                                |
| `--num_classes`   | Total classes including background (default: 5)                                                     |
| `--checkpoint`    | Path to pretrained checkpoint for fine-tuning                                                       |
| `--use_attn`      | Enable FPN→Transformer & Spatial–Channel attention                                                  |
| `--use_amp`       | Enable FP16 mixed-precision training via AMP to reduce memory usage and accelerate training.                                                                    |
| `--use_cosine`    | Use `CosineAnnealingLR` scheduler                                                                   |
| `--eta_min`       | Minimum learning rate for cosine annealing (default: 5e-6)                                          |
| `--warmup_epochs` | Warm-up epochs before cosine decay (default: 5)                                                     |
| `--backbone`      | Choose backbone: `resnet` or `effnet`                                                               |
| `--mode`          | Visualization mode: `val` or `test`                                                                 |
| `--num_images`    | Number of images to process (for visualization)                                                     |
| `--score_thresh`  | Confidence threshold for predictions (default: 0.5)                                                 |
| `--draw_gt_mask`  | Overlay ground-truth masks in visualization                                                         |
| `--vis_dir`       | Directory to save visualization outputs                                                             |


---

## Usage

### Training

```bash
python train_resnet.py \
  --use_cosine \
  --eta_min 5e-6 \
  --warmup_epochs 5 \
  --epochs 50 \
  --batch_size 2 \
  --lr 2e-4 \
  --use_attn
```

### Inference

```bash
python inference.py \
  --checkpoint output_resnet_attn/epoch25_map0.4132.pth \
  --backbone resnet \
  --test_dir dataset/test_release \
  --id_map_json dataset/test_image_name_to_ids.json \
  --output_dir inference_resnet_attn \
  --score_thresh 0.5 \
  --num_workers 4 \
  --seed 42 \
  --use_attn
```

### Visualization

```bash
python visualization.py \
  --checkpoint output_resnet_attn/epoch25_map0.4132.pth \
  --backbone resnet \
  --mode val \
  --num_images 3 \
  --score_thresh 0.5 \
  --vis_dir vis_resnet_attn \
  --use_attn \
  --draw_gt_mask
```

### Parameter Counting

```bash
python cal_param.py \
  --backbone effnet \
  --num_classes 5 \
  --use_attn
```

---

## Experimental Results

![Loss & mAP Curves](https://github.com/user-attachments/assets/114b19fb-34b9-4549-8067-1f3c1f79af8e)

---

## Performance Snapshot

![image](https://github.com/user-attachments/assets/597da48a-ff32-4b90-92c7-562a64b0c456)

---

## Comparison of Model Parameters

![image](https://github.com/user-attachments/assets/182d2835-3507-4f78-92b2-1b8adbadf67d)

---
*Adjust paths and parameters as needed for your environment.*
