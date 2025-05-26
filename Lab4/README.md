# NYCU_CV 2025 Spring Lab4

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction

This repository implements a **blind image restoration task** targeting two types of degradations—**rain and snow**—by leveraging the **PromptIR model’s trainable prompt mechanism** to adapt a single set of network weights across different degradation scenarios.

We focus our contributions on:

- **Loss design**  
  - Use **Charbonnier Loss** as our baseline  
  - Reproduce and compare **Guided Frequency Loss**

- **Fine-tuning strategies**  
  - Adopt a **two-stage, hierarchical fine-tuning protocol** (freeze most of the network, selectively unfreeze core modules for retraining)  
  - Introduce **test-time augmentation** during inference

With this combined training and fine-tuning approach, we achieve **PSNR results that surpass the strong baseline**.  

---

## Environment Setup

We recommend **Python 3.12.x**. Install required dependencies via:

```bash
pip install -r requirements.txt
```

---

## File Structure

| File                     | Description                                                                                                                        |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| **GFL_Loss.py**          | Implements Guided Frequency Loss (GFL) and helper functions for computing and comparing the frequency-domain guided loss.          |
| **augment.py**           | Data augmentation pipeline: includes flipping, rotation, CutBlur, Mixup, and Blend operations.                                      |
| **dataloader.py**        | Data loading and preprocessing module, wrapping PyTorch Dataset and DataLoader for training and validation.                         |
| **first_stage_ft.py**    | First-stage fine-tuning script: disables strong augmentations and retains only flipping and rotation for continued fine-tuning.    |
| **second_stage_ft.py**   | Second-stage fine-tuning script: builds on the first stage by unfreezing six core modules (refinement, first decoder layer, etc.) for gradient updates and fine-tuning. |
| **model.py**             | Defines the PromptIR model architecture and trainable prompt mechanism, including assembly of the backbone and prompt layers with forward logic. |
| **trainer.py**           | Training workflow controller: handles loop scheduling, loss/metric computation, learning-rate scheduling, checkpoint saving, and logging. |
| **inference.py**         | Inference script: loads the trained model and restores test or real images using test-time augmentation (TTA) to improve PSNR.     |
| **utils.py**             | Draws training/validation loss curves and validation PSNR curves for evaluating and analyzing the training process.                |
| **vis.py**               | Visualization tool: displays/saves degraded test images side by side with restored outputs from `pred.npy` for quality comparison. |

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

| PromptIR[1] w/ Loss Function & Fine-Tuning Settings         | val PSNR | test PSNR (w/o TTA[4]) | test PSNR (w/ TTA[4]) |
|-------------------------------------------------------------|----------|-------------------------|---------------------------------------------|
| Charbonnier Loss[2] w/o FT                                  | 29.534   | 29.481                  | 30.001                                      |
| Charbonnier Loss[2] w/ first-stage FT                       | 29.830   | 29.833                  | 30.385                                      |
| Charbonnier Loss[2] w/ second-stage FT                      | 29.949   | 29.963                  | 30.517                                      |
| Charbonnier Loss[2] w/ second-stage FT [FT again]           | 30.024   | 30.042                  | 30.586                                      |
| GFL Loss[3] w/o FT                                          | 29.620   | 29.556                  | 30.229                                      |
| GFL Loss[3] w/ first-stage FT                               | 29.879   | 29.878                  | 30.491                                      |
| GFL Loss[3] w/ second-stage FT                              | 29.995   | 30.028                  | 30.630                                      |
| **GFL Loss[3] w/ second-stage FT [FT again]**                   | **30.062**   | **30.140**                  | **30.733**                                  |

---

## Performance Snapshot

![image](https://github.com/user-attachments/assets/441fc94e-ec81-4a43-a01f-33b7ff947a41)


---
*Adjust paths and parameters as needed for your environment.*
