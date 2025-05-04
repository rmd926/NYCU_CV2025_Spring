# NYCU_CV 2025 Spring Lab3

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction

This repository provides the code and instructions for **training** and **evaluating** an Instance Segmentation model (**Mask R-CNN**). Initially, **resnet-101** and **EfficientNet_V2** serve as the bacckbone of baseline models. Once experiments with these baseline models are completed, we  enhance them with two compact attention modules which we proposed: an FPNTransformer—placed between FPN and RPN for global, cross-scale feature fusion—and a Spatial–Channel Transformer in the Mask Head for adaptive channel recalibration and boundary-aware sampling. The pipeline supports mixed-precision training, cosine-annealed learning-rate schedules with warm-up, and extensive augmentations. Utility scripts cover data loading, augmentation, model building, COCO-style evaluation, inference, parameter profiling, dataset statistics, and visualization.

---

## How to install the required libraries / Environment Setup

We recommend using **Python 3.12.x**.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

---


  --use_cosine \
  --eta_min 5e-6 \
  --warmup_epochs 5
```

---

### Inference

Run the trained model on the test set and export COCO-style RLE JSON.

- **Script:** `inference.py`

**Key arguments:**
```
--checkpoint     Path to your trained .pth model
--backbone       Backbone type: resnet or effnet
--train_dir      (for loader config) 
--test_dir       Directory of test images
--id_map_json    JSON mapping test filenames to IDs
--output_dir     Directory to save test-results.json
--score_thresh   Minimum score to keep detections (default: 0.5)
--num_workers    DataLoader workers (default: 4)
--seed           Random seed (default: 42)
--use_attn       Enable Spatial-Channel attention in mask head
```

**Example:**
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

---

### Visualization

Overlay predictions and/or ground‐truth masks on sample images.

- **Script:** `visualization.py`

**Arguments:**
```
--checkpoint     Path to model checkpoint
--backbone       resnet or effnet
--train_dir      Training/validation directory
--test_dir       Test images directory
--id_map_json    JSON mapping test filenames to IDs
--vis_dir        Directory to save visualization PNGs
--mode           val or test (default: val)
--num_images     Number of images to visualize    (default: 2)
--score_thresh   Prediction score threshold       (default: 0.5)
--use_attn       Enable Spatial-Channel attention
--draw_gt_mask   Overlay GT masks in GT figures
```

**Example (visualize first 3 val samples with GT masks):**
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

---

### Parameter Counting

Compute total and trainable parameters for each backbone.

- **Script:** `cal_param.py`

**Arguments:**
```
--backbone     resnet or effnet
--num_classes  Number of classes incl. background
--use_attn     Include attention modules
```

**Example:**
```bash
python cal_param.py --backbone effnet --num_classes 5 --use_attn
```
```
---



## Experiment Results 
![image](https://github.com/user-attachments/assets/114b19fb-34b9-4549-8067-1f3c1f79af8e)
![image](https://github.com/user-attachments/assets/7efae9ac-d382-4c9a-a245-ce30f94fd0a5)



## Performance Snapshot
![image](https://github.com/user-attachments/assets/e2a956e7-08f6-4e2a-992b-56e89f3b7768)



## Additional Notes
- Adjust the paths and parameters according to your dataset and system configuration.
- Ensure the checkpoint paths correctly reference your trained models.

---
