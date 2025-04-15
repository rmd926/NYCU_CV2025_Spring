# NYCU_CV 2025 Spring

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction

This repository provides the code and instructions for training, evaluating, and fine-tuning an object detection model (Faster R-CNN) with an optional custom Residual Feature Pyramid Network (ResFPN) enhancement.

---

## Environment Setup

We recommend using **Python 3.12.x**.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

Run the training script with the following arguments:

- **`--use_res_fpn`**: Enables the custom Residual FPN enhancement.
- **`--train_root`**: Root directory containing training images.
- **`--train_ann`**: COCO-format JSON annotations for the training set.
- **`--valid_root`**: Root directory containing validation images.
- **`--valid_ann`**: COCO-format JSON annotations for the validation set.
- **`--train_bs`**: Batch size for training.
- **`--valid_bs`**: Batch size for validation.
- **`--num_epochs`**: Total number of epochs.
- **`--warmup_epochs`**: Number of epochs for learning rate warmup.
- **`--lr`**: Initial learning rate.
- **`--eta_min`**: Minimum learning rate for cosine annealing.
- **`--num_workers`**: Number of data-loading worker processes.

**Example command:**
```bash
python model.py --use_res_fpn --train_root dataset/train --train_ann dataset/train.json --valid_root dataset/valid --valid_ann dataset/valid.json --train_bs 2 --valid_bs 4 --num_epochs 30 --warmup_epochs 5 --lr 1e-4 --eta_min 5e-6 --num_workers 0
```

---

### Find Optimal Threshold

This script identifies the best confidence threshold for inference:

**Example command:**
```bash
python Find_threshold.py --use_res_fpn --train_root dataset/train --train_ann dataset/train.json --valid_root dataset/valid --valid_ann dataset/valid.json --batch_size 4
```

---

### Inference

Use trained weights for making predictions:

- **`--checkpoint`**: Path to the trained model checkpoint.

**Example command:**
```bash
python inference.py --use_res_fpn --batch_size 8 --test_root dataset/test --checkpoint best_resfpn_v2.pth
```

---

### Fine-Tuning

Start training from pre-trained weights:

- **`--finetune_weights`**: Pre-trained weights file.

**Example command:**
```bash
python model.py --use_res_fpn --train_root dataset/train --train_ann dataset/train.json --valid_root dataset/valid --valid_ann dataset/valid.json --train_bs 2 --valid_bs 4 --num_epochs 20 --warmup_epochs 1 --lr 5e-6 --eta_min 5e-8 --num_workers 0 --finetune_weights best.pth
```

---

## Additional Notes
- Adjust the paths and parameters according to your dataset and system configuration.
- Ensure the checkpoint paths correctly reference your trained models.

---

## Author

Po-Jui Su (蘇柏叡)
