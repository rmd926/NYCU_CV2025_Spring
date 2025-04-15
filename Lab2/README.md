# NYCU_CV 2025 Spring

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction

This repository provides the code and instructions for **training**, **evaluating**, and **fine-tuning** an object detection model (**Faster R-CNN**). Initially, **fasterrcnn_res50_fpn** and **fasterrcnn_res50_fpn_v2** serve as baseline models. Once experiments with these baseline models are completed, the **Residual Feature Pyramid Network (ResFPN)** enhancement can be applied for further **performance improvements**.

---

## Environment Setup

We recommend using **Python 3.12.x**.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Usage

---

### Training

To train the model, you can use a variety of command-line arguments to configure the process. The main arguments are as follows:

- **`--use_res_fpn`**:  
  Enables the custom Residual FPN enhancement.

- **`--train_root`** and **`--train_ann`**:  
  Specify the root directory containing training images and the COCO-format JSON annotations for the training set.

- **`--valid_root`** and **`--valid_ann`**:  
  Specify the root directory containing validation images and the COCO-format JSON annotations for the validation set.

- **`--train_bs`** and **`--valid_bs`**:  
  Set the batch size for training and validation, respectively.

- **`--num_epochs`**:  
  The total number of training epochs.

- **`--warmup_epochs`**:  
  The number of epochs for a learning rate warmup phase.

- **`--lr`** and **`--eta_min`**:  
  Define the initial learning rate and the minimum learning rate used in cosine annealing.

- **`--num_workers`**:  
  The number of worker processes to use for data loading.

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

- **`--checkpoint`**:
Path to the trained model checkpoint.

**Example command:**
```bash
python inference.py --use_res_fpn --batch_size 8 --test_root dataset/test --checkpoint best.pth
```

---

### Fine-Tuning

Start training from pre-trained weights:

- **`--finetune_weights`**:
Path to the Pre-trained weights file.

**Example command:**
```bash
python model.py --use_res_fpn --train_root dataset/train --train_ann dataset/train.json --valid_root dataset/valid --valid_ann dataset/valid.json --train_bs 2 --valid_bs 4 --num_epochs 20 --warmup_epochs 1 --lr 5e-6 --eta_min 5e-8 --num_workers 0 --finetune_weights best.pth
```

---



## Experiment Results 
![image](https://github.com/user-attachments/assets/c2d54845-f7f6-40a4-a2f8-9681afa8f45c)


## Performance Snapshot
We finally got a 0.38 mAP in Task1 and a 0.80 accuracy in Task2 and beated the strong baseline on the Codabench.
![image](https://github.com/user-attachments/assets/72a7dcd1-5c1f-4a5a-a891-8333bc398ef6)


## Additional Notes
- Adjust the paths and parameters according to your dataset and system configuration.
- Ensure the checkpoint paths correctly reference your trained models.

---
