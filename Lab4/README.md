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

| 檔案                     | Description                                                                                                        |
|--------------------------|--------------------------------------------------------------------------------------------------------------------|
| **GFL_Loss.py**          | 實現 Guided Frequency Loss（GFL）及其輔助函數，用於頻域引導損失的計算與比較。                                              |
| **README.md**            | 專案總覽與使用說明，包括環境依賴、安裝流程、訓練／推論命令範例以及結果再現指引。                                            |
| **augment.py**           | 數據增強管線：包含**翻轉、旋轉、CutBlur、Mixup、Blend**等操作，不含 Mosaic。                                             |
| **dataloader.py**        | 數據載入與預處理模組，封裝為 PyTorch Dataset 和 DataLoader，用於訓練與驗證。                                          |
| **first_stage_ft.py**    | 第一階段微調腳本：關閉強增強，只保留翻轉與旋轉，並凍結模型大部分結構，只訓練可學習 prompt 模塊。                            |
| **second_stage_ft.py**   | 第二階段微調腳本：在第一階段基礎上解凍 refinement、第一層 decoder 等共六個核心模塊進行梯度更新與微調。                       |
| **model.py**             | 定義 PromptIR 模型架構及可訓練 prompt 機制，包括主網與 prompt 層的組裝與前向傳播邏輯。                                      |
| **trainer.py**           | 訓練流程控制：負責迴圈調度、損失／度量計算、學習率調度、檢查點保存和日誌記錄。                                            |
| **inference.py**         | 推論腳本：載入訓練好模型，對測試或實際圖片進行復原，並計算／輸出 PSNR、保存復原結果。                                      |
| **utils.py**             | 繪製訓練與驗證損失曲線，以及驗證集 PSNR 曲線，用於評估與分析訓練過程。                                                     |
| **vis.py**               | 結果可視化：將測試集的退化圖片與由 `pred.npy` 生成的恢復圖像並排顯示／保存，便於質量對比。                                  |


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
