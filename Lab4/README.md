# NYCU_CV 2025 Spring Lab4

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction

This repository implements a **blind image restoration task** targeting two types of degradations—**rain and snow**—by leveraging the **PromptIR[1] model’s trainable prompt mechanism** to adapt a single set of network weights across different degradation scenarios.

We focus our contributions on:

- **Loss design**  
  - Use **Charbonnier Loss[2]** as our baseline  
  - Reproduce and compare **Guided Frequency Loss[3] as our additional expeeriments**

- **Fine-tuning strategies**  
  - Adopt a **two-stage, hierarchical fine-tuning protocol** (freeze most of the network, selectively unfreeze core modules for retraining)  
  - Introduce **test-time augmentation[4]** during inference

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
| **augment.py**           | Data augmentation pipeline: includes flipping, rotation, CutBlur[5], Mixup[6], and Blend[7] operations.                                      |
| **dataloader.py**        | Data loading and preprocessing module, wrapping PyTorch Dataset and DataLoader for training and validation.                         |
| **first_stage_ft.py**    | First-stage fine-tuning script: disables strong augmentations and retains only flipping and rotation for continued fine-tuning.    |
| **second_stage_ft.py**   | Second-stage fine-tuning script: builds on the first stage by unfreezing six core modules (refinement, first decoder layer, etc.) for gradient updates and fine-tuning. |
| **model.py**             | Defines the PromptIR model architecture and trainable prompt mechanism, including assembly of the backbone and prompt layers with forward logic. |
| **trainer.py**           | Training workflow controller: handles loop scheduling, loss/metric computation, learning-rate scheduling, checkpoint saving, and logging. |
| **inference.py**         | Inference script: loads the trained model and restores test or real images using test-time augmentation[4] (TTA) to improve PSNR.     |
| **utils.py**             | Draws training/validation loss curves and validation PSNR curves for evaluating and analyzing the training process.                |
| **vis.py**               | Visualization tool: displays/saves degraded test images side by side with restored outputs from `pred.npy` for quality comparison. |

---

## Command-Line Options

All scripts share common arguments:

| Command Option    | Default             | Description                                                 |
|-------------------|---------------------|-------------------------------------------------------------|
| `--data_dir`      | —                   | Path to the dataset root directory (required)               |
| `--batch_size`    | `2`                 | Batch size per GPU                                          |
| `--accum_steps`   | `4`                 | Number of gradient accumulation steps                       |
| `--num_workers`   | `4`                 | Number of data loader worker processes                      |
| `--lr`            | `2e-4`   or     `1.5e-4`     | Initial learning rate                                       |
| `--epochs`        | `250`    or     `100`      | Total number of training epochs                             |
| `--warmup_epochs` | `10`     or    `5`      | Number of warmup epochs                                     |
| `--save_dir`      | `"checkpoints"` | Directory to save checkpoints (model weights)               |
| `--fp16`          | `False`             | Enable mixed precision training (fp16)                      |
| `--loss`          | `"gfl"`     | Loss function choice: `"charbonnier"` or `"gfl"`            |
| `--input_folder`  | —                   | Path to degraded test images folder (required)              |
| `--output_npz`    | `"pred.npz"`        | Filename for the output `.npz` file                         |



---

## Usage

### Training

```bash
python trainer.py --data_dir datasets --batch_size 2 --accum_steps 4 --epochs 250 --lr 2e-4 --fp16 --warmup 10  --loss gfl --save_dir gfl_ckpt --num_workers 4
```

### First-Stage Fine-Tuning
```bash
python first_stage_ft.py --data_dir datasets --batch_size 2 --accum_steps 4 --epochs 100 --lr 1.5e-4 --fp16 --warmup 5  --loss gfl --save_dir gfl_ft_first --num_workers 4 --checkpoint best.pth
```

### Second-Stage Fine-Tuning
```bash
python second_stage_ft.py --data_dir datasets --batch_size 2 --accum_steps 4 --epochs 100 --lr 1.5e-4 --fp16 --warmup 5  --loss gfl --save_dir gfl_ft_second --num_workers 4 --checkpoint best.pth
```

### Inference

```bash
python inference.py  --checkpoint best.pth --input_folder datasets/test/degraded
```

### Visualization

```bash
python vis.py
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

## References
[1] Potlapalli, V., Zamir, S. W., Khan, S. H., & Shahbaz Khan, F. (2023). Promptir: Prompting for all-in-one image restoration. Advances in Neural Information Processing Systems, 36, 71275-71293.

[2] Charbonnier, P., Blanc-Feraud, L., Aubert, G., & Barlaud, M. (1994, November). Two deterministic half-quadratic regularization algorithms for computed imaging. In Proceedings of 1st international conference on image processing (Vol. 2, pp. 168-172). IEEE.

[3] Benjdira, B., Ali, A. M., & Koubaa, A. (2023). Guided frequency loss for image restoration. arXiv preprint arXiv:2309.15563.

[4] Shanmugam, D., Blalock, D., Balakrishnan, G., & Guttag, J. (2021). Better aggregation in test-time augmentation. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1214-1223).

[5] Yoo, J., Ahn, N., & Sohn, K. A. (2020). Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8375-8384).

[6] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.

[7] Shyam, P., Sengar, S. S., Yoon, K. J., & Kim, K. S. (2021). Evaluating copy-blend augmentation for low level vision tasks. arXiv preprint arXiv:2103.05889.

---
*Adjust paths and parameters as needed for your environment.*
