# NYCU_CV 2025 Spring

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction
This project uses **ResNeSt** as the backbone network and applies two types of multi-scale feature fusion—**Gate Fusion** and **Pyramid Feature Fusion**. Both proposed methods slightly outperform the baseline ResNeSt architecture in our experiments.

# How to install the required libraries 
We recommend you using `Python 3.12.x` and using this command line to install some libraries for this task.
'pip install -r requirements.txt'

## How to Train Our Model
1. Download and extract the dataset files, then place them under `datasets/data`.
2. Run `resnest.py` (or the other training scripts) directly; no additional command-line arguments are required for training.

## How to Use `weight.pt` for Inference
To perform inference, note the following arguments:

- **--model**  
  Three model options are available: `resnest`, `pyramid`, and `gate`.
- **--weights**  
  Specify the file path to the trained model weights.

Below are example commands for the three models. Be sure to replace the placeholder weights file with the one you have trained:

- **ResNeSt**  
  
  python inference.py --model resnest --weights resnest_best.pt

- **Pyramid**

  python inference.py --model pyramid --weights pyramid_best.pt

- **Gate**

  python inference.py --model gate --weights gate_best.pt

## Performance Snapshot
We finally got a 0.95 accuracy and beated the strong baseline on the Codabench.
![image](https://github.com/user-attachments/assets/44675fbd-e8c1-477d-b451-3efe2dc43b18)
