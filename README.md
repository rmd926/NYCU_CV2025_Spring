# NYCU_CV2025_Spring

Student ID: 313553024
Name: Po-Jui Su(蘇柏叡)

## Introduction
We use the ResNest as our backbone in this task, and do some modifications on it.
In brief, we perform two kinds of Multi-Scaled feature fusion(Gate Fusion and Pyramid Feature Fusion) in this task and both of the proposed methods outperform slightly than using the ResNeSt alone.

## How to train our model?
After downloading the datasets files and putting them in the datasets/data, you can start executing the resnest.py, etc directly without any command lines.


## How to use the weight.pt to inference the result?
As for the Inference phase, we have some explanations and suggestions.
--model: Three model options are available: ResNeSt, Pyramid, and Gate.
--weights: Specify the file path of the trained model weights.

Below are example command lines for the three models. Please ensure that you replace the placeholder for the trained model weights with the file you have generated and named.
ResNeSt:
inference.py --model resnest --weights resnest_best.pt

Pyramid:
inference.py --model pyramid --weights pyramid_best.pt

Gate:
inference.py --model gate --weights gate_best.pt
