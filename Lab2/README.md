# NYCU_CV 2025 Spring

**Student ID:** 313553024  
**Name:** Po-Jui Su (蘇柏叡)

---

## Introduction


# How to install the required libraries 
We recommend you using `Python 3.12.x` and using this command line to install some libraries for this task.
`pip install -r requirements.txt`

## How to Train Our Model


## How to Use `weight.pt` for Inference
--use_res_fpn:
 Enables the custom Residual FPN enhancement, replacing the default FPN with a stronger version. If you do not want to use it, you do not need to type --use_re_fpn. Then you can use the default FPN.


--train_root:
 Specifies the root directory containing training images.


--train_ann:
 Specifies the COCO-format JSON annotation file for the training set.


--valid_root:
 Specifies the root directory containing validation images.


--valid_ann:
 Specifies the COCO-format JSON annotation file for the validation set.


--train_bs:
 Sets the batch size for training.


--valid_bs:
 Sets the batch size for validation.


--num_epochs:
 Defines the total number of epochs to train the model.


--warmup_epochs:
 Specifies the number of warmup epochs during which the learning rate gradually increases.


--lr:
 Sets the initial learning rate for training.


--eta_min:
 Defines the minimum learning rate for the cosine annealing scheduler.


--num_workers:
 Specifies the number of worker processes to use for data loading.


--batch_size (in Find_threshold.py and inference.py):
 Sets the batch size for processing images during threshold finding or inference.


--checkpoint:
 Specifies the model checkpoint file to load during the inference phase.


--finetune_weights:
 Specifies a pre-trained weight file to load for fine-tuning instead of starting from scratch.

Below are example commands for the three models. Be sure to replace the placeholder weights file with the one you have trained:

- **ResNeSt**  
  
  `python inference.py --model resnest --weights resnest_best.pt`

- **Pyramid**

  `python inference.py --model pyramid --weights pyramid_best.pt`

- **Gate**

  `python inference.py --model gate --weights gate_best.pt`




