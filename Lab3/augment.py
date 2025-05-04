import random
import numpy as np
import torch
import torchvision.transforms.functional as TF


class TransformConfig:
    """
    Configuration for data augmentation parameters.
    """
    def __init__(self):
        self.hsv_prob = 0.5        # probability of brightness jitter
        self.flip_prob = 0.3       # probability of horizontal/vertical flip
        self.cutout_prob = 0.5     # probability of applying CutOut
        self.cutout_size = 50      # side length of CutOut square


class CustomAugmentation:
    """
    Combines brightness jitter, flips, and CutOut.
    """
    def __init__(self, config: TransformConfig):
        self.config = config

    def __call__(self, image: torch.Tensor, target: dict):
        _, H, W = image.shape

        # 1) Brightness jitter
        if random.random() < self.config.hsv_prob:
            factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, factor)

        # 2) Horizontal flip
        if random.random() < self.config.flip_prob:
            image = TF.hflip(image)
            boxes = target["boxes"]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            target["boxes"] = boxes
            target["masks"] = TF.hflip(target["masks"])

        # 3) Vertical flip
        if random.random() < self.config.flip_prob:
            image = TF.vflip(image)
            boxes = target["boxes"]
            boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
            target["boxes"] = boxes
            target["masks"] = TF.vflip(target["masks"])

        # 4) CutOut
        if random.random() < self.config.cutout_prob:
            size = self.config.cutout_size
            y1 = random.randint(0, H - size)
            x1 = random.randint(0, W - size)
            y2, x2 = y1 + size, x1 + size

            # apply CutOut to image
            image[:, y1:y2, x1:x2] = 0

            # apply CutOut to masks
            masks = target["masks"]
            masks[:, y1:y2, x1:x2] = 0
            target["masks"] = masks

        return image, target


class TransformWrapper:
    """
    Wraps a simple (img, target) -> (img, target) function.
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, image, target):
        return self.fn(image, target)


class ComposeTransforms:
    """
    Composes multiple (img, target) transforms in sequence.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_train_transform():
    """
    Returns the training augmentation pipeline.
    """
    return CustomAugmentation(TransformConfig())


def get_val_transform():
    """
    No augmentation for validation.
    """
    return None
