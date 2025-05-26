# """
# CutBlur & Related Augmentations
# Copyright 2020-present NAVER corp.
# MIT license

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['apply_augment', 'blend', 'mixup', 'cutblur', 'rgb']


def apply_augment(
    im1, im2,
    augs, probs, alphas,
    aux_prob=None, aux_alpha=None,
    mix_p=None
):
    """Apply selected augmentations and geometric transforms to image pair."""
    # Random horizontal flip
    if np.random.rand() < 0.5:
        im1 = torch.flip(im1, dims=[3])
        im2 = torch.flip(im2, dims=[3])

    # Random vertical flip
    if np.random.rand() < 0.5:
        im1 = torch.flip(im1, dims=[2])
        im2 = torch.flip(im2, dims=[2])

    # Random 90° rotation
    k = np.random.randint(0, 4)
    if k != 0:
        im1 = torch.rot90(im1, k, dims=[2, 3])
        im2 = torch.rot90(im2, k, dims=[2, 3])

    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "blend":
        im1_aug, im2_aug = blend(im1.clone(), im2.clone(), prob, alpha)
    elif aug == "mixup":
        im1_aug, im2_aug = mixup(im1.clone(), im2.clone(), prob, alpha)
    elif aug == "cutblur":
        im1_aug, im2_aug = cutblur(im1.clone(), im2.clone(), prob, alpha)
    elif aug == "rgb":
        im1_aug, im2_aug = rgb(im1.clone(), im2.clone(), prob)
    else:
        raise ValueError(f"{aug} is not a valid augmentation.")

    return im1_aug, im2_aug, None, aug


def blend(im1, im2, prob=1.0, alpha=0.6):
    """Apply global photometric blending with random color shift."""
    if alpha <= 0 or np.random.rand() >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim1 = c.expand(-1, -1, im1.size(2), im1.size(3))
    rim2 = c.expand(-1, -1, im2.size(2), im2.size(3))
    v = np.random.uniform(alpha, 1.0)

    return v * im1 + (1 - v) * rim1, v * im2 + (1 - v) * rim2


def mixup(im1, im2, prob=1.0, alpha=1.2):
    """Perform linear interpolation between pairs of images and labels."""
    if alpha <= 0 or np.random.rand() >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    rindex = torch.randperm(im1.size(0), device=im1.device)

    return (
        v * im1 + (1 - v) * im1[rindex],
        v * im2 + (1 - v) * im2[rindex],
    )


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    """Randomly replace a region in im2 with the corresponding region from im1."""
    if alpha <= 0 or np.random.rand() >= prob:
        return im1, im2

    h, w = im2.size(2), im2.size(3)
    ch = int(h * alpha)
    cw = int(w * alpha)
    cy = np.random.randint(0, h - ch + 1)
    cx = np.random.randint(0, w - cw + 1)

    if np.random.rand() > 0.5:
        im2[:, :, cy:cy + ch, cx:cx + cw] = im1[:, :, cy:cy + ch, cx:cx + cw]
    else:
        tmp = im2.clone()
        tmp[:, :, cy:cy + ch, cx:cx + cw] = im1[:, :, cy:cy + ch, cx:cx + cw]
        im2 = tmp

    return im1, im2


def rgb(im1, im2, prob=1.0):
    """Randomly permute RGB channels."""
    if np.random.rand() >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    return im1[:, perm], im2[:, perm]
