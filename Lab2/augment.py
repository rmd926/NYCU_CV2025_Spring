import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class TransformConfig:
    def __init__(self):
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0


def get_forward_affine_matrix(angle, translate, scale, shear, center):
    cx, cy = center
    angle_rad = np.deg2rad(angle)
    shear_rad = np.deg2rad(shear)
    R = np.array([
        [np.cos(angle_rad) * scale, -np.sin(angle_rad) * scale],
        [np.sin(angle_rad) * scale, np.cos(angle_rad) * scale]
    ])
    S = np.array([
        [1, np.tan(shear_rad)],
        [0, 1]
    ])
    A = R.dot(S)
    t = np.array([cx, cy]) - A.dot(np.array([cx, cy])) + np.array(translate)
    M = np.hstack([A, t.reshape(2, 1)])
    return M


class CustomAugmentation:
    def __init__(self, config, apply_affine=True):
        self.config = config
        self.apply_affine = apply_affine
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

    def __call__(self, image, target):
        if random.random() < self.config.hsv_prob:
            image = self.color_jitter(image)
        image, target = self.flip_with_target(image, target)
        if self.apply_affine:
            image, target = self.affine_with_target(image, target)
        return image, target

    def flip_with_target(self, image, target):
        if random.random() < self.config.flip_prob:
            image = T.functional.hflip(image)
            w, _ = image.size
            boxes = target["boxes"]
            boxes_flipped = boxes.clone()
            boxes_flipped[:, 0] = w - boxes[:, 2]
            boxes_flipped[:, 2] = w - boxes[:, 0]
            target["boxes"] = boxes_flipped
        return image, target

    def affine_with_target(self, image, target):
        w, h = image.size
        center = (w * 0.5, h * 0.5)
        angle = random.uniform(-self.config.degrees, self.config.degrees)
        max_dx = self.config.translate * w
        max_dy = self.config.translate * h
        translate = (
            random.uniform(-max_dx, max_dx),
            random.uniform(-max_dy, max_dy)
        )
        scale = 1.0
        shear = random.uniform(-self.config.shear, self.config.shear)
        M = get_forward_affine_matrix(angle, translate, scale, shear, center)

        boxes = target["boxes"].numpy()
        new_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])
            transformed = (M.dot(corners_h.T)).T
            x_new = transformed[:, 0]
            y_new = transformed[:, 1]
            new_box = [
                float(x_new.min()),
                float(y_new.min()),
                float(x_new.max()),
                float(y_new.max())
            ]
            new_boxes.append(new_box)
        target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
        image = F.affine(image, angle=angle, translate=translate,
                         scale=scale, shear=shear)
        return image, target

# reference: https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
class SubPolicy:
    def __init__(self, p1, operation1, magnitude_idx1,
                 p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "color": np.linspace(0.0, 0.9, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        func = {
            "color": lambda img, mag: T.ColorJitter(
                brightness=0.0, contrast=mag, saturation=0.0, hue=0.0
            )(img),
            "contrast": lambda img, mag: T.ColorJitter(
                brightness=0.0, contrast=mag, saturation=0.0, hue=0.0
            )(img),
            "brightness": lambda img, mag: T.ColorJitter(
                brightness=mag, contrast=0.0, saturation=0.0, hue=0.0
            )(img),
            "sharpness": lambda img, mag: T.RandomAdjustSharpness(
                sharpness_factor=2, p=1.0
            )(img),
            "posterize": lambda img, mag: T.functional.posterize(img, int(mag)),
            "solarize": lambda img, mag: T.functional.solarize(img, mag),
            "autocontrast": lambda img, _: T.functional.autocontrast(img),
            "equalize": lambda img, _: T.functional.equalize(img),
            "invert": lambda img, _: T.functional.invert(img)
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class SVHNColorPolicy:
    """只針對img做色彩變化，並不會改變幾何資訊"""
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.6, "equalize", 5, 0.6, "invert", 5, fillcolor),
            SubPolicy(0.8, "autocontrast", 5, 0.6, "equalize", 5, fillcolor),
            SubPolicy(0.7, "color", 3, 0.6, "contrast", 3, fillcolor),
            SubPolicy(0.8, "sharpness", 4, 0.6, "brightness", 4, fillcolor),
            SubPolicy(0.8, "posterize", 3, 0.6, "solarize", 4, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Color Policy"


class TransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        return self.transform(image), target


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
