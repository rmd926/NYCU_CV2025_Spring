import os
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image


class CocoDigitsDataset(Dataset):
    """
    A dataset for COCO-formatted digit data.

    This dataset reads COCO-formatted annotations and creates dictionaries
    for image information and annotations (grouped by image_id). Additionally,
    it builds a mapping between image_id and file name for use in evaluation.
    """

    def __init__(self, root, ann_file, transforms=None):
        """
        Initialize the CocoDigitsDataset.

        Args:
            root (str): Path to the directory containing the images.
            ann_file (str): Path to the COCO-format JSON annotation file.
            transforms (callable, optional): A function or transform that
                takes in an image and target and returns a transformed version.
        """
        self.root = root
        self.transforms = transforms

        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Create a dictionary for image information where the key is the image id.
        self.image_info = {img['id']: img for img in coco_data['images']}

        # Create a defaultdict for annotations, grouping annotations by image_id.
        self.annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)

        # Get a list of all image ids.
        self.ids = list(self.image_info.keys())

        # Build a mapping from image id to file name for evaluation purposes.
        self.id_to_filename = {
            img['id']: img['file_name'] for img in coco_data['images']
        }

    def __getitem__(self, idx):
        """
        Retrieve the image and target information for a given index.

        Args:
            idx (int): The index of the data sample.

        Returns:
            tuple: A tuple (image, target), where image is a PIL.Image in RGB
                   mode and target is a dictionary containing "boxes", "labels",
                   and "image_id".
        """
        image_id = self.ids[idx]
        info = self.image_info[image_id]
        file_name = info['file_name']
        img_path = os.path.join(self.root, file_name)
        img = Image.open(img_path).convert("RGB")

        ann_list = self.annotations[image_id]
        boxes, labels = [], []
        for ann in ann_list:
            # COCO format bbox: [x, y, w, h]
            x_min, y_min, w, h = ann['bbox']
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64)
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.ids)


def collate_fn(batch):
    """
    Custom collate_fn for use with DataLoader.

    This function organizes a batch of samples (each a tuple of image and target)
    into a tuple of lists: one for images and one for targets.

    Args:
        batch (list): A list of (image, target) tuples.

    Returns:
        tuple: A tuple containing a list of images and a list of targets.
    """
    return tuple(zip(*batch))
