import os
import glob
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tifffile


def collate_fn(batch):
    """
    Custom collate function to combine samples into a batch.
    """
    return tuple(zip(*batch))


class MedicalInstanceDataset(Dataset):
    """
    Dataset for training/validation of colored medical images instance segmentation.
    Each sample folder under `root` must contain:
      - image.tif
      - class1.tif, class2.tif, … (each mask with multiple instance IDs)
    Returns (image_tensor, target_dict).
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.ids = sorted(os.listdir(root))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        folder = os.path.join(self.root, sample_id)

        # 1) Load image with tifffile, convert to [C, H, W] tensor
        img_np = tifffile.imread(os.path.join(folder, "image.tif"))
        # Convert grayscale or RGBA to RGB
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # 2) Load masks and labels
        masks_list = []
        labels_list = []
        mask_paths = glob.glob(os.path.join(folder, "class*.tif"))
        for mask_path in mask_paths:
            cls_idx = int(os.path.basename(mask_path)
                          .replace("class", "")
                          .split(".")[0])
            arr = tifffile.imread(mask_path)
            for inst_id in np.unique(arr):
                if inst_id == 0:
                    continue
                mask = (arr == inst_id).astype(np.uint8)
                masks_list.append(mask)
                labels_list.append(cls_idx)

        # 3) Compute bounding boxes from masks
        boxes = []
        for mask in masks_list:
            ys, xs = np.where(mask)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            boxes.append([xmin, ymin, xmax, ymax])

        # 4) Convert lists to tensors, handle empty case
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        if masks_list:
            masks_np = np.stack(masks_list, axis=0)  # [N, H, W]
            masks = torch.as_tensor(masks_np, dtype=torch.uint8)
            labels = torch.as_tensor(np.array(labels_list),
                                     dtype=torch.int64)
        else:
            _, H, W = img.shape
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # 5) Apply transforms if provided
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


class MedicalTestDataset(Dataset):
    """
    Test dataset for colored medical images instance segmentation (no masks).
    Loads each .tif under `root`, uses `id_map_json` to get image_id.
    Returns (image_tensor, image_id, filename).
    """

    def __init__(self, root, id_map_json, transforms=None):
        self.root = root
        self.images = sorted([f for f in os.listdir(root) if f.endswith(".tif")])
        raw_map = json.load(open(id_map_json, "r"))

        # Normalize to filename → id dict
        if isinstance(raw_map, dict):
            self.id_map = raw_map
        elif isinstance(raw_map, list):
            if all(isinstance(x, dict) and "file_name" in x and "id" in x
                   for x in raw_map):
                self.id_map = {entry["file_name"]: entry["id"] for entry in raw_map}
            else:
                self.id_map = {
                    fname: raw_map[i] if i < len(raw_map) else i
                    for i, fname in enumerate(self.images)
                }
        else:
            raise ValueError("Unsupported id_map_json format")

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img_np = tifffile.imread(os.path.join(self.root, fname))
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Apply transforms if provided
        if self.transforms:
            img = self.transforms(img)

        image_id = torch.tensor([self.id_map[fname]])
        return img, image_id, fname


def get_loaders(
    train_dir: str,
    test_dir: str,
    id_map_json: str,
    batch_size: int = 4,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create DataLoader objects for training, validation, and testing.
    """
    full_ds = MedicalInstanceDataset(train_dir)
    n_total = len(full_ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_ds = MedicalTestDataset(test_dir, id_map_json)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        train_dir="dataset/train",
        test_dir="dataset/test_release",
        id_map_json="dataset/test_image_name_to_ids.json",
        batch_size=2,
        val_split=0.2,
        num_workers=2,
        seed=42
    )

    print(
        f"Total = {len(train_loader.dataset) + len(val_loader.dataset)} samples"
    )
    print(f"  Train = {len(train_loader.dataset)}")
    print(f"  Val   = {len(val_loader.dataset)}")
    print(f"  Test  = {len(test_loader.dataset)}\n")

    # Iterate and print all test items
    print("All Test items:")
    for images, image_ids, fnames in test_loader:
        img = images[0]      # batch_size=1
        image_id = image_ids[0]
        fname = fnames[0]
        print(f"  {fname} → id={int(image_id)}, shape={tuple(img.shape)}")
