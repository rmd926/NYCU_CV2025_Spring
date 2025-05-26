import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms.functional as TF

# ------------------------------
# Dataset Definition: Stratified Random Split
# ------------------------------
class Dataset(TorchDataset):
    def __init__(self, root_dir, split='train', seed=42, val_per_class=320):
        """
        Args:
            root_dir (str): Dataset root directory containing 'train' and 'test' subfolders.
            split (str): One of 'train', 'val', or 'test'.
            seed (int): Random seed for reproducibility.
            val_per_class (int): Number of validation samples per class (rain, snow).
        """
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"
        self.split = split

        base = 'train' if split in ('train', 'val') else 'test'
        self.degraded_dir = os.path.join(root_dir, base, 'degraded')
        if base == 'train':
            self.clean_dir = os.path.join(root_dir, 'train', 'clean')

        # Read all degraded image filenames
        all_files = sorted([
            f for f in os.listdir(self.degraded_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if split in ('train', 'val'):
            # Separate into rain and snow categories
            rain_files = [f for f in all_files if f.startswith('rain-')]
            snow_files = [f for f in all_files if f.startswith('snow-')]

            # Shuffle with fixed seed for reproducibility
            random.seed(seed)
            random.shuffle(rain_files)
            random.shuffle(snow_files)

            # Select val_per_class samples for validation; use the rest for training
            val_rain = rain_files[:val_per_class]
            val_snow = snow_files[:val_per_class]
            train_rain = rain_files[val_per_class:]
            train_snow = snow_files[val_per_class:]

            if split == 'train':
                selected = train_rain + train_snow
            else:  # val
                selected = val_rain + val_snow

            self.filenames = sorted(selected)
        else:
            # For test split, use all images
            self.filenames = all_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load degraded image
        deg_path = os.path.join(self.degraded_dir, fname)
        degraded = Image.open(deg_path).convert('RGB')
        degraded = TF.to_tensor(degraded)

        if self.split in ('train', 'val'):
            # Get the corresponding clean image
            name, ext = os.path.splitext(fname)
            prefix, num = name.split('-', 1)
            clean_name = f"{prefix}_clean-{num}{ext}"
            clean_path = os.path.join(self.clean_dir, clean_name)
            clean = Image.open(clean_path).convert('RGB')
            clean = TF.to_tensor(clean)
            return degraded, clean, fname
        else:
            return degraded, fname


# ------------------------------
# DataLoader Factory
# ------------------------------
def get_dataloader(root_dir,
                   split='train',
                   batch_size=1,
                   num_workers=0,
                   seed=42,
                   val_per_class=320):
    """
    Create DataLoader supporting 'train', 'val', and 'test' modes.

    Args:
        root_dir (str): Dataset root directory.
        split (str): Dataset split - 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        seed (int): Random seed for stratified split.
        val_per_class (int): Number of validation samples per class.
    """
    dataset = Dataset(root_dir, split, seed=seed, val_per_class=val_per_class)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return loader