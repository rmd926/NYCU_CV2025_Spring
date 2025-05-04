import numpy as np
from collections import Counter
from tqdm import tqdm

from dataloader import MedicalInstanceDataset, MedicalTestDataset


def check_trainval_stats(dataset):
    """
    Iterate through a train/val dataset with a tqdm progress bar,
    validating each sample and collecting instance statistics.
    """
    num_samples = len(dataset)
    empty_samples = 0
    instance_counts = []
    class_counter = Counter()

    for idx in tqdm(range(num_samples), desc="Checking Train+Val"):
        _, target = dataset[idx]
        boxes = target['boxes']
        labels = target['labels']
        masks = target['masks']

        # consistency checks
        n = boxes.shape[0]
        assert labels.shape[0] == n, f"Label/box mismatch at idx {idx}"
        assert masks.shape[0] == n, f"Mask/box mismatch at idx {idx}"

        if n == 0:
            empty_samples += 1
        instance_counts.append(n)
        class_counter.update(labels.tolist())

    print(f"\nTrain+Val: {num_samples} samples")
    print(f"- Empty: {empty_samples} ({empty_samples / num_samples * 100:.1f}%)")
    total_instances = sum(instance_counts)
    print(f"- Total instances: {total_instances}")
    print("- Instances per class:")
    for cls, cnt in sorted(class_counter.items()):
        print(f"   Class {cls}: {cnt}")


def check_test_alignment(test_ds):
    """
    Verify that test filenames and ID mappings align.
    """
    n_images = len(test_ds)
    n_mapping = len(test_ds.id_map)
    print(f"\nTest: {n_images} images found")
    print(f"- {n_mapping} mapping entries")

    missing = [f for f in test_ds.images if f not in test_ds.id_map]
    extra = [k for k in test_ds.id_map if k not in test_ds.images]

    if not missing and not extra:
        print("Test filenames and ID mapping are perfectly aligned.")
    else:
        if missing:
            print("Missing mappings for:", missing)
        if extra:
            print("Extra mappings for:", extra)


def main():
    # Train+Val statistics
    full_ds = MedicalInstanceDataset('dataset/train')
    check_trainval_stats(full_ds)

    # Test set alignment
    test_ds = MedicalTestDataset(
        root='dataset/test_release',
        id_map_json='dataset/test_image_name_to_ids.json'
    )
    check_test_alignment(test_ds)


if __name__ == '__main__':
    main()
