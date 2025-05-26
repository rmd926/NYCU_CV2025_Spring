import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
degraded_dir = 'datasets/test/degraded'
npz_path = 'pred.npz'

# 1. Load restored results
data = np.load(npz_path)
filenames = sorted(data.keys())

# 2. Select only images 0–1 and 50–51 (total of 4 images)
sample_fns = filenames[:2] + filenames[50:52]

# 3. Plot in 2 rows × 4 columns
rows, cols = 2, len(sample_fns)
plt.figure(figsize=(cols * 4, rows * 4))

for idx, fn in enumerate(sample_fns):
    # top row: degraded images
    ax = plt.subplot(rows, cols, idx + 1)
    img_path = os.path.join(degraded_dir, fn)
    img = Image.open(img_path).convert('RGB')
    ax.imshow(np.array(img))
    ax.axis('off')

    # bottom row: restored images
    ax = plt.subplot(rows, cols, cols + idx + 1)
    # shape = (3, H, W)
    res_arr = data[fn]
    # H x W x 3
    res_img = res_arr.transpose(1, 2, 0)
    ax.imshow(res_img)
    ax.axis('off')

plt.tight_layout()
plt.show()
