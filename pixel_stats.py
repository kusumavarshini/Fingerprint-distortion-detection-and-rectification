import cv2
import glob
import numpy as np
import os

roots = [
    "data/train_distorted",
    "data/val_distorted",
    "data/test_distorted",
]

print("PIXEL STATISTICS")
print("=" * 80)

for root in roots:
    print(f"\n{root}")

    groups = {
        "clean": os.path.join(root, "clean", "**", "*.png"),
        "elastic": os.path.join(root, "distorted", "elastic", "**", "*.png"),
        "local": os.path.join(root, "distorted", "local", "**", "*.png"),
        "bending": os.path.join(root, "distorted", "bending", "**", "*.png"),
    }

    for name, pattern in groups.items():
        paths = glob.glob(pattern, recursive=True)

        means = []
        stds = []

        for f in paths:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                means.append(float(img.mean()))
                stds.append(float(img.std()))

        if means:
            print(
                f"  {name:<10} "
                f"n={len(means):4d} "
                f"mean={np.mean(means):7.2f} "
                f"std={np.mean(stds):7.2f}"
            )
        else:
            print(f"  {name:<10} n=0")