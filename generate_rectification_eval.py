import os
from attrs import field
import cv2
from fastapi import params
import numpy as np
import pandas as pd

from generate_rectification_dataset import (
    distort_elastic,
    distort_local,
    distort_bending,
    sampling_to_ddrnet_field,
)

IMAGE_SIZE = 224

DATASETS = [
    ("val", "data/split/val", "data/rectification_val", 7777),
    ("test", "data/split/test", "data/rectification_test", 9999),
]


def process_dataset(split_name, input_dir, output_dir, master_seed):
    clean_dir = os.path.join(output_dir, "clean")
    distorted_dir = os.path.join(output_dir, "distorted")
    field_dir = os.path.join(output_dir, "fields")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(distorted_dir, exist_ok=True)
    os.makedirs(field_dir, exist_ok=True)

    records = []
    image_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".png"):
                image_paths.append(os.path.join(root, file))

    image_paths.sort()

    counts = {
        "clean": 0,
        "elastic": 0,
        "local": 0,
        "bending": 0,
    }

    for idx, input_path in enumerate(image_paths):
        rel_path = os.path.relpath(input_path, input_dir)
        sample_name = os.path.splitext(rel_path.replace("\\", "__").replace("/", "__"))[0]

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Failed to read: {input_path}")

        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(
                img,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        # Save clean target
        clean_path = os.path.join(clean_dir, f"{sample_name}.png")
        cv2.imwrite(clean_path, img)
        counts["clean"] += 1

        distortions = [
            ("elastic", distort_elastic, 0),
            ("local", distort_local, 1),
            ("bending", distort_bending, 2),
        ]

        for distortion_name, distortion_fn, type_offset in distortions:
            seed = master_seed * 100000 + idx * 10 + type_offset
            rng = np.random.default_rng(seed)

            distorted_img, dx, dy, params = distortion_fn(img, rng)
            dx_rect, dy_rect = sampling_to_ddrnet_field(dx, dy)

            distorted_filename = (
                f"{sample_name}__{distortion_name}.png"
            )
            field_filename = (
                f"{sample_name}__{distortion_name}.npz"
            )

            distorted_path = os.path.join(
                distorted_dir,
                distorted_filename,
            )

            field_path = os.path.join(
                field_dir,
                field_filename,
            )

            cv2.imwrite(distorted_path, distorted_img)

            np.savez_compressed(
                field_path,
                dx=dx_rect.astype(np.float32),
                dy=dy_rect.astype(np.float32),
            )

            magnitude = np.sqrt(dx_rect ** 2 + dy_rect ** 2)

            records.append({
                "split": split_name,
                "sample_name": sample_name,
                "distortion_type": distortion_name,
                "seed": seed,
                "mean_displacement": float(np.mean(magnitude)),
                "max_displacement": float(np.max(magnitude)),
                **params,
            })

            counts[distortion_name] += 1

    metadata_path = os.path.join(output_dir, "metadata.csv")
    pd.DataFrame(records).to_csv(metadata_path, index=False)

    print(f"\n{split_name.upper()} RECTIFICATION DATASET")
    print("=" * 50)
    print(f"Source images : {len(image_paths)}")
    print(f"Clean         : {counts['clean']}")
    print(f"Elastic       : {counts['elastic']}")
    print(f"Local         : {counts['local']}")
    print(f"Bending       : {counts['bending']}")
    print(f"Total distorted: {counts['elastic'] + counts['local'] + counts['bending']}")
    print(f"Output        : {output_dir}")

    expected_clean = len(image_paths)
    expected_distorted = len(image_paths) * 3

    assert counts["clean"] == expected_clean
    assert counts["elastic"] == len(image_paths)
    assert counts["local"] == len(image_paths)
    assert counts["bending"] == len(image_paths)

    print("STATUS: PASS")


if __name__ == "__main__":
    for split_name, input_dir, output_dir, seed in DATASETS:
        process_dataset(
            split_name,
            input_dir,
            output_dir,
            seed,
        )

    print("\nALL RECTIFICATION EVAL DATASETS GENERATED SUCCESSFULLY.")