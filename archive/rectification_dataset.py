"""
Rectification Dataset Loader
============================

Loads samples for supervised fingerprint distortion rectification.

Each sample contains:
    distorted image -> model input
    dx, dy          -> ground-truth displacement field
    clean image     -> reconstruction/reference target

Dataset structure:
    data/rectification_train/
        clean/
        distorted/
        fields/
        metadata.csv
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "rectification_train"
)

CLEAN_DIR = os.path.join(DATA_ROOT, "clean")
DISTORTED_DIR = os.path.join(DATA_ROOT, "distorted")
FIELDS_DIR = os.path.join(DATA_ROOT, "fields")

IMAGE_SIZE = 224


# ============================================================
# DATASET
# ============================================================

class FingerprintRectificationDataset(Dataset):
    """
    PyTorch dataset for fingerprint distortion rectification.

    Returns:
        distorted : torch.Tensor, shape (1, 224, 224)
        field     : torch.Tensor, shape (2, 224, 224)
        clean     : torch.Tensor, shape (1, 224, 224)
    """

    def __init__(self, data_root=DATA_ROOT):
        self.data_root = data_root

        self.clean_dir = os.path.join(
            data_root,
            "clean"
        )

        self.distorted_dir = os.path.join(
            data_root,
            "distorted"
        )

        self.fields_dir = os.path.join(
            data_root,
            "fields"
        )

        if not os.path.isdir(self.clean_dir):
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            )

        if not os.path.isdir(self.distorted_dir):
            raise FileNotFoundError(
                f"Distorted directory not found: {self.distorted_dir}"
            )

        if not os.path.isdir(self.fields_dir):
            raise FileNotFoundError(
                f"Fields directory not found: {self.fields_dir}"
            )

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid rectification samples found."
            )

    # --------------------------------------------------------
    # SAMPLE COLLECTION
    # --------------------------------------------------------

    def _collect_samples(self):

        samples = []

        distorted_files = sorted(
            f for f in os.listdir(self.distorted_dir)
            if f.lower().endswith(".png")
        )

        for distorted_file in distorted_files:

            base_name = os.path.splitext(
                distorted_file
            )[0]

            # Example:
            # FAMILY-001__FATHER__image__elastic
            #
            # Clean filename is the part before "__elastic",
            # "__local", or "__bending".

            distortion_types = [
                "__elastic",
                "__local",
                "__bending"
            ]

            clean_base = None

            for suffix in distortion_types:
                if base_name.endswith(suffix):
                    clean_base = base_name[
                        :-len(suffix)
                    ]
                    break

            if clean_base is None:
                continue

            clean_file = clean_base + ".png"
            field_file = base_name + ".npz"

            clean_path = os.path.join(
                self.clean_dir,
                clean_file
            )

            distorted_path = os.path.join(
                self.distorted_dir,
                distorted_file
            )

            field_path = os.path.join(
                self.fields_dir,
                field_file
            )

            if not os.path.isfile(clean_path):
                continue

            if not os.path.isfile(field_path):
                continue

            samples.append({
                "clean": clean_path,
                "distorted": distorted_path,
                "field": field_path,
                "name": base_name
            })

        return samples

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    # GET ITEM
    # --------------------------------------------------------

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load distorted fingerprint
        # ----------------------------------------------------

        distorted = cv2.imread(
            sample["distorted"],
            cv2.IMREAD_GRAYSCALE
        )

        if distorted is None:
            raise RuntimeError(
                f"Could not read distorted image: "
                f"{sample['distorted']}"
            )

        # ----------------------------------------------------
        # Load clean fingerprint
        # ----------------------------------------------------

        clean = cv2.imread(
            sample["clean"],
            cv2.IMREAD_GRAYSCALE
        )

        if clean is None:
            raise RuntimeError(
                f"Could not read clean image: "
                f"{sample['clean']}"
            )

        # ----------------------------------------------------
        # Load displacement field
        # ----------------------------------------------------

        field_data = np.load(
            sample["field"]
        )

        if "dx" not in field_data or "dy" not in field_data:
            raise RuntimeError(
                f"Field file missing dx/dy: "
                f"{sample['field']}"
            )

        dx = field_data["dx"].astype(
            np.float32
        )

        dy = field_data["dy"].astype(
            np.float32
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if distorted.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Invalid distorted shape "
                f"{distorted.shape}: "
                f"{sample['distorted']}"
            )

        if clean.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Invalid clean shape "
                f"{clean.shape}: "
                f"{sample['clean']}"
            )

        if dx.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Invalid dx shape "
                f"{dx.shape}: "
                f"{sample['field']}"
            )

        if dy.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Invalid dy shape "
                f"{dy.shape}: "
                f"{sample['field']}"
            )

        # ----------------------------------------------------
        # Normalize images
        # ----------------------------------------------------

        distorted = (
            distorted.astype(np.float32) / 255.0
        )

        clean = (
            clean.astype(np.float32) / 255.0
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        distorted = torch.from_numpy(
            distorted
        ).unsqueeze(0)

        clean = torch.from_numpy(
            clean
        ).unsqueeze(0)

        field = torch.from_numpy(
            np.stack([dx, dy], axis=0)
        )

        return {
            "distorted": distorted,
            "field": field,
            "clean": clean,
            "name": sample["name"]
        }


# ============================================================
# STANDALONE VERIFICATION
# ============================================================

def main():

    print("=" * 70)
    print("RECTIFICATION DATASET LOADER VERIFICATION")
    print("=" * 70)

    dataset = FingerprintRectificationDataset()

    print(f"\nDataset root : {DATA_ROOT}")
    print(f"Samples      : {len(dataset)}")

    sample = dataset[0]

    print("\nFirst sample:")
    print(f"  Name       : {sample['name']}")
    print(
        f"  Distorted  : "
        f"{tuple(sample['distorted'].shape)} "
        f"{sample['distorted'].dtype}"
    )
    print(
        f"  Field      : "
        f"{tuple(sample['field'].shape)} "
        f"{sample['field'].dtype}"
    )
    print(
        f"  Clean      : "
        f"{tuple(sample['clean'].shape)} "
        f"{sample['clean'].dtype}"
    )

    print("\nValue ranges:")

    print(
        f"  Distorted  : "
        f"{sample['distorted'].min().item():.4f} "
        f"to "
        f"{sample['distorted'].max().item():.4f}"
    )

    print(
        f"  dx         : "
        f"{sample['field'][0].min().item():.4f} "
        f"to "
        f"{sample['field'][0].max().item():.4f}"
    )

    print(
        f"  dy         : "
        f"{sample['field'][1].min().item():.4f} "
        f"to "
        f"{sample['field'][1].max().item():.4f}"
    )

    print(
        f"  Clean      : "
        f"{sample['clean'].min().item():.4f} "
        f"to "
        f"{sample['clean'].max().item():.4f}"
    )

    print("\n[PASS] Rectification dataset loader is working.")


if __name__ == "__main__":
    main()