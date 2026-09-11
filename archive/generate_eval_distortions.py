from pathlib import Path
import csv
import json
import shutil

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_ROOT = PROJECT_ROOT / "data" / "split"

MASTER_SEED = 20260901

# One distorted image per clean image for validation/test.
# We deliberately use NEW seeds, independent of training generation.
SPLITS = {
    "val": 101,
    "test": 202,
}


# ============================================================
# Utility functions
# ============================================================

def load_image(path):
    """Load an image as a single-channel grayscale image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if img.shape != (224, 224):
        raise ValueError(
            f"Unexpected shape {img.shape} for {path}; "
            "expected (224, 224)"
        )

    return img


def apply_displacement(img, dx, dy):
    """Warp image using dense displacement fields."""
    h, w = img.shape

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    map_x = x + dx.astype(np.float32)
    map_y = y + dy.astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return warped


def displacement_stats(dx, dy):
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    return {
        "mean_displacement": float(np.mean(magnitude)),
        "max_displacement": float(np.max(magnitude)),
    }


# ============================================================
# 1. Elastic deformation
# ============================================================

def elastic_distortion(img, rng):
    h, w = img.shape

    # Coarse displacement grid.
    spacing = int(rng.integers(32, 65))

    gh = int(np.ceil(h / spacing)) + 2
    gw = int(np.ceil(w / spacing)) + 2

    alpha = float(rng.uniform(8.0, 16.0))
    sigma = float(rng.uniform(5.0, 8.0))

    # Random displacement at coarse points.
    dx_small = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
    dy_small = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)

    # Upsample to image size.
    dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Light smoothing.
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma)

    # Normalize so alpha actually controls displacement magnitude.
    current = np.sqrt(dx ** 2 + dy ** 2)
    current_max = float(np.max(current))

    if current_max > 1e-6:
        dx = dx / current_max * alpha
        dy = dy / current_max * alpha

    distorted = apply_displacement(img, dx, dy)

    stats = displacement_stats(dx, dy)

    params = {
        "distortion_type": "elastic",
        "alpha": alpha,
        "sigma": sigma,
        "grid_spacing": spacing,
        **stats,
    }

    return distorted, params


# ============================================================
# 2. Local nonlinear displacement
# ============================================================

def local_distortion(img, rng):
    h, w = img.shape

    max_disp = float(rng.uniform(5.0, 10.0))
    sigma = float(rng.uniform(8.0, 12.0))

    # 5x5 control grid.
    grid_size = 5

    x_control = np.linspace(0, w - 1, grid_size)
    y_control = np.linspace(0, h - 1, grid_size)

    dx_control = rng.uniform(
        -max_disp,
        max_disp,
        (grid_size, grid_size),
    ).astype(np.float32)

    dy_control = rng.uniform(
        -max_disp,
        max_disp,
        (grid_size, grid_size),
    ).astype(np.float32)

    # Interpolate the control-point displacement fields.
    dx = cv2.resize(
        dx_control,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    dy = cv2.resize(
        dy_control,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    # Smooth transitions between regions.
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma)

    distorted = apply_displacement(img, dx, dy)

    stats = displacement_stats(dx, dy)

    params = {
        "distortion_type": "local",
        "control_grid": "5x5",
        "max_disp_param": max_disp,
        "smooth_sigma": sigma,
        **stats,
    }

    return distorted, params


# ============================================================
# 3. Global nonlinear bending
# ============================================================

def bending_distortion(img, rng):
    h, w = img.shape

    max_disp = float(rng.uniform(15.0, 25.0))
    strength = float(rng.uniform(0.5, 1.0))

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    # Normalize coordinates around the image center.
    xn = (x - w / 2) / (w / 2)
    yn = (y - h / 2) / (h / 2)

    # Random bending direction.
    direction = int(rng.integers(0, 2))

    if direction == 0:
        # Horizontal displacement varying smoothly with y.
        dx = (
            max_disp
            * strength
            * np.sin(np.pi * yn)
            * (0.5 + 0.5 * np.abs(xn))
        )

        dy = np.zeros_like(dx)

    else:
        # Vertical displacement varying smoothly with x.
        dy = (
            max_disp
            * strength
            * np.sin(np.pi * xn)
            * (0.5 + 0.5 * np.abs(yn))
        )

        dx = np.zeros_like(dy)

    # Add a mild secondary nonlinear component.
    harmonic = int(rng.integers(1, 3))

    if direction == 0:
        dx += (
            0.25
            * max_disp
            * np.sin(harmonic * np.pi * yn)
        )
    else:
        dy += (
            0.25
            * max_disp
            * np.sin(harmonic * np.pi * xn)
        )

    distorted = apply_displacement(img, dx, dy)

    stats = displacement_stats(dx, dy)

    params = {
        "distortion_type": "bending",
        "max_disp_param": max_disp,
        "strength": strength,
        "harmonic": harmonic,
        "direction": "horizontal" if direction == 0 else "vertical",
        **stats,
    }

    return distorted, params


# ============================================================
# Distortion dispatcher
# ============================================================

def generate_distortion(img, distortion_type, rng):

    if distortion_type == "elastic":
        return elastic_distortion(img, rng)

    if distortion_type == "local":
        return local_distortion(img, rng)

    if distortion_type == "bending":
        return bending_distortion(img, rng)

    raise ValueError(f"Unknown distortion type: {distortion_type}")


# ============================================================
# Generate validation/test datasets
# ============================================================

def process_split(split_name, seed_offset):

    source_root = SPLIT_ROOT / split_name

    output_root = PROJECT_ROOT / "data" / f"{split_name}_distorted"

    if output_root.exists():
        print(f"\nRemoving previous {output_root}")
        shutil.rmtree(output_root)

    clean_root = output_root / "clean"
    distorted_root = output_root / "distorted"

    clean_root.mkdir(parents=True)
    distorted_root.mkdir(parents=True)

    image_paths = sorted(source_root.rglob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No PNG images found in {source_root}")

    print(f"\n{'=' * 60}")
    print(f"Processing {split_name.upper()}")
    print(f"Source images: {len(image_paths)}")
    print(f"{'=' * 60}")

    # Exactly equal representation of the three distortion types.
    distortion_types = (
        ["elastic"] * (len(image_paths) // 3)
        + ["local"] * (len(image_paths) // 3)
        + ["bending"] * (len(image_paths) // 3)
    )

    remainder = len(image_paths) - len(distortion_types)

    # Distribute any remainder.
    for i in range(remainder):
        distortion_types.append(
            ["elastic", "local", "bending"][i]
        )

    master_rng = np.random.default_rng(
        MASTER_SEED + seed_offset
    )

    master_rng.shuffle(distortion_types)

    metadata = []

    for idx, (source_path, distortion_type) in enumerate(
        zip(image_paths, distortion_types)
    ):

        img = load_image(source_path)

        # Unique deterministic seed for every image.
        image_seed = (
            MASTER_SEED
            + seed_offset * 100000
            + idx
        )

        rng = np.random.default_rng(image_seed)

        # Relative path inside train/val/test.
        relative_path = source_path.relative_to(source_root)

        # ----------------------------------------------------
        # Save clean copy
        # ----------------------------------------------------

        clean_output = clean_root / relative_path
        clean_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_path,
            clean_output,
        )

        # ----------------------------------------------------
        # Generate distortion
        # ----------------------------------------------------

        distorted_img, params = generate_distortion(
            img,
            distortion_type,
            rng,
        )

        # Change filename:
        # FM001_C1.png -> FM001_C1_distorted.png
        distorted_filename = (
            source_path.stem
            + "_distorted.png"
        )

        distorted_relative = (
            relative_path.parent
            / distorted_filename
        )

        distorted_output = (
            distorted_root
            / distorted_relative
        )

        distorted_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(distorted_output),
            distorted_img,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        parts = relative_path.parts

        family_id = parts[0] if len(parts) >= 1 else ""
        member = parts[1] if len(parts) >= 2 else ""

        metadata.append({
            "split": split_name,
            "original_path": str(source_path),
            "clean_output_path": str(clean_output),
            "distorted_output_path": str(distorted_output),
            "family_id": family_id,
            "member": member,
            "distortion_type": distortion_type,
            "seed": image_seed,
            **params,
        })

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    csv_path = output_root / "distortion_metadata.csv"

    fieldnames = sorted({
        key
        for row in metadata
        for key in row.keys()
    })

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata)

    json_path = output_root / "distortion_metadata.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    clean_count = len(
        list(clean_root.rglob("*.png"))
    )

    distorted_count = len(
        list(distorted_root.rglob("*.png"))
    )

    type_counts = {
        "elastic": sum(
            r["distortion_type"] == "elastic"
            for r in metadata
        ),
        "local": sum(
            r["distortion_type"] == "local"
            for r in metadata
        ),
        "bending": sum(
            r["distortion_type"] == "bending"
            for r in metadata
        ),
    }

    print(f"\n{split_name.upper()} COMPLETE")
    print(f"Clean:       {clean_count}")
    print(f"Distorted:   {distorted_count}")
    print(f"Elastic:     {type_counts['elastic']}")
    print(f"Local:       {type_counts['local']}")
    print(f"Bending:     {type_counts['bending']}")

    expected = len(image_paths)

    assert clean_count == expected
    assert distorted_count == expected

    for distortion_type in (
        "elastic",
        "local",
        "bending",
    ):
        assert type_counts[distortion_type] > 0

    # Check all output images are 224x224 grayscale.
    for output_path in distorted_root.rglob("*.png"):

        check = cv2.imread(
            str(output_path),
            cv2.IMREAD_UNCHANGED,
        )

        if check is None:
            raise RuntimeError(
                f"Unreadable output: {output_path}"
            )

        if check.shape != (224, 224):
            raise RuntimeError(
                f"Wrong shape {check.shape}: {output_path}"
            )

        if len(check.shape) != 2:
            raise RuntimeError(
                f"Not grayscale: {output_path}"
            )

    print("Verification: PASSED")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    for split_name, seed_offset in SPLITS.items():
        process_split(
            split_name,
            seed_offset,
        )

    print("\n" + "=" * 60)
    print("VALIDATION + TEST DISTORTION GENERATION COMPLETE")
    print("=" * 60)