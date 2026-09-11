from pathlib import Path
import csv

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path("DDRNet").resolve()))

from models.DDRNet_DIR import DDRNet_DIR


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("data/ddrnet_val")
CHECKPOINT = Path("experiments/ddrnet_baseline/best_model.pth")

OUTPUT_DIR = Path("experiments/ddrnet_baseline/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("DDRNet RECTIFICATION EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")

model = DDRNet_DIR(dis_const=16)
model = model.to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")


# ============================================================
# Helpers
# ============================================================

def load_image(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read: {path}"
        )

    return image


def apply_rectification(image, dx, dy):
    """
    DDRNet displacement convention for our dataset.

    Rectification uses:
        map_x = x - dx
        map_y = y - dy
    """

    h, w = image.shape

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = grid_x - dx
    map_y = grid_y - dy

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    return rectified


def calculate_metrics(reference, result):
    reference = reference.astype(np.float32)
    result = result.astype(np.float32)

    error = result - reference

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return mae, rmse


# ============================================================
# Find samples
# ============================================================

image_files = sorted(
    (DATASET_DIR / "images").glob("*.png")
)

print(f"Validation samples: {len(image_files)}")

if len(image_files) == 0:
    raise RuntimeError("No validation images found.")


# ============================================================
# Evaluation
# ============================================================

results = []

type_stats = {
    "elastic": [],
    "local": [],
    "bending": []
}


with torch.no_grad():

    for index, image_path in enumerate(image_files):

        name = image_path.name

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        field_path = (
            DATASET_DIR /
            "fields" /
            image_path.with_suffix(".npz").name
        )

        mask_path = (
            DATASET_DIR /
            "masks" /
            name
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        distorted = load_image(image_path)

        # ----------------------------------------------------
        # Ground-truth displacement
        # ----------------------------------------------------

        field = np.load(field_path)

        gt_dx = field["dx_rect16"].astype(
            np.float32
        )

        gt_dy = field["dy_rect16"].astype(
            np.float32
        )

        # DDRNet predicts at 14x14.
        # Upsample to 224x224.

        gt_dx_full = cv2.resize(
            gt_dx,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        gt_dy_full = cv2.resize(
            gt_dy,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------------------------
        # Clean reference
        # ----------------------------------------------------

        # Naming convention:
        #
        # distorted:
        # FAMILY...__type.png
        #
        # clean:
        # corresponding family/member image
        #
        # Extract family/member from filename.

        parts = name.split("__")

        if len(parts) < 3:
            raise RuntimeError(
                f"Unexpected filename: {name}"
            )

        family = parts[0]
        member = parts[1]

        # Example:
        # FAMILY-100__CHILD__FM100_C1__bending.png
        # clean source:
        # data/rectification_val/clean/FAMILY-100/CHILD/FM100_C1.png

        clean_candidates = list(
            (DATASET_DIR / "clean" / family / member).glob(
                "*.png"
            )
        )

        # The clean filename is the third component
        # without distortion type.

        clean_stem = parts[2]

        clean_path = (
            Path("data/split/val") /
            family /
            member /
            f"{clean_stem}.png"
        )

        if not clean_path.exists():
            raise RuntimeError(
                f"Clean image not found: {clean_path}"
            )

        clean = load_image(clean_path)

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = load_image(mask_path)

        mask_tensor = torch.from_numpy(
            (mask > 0).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        # ----------------------------------------------------
        # DDRNet input
        # ----------------------------------------------------

        input_image = (
            255.0 - distorted.astype(np.float32)
        ) / 255.0

        input_tensor = torch.from_numpy(
            input_image
        ).unsqueeze(0).unsqueeze(0)

        input_tensor = input_tensor.to(DEVICE)
        mask_tensor = mask_tensor.to(DEVICE)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        displacement_pred, orientation_pred = model(
            input_tensor,
            mask_tensor
        )

        predicted_field = (
            displacement_pred[0]
            .detach()
            .cpu()
            .numpy()
        )

        pred_dx = predicted_field[0]
        pred_dy = predicted_field[1]

        # ----------------------------------------------------
        # Upsample predicted field
        # ----------------------------------------------------

        pred_dx_full = cv2.resize(
            pred_dx,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        pred_dy_full = cv2.resize(
            pred_dy,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------------------------
        # Rectify
        # ----------------------------------------------------

        rectified = apply_rectification(
            distorted,
            pred_dx_full,
            pred_dy_full
        )

        # ----------------------------------------------------
        # Image metrics
        # ----------------------------------------------------

        distorted_mae, distorted_rmse = calculate_metrics(
            clean,
            distorted
        )

        rectified_mae, rectified_rmse = calculate_metrics(
            clean,
            rectified
        )

        mae_improvement = (
            (distorted_mae - rectified_mae)
            / max(distorted_mae, 1e-8)
            * 100.0
        )

        rmse_improvement = (
            (distorted_rmse - rectified_rmse)
            / max(distorted_rmse, 1e-8)
            * 100.0
        )

        # ----------------------------------------------------
        # Distortion type
        # ----------------------------------------------------

        if "__elastic" in name:
            distortion_type = "elastic"
        elif "__local" in name:
            distortion_type = "local"
        elif "__bending" in name:
            distortion_type = "bending"
        else:
            distortion_type = "unknown"

        record = {
            "filename": name,
            "type": distortion_type,
            "distorted_mae": distorted_mae,
            "rectified_mae": rectified_mae,
            "mae_improvement_percent": mae_improvement,
            "distorted_rmse": distorted_rmse,
            "rectified_rmse": rectified_rmse,
            "rmse_improvement_percent": rmse_improvement
        }

        results.append(record)

        if distortion_type in type_stats:
            type_stats[distortion_type].append(
                record
            )

        # ----------------------------------------------------
        # Save selected visualizations
        # ----------------------------------------------------

        if len(results) <= 6:

            separator = np.full(
                (224, 4),
                255,
                dtype=np.uint8
            )

            comparison = np.hstack(
                [
                    clean,
                    separator,
                    distorted,
                    separator,
                    rectified
                ]
            )

            output_path = (
                VIS_DIR /
                f"{len(results):02d}_{distortion_type}.png"
            )

            cv2.imwrite(
                str(output_path),
                comparison
            )

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(image_files)}"
            )


# ============================================================
# Overall results
# ============================================================

def average(records, key):
    if not records:
        return 0.0

    return float(
        np.mean([
            r[key]
            for r in records
        ])
    )


print()
print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

overall = {
    "distorted_mae": average(
        results,
        "distorted_mae"
    ),
    "rectified_mae": average(
        results,
        "rectified_mae"
    ),
    "distorted_rmse": average(
        results,
        "distorted_rmse"
    ),
    "rectified_rmse": average(
        results,
        "rectified_rmse"
    )
}

overall["mae_improvement_percent"] = (
    (
        overall["distorted_mae"]
        - overall["rectified_mae"]
    )
    / overall["distorted_mae"]
    * 100.0
)

overall["rmse_improvement_percent"] = (
    (
        overall["distorted_rmse"]
        - overall["rectified_rmse"]
    )
    / overall["distorted_rmse"]
    * 100.0
)

print(
    f"MAE  : "
    f"{overall['distorted_mae']:.4f} -> "
    f"{overall['rectified_mae']:.4f} "
    f"({overall['mae_improvement_percent']:.2f}%)"
)

print(
    f"RMSE : "
    f"{overall['distorted_rmse']:.4f} -> "
    f"{overall['rectified_rmse']:.4f} "
    f"({overall['rmse_improvement_percent']:.2f}%)"
)


# ============================================================
# Per-type results
# ============================================================

print()
print("=" * 60)
print("RESULTS BY DISTORTION TYPE")
print("=" * 60)

for distortion_type in [
    "elastic",
    "local",
    "bending"
]:

    records = type_stats[distortion_type]

    d_mae = average(
        records,
        "distorted_mae"
    )

    r_mae = average(
        records,
        "rectified_mae"
    )

    d_rmse = average(
        records,
        "distorted_rmse"
    )

    r_rmse = average(
        records,
        "rectified_rmse"
    )

    mae_imp = (
        (d_mae - r_mae)
        / d_mae
        * 100.0
    )

    rmse_imp = (
        (d_rmse - r_rmse)
        / d_rmse
        * 100.0
    )

    print()
    print(distortion_type.upper())

    print(
        f"Samples: {len(records)}"
    )

    print(
        f"MAE  : "
        f"{d_mae:.4f} -> "
        f"{r_mae:.4f} "
        f"({mae_imp:.2f}%)"
    )

    print(
        f"RMSE : "
        f"{d_rmse:.4f} -> "
        f"{r_rmse:.4f} "
        f"({rmse_imp:.2f}%)"
    )


# ============================================================
# Save CSV
# ============================================================

csv_path = OUTPUT_DIR / "evaluation_results.csv"

fieldnames = [
    "filename",
    "type",
    "distorted_mae",
    "rectified_mae",
    "mae_improvement_percent",
    "distorted_rmse",
    "rectified_rmse",
    "rmse_improvement_percent"
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Results: {csv_path}"
)

print(
    f"Visualizations: {VIS_DIR}"
)