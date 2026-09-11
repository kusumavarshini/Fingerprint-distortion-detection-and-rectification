import os
import glob
import json
import numpy as np
import cv2
import torch
from rectification_model import FingerprintRectificationNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"data\rectification_test"
CHECKPOINT = r"experiments\rectification_baseline\best_model.pth"
OUTPUT_DIR = r"experiments\rectification_field_analysis"

IMAGE_SIZE = 224
NUM_SAMPLES_PER_TYPE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# UTILITIES
# ============================================================

def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def displacement_stats(gt_dx, gt_dy, pred_dx, pred_dy):
    gt_mag = np.sqrt(gt_dx ** 2 + gt_dy ** 2)
    pred_mag = np.sqrt(pred_dx ** 2 + pred_dy ** 2)

    dx_error = pred_dx - gt_dx
    dy_error = pred_dy - gt_dy
    mag_error = pred_mag - gt_mag

    return {
        "dx_mae": float(np.mean(np.abs(dx_error))),
        "dx_rmse": float(np.sqrt(np.mean(dx_error ** 2))),

        "dy_mae": float(np.mean(np.abs(dy_error))),
        "dy_rmse": float(np.sqrt(np.mean(dy_error ** 2))),

        "magnitude_gt_mean": float(np.mean(gt_mag)),
        "magnitude_pred_mean": float(np.mean(pred_mag)),

        "magnitude_mae": float(np.mean(np.abs(mag_error))),
        "magnitude_rmse": float(np.sqrt(np.mean(mag_error ** 2))),

        "dx_correlation": safe_corr(gt_dx, pred_dx),
        "dy_correlation": safe_corr(gt_dy, pred_dy),
        "magnitude_correlation": safe_corr(gt_mag, pred_mag),

        "vector_rmse": float(
            np.sqrt(np.mean(dx_error ** 2 + dy_error ** 2))
        ),
    }


def save_field_visualization(
    image,
    gt_dx,
    gt_dy,
    pred_dx,
    pred_dy,
    output_path
):
    gt_mag = np.sqrt(gt_dx ** 2 + gt_dy ** 2)
    pred_mag = np.sqrt(pred_dx ** 2 + pred_dy ** 2)

    error_mag = np.sqrt(
        (pred_dx - gt_dx) ** 2 +
        (pred_dy - gt_dy) ** 2
    )

    def normalize_field(field):
        field = np.abs(field)
        max_value = np.percentile(field, 99)

        if max_value < 1e-8:
            max_value = 1.0

        field = np.clip(field / max_value, 0, 1)
        return (field * 255).astype(np.uint8)

    image_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)

    gt_mag_u8 = normalize_field(gt_mag)
    pred_mag_u8 = normalize_field(pred_mag)
    error_u8 = normalize_field(error_mag)

    gt_dx_u8 = normalize_field(gt_dx)
    pred_dx_u8 = normalize_field(pred_dx)

    gt_dy_u8 = normalize_field(gt_dy)
    pred_dy_u8 = normalize_field(pred_dy)

    # Add labels
    panels = [
        ("Input", image_u8),
        ("GT Magnitude", gt_mag_u8),
        ("Pred Magnitude", pred_mag_u8),
        ("Magnitude Error", error_u8),
        ("GT DX", gt_dx_u8),
        ("Pred DX", pred_dx_u8),
        ("GT DY", gt_dy_u8),
        ("Pred DY", pred_dy_u8),
    ]

    panel_images = []

    for label, img in panels:
        panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            panel,
            label,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        panel_images.append(panel)

    row1 = cv2.hconcat(panel_images[:4])
    row2 = cv2.hconcat(panel_images[4:])
    final = cv2.vconcat([row1, row2])

    cv2.imwrite(output_path, final)


# ============================================================
# LOAD MODEL
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

print("=" * 90)
print("PREDICTED vs GROUND-TRUTH DISPLACEMENT FIELD ANALYSIS")
print("=" * 90)

print(f"\nDevice: {DEVICE}")

model = FingerprintRectificationNet()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print(f"Checkpoint loaded: {CHECKPOINT}")


# ============================================================
# FIND TEST SAMPLES
# ============================================================

distorted_dir = os.path.join(DATASET_DIR, "distorted")

all_images = sorted(
    glob.glob(os.path.join(distorted_dir, "*.png"))
)

print(f"Total distorted test images: {len(all_images)}")


# ============================================================
# GROUP BY DISTORTION TYPE
# ============================================================

types = {
    "elastic": [],
    "local": [],
    "bending": []
}

for path in all_images:
    name = os.path.basename(path).lower()

    for distortion_type in types:
        if f"__{distortion_type}.png" in name:
            types[distortion_type].append(path)
            break


# ============================================================
# RUN EVALUATION
# ============================================================

all_results = []
summary = {}

for distortion_type, paths in types.items():

    print("\n" + "=" * 90)
    print(f"ANALYZING: {distortion_type.upper()}")
    print("=" * 90)

    paths = paths[:NUM_SAMPLES_PER_TYPE]

    type_results = []

    for index, image_path in enumerate(paths):

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        image = image.astype(np.float32) / 255.0

        field_path = os.path.join(
            DATASET_DIR,
            "fields",
            os.path.splitext(os.path.basename(image_path))[0] + ".npz"
        )

        if not os.path.exists(field_path):
            print(f"WARNING: Missing field: {field_path}")
            continue

        data = np.load(field_path)

        gt_dx = data["dx"].astype(np.float32)
        gt_dy = data["dy"].astype(np.float32)

        input_tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = model(input_tensor)

        prediction = prediction.squeeze(0).cpu().numpy()

        pred_dx = prediction[0]
        pred_dy = prediction[1]

        stats = displacement_stats(
            gt_dx,
            gt_dy,
            pred_dx,
            pred_dy
        )

        stats["distortion_type"] = distortion_type
        stats["image"] = os.path.basename(image_path)

        type_results.append(stats)
        all_results.append(stats)

        if index < 3:
            visualization_path = os.path.join(
                OUTPUT_DIR,
                "visualizations",
                f"{distortion_type}_{index + 1:02d}.png"
            )

            save_field_visualization(
                image,
                gt_dx,
                gt_dy,
                pred_dx,
                pred_dy,
                visualization_path
            )

        if (index + 1) % 5 == 0:
            print(f"  Processed: {index + 1}/{len(paths)}")

    if type_results:

        keys = [
            "dx_mae",
            "dx_rmse",
            "dy_mae",
            "dy_rmse",
            "magnitude_gt_mean",
            "magnitude_pred_mean",
            "magnitude_mae",
            "magnitude_rmse",
            "dx_correlation",
            "dy_correlation",
            "magnitude_correlation",
            "vector_rmse"
        ]

        type_summary = {}

        for key in keys:
            type_summary[key] = float(
                np.mean([r[key] for r in type_results])
            )

        summary[distortion_type] = type_summary


# ============================================================
# OVERALL SUMMARY
# ============================================================

if all_results:

    keys = [
        "dx_mae",
        "dx_rmse",
        "dy_mae",
        "dy_rmse",
        "magnitude_gt_mean",
        "magnitude_pred_mean",
        "magnitude_mae",
        "magnitude_rmse",
        "dx_correlation",
        "dy_correlation",
        "magnitude_correlation",
        "vector_rmse"
    ]

    overall = {}

    for key in keys:
        overall[key] = float(
            np.mean([r[key] for r in all_results])
        )

    summary["overall"] = overall


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FIELD PREDICTION SUMMARY")
print("=" * 90)

for distortion_type, result in summary.items():

    print(f"\n{distortion_type.upper()}")

    print(f"  DX MAE:                 {result['dx_mae']:.4f}")
    print(f"  DX RMSE:                {result['dx_rmse']:.4f}")

    print(f"  DY MAE:                 {result['dy_mae']:.4f}")
    print(f"  DY RMSE:                {result['dy_rmse']:.4f}")

    print(
        f"  GT magnitude mean:      "
        f"{result['magnitude_gt_mean']:.4f}"
    )

    print(
        f"  Pred magnitude mean:    "
        f"{result['magnitude_pred_mean']:.4f}"
    )

    print(
        f"  Magnitude MAE:          "
        f"{result['magnitude_mae']:.4f}"
    )

    print(
        f"  Magnitude RMSE:         "
        f"{result['magnitude_rmse']:.4f}"
    )

    print(
        f"  DX correlation:         "
        f"{result['dx_correlation']:.4f}"
    )

    print(
        f"  DY correlation:         "
        f"{result['dy_correlation']:.4f}"
    )

    print(
        f"  Magnitude correlation:  "
        f"{result['magnitude_correlation']:.4f}"
    )

    print(
        f"  Vector RMSE:            "
        f"{result['vector_rmse']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "field_analysis_summary.json"
    ),
    "w"
) as f:
    json.dump(summary, f, indent=2)


with open(
    os.path.join(
        OUTPUT_DIR,
        "field_analysis_samples.json"
    ),
    "w"
) as f:
    json.dump(all_results, f, indent=2)


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nVisualizations saved to:\n"
    f"{os.path.join(OUTPUT_DIR, 'visualizations')}"
)