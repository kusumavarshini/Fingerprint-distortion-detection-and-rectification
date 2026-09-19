import os
import glob
import time
import warnings

import cv2
import numpy as np
import torch

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = r"experiments\ddrnet_baseline\best_model.pth"

DISTORTED_DIR = r"data\rectification_test\distorted"
CLEAN_DIR = r"data\rectification_test\clean"

OUTPUT_DIR = r"experiments\rectification_accuracy\full_test"

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "rectification_accuracy_results.csv"
)

IMAGE_SIZE = 224

# Full test set:
# 225 elastic + 225 local + 225 bending = 675
SAMPLES_PER_TYPE = 225

DEVICE = torch.device("cpu")


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

torch.set_num_threads(16)
torch.set_num_interop_threads(2)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DDRNET
# ============================================================

print("=" * 70)
print("LOADING DDRNET")
print("=" * 70)

model = DDRNet_DIR()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded.")
print()


# ============================================================
# COLLECT FILES
# ============================================================

all_files = sorted(
    glob.glob(
        os.path.join(
            DISTORTED_DIR,
            "*.png"
        )
    )
)

selected = []

for distortion_type in [
    "elastic",
    "local",
    "bending"
]:

    files = [
        f
        for f in all_files
        if f"__{distortion_type}.png"
        in os.path.basename(f)
    ]

    files = sorted(files)

    if len(files) != SAMPLES_PER_TYPE:
        raise RuntimeError(
            f"Expected {SAMPLES_PER_TYPE} "
            f"{distortion_type} images, "
            f"but found {len(files)}"
        )

    selected.extend(
        [
            (f, distortion_type)
            for f in files
        ]
    )


print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"Elastic : "
    f"{sum(d == 'elastic' for _, d in selected)}"
)

print(
    f"Local   : "
    f"{sum(d == 'local' for _, d in selected)}"
)

print(
    f"Bending : "
    f"{sum(d == 'bending' for _, d in selected)}"
)

print(
    f"Total   : "
    f"{len(selected)}"
)

print()


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_metrics(
    clean,
    rectified
):

    clean_f = clean.astype(
        np.float32
    )

    rectified_f = rectified.astype(
        np.float32
    )

    difference = (
        clean_f - rectified_f
    )

    mae = np.mean(
        np.abs(difference)
    )

    rmse = np.sqrt(
        np.mean(
            difference ** 2
        )
    )

    return (
        float(mae),
        float(rmse)
    )


# ============================================================
# CSV HEADER
# ============================================================

with open(
    RESULTS_CSV,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "filename,distortion,mae,rmse\n"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

print("=" * 70)
print("RECTIFICATION ACCURACY")
print("=" * 70)

print(
    "Comparison: CLEAN vs RECTIFIED"
)

print(
    "Distorted image is used only as DDRNet input."
)

print("=" * 70)
print()


results = []

start_total = time.time()


with torch.no_grad():

    for idx, (
        distorted_path,
        distortion_type
    ) in enumerate(
        selected,
        start=1
    ):

        filename = os.path.basename(
            distorted_path
        )

        # ----------------------------------------------------
        # Determine corresponding clean filename
        # ----------------------------------------------------

        clean_filename = filename.replace(
            f"__{distortion_type}.png",
            ".png"
        )

        clean_path = os.path.join(
            CLEAN_DIR,
            clean_filename
        )

        if not os.path.exists(
            clean_path
        ):

            raise FileNotFoundError(
                f"Clean image not found:\n"
                f"{clean_path}"
            )

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        distorted = cv2.imread(
            distorted_path,
            cv2.IMREAD_GRAYSCALE
        )

        clean = cv2.imread(
            clean_path,
            cv2.IMREAD_GRAYSCALE
        )

        if distorted is None:
            raise RuntimeError(
                f"Could not read:\n"
                f"{distorted_path}"
            )

        if clean is None:
            raise RuntimeError(
                f"Could not read:\n"
                f"{clean_path}"
            )

        if distorted.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):

            raise RuntimeError(
                f"Unexpected distorted "
                f"shape {distorted.shape} "
                f"for {filename}"
            )

        if clean.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):

            raise RuntimeError(
                f"Unexpected clean "
                f"shape {clean.shape} "
                f"for {clean_filename}"
            )

        # ----------------------------------------------------
        # Exact DDRNet segmentation
        # ----------------------------------------------------

        mask = segmentation_coherence(
            distorted,
            win_size=16,
            stride=8
        )

        # ----------------------------------------------------
        # Convert mask to 14x14
        #
        # This matches the DDRNet input requirement.
        # ----------------------------------------------------

        mask16 = cv2.resize(
            mask.astype(
                np.uint8
            ),
            (14, 14),
            interpolation=cv2.INTER_NEAREST
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Exact DDRNet image preprocessing
        #
        # transform_img(img):
        #     (255 - img) / 255
        # ----------------------------------------------------

        transformed = (
            255.0 - distorted
        ) / 255.0

        x = torch.from_numpy(
            np.float32(
                transformed
            )[None, None, :, :]
        ).to(DEVICE)

        mask_tensor = torch.from_numpy(
            mask16[None, None, :, :]
        ).to(DEVICE)

        # ----------------------------------------------------
        # DDRNet inference
        # ----------------------------------------------------

        field, direction = model(
            x,
            mask_tensor
        )

        # field:
        # (1, 2, 14, 14)

        field = field[
            0
        ].cpu().numpy()

        dx = field[0]
        dy = field[1]

        # ----------------------------------------------------
        # Upsample predicted field
        #
        # 14x14 -> 224x224
        # ----------------------------------------------------

        dx = cv2.resize(
            dx,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        dy = cv2.resize(
            dy,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # CREATE SAMPLING MAP
        #
        # Established convention:
        #
        # map_x = grid_x - dx
        # map_y = grid_y - dy
        # ----------------------------------------------------

        h, w = distorted.shape

        grid_x, grid_y = np.meshgrid(
            np.arange(
                w,
                dtype=np.float32
            ),
            np.arange(
                h,
                dtype=np.float32
            )
        )

        map_x = (
            grid_x - dx
        ).astype(
            np.float32
        )

        map_y = (
            grid_y - dy
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # RECTIFY
        #
        # IMPORTANT:
        # Use the original uint8 distorted image directly.
        #
        # Do NOT multiply by 255 here.
        # ----------------------------------------------------

        rectified = cv2.remap(
            distorted,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        # ----------------------------------------------------
        # ACCURACY
        #
        # ONLY:
        #
        # CLEAN <-> RECTIFIED
        #
        # Distorted is NOT used here.
        # ----------------------------------------------------

        mae, rmse = calculate_metrics(
            clean,
            rectified
        )

        # ----------------------------------------------------
        # SAVE RECTIFIED IMAGE
        # ----------------------------------------------------

        rectified_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            rectified_path,
            rectified
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "filename": filename,
                "distortion": distortion_type,
                "mae": mae,
                "rmse": rmse
            }
        )

        # ----------------------------------------------------
        # Append CSV
        # ----------------------------------------------------

        with open(
            RESULTS_CSV,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{filename},"
                f"{distortion_type},"
                f"{mae:.6f},"
                f"{rmse:.6f}\n"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_total
        )

        avg_time = (
            elapsed / idx
        )

        remaining = (
            avg_time
            * (len(selected) - idx)
        )

        print(
            f"[{idx:3d}/{len(selected)}] "
            f"{distortion_type:7s} | "
            f"MAE={mae:8.4f} | "
            f"RMSE={rmse:8.4f} | "
            f"ETA={remaining:6.1f}s"
        )


# ============================================================
# SUMMARY
# ============================================================

total_time = (
    time.time()
    - start_total
)


print()
print("=" * 70)
print("RECTIFICATION ACCURACY SUMMARY")
print("=" * 70)

for distortion_type in [
    "elastic",
    "local",
    "bending"
]:

    subset = [
        r
        for r in results
        if r["distortion"]
        == distortion_type
    ]

    mean_mae = np.mean(
        [
            r["mae"]
            for r in subset
        ]
    )

    mean_rmse = np.mean(
        [
            r["rmse"]
            for r in subset
        ]
    )

    print(
        f"{distortion_type:7s} | "
        f"MAE={mean_mae:.4f} | "
        f"RMSE={mean_rmse:.4f}"
    )


overall_mae = np.mean(
    [
        r["mae"]
        for r in results
    ]
)

overall_rmse = np.mean(
    [
        r["rmse"]
        for r in results
    ]
)

print("-" * 70)

print(
    f"OVERALL  | "
    f"MAE={overall_mae:.4f} | "
    f"RMSE={overall_rmse:.4f}"
)

print()
print(
    f"Processed : {len(results)} / "
    f"{len(selected)}"
)

print(
    f"Runtime   : {total_time:.2f} seconds"
)

print(
    f"Avg/image : "
    f"{total_time / len(results):.4f} seconds"
)

print()
print(
    "Rectified images:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Results CSV:"
)

print(
    RESULTS_CSV
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)