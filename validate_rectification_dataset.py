import os
import glob
import cv2
import numpy as np
from scipy.interpolate import griddata


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "data/rectification_test"

NUM_SAMPLES = 10


# ============================================================
# DDRNET-STYLE DISTORTION / RECTIFICATION
# ============================================================

def apply_distortion(img, dx, dy, method="nearest"):

    img_shape = img.shape

    x, y = np.meshgrid(
        np.arange(0, img_shape[0]),
        np.arange(0, img_shape[1])
    )

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    distorted = griddata(
        np.hstack((
            x + dx.reshape((-1, 1)),
            y + dy.reshape((-1, 1))
        )),
        img.reshape((-1, 1)),
        np.hstack((x, y)),
        method=method,
        fill_value=255
    ).reshape(img_shape)

    return distorted.clip(0, 255)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(clean, result):

    clean = clean.astype(np.float32)
    result = result.astype(np.float32)

    diff = clean - result

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    return mae, rmse


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECTIFICATION DATASET VALIDATION")
    print("=" * 70)

    distorted_dir = os.path.join(
        DATASET_DIR,
        "distorted"
    )

    fields_dir = os.path.join(
        DATASET_DIR,
        "fields"
    )

    clean_dir = os.path.join(
        DATASET_DIR,
        "clean"
    )

    distorted_files = sorted(
        glob.glob(
            os.path.join(
                distorted_dir,
                "*.png"
            )
        )
    )

    print(f"\nFound distorted samples: {len(distorted_files)}")

    if len(distorted_files) == 0:
        raise RuntimeError(
            "No distorted images found."
        )

    samples = distorted_files[:NUM_SAMPLES]

    print(f"Testing samples: {len(samples)}")

    all_results = []

    for i, distorted_path in enumerate(samples):

        filename = os.path.basename(
            distorted_path
        )

        base_name = os.path.splitext(
            filename
        )[0]

        clean_filename = (
            base_name.rsplit("__", 1)[0]
            + ".png"
        )

        clean_path = os.path.join(
            clean_dir,
            clean_filename
        )

        field_path = os.path.join(
            fields_dir,
            base_name + ".npz"
        )

        clean = cv2.imread(
            clean_path,
            cv2.IMREAD_GRAYSCALE
        )

        distorted = cv2.imread(
            distorted_path,
            cv2.IMREAD_GRAYSCALE
        )

        if clean is None:
            print(f"WARNING: Missing clean image:")
            print(f"  {clean_path}")
            continue

        if distorted is None:
            print(f"WARNING: Cannot read:")
            print(f"  {distorted_path}")
            continue

        fields = np.load(field_path)

        dx = fields["dx"]
        dy = fields["dy"]

        # Saved fields are DDRNet rectification fields.
        rectified = apply_distortion(
            distorted,
            dx,
            dy,
            method="nearest"
        )

        distorted_mae, distorted_rmse = calculate_metrics(
            clean,
            distorted
        )

        rectified_mae, rectified_rmse = calculate_metrics(
            clean,
            rectified
        )

        improvement = (
            (distorted_rmse - rectified_rmse)
            / distorted_rmse
            * 100
        )

        print("\n" + "-" * 70)
        print(f"Sample {i + 1}: {filename}")
        print(f"Distortion RMSE : {distorted_rmse:.4f}")
        print(f"Rectified RMSE  : {rectified_rmse:.4f}")
        print(f"RMSE improvement: {improvement:.2f}%")

        all_results.append({
            "distorted_rmse": distorted_rmse,
            "rectified_rmse": rectified_rmse,
            "improvement": improvement
        })

    if len(all_results) == 0:
        raise RuntimeError(
            "No samples were successfully validated."
        )

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    avg_distorted = np.mean([
        r["distorted_rmse"]
        for r in all_results
    ])

    avg_rectified = np.mean([
        r["rectified_rmse"]
        for r in all_results
    ])

    avg_improvement = np.mean([
        r["improvement"]
        for r in all_results
    ])

    print(
        f"\nAverage distorted RMSE : "
        f"{avg_distorted:.4f}"
    )

    print(
        f"Average rectified RMSE  : "
        f"{avg_rectified:.4f}"
    )

    print(
        f"Average RMSE improvement: "
        f"{avg_improvement:.2f}%"
    )

    if avg_rectified < avg_distorted:
        print("\nRESULT: PASS")
        print(
            "The saved DDRNet rectification fields "
            "improve the distorted fingerprints."
        )
    else:
        print("\nRESULT: FAIL")
        print(
            "Rectification did not improve the "
            "images. Field convention needs investigation."
        )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()