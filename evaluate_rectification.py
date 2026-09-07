import os
import glob
import cv2
import numpy as np
import torch
from scipy.interpolate import griddata

from rectification_model import FingerprintRectificationNet


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_ROOT = "data/rectification_test"

CHECKPOINT_PATH = (
    "experiments/rectification_baseline/best_model.pth"
)


# ============================================================
# DDRNET-STYLE FIELD APPLICATION
# ============================================================

def apply_distortion(img, dx, dy, method="nearest"):

    img_shape = img.shape

    x, y = np.meshgrid(
        np.arange(0, img_shape[0]),
        np.arange(0, img_shape[1])
    )

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    rectified = griddata(
        np.hstack((
            x + dx.reshape((-1, 1)),
            y + dy.reshape((-1, 1))
        )),
        img.reshape((-1, 1)),
        np.hstack((x, y)),
        method=method,
        fill_value=255
    ).reshape(img_shape)

    return rectified.clip(0, 255)


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
# LOAD MODEL
# ============================================================

def load_model(device):

    model = FingerprintRectificationNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nLoaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINGERPRINT RECTIFICATION MODEL EVALUATION")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Locate test images
    # --------------------------------------------------------

    distorted_dir = os.path.join(
        TEST_DATA_ROOT,
        "distorted"
    )

    clean_dir = os.path.join(
        TEST_DATA_ROOT,
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

    print(
        f"\nTest samples found: "
        f"{len(distorted_files)}"
    )

    if len(distorted_files) != 675:

        raise RuntimeError(
            f"Expected 675 test samples, "
            f"found {len(distorted_files)}."
        )

    # --------------------------------------------------------
    # Result storage
    # --------------------------------------------------------

    overall_results = []

    distortion_results = {
        "elastic": [],
        "local": [],
        "bending": []
    }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating all test samples...")

    with torch.no_grad():

        for index, distorted_path in enumerate(
            distorted_files
        ):

            filename = os.path.basename(
                distorted_path
            )

            base_name = os.path.splitext(
                filename
            )[0]

            # Identify distortion type
            if "__elastic" in base_name:
                distortion_type = "elastic"

            elif "__local" in base_name:
                distortion_type = "local"

            elif "__bending" in base_name:
                distortion_type = "bending"

            else:
                print(
                    f"WARNING: Unknown distortion: "
                    f"{filename}"
                )
                continue

            # ------------------------------------------------
            # Clean image path
            # ------------------------------------------------

            clean_filename = (
                base_name.rsplit("__", 1)[0]
                + ".png"
            )

            clean_path = os.path.join(
                clean_dir,
                clean_filename
            )

            # ------------------------------------------------
            # Read images
            # ------------------------------------------------

            clean = cv2.imread(
                clean_path,
                cv2.IMREAD_GRAYSCALE
            )

            distorted = cv2.imread(
                distorted_path,
                cv2.IMREAD_GRAYSCALE
            )

            if clean is None:
                print(
                    f"WARNING: Missing clean image: "
                    f"{clean_path}"
                )
                continue

            if distorted is None:
                print(
                    f"WARNING: Cannot read: "
                    f"{distorted_path}"
                )
                continue

            # ------------------------------------------------
            # Prepare model input
            # ------------------------------------------------

            input_image = (
                distorted.astype(np.float32)
                / 255.0
            )

            input_tensor = torch.from_numpy(
                input_image
            ).unsqueeze(0).unsqueeze(0).to(device)

            # ------------------------------------------------
            # Predict displacement field
            # ------------------------------------------------

            predicted_field = model(
                input_tensor
            )

            predicted_field = (
                predicted_field
                .squeeze(0)
                .cpu()
                .numpy()
            )

            predicted_dx = predicted_field[0]
            predicted_dy = predicted_field[1]

            # ------------------------------------------------
            # Apply predicted field
            # ------------------------------------------------

            rectified = apply_distortion(
                distorted,
                predicted_dx,
                predicted_dy,
                method="nearest"
            )

            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            distorted_mae, distorted_rmse = (
                calculate_metrics(
                    clean,
                    distorted
                )
            )

            rectified_mae, rectified_rmse = (
                calculate_metrics(
                    clean,
                    rectified
                )
            )

            rmse_improvement = (
                (
                    distorted_rmse
                    - rectified_rmse
                )
                / distorted_rmse
                * 100
            )

            mae_improvement = (
                (
                    distorted_mae
                    - rectified_mae
                )
                / distorted_mae
                * 100
            )

            result = {
                "distorted_mae": distorted_mae,
                "distorted_rmse": distorted_rmse,
                "rectified_mae": rectified_mae,
                "rectified_rmse": rectified_rmse,
                "rmse_improvement": rmse_improvement,
                "mae_improvement": mae_improvement
            }

            overall_results.append(result)

            distortion_results[
                distortion_type
            ].append(result)

            # Progress
            if (index + 1) % 50 == 0:

                print(
                    f"  Processed "
                    f"{index + 1}/675"
                )

    # ========================================================
    # REPORT FUNCTION
    # ========================================================

    def report_results(title, results):

        if len(results) == 0:
            return

        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)

        avg_distorted_mae = np.mean([
            r["distorted_mae"]
            for r in results
        ])

        avg_rectified_mae = np.mean([
            r["rectified_mae"]
            for r in results
        ])

        avg_distorted_rmse = np.mean([
            r["distorted_rmse"]
            for r in results
        ])

        avg_rectified_rmse = np.mean([
            r["rectified_rmse"]
            for r in results
        ])

        avg_mae_improvement = np.mean([
            r["mae_improvement"]
            for r in results
        ])

        avg_rmse_improvement = np.mean([
            r["rmse_improvement"]
            for r in results
        ])

        improved_count = sum(
            r["rectified_rmse"]
            < r["distorted_rmse"]
            for r in results
        )

        print(
            f"Samples                 : "
            f"{len(results)}"
        )

        print(
            f"Distorted MAE           : "
            f"{avg_distorted_mae:.4f}"
        )

        print(
            f"Rectified MAE           : "
            f"{avg_rectified_mae:.4f}"
        )

        print(
            f"MAE improvement         : "
            f"{avg_mae_improvement:.2f}%"
        )

        print(
            f"Distorted RMSE          : "
            f"{avg_distorted_rmse:.4f}"
        )

        print(
            f"Rectified RMSE          : "
            f"{avg_rectified_rmse:.4f}"
        )

        print(
            f"RMSE improvement       : "
            f"{avg_rmse_improvement:.2f}%"
        )

        print(
            f"Samples improved        : "
            f"{improved_count}/{len(results)}"
        )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    report_results(
        "OVERALL TEST SET",
        overall_results
    )

    report_results(
        "ELASTIC DISTORTION",
        distortion_results["elastic"]
    )

    report_results(
        "LOCAL DISTORTION",
        distortion_results["local"]
    )

    report_results(
        "BENDING DISTORTION",
        distortion_results["bending"]
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if len(overall_results) > 0:

        avg_before = np.mean([
            r["distorted_rmse"]
            for r in overall_results
        ])

        avg_after = np.mean([
            r["rectified_rmse"]
            for r in overall_results
        ])

        print("\n" + "=" * 70)

        if avg_after < avg_before:

            print(
                "RESULT: RECTIFICATION IMPROVES "
                "THE TEST SET"
            )

        else:

            print(
                "RESULT: RECTIFICATION DOES NOT "
                "IMPROVE THE TEST SET"
            )

        print("=" * 70)


if __name__ == "__main__":
    main()