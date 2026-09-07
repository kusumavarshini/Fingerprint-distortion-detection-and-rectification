import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from rectification_model import FingerprintRectificationNet


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_ROOT = "data/rectification_test"

CHECKPOINT_PATH = (
    "experiments/rectification_baseline/best_model.pth"
)

OUTPUT_DIR = (
    "experiments/rectification_baseline/visualizations"
)

NUM_PER_TYPE = 2


# ============================================================
# APPLY PREDICTED RECTIFICATION FIELD
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
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    return model


# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

def save_comparison(
    clean,
    distorted,
    rectified,
    distortion_type,
    filename
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        clean,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title("Clean")

    axes[1].imshow(
        distorted,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        f"Distorted\n{distortion_type}"
    )

    axes[2].imshow(
        rectified,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].set_title("Rectified")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Fingerprint Rectification Comparison",
        fontsize=14
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINGERPRINT RECTIFICATION VISUALIZATION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    model = load_model(device)

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

    selected = {
        "elastic": [],
        "local": [],
        "bending": []
    }

    # --------------------------------------------------------
    # Select examples
    # --------------------------------------------------------

    for path in distorted_files:

        name = os.path.basename(path)

        if "__elastic" in name:
            distortion_type = "elastic"

        elif "__local" in name:
            distortion_type = "local"

        elif "__bending" in name:
            distortion_type = "bending"

        else:
            continue

        if len(selected[distortion_type]) < NUM_PER_TYPE:
            selected[distortion_type].append(path)

    print("\nSelected examples:")

    for distortion_type, paths in selected.items():

        print(
            f"  {distortion_type}: "
            f"{len(paths)}"
        )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    counter = 1

    with torch.no_grad():

        for distortion_type in [
            "elastic",
            "local",
            "bending"
        ]:

            for distorted_path in selected[
                distortion_type
            ]:

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

                clean = cv2.imread(
                    clean_path,
                    cv2.IMREAD_GRAYSCALE
                )

                distorted = cv2.imread(
                    distorted_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if clean is None or distorted is None:

                    print(
                        f"WARNING: Could not read "
                        f"{filename}"
                    )

                    continue

                # --------------------------------------------
                # Model input
                # --------------------------------------------

                input_image = (
                    distorted.astype(np.float32)
                    / 255.0
                )

                input_tensor = torch.from_numpy(
                    input_image
                ).unsqueeze(0).unsqueeze(0).to(device)

                # --------------------------------------------
                # Predict field
                # --------------------------------------------

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

                # --------------------------------------------
                # Rectify
                # --------------------------------------------

                rectified = apply_distortion(
                    distorted,
                    predicted_dx,
                    predicted_dy,
                    method="nearest"
                )

                # --------------------------------------------
                # Save
                # --------------------------------------------

                output_filename = (
                    f"{counter:02d}_"
                    f"{distortion_type}.png"
                )

                save_comparison(
                    clean,
                    distorted,
                    rectified,
                    distortion_type,
                    output_filename
                )

                counter += 1

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()