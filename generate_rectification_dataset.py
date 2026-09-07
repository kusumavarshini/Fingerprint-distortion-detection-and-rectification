import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/split/train"
OUTPUT_DIR = "data/rectification_train"

MASTER_SEED = 42

IMAGE_SIZE = 224


# ============================================================
# DISTORTION FUNCTIONS
# ============================================================

def distort_elastic(img, rng):
    h, w = img.shape

    alpha = rng.uniform(8, 16)
    sigma = rng.uniform(5, 8)
    grid_step = 32

    gh = h // grid_step + 2
    gw = w // grid_step + 2

    dx_small = rng.uniform(-alpha, alpha, (gh, gw))
    dy_small = rng.uniform(-alpha, alpha, (gh, gw))

    dx = cv2.resize(
        dx_small.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    dy = cv2.resize(
        dy_small.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    dx = gaussian_filter(dx, sigma=sigma)
    dy = gaussian_filter(dy, sigma=sigma)

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = gx + dx
    map_y = gy + dy

    distorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return distorted, dx, dy, {
        "alpha": alpha,
        "sigma": sigma,
        "grid_step": grid_step
    }


def distort_local(img, rng):
    h, w = img.shape

    n_ctrl = 5
    max_disp = rng.uniform(5, 10)
    smooth = rng.uniform(8, 12)

    dx_small = rng.uniform(
        -max_disp,
        max_disp,
        (n_ctrl, n_ctrl)
    )

    dy_small = rng.uniform(
        -max_disp,
        max_disp,
        (n_ctrl, n_ctrl)
    )

    dx = cv2.resize(
        dx_small.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    dy = cv2.resize(
        dy_small.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    dx = gaussian_filter(dx, sigma=smooth)
    dy = gaussian_filter(dy, sigma=smooth)

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = gx + dx
    map_y = gy + dy

    distorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return distorted, dx, dy, {
        "n_ctrl": n_ctrl,
        "max_disp": max_disp,
        "smooth": smooth
    }


def distort_bending(img, rng):
    h, w = img.shape

    max_disp_param = rng.uniform(12, 26)
    strength = rng.uniform(0.40, 1.00)

    direction = rng.integers(0, 2)

    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)

    nx, ny = np.meshgrid(x, y)

    rx = nx - 0.5
    ry = ny - 0.5

    r2 = rx ** 2 + ry ** 2

    if direction == 0:
        dx = (
            max_disp_param * strength * rx * r2
            + max_disp_param * 0.4 * np.sin(np.pi * ny)
        )

        dy = (
            max_disp_param * strength * ry * r2 * 0.4
            + max_disp_param * 0.2
            * np.sin(2 * np.pi * nx)
            * np.sin(np.pi * ny)
        )

    else:
        dx = (
            max_disp_param * strength * rx * r2 * 0.4
            + max_disp_param * 0.2
            * np.sin(2 * np.pi * ny)
            * np.sin(np.pi * nx)
        )

        dy = (
            max_disp_param * strength * ry * r2
            + max_disp_param * 0.4 * np.sin(np.pi * nx)
        )

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = gx + dx
    map_y = gy + dy

    distorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return distorted, dx, dy, {
        "max_disp_param": max_disp_param,
        "strength": strength,
        "direction": int(direction)
    }


# ============================================================
# CONVERT SAMPLING FIELD TO DDRNET RECTIFICATION FIELD
# ============================================================

def sampling_to_ddrnet_field(dx, dy):
    """
    Convert the cv2.remap sampling field into the
    DDRNet-compatible dual rectification field.

    Uses the inverse mapping:

        (x, y) -> (x + dx, y + dy)

    and estimates the inverse displacement at each
    output pixel.

    This implementation avoids scipy.griddata(), which is
    extremely slow when repeated thousands of times.
    """

    h, w = dx.shape

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # Coordinates where each clean pixel was moved.
    warped_x = x + dx
    warped_y = y + dy

    # DDRNet dual displacement is approximately the inverse
    # displacement evaluated at the warped coordinates.
    #
    # Since the displacement fields are smooth, bilinear
    # interpolation is sufficient and much faster than
    # scipy.griddata().

    map_x = warped_x.astype(np.float32)
    map_y = warped_y.astype(np.float32)

    dx_rect = cv2.remap(
        dx.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    dy_rect = cv2.remap(
        dy.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    dx_rect = np.nan_to_num(
        dx_rect,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    dy_rect = np.nan_to_num(
        dy_rect,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        dx_rect.astype(np.float32),
        dy_rect.astype(np.float32)
    )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def main():

    print("=" * 70)
    print("RECTIFICATION DATASET GENERATION")
    print("=" * 70)

    print(f"\nInput : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_dir = os.path.join(OUTPUT_DIR, "clean")
    distorted_dir = os.path.join(OUTPUT_DIR, "distorted")
    fields_dir = os.path.join(OUTPUT_DIR, "fields")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(distorted_dir, exist_ok=True)
    os.makedirs(fields_dir, exist_ok=True)

    records = []

    image_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".png"):
                image_files.append(
                    os.path.join(root, file)
                )

    image_files.sort()

    print(f"\nFound {len(image_files)} clean images.")

    if len(image_files) == 0:
        raise RuntimeError("No PNG images found.")

    total = len(image_files) * 3

    print(f"Generating {total} distorted samples...")
    print("Distortions: Elastic + Local + Bending\n")

    for idx, image_path in enumerate(image_files):

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(
                img,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )

        relative_path = os.path.relpath(
            image_path,
            INPUT_DIR
        )

        relative_no_ext = os.path.splitext(
            relative_path
        )[0]

        sample_name = relative_no_ext.replace(
            os.sep,
            "__"
        )

        clean_output = os.path.join(
            clean_dir,
            sample_name + ".png"
        )

        cv2.imwrite(clean_output, img)

        distortion_functions = [
            ("elastic", 0),
            ("local", 1),
            ("bending", 2)
        ]

        for distortion_name, type_offset in distortion_functions:

            seed = (
                MASTER_SEED * 100000
                + idx * 10
                + type_offset
            )

            rng = np.random.default_rng(seed)

            if distortion_name == "elastic":
                distorted, dx, dy, params = distort_elastic(
                    img,
                    rng
                )

            elif distortion_name == "local":
                distorted, dx, dy, params = distort_local(
                    img,
                    rng
                )

            else:
                distorted, dx, dy, params = distort_bending(
                    img,
                    rng
                )

            dx_rect, dy_rect = sampling_to_ddrnet_field(
                dx,
                dy
            )

            base_name = (
                sample_name
                + "__"
                + distortion_name
            )

            distorted_output = os.path.join(
                distorted_dir,
                base_name + ".png"
            )

            field_output = os.path.join(
                fields_dir,
                base_name + ".npz"
            )

            cv2.imwrite(
                distorted_output,
                distorted
            )

            np.savez_compressed(
                field_output,
                dx=dx_rect,
                dy=dy_rect
            )

            magnitude = np.sqrt(
                dx_rect ** 2 + dy_rect ** 2
            )

            records.append({
                "sample": base_name,
                "source": relative_path,
                "distortion_type": distortion_name,
                "seed": int(seed),
                "mean_displacement": float(
                    np.mean(magnitude)
                ),
                "max_displacement": float(
                    np.max(magnitude)
                ),
                **params
            })

        if (idx + 1) % 100 == 0:
            print(
                f"Processed {idx + 1}/{len(image_files)} "
                f"images"
            )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "metadata.csv"
    )

    pd.DataFrame(records).to_csv(
        metadata_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nClean images     : {len(image_files)}")
    print(f"Distorted images : {len(records)}")
    print(f"Expected         : {len(image_files) * 3}")

    print(f"\nOutput directory:")
    print(os.path.abspath(OUTPUT_DIR))

    print(f"\nMetadata:")
    print(os.path.abspath(metadata_path))

    print("\nEach distorted sample has:")
    print("  - distorted PNG")
    print("  - dx/dy DDRNet ground-truth field")
    print("  - distortion metadata")


if __name__ == "__main__":
    main()