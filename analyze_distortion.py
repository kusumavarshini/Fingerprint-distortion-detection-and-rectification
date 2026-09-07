import os
import csv
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_ROOT = os.path.join(BASE_DIR, "data", "train_distorted")
METADATA_FILE = os.path.join(TRAIN_ROOT, "distortion_metadata.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "distortion_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISTORTION_TYPES = ["elastic", "local", "bending"]


# ============================================================
# DISTORTION FUNCTIONS
# Same logic and parameter ranges as the existing generator.
# ============================================================

def distort_elastic(img, rng):
    h, w = img.shape

    alpha = rng.uniform(8, 16)
    sigma = rng.uniform(5, 8)
    grid_step = 32

    gh = h // grid_step + 2
    gw = w // grid_step + 2

    coarse_dx = rng.uniform(-alpha, alpha, (gh, gw)).astype(np.float32)
    coarse_dy = rng.uniform(-alpha, alpha, (gh, gw)).astype(np.float32)

    dx = cv2.resize(
        coarse_dx,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    dy = cv2.resize(
        coarse_dy,
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

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return warped, dx, dy


def distort_local(img, rng):
    h, w = img.shape

    n_ctrl = 5
    max_disp = rng.uniform(5, 10)
    smooth = rng.uniform(8, 12)

    ctrl_dx = rng.uniform(
        -max_disp,
        max_disp,
        (n_ctrl, n_ctrl)
    ).astype(np.float32)

    ctrl_dy = rng.uniform(
        -max_disp,
        max_disp,
        (n_ctrl, n_ctrl)
    ).astype(np.float32)

    dx = cv2.resize(
        ctrl_dx,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    dy = cv2.resize(
        ctrl_dy,
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

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return warped, dx, dy


def distort_bending(img, rng):
    h, w = img.shape

    max_disp_param = rng.uniform(12, 26)
    strength = rng.uniform(0.40, 1.00)
    direction = int(rng.integers(0, 2))

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    nx = gx / w
    ny = gy / h

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
        dy = (
            max_disp_param * strength * ry * r2
            + max_disp_param * 0.4 * np.sin(np.pi * nx)
        )

        dx = (
            max_disp_param * strength * rx * r2 * 0.4
            + max_disp_param * 0.2
            * np.sin(2 * np.pi * ny)
            * np.sin(np.pi * nx)
        )

    map_x = gx + dx
    map_y = gy + dy

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return warped, dx, dy


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(dx, dy):
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    metrics = {
        "mean_disp": float(np.mean(magnitude)),
        "median_disp": float(np.median(magnitude)),
        "p90_disp": float(np.percentile(magnitude, 90)),
        "p95_disp": float(np.percentile(magnitude, 95)),
        "max_disp": float(np.max(magnitude)),
        "std_disp": float(np.std(magnitude)),
    }

    # Useful for later severity analysis.
    # These thresholds are NOT severity definitions.
    metrics["affected_2px_pct"] = float(
        np.mean(magnitude > 2.0) * 100
    )

    metrics["affected_4px_pct"] = float(
        np.mean(magnitude > 4.0) * 100
    )

    metrics["affected_6px_pct"] = float(
        np.mean(magnitude > 6.0) * 100
    )

    return metrics


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    print("=" * 70)
    print("DISTORTION EXTENT ANALYSIS")
    print("=" * 70)

    records = []

    with open(METADATA_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            records.append(row)

    print(f"Metadata records found: {len(records)}")

    results = []

    distortion_functions = {
        "elastic": distort_elastic,
        "local": distort_local,
        "bending": distort_bending,
    }

    for i, row in enumerate(records, start=1):

        distortion_type = row["distortion_type"]

        if distortion_type not in DISTORTION_TYPES:
            continue

        image_path = row["original_path"]

        # Handle paths stored relative to project root.
        if not os.path.isabs(image_path):
            image_path = os.path.join(BASE_DIR, image_path)

        if not os.path.exists(image_path):
            # Fallback: derive from family/member/stem
            family = row["family"]
            member = row["member"]

            filename = os.path.basename(image_path)

            image_path = os.path.join(
                BASE_DIR,
                "data",
                "split",
                "train",
                family,
                member,
                filename
            )

        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            continue

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print(f"[WARNING] Could not read: {image_path}")
            continue

        seed = int(row["seed"])
        rng = np.random.default_rng(seed)

        _, dx, dy = distortion_functions[distortion_type](
            img,
            rng
        )

        metrics = calculate_metrics(dx, dy)

        result = {
            "family": row["family"],
            "member": row["member"],
            "distortion_type": distortion_type,
            "seed": seed,
            **metrics
        }

        results.append(result)

        if i % 250 == 0:
            print(f"Processed {i}/{len(records)}")

    # ========================================================
    # SAVE PER-IMAGE RESULTS
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "distortion_metrics.csv"
    )

    fieldnames = [
        "family",
        "member",
        "distortion_type",
        "seed",
        "mean_disp",
        "median_disp",
        "p90_disp",
        "p95_disp",
        "max_disp",
        "std_disp",
        "affected_2px_pct",
        "affected_4px_pct",
        "affected_6px_pct",
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
    print(f"Saved: {csv_path}")

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    summary = {}

    for distortion_type in DISTORTION_TYPES:

        subset = [
            r for r in results
            if r["distortion_type"] == distortion_type
        ]

        if not subset:
            continue

        summary[distortion_type] = {}

        for metric in fieldnames[4:]:

            values = np.array(
                [r[metric] for r in subset],
                dtype=np.float64
            )

            summary[distortion_type][metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }

    json_path = os.path.join(
        OUTPUT_DIR,
        "distortion_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(f"Saved: {json_path}")

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for distortion_type in DISTORTION_TYPES:

        if distortion_type not in summary:
            continue

        print()
        print(f"{distortion_type.upper()}")

        for metric in [
            "mean_disp",
            "median_disp",
            "p90_disp",
            "p95_disp",
            "max_disp",
        ]:

            s = summary[distortion_type][metric]

            print(
                f"  {metric:15s} "
                f"mean={s['mean']:.3f} "
                f"median={s['median']:.3f} "
                f"p95={s['p95']:.3f} "
                f"max={s['max']:.3f}"
            )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()