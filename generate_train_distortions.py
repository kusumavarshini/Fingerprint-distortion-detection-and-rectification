"""
Training Set Distortion Generator
===================================
Generates the complete synthetic distorted training dataset for binary
classification (Clean vs Distorted).

Input  : data/split/train/  (70 families, 1050 images, 224x224 grayscale)
Output : data/train_distorted/
           clean/FAMILY-*/MEMBER/*.png          (1050 copies)
           distorted/FAMILY-*/MEMBER/*.png      (3150 distorted)
           distortion_metadata.csv
           distortion_metadata.json
           train_distortion_preview.png

Distortion types (geometric/spatial only, via cv2.remap):
  1. elastic  -- smooth coarse-grid displacement field
  2. local    -- sparse control-point interpolated displacement
  3. bending  -- global nonlinear barrel + sinusoidal curvature

Nothing in data/split/train/, data/processed/, or the raw dataset is modified.
"""

import os
import sys
import csv
import json
import time
import shutil
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Configuration -----------------------------------------------------------
MASTER_SEED = 42
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT    = os.path.join(BASE_DIR, "data", "split", "train")
DST_ROOT    = os.path.join(BASE_DIR, "data", "train_distorted")
CLEAN_DIR   = os.path.join(DST_ROOT, "clean")
DIST_DIR    = os.path.join(DST_ROOT, "distorted")

DISTORTION_TYPES = ["elastic", "local", "bending"]


# ============================================================================
#  DISTORTION ALGORITHMS
# ============================================================================

def distort_elastic(img, rng):
    """
    Smooth elastic deformation via coarse-grid random displacements.

    Parameters (sampled per image):
      alpha      : 8-16 px   (displacement magnitude)
      sigma      : 5-8       (Gaussian smoothing)
      grid_step  : 32 px     (coarse grid spacing)
    """
    h, w = img.shape
    alpha      = rng.uniform(8.0, 16.0)
    sigma      = rng.uniform(5.0, 8.0)
    grid_step  = 32

    gh = h // grid_step + 2
    gw = w // grid_step + 2
    dx_coarse = rng.uniform(-alpha, alpha, (gh, gw)).astype(np.float32)
    dy_coarse = rng.uniform(-alpha, alpha, (gh, gw)).astype(np.float32)

    dx = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    dx = gaussian_filter(dx, sigma=sigma).astype(np.float32)
    dy = gaussian_filter(dy, sigma=sigma).astype(np.float32)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    map_x = gx + dx
    map_y = gy + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    mag = np.sqrt(dx**2 + dy**2)
    params = {
        "alpha": round(float(alpha), 3),
        "sigma": round(float(sigma), 3),
        "grid_step": grid_step,
        "mean_disp": round(float(mag.mean()), 2),
        "max_disp": round(float(mag.max()), 2),
    }
    return warped, params


def distort_local(img, rng):
    """
    Sparse control-point displacement field.

    Parameters (sampled per image):
      n_ctrl   : 5 per axis
      max_disp : 5-10 px
      smooth   : 8-12
    """
    h, w = img.shape
    n_ctrl   = 5
    max_disp = rng.uniform(5.0, 10.0)
    smooth   = rng.uniform(8.0, 12.0)

    dx_ctrl = rng.uniform(-max_disp, max_disp,
                          (n_ctrl, n_ctrl)).astype(np.float32)
    dy_ctrl = rng.uniform(-max_disp, max_disp,
                          (n_ctrl, n_ctrl)).astype(np.float32)

    dx = cv2.resize(dx_ctrl, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy_ctrl, (w, h), interpolation=cv2.INTER_LINEAR)

    dx = gaussian_filter(dx, sigma=smooth).astype(np.float32)
    dy = gaussian_filter(dy, sigma=smooth).astype(np.float32)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    map_x = gx + dx
    map_y = gy + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    mag = np.sqrt(dx**2 + dy**2)
    params = {
        "n_ctrl": n_ctrl,
        "max_disp_param": round(float(max_disp), 3),
        "smooth": round(float(smooth), 3),
        "mean_disp": round(float(mag.mean()), 2),
        "max_disp": round(float(mag.max()), 2),
    }
    return warped, params


def distort_bending(img, rng):
    """
    Smooth global nonlinear spatial curvature (barrel + sinusoidal).

    Parameters (sampled per image):
      max_disp  : 15-25 px
      strength  : 0.5-1.0
      direction : random horizontal or vertical emphasis
    """
    h, w = img.shape
    max_disp_param = rng.uniform(12.0, 26.0)
    strength = rng.uniform(0.40, 1.00)

    # Random direction: 0 = horizontal emphasis, 1 = vertical emphasis
    direction = int(rng.integers(0, 2))

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    nx = gx / w
    ny = gy / h

    cx, cy = 0.5, 0.5
    rx = nx - cx
    ry = ny - cy
    r2 = rx**2 + ry**2

    # Barrel distortion component
    if direction == 0:
        # Horizontal emphasis
        dx = (
            max_disp_param * strength * rx * r2
        ).astype(np.float32)

        dy = (
            max_disp_param * strength * ry * r2 * 0.4
        ).astype(np.float32)

        dx += (
            max_disp_param * 0.4 * np.sin(np.pi * ny)
        ).astype(np.float32)

        dy += (
            max_disp_param * 0.2
            * np.sin(2 * np.pi * nx)
            * np.sin(np.pi * ny)
        ).astype(np.float32)

    else:
        # Vertical emphasis
        dx = (
            max_disp_param * strength * rx * r2 * 0.4
        ).astype(np.float32)

        dy = (
            max_disp_param * strength * ry * r2
        ).astype(np.float32)

        dy += (
            max_disp_param * 0.4 * np.sin(np.pi * nx)
        ).astype(np.float32)

        dx += (
            max_disp_param * 0.2
            * np.sin(2 * np.pi * ny)
            * np.sin(np.pi * nx)
        ).astype(np.float32)

    map_x = gx + dx
    map_y = gy + dy

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mag = np.sqrt(dx**2 + dy**2)

    params = {
        "max_disp_param": round(float(max_disp_param), 3),
        "strength": round(float(strength), 3),
        "direction": (
            "horizontal" if direction == 0 else "vertical"
        ),
        "mean_disp": round(float(mag.mean()), 2),
        "max_disp": round(float(mag.max()), 2),
    }

    return warped, params


DISTORT_FN = {
    "elastic": distort_elastic,
    "local":   distort_local,
    "bending": distort_bending,
}


# ============================================================================
#  IMAGE COLLECTION
# ============================================================================

def collect_images(root):
    """Collect all PNG paths with their family/member metadata."""
    images = []
    for family in sorted(os.listdir(root)):
        fam_path = os.path.join(root, family)
        if not os.path.isdir(fam_path) or not family.startswith("FAMILY-"):
            continue
        for member in sorted(os.listdir(fam_path)):
            mem_path = os.path.join(fam_path, member)
            if not os.path.isdir(mem_path):
                continue
            for fname in sorted(os.listdir(mem_path)):
                if not fname.lower().endswith(".png"):
                    continue
                images.append({
                    "path": os.path.join(mem_path, fname),
                    "family": family,
                    "member": member,
                    "filename": fname,
                    "stem": os.path.splitext(fname)[0],
                    "rel_dir": os.path.join(family, member),
                })
    return images


# ============================================================================
#  CONTACT SHEET
# ============================================================================

def make_contact_sheet(preview_records, dst_path):
    """
    4-column contact sheet: Original | Elastic | Local | Bending
    for a selection of preview images.
    """
    n = len(preview_records)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Elastic", "Local", "Bending"]

    for i, rec in enumerate(preview_records):
        for j, col_key in enumerate(["clean", "elastic", "local", "bending"]):
            img = cv2.imread(rec[col_key], cv2.IMREAD_GRAYSCALE)
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=255)
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=10,
                                     fontweight="bold")
            axes[i, j].axis("off")
        # Row label
        axes[i, 0].set_ylabel(rec["stem"], fontsize=7, rotation=0,
                               labelpad=60, va="center")

    fig.suptitle("Training Distortion Preview (12 samples)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(dst_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ============================================================================
#  MAIN
# ============================================================================

def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: training split not found at {SRC_ROOT}")
        sys.exit(1)

    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED)

    # -- Collect source images ------------------------------------------------
    images = collect_images(SRC_ROOT)
    n_images = len(images)
    print(f"Source: {os.path.relpath(SRC_ROOT, BASE_DIR)}/")
    print(f"Found {n_images} training images\n")

    if n_images == 0:
        print("ERROR: no images found"); sys.exit(1)

    # -- Select 12 random images for preview ----------------------------------
    preview_indices = set(
        rng.choice(n_images, size=min(12, n_images), replace=False))

    # -- Process each image ---------------------------------------------------
    metadata_records = []
    counts = {"clean": 0, "elastic": 0, "local": 0, "bending": 0}
    preview_records = []
    errors = []

    for idx, info in enumerate(images):
        src_path = info["path"]
        stem     = info["stem"]
        rel_dir  = info["rel_dir"]

        # Load image
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors.append(f"Cannot read: {src_path}")
            continue

        # -- Copy clean image -------------------------------------------------
        clean_dst_dir = os.path.join(CLEAN_DIR, rel_dir)
        os.makedirs(clean_dst_dir, exist_ok=True)
        clean_fname = f"{stem}_clean.png"
        clean_path  = os.path.join(clean_dst_dir, clean_fname)
        shutil.copy2(src_path, clean_path)
        counts["clean"] += 1

        # Track preview paths
        preview_entry = None
        if idx in preview_indices:
            preview_entry = {
                "stem": stem,
                "clean": clean_path,
            }

        # -- Generate 3 distorted versions ------------------------------------
        dist_dst_dir = os.path.join(DIST_DIR, rel_dir)
        os.makedirs(dist_dst_dir, exist_ok=True)

        for dist_type in DISTORTION_TYPES:
            # Unique reproducible seed per image + distortion type
            type_offset = DISTORTION_TYPES.index(dist_type)
            per_seed = MASTER_SEED * 100000 + idx * 10 + type_offset
            img_rng = np.random.default_rng(per_seed)

            warped, params = DISTORT_FN[dist_type](img, img_rng)

            dist_fname = f"{stem}_{dist_type}.png"
            dist_path  = os.path.join(dist_dst_dir, dist_fname)
            cv2.imwrite(dist_path, warped)
            counts[dist_type] += 1

            # Metadata record
            metadata_records.append({
                "original_path": os.path.relpath(src_path, BASE_DIR)
                                     .replace("\\", "/"),
                "distorted_path": os.path.relpath(dist_path, BASE_DIR)
                                      .replace("\\", "/"),
                "family": info["family"],
                "member": info["member"],
                "distortion_type": dist_type,
                "seed": per_seed,
                "mean_disp": params.pop("mean_disp"),
                "max_disp": params.pop("max_disp"),
                **{f"param_{k}": v for k, v in params.items()},
            })

            if preview_entry is not None:
                preview_entry[dist_type] = dist_path

        if preview_entry is not None:
            preview_records.append(preview_entry)

        # Progress
        if (idx + 1) % 200 == 0 or idx == n_images - 1:
            print(f"  Processed {idx + 1}/{n_images} images ...")

    elapsed = time.time() - t0

    # -- Save metadata CSV ----------------------------------------------------
    csv_path = os.path.join(DST_ROOT, "distortion_metadata.csv")
    if metadata_records:
        fieldnames = list(metadata_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            for rec in metadata_records:
                writer.writerow(rec)
    print(f"\n  CSV  saved -> {os.path.relpath(csv_path, BASE_DIR)}")

    # -- Save metadata JSON ---------------------------------------------------
    json_path = os.path.join(DST_ROOT, "distortion_metadata.json")
    json_obj = {
        "master_seed": MASTER_SEED,
        "source": os.path.relpath(SRC_ROOT, BASE_DIR).replace("\\", "/"),
        "total_source_images": n_images,
        "total_clean": counts["clean"],
        "total_distorted": counts["elastic"] + counts["local"] + counts["bending"],
        "counts": counts,
        "records": metadata_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)
    print(f"  JSON saved -> {os.path.relpath(json_path, BASE_DIR)}")

    # -- Contact sheet --------------------------------------------------------
    if preview_records:
        sheet_path = os.path.join(DST_ROOT, "train_distortion_preview.png")
        make_contact_sheet(preview_records, sheet_path)
        print(f"  Preview saved -> {os.path.relpath(sheet_path, BASE_DIR)}")

    # -- Summary report -------------------------------------------------------
    total_distorted = counts["elastic"] + counts["local"] + counts["bending"]
    total_all = counts["clean"] + total_distorted

    sep = "=" * 60
    report_lines = [
        sep,
        "  TRAINING DISTORTION GENERATION REPORT",
        sep,
        f"  Source          : {os.path.relpath(SRC_ROOT, BASE_DIR)}/",
        f"  Output          : {os.path.relpath(DST_ROOT, BASE_DIR)}/",
        f"  Master seed     : {MASTER_SEED}",
        f"  Time elapsed    : {elapsed:.1f}s",
        "",
        f"  {'Category':<20} {'Count':>8}",
        f"  {'-'*20} {'-'*8}",
        f"  {'Clean copies':<20} {counts['clean']:>8}",
        f"  {'Elastic':<20} {counts['elastic']:>8}",
        f"  {'Local':<20} {counts['local']:>8}",
        f"  {'Bending':<20} {counts['bending']:>8}",
        f"  {'-'*20} {'-'*8}",
        f"  {'Total distorted':<20} {total_distorted:>8}",
        f"  {'TOTAL images':<20} {total_all:>8}",
        "",
    ]
    if errors:
        report_lines.append(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            report_lines.append(f"    - {e}")
    else:
        report_lines.append("  Errors: 0")

    report_lines.extend(["", sep,
                         "  Original training split is UNCHANGED.", sep])
    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    # Save report
    report_path = os.path.join(DST_ROOT, "generation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")


if __name__ == "__main__":
    main()
