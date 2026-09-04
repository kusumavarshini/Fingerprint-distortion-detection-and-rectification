"""
Synthetic Fingerprint Distortion Generator — Preview (10 images)
================================================================
Reads 10 randomly-selected preprocessed 224×224 grayscale fingerprint PNGs
from data/processed/, applies one of three geometric distortion types, and
saves clean + distorted pairs to data/distorted_preview/.

Distortion types (geometric / spatial only):
  1. elastic    — Random displacement fields smoothed with Gaussian
  2. local      — Sparse random control-point displacements, interpolated
  3. bending    — Smooth global nonlinear curvature warp

Outputs:
  data/distorted_preview/FAMILY-*/MEMBER/*_clean.png
  data/distorted_preview/FAMILY-*/MEMBER/*_distorted.png
  data/distorted_preview/distortion_params.csv
  data/distorted_preview/distortion_params.json
  data/distorted_preview/contact_sheet.png

Nothing in data/ or data/processed/ is modified.
"""

import os
import sys
import csv
import json
import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
GLOBAL_SEED   = 42
NUM_SAMPLES   = 10
IMG_SIZE      = 224                          # images are 224×224

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT      = os.path.join(BASE_DIR, "data", "processed")
DST_ROOT      = os.path.join(BASE_DIR, "data", "distorted_preview")

DISTORTION_TYPES = ["elastic", "local", "bending"]


# ══════════════════════════════════════════════════════════════════════════════
#  DISTORTION ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def distort_elastic(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Elastic deformation via random displacement fields smoothed by a Gaussian.

    1. Sample dense random displacement fields (dx, dy) from uniform distribution.
    2. Smooth with Gaussian filter (sigma) to create spatially coherent warps.
    3. Scale by alpha to control displacement strength.
    4. Apply with cv2.remap().
    """
    h, w = img.shape[:2]

    # Sample parameters from the specified ranges
    alpha  = rng.uniform(6.0, 16.0)         # displacement strength (pixels)
    sigma  = rng.uniform(8.0, 16.0)         # Gaussian smoothing kernel width
    grid   = rng.integers(32, 65)           # control grid spacing (pixels)

    # Generate coarse random displacement fields and up-sample
    grid_h = h // grid + 1
    grid_w = w // grid + 1
    dx_coarse = rng.uniform(-1.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)
    dy_coarse = rng.uniform(-1.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)

    # Resize to full image size (bilinear) then smooth
    dx = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    dx = gaussian_filter(dx, sigma=sigma) * alpha
    dy = gaussian_filter(dy, sigma=sigma) * alpha

    # Build remap coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + dx
    map_y = grid_y + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    params = {
        "alpha": round(float(alpha), 3),
        "sigma": round(float(sigma), 3),
        "grid_spacing": int(grid),
    }
    return warped, params


def distort_local(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Local nonlinear displacement via sparse random control points.

    1. Place a small grid of control points across the image.
    2. Assign random displacement vectors to each control point.
    3. Interpolate/smooth the displacement field across the full image.
    4. Apply with cv2.remap().
    """
    h, w = img.shape[:2]

    # Sample parameters
    n_ctrl     = rng.integers(5, 10)        # control points per axis
    max_disp   = rng.uniform(5.0, 15.0)     # max displacement (pixels)
    smooth_sig = rng.uniform(10.0, 20.0)    # Gaussian smoothing sigma

    # Generate control-point displacements on a coarse grid
    dx_ctrl = rng.uniform(-max_disp, max_disp,
                          size=(n_ctrl, n_ctrl)).astype(np.float32)
    dy_ctrl = rng.uniform(-max_disp, max_disp,
                          size=(n_ctrl, n_ctrl)).astype(np.float32)

    # Bilinear upsample to full resolution
    dx = cv2.resize(dx_ctrl, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy_ctrl, (w, h), interpolation=cv2.INTER_LINEAR)

    # Smooth for coherence
    dx = gaussian_filter(dx, sigma=smooth_sig)
    dy = gaussian_filter(dy, sigma=smooth_sig)

    # Build remap coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + dx
    map_y = grid_y + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    params = {
        "n_ctrl_per_axis": int(n_ctrl),
        "max_displacement_px": round(float(max_disp), 3),
        "smoothing_sigma": round(float(smooth_sig), 3),
    }
    return warped, params


def distort_bending(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Smooth global bending / nonlinear curvature warp.

    Applies a combination of sinusoidal warps along both axes to simulate
    the natural bending of a finger pressed on a sensor.  The deformation
    is smooth and globally coherent.
    """
    h, w = img.shape[:2]

    # Sample parameters
    max_disp = rng.uniform(10.0, 25.0)      # peak displacement (pixels)
    strength = rng.uniform(0.2, 1.0)         # bending strength multiplier
    # Random number of sine components for variety (1–3 harmonics)
    n_harmonics = rng.integers(1, 4)

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))

    # Normalised coordinates in [0, 1]
    nx = grid_x / w
    ny = grid_y / h

    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    for k in range(1, int(n_harmonics) + 1):
        freq_x  = rng.uniform(0.5, 2.0) * k
        freq_y  = rng.uniform(0.5, 2.0) * k
        phase_x = rng.uniform(0.0, 2 * np.pi)
        phase_y = rng.uniform(0.0, 2 * np.pi)
        amp     = max_disp * strength / k   # higher harmonics get weaker

        dx += (amp * np.sin(2 * np.pi * freq_y * ny + phase_y)
               * np.cos(2 * np.pi * freq_x * nx * 0.5 + phase_x)).astype(np.float32)
        dy += (amp * np.sin(2 * np.pi * freq_x * nx + phase_x)
               * np.cos(2 * np.pi * freq_y * ny * 0.5 + phase_y)).astype(np.float32)

    # Taper edges to avoid hard boundary artefacts
    taper_x = np.sin(np.linspace(0, np.pi, w, dtype=np.float32))
    taper_y = np.sin(np.linspace(0, np.pi, h, dtype=np.float32))
    taper = np.outer(taper_y, taper_x)
    dx *= taper
    dy *= taper

    map_x = grid_x + dx
    map_y = grid_y + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    params = {
        "max_displacement_px": round(float(max_disp), 3),
        "bending_strength": round(float(strength), 3),
        "n_harmonics": int(n_harmonics),
    }
    return warped, params


# Dispatcher
DISTORT_FN = {
    "elastic": distort_elastic,
    "local":   distort_local,
    "bending": distort_bending,
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONTACT SHEET
# ══════════════════════════════════════════════════════════════════════════════

def make_contact_sheet(records: list[dict], dst_path: str):
    """
    2-column contact sheet:  clean (left) | distorted (right), one row per sample.
    """
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, rec in enumerate(records):
        clean     = cv2.imread(rec["clean_path"], cv2.IMREAD_GRAYSCALE)
        distorted = cv2.imread(rec["distorted_path"], cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(clean, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"Clean - {rec['source_basename']}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(distorted, cmap="gray", vmin=0, vmax=255)
        label = rec["distortion_type"]
        axes[i, 1].set_title(f"Distorted - {label}", fontsize=8)
        axes[i, 1].axis("off")

    fig.suptitle("Fingerprint Distortion Preview  (10 samples)",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(dst_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Contact sheet saved -> {os.path.relpath(dst_path, BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_images(root: str) -> list[str]:
    """Return sorted list of all PNG paths under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: processed dataset not found at\n  {SRC_ROOT}")
        sys.exit(1)

    rng = np.random.default_rng(GLOBAL_SEED)
    t0  = time.time()

    # ── Collect & sample ──────────────────────────────────────────────────
    all_images = collect_all_images(SRC_ROOT)
    print(f"Found {len(all_images)} processed images in {SRC_ROOT}")

    indices = rng.choice(len(all_images), size=NUM_SAMPLES, replace=False)
    selected = [all_images[i] for i in sorted(indices)]
    print(f"Selected {NUM_SAMPLES} images (seed={GLOBAL_SEED})\n")

    # ── Process each image ────────────────────────────────────────────────
    records = []

    for idx, src_path in enumerate(selected):
        # Determine relative path within processed/ to mirror in output
        rel = os.path.relpath(src_path, SRC_ROOT)           # e.g. FAMILY-3/FATHER/FM003_F2.png
        stem = os.path.splitext(os.path.basename(rel))[0]    # e.g. FM003_F2

        dst_dir = os.path.join(DST_ROOT, os.path.dirname(rel))
        os.makedirs(dst_dir, exist_ok=True)

        # Load (already 224×224 grayscale uint8)
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  SKIP (unreadable): {rel}")
            continue

        # Round-robin distortion type so we see all three
        dist_type = DISTORTION_TYPES[idx % len(DISTORTION_TYPES)]

        # Per-image reproducible seed derived from global seed + index
        per_seed = GLOBAL_SEED * 1000 + idx
        img_rng  = np.random.default_rng(per_seed)

        distorted, params = DISTORT_FN[dist_type](img, img_rng)

        # Save clean + distorted
        clean_path     = os.path.join(dst_dir, f"{stem}_clean.png")
        distorted_path = os.path.join(dst_dir, f"{stem}_distorted.png")

        cv2.imwrite(clean_path, img)
        cv2.imwrite(distorted_path, distorted)

        rec = {
            "index":            idx,
            "source_path":      rel.replace("\\", "/"),
            "source_basename":  os.path.basename(rel),
            "distortion_type":  dist_type,
            "per_image_seed":   per_seed,
            "params":           params,
            "clean_output":     os.path.relpath(clean_path, BASE_DIR).replace("\\", "/"),
            "distorted_output": os.path.relpath(distorted_path, BASE_DIR).replace("\\", "/"),
            # Absolute paths kept internally for contact-sheet loading
            "clean_path":       clean_path,
            "distorted_path":   distorted_path,
        }
        records.append(rec)

    # ── Save distortion_params.csv ────────────────────────────────────────
    csv_path = os.path.join(DST_ROOT, "distortion_params.csv")
    csv_fields = [
        "index", "source_path", "distortion_type", "per_image_seed",
        "params", "clean_output", "distorted_output",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in csv_fields}
            row["params"] = json.dumps(rec["params"])       # serialise dict
            writer.writerow(row)
    print(f"  CSV  saved -> {os.path.relpath(csv_path, BASE_DIR)}")

    # ── Save distortion_params.json ───────────────────────────────────────
    json_path = os.path.join(DST_ROOT, "distortion_params.json")
    json_data = []
    for rec in records:
        json_data.append({
            "index":           rec["index"],
            "source_path":     rec["source_path"],
            "distortion_type": rec["distortion_type"],
            "per_image_seed":  rec["per_image_seed"],
            "parameters":      rec["params"],
            "clean_output":    rec["clean_output"],
            "distorted_output":rec["distorted_output"],
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"global_seed": GLOBAL_SEED, "num_samples": NUM_SAMPLES,
                    "records": json_data}, f, indent=2)
    print(f"  JSON saved -> {os.path.relpath(json_path, BASE_DIR)}")

    # ── Contact sheet ─────────────────────────────────────────────────────
    sheet_path = os.path.join(DST_ROOT, "contact_sheet.png")
    make_contact_sheet(records, sheet_path)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  DISTORTION PREVIEW — SUMMARY")
    print(sep)
    print(f"  Global seed        : {GLOBAL_SEED}")
    print(f"  Images processed   : {len(records)}")
    print(f"  Output directory   : {os.path.relpath(DST_ROOT, BASE_DIR)}/")
    print(f"  Time elapsed       : {elapsed:.1f}s\n")

    print(f"  {'#':<4} {'Source':<24} {'Type':<10} {'Seed':<8} Parameters")
    print(f"  {'-'*4} {'-'*24} {'-'*10} {'-'*8} {'-'*36}")
    for rec in records:
        p_str = ", ".join(f"{k}={v}" for k, v in rec["params"].items())
        print(f"  {rec['index']:<4} {rec['source_basename']:<24} "
              f"{rec['distortion_type']:<10} {rec['per_image_seed']:<8} {p_str}")

    print(f"\n{sep}")
    print("  Original data/processed/ images are UNCHANGED.")
    print(sep)


if __name__ == "__main__":
    main()
