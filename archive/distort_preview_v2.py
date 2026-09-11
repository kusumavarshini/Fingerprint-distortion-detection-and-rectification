"""
Distortion Preview v2 -- Visible Geometric Deformations (revised)
==================================================================
Applies 3 geometric distortion types to ONE fingerprint with
moderate-to-strong parameters.  Ridge structure remains recognisable.

For each type produces a 3-panel comparison:
   Original  |  Displacement Field (heatmap + quiver)  |  Distorted + Grid

All transformations are purely spatial (cv2.remap with map_x, map_y).
No intensity, noise, contrast, or texture manipulation.

Outputs:
  data/distorted_preview_v2/comparison_sheet.png
  data/distorted_preview_v2/{type}_original.png
  data/distorted_preview_v2/{type}_distorted.png
  data/distorted_preview_v2/{type}_grid.png
  data/distorted_preview_v2/distortion_params.json
"""

import os
import sys
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(BASE_DIR, "data", "processed")
DST_ROOT = os.path.join(BASE_DIR, "data", "distorted_preview_v2")

SEED     = 42
IMG_SIZE = 224


# ============================================================================
#  DISPLACEMENT FIELD VISUALISATION
# ============================================================================

def draw_warped_grid(ax, map_x, map_y, step=12, color="deepskyblue", lw=0.5):
    """Draw a warped coordinate grid showing geometric deformation."""
    h, w = map_x.shape
    for r in range(0, h, step):
        ax.plot(map_x[r, :], map_y[r, :], color=color, lw=lw, alpha=0.7)
    for c in range(0, w, step):
        ax.plot(map_x[:, c], map_y[:, c], color=color, lw=lw, alpha=0.7)


def visualise_displacement(ax, dx, dy, step=14):
    """Displacement magnitude heatmap with quiver arrows."""
    h, w = dx.shape
    mag = np.sqrt(dx**2 + dy**2)

    im = ax.imshow(mag, cmap="inferno", interpolation="bilinear")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pixels")

    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    X, Y = np.meshgrid(xs, ys)
    U = dx[Y, X]
    V = dy[Y, X]
    ax.quiver(X, Y, U, V, color="white", scale_units="xy", scale=1,
              width=0.003, headwidth=4, alpha=0.85)


# ============================================================================
#  DISTORTION 1 -- ELASTIC DEFORMATION
# ============================================================================

def distort_elastic(img, rng):
    """
    Elastic deformation via coarse-grid random displacements.

    Strategy:
      - Generate random displacements on a sparse grid (~32px spacing).
      - Bicubic-upsample to full resolution.
      - Moderate Gaussian smoothing (sigma=5) for spatial continuity.

    This produces smooth local stretching/compression of ridge positions
    while keeping the overall fingerprint structure intact.
    """
    h, w = img.shape
    amplitude   = 10.0    # displacement magnitude per control point (px)
    grid_step   = 32      # control-point spacing (px)
    smooth_sigma = 5.0    # post-upsample smoothing

    gh = h // grid_step + 2
    gw = w // grid_step + 2
    dx_coarse = rng.uniform(-amplitude, amplitude,
                            (gh, gw)).astype(np.float32)
    dy_coarse = rng.uniform(-amplitude, amplitude,
                            (gh, gw)).astype(np.float32)

    # Bicubic upsample preserves displacement magnitude
    dx = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    # Smooth for spatial continuity (moderate -- not too aggressive)
    dx = gaussian_filter(dx, sigma=smooth_sigma).astype(np.float32)
    dy = gaussian_filter(dy, sigma=smooth_sigma).astype(np.float32)

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = grid_x + dx
    map_y = grid_y + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    params = {"amplitude_px": amplitude, "grid_step": grid_step,
              "smooth_sigma": smooth_sigma}
    return warped, dx, dy, map_x, map_y, params


# ============================================================================
#  DISTORTION 2 -- LOCAL NONLINEAR DISPLACEMENT
# ============================================================================

def distort_local(img, rng):
    """
    Sparse control-point displacements interpolated across the image.

    Strategy:
      - Place a 5x5 grid of control points across the image.
      - Assign random displacement vectors (up to 12px each).
      - Bilinear-upsample to full resolution, then smooth (sigma=8).

    This makes different fingerprint regions shift by different amounts,
    with smooth transitions between displaced areas.
    """
    h, w = img.shape
    n_ctrl   = 5         # control points per axis
    max_disp = 12.0      # maximum displacement per control point (px)
    smooth   = 8.0       # post-interpolation smoothing

    dx_ctrl = rng.uniform(-max_disp, max_disp,
                          (n_ctrl, n_ctrl)).astype(np.float32)
    dy_ctrl = rng.uniform(-max_disp, max_disp,
                          (n_ctrl, n_ctrl)).astype(np.float32)

    dx = cv2.resize(dx_ctrl, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy_ctrl, (w, h), interpolation=cv2.INTER_LINEAR)

    dx = gaussian_filter(dx, sigma=smooth).astype(np.float32)
    dy = gaussian_filter(dy, sigma=smooth).astype(np.float32)

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = grid_x + dx
    map_y = grid_y + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    params = {"n_ctrl": n_ctrl, "max_disp_px": max_disp, "smooth": smooth}
    return warped, dx, dy, map_x, map_y, params


# ============================================================================
#  DISTORTION 3 -- GLOBAL NONLINEAR BENDING
# ============================================================================

def distort_bending(img, rng):
    """
    Smooth global spatial curvature (barrel + sinusoidal warp).

    Strategy:
      - Radial barrel distortion centred on the image.
      - Additive sinusoidal bending component.

    Simulates a finger pressed at an angle on a sensor.
    Deformation is strongest at centre, tapers at edges.
    """
    h, w = img.shape
    max_disp = 25.0     # peak displacement (px)
    strength = 0.9      # bending strength

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    nx = grid_x / w     # normalised [0, 1]
    ny = grid_y / h

    cx, cy = 0.5, 0.5
    rx = nx - cx
    ry = ny - cy
    r2 = rx**2 + ry**2

    # Radial barrel distortion
    dx = (max_disp * strength * rx * r2).astype(np.float32)
    dy = (max_disp * strength * ry * r2 * 0.6).astype(np.float32)

    # Additive sinusoidal bend
    dx += (max_disp * 0.4 * np.sin(np.pi * ny)).astype(np.float32)
    dy += (max_disp * 0.3 * np.sin(2 * np.pi * nx) *
           np.sin(np.pi * ny)).astype(np.float32)

    map_x = grid_x + dx
    map_y = grid_y + dy

    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    params = {"max_disp_px": max_disp, "strength": strength}
    return warped, dx, dy, map_x, map_y, params


# ============================================================================
#  MAIN
# ============================================================================

def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: {SRC_ROOT} not found"); sys.exit(1)

    os.makedirs(DST_ROOT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # -- Pick one image with good ridge detail --------------------------------
    sample_rel  = os.path.join("FAMILY-1", "FATHER", "FM001_F1.png")
    sample_path = os.path.join(SRC_ROOT, sample_rel)
    if not os.path.isfile(sample_path):
        print(f"ERROR: sample not found at {sample_path}"); sys.exit(1)

    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded: {sample_rel}  ({img.shape[1]}x{img.shape[0]}, "
          f"dtype={img.dtype})")
    print(f"Seed:   {SEED}\n")

    # -- Apply each distortion ------------------------------------------------
    distortions = [
        ("elastic",  distort_elastic),
        ("local",    distort_local),
        ("bending",  distort_bending),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    all_params = {}

    for row, (name, fn) in enumerate(distortions):
        print(f"  [{row+1}/3] {name} ...", end=" ")
        warped, dx, dy, map_x, map_y, params = fn(img, rng)

        # Displacement statistics
        mag = np.sqrt(dx**2 + dy**2)
        params["_stats"] = {
            "mean_disp_px": round(float(mag.mean()), 2),
            "max_disp_px":  round(float(mag.max()), 2),
            "std_disp_px":  round(float(mag.std()), 2),
        }
        all_params[name] = params

        # Save individual PNGs
        cv2.imwrite(os.path.join(DST_ROOT, f"{name}_original.png"), img)
        cv2.imwrite(os.path.join(DST_ROOT, f"{name}_distorted.png"), warped)

        # -- Column 0: Original -----------------------------------------------
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("Original", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        # -- Column 1: Displacement field (heatmap + quiver) ------------------
        ax_mid = axes[row, 1]
        visualise_displacement(ax_mid, dx, dy, step=14)
        ax_mid.set_title(f"Displacement field  ({name})",
                         fontsize=10, fontweight="bold")
        ax_mid.set_xlim(0, img.shape[1])
        ax_mid.set_ylim(img.shape[0], 0)
        ax_mid.set_aspect("equal")

        # -- Column 2: Distorted with warped-grid overlay ---------------------
        ax_right = axes[row, 2]
        ax_right.imshow(warped, cmap="gray", vmin=0, vmax=255)
        draw_warped_grid(ax_right, map_x, map_y, step=16,
                         color="cyan", lw=0.4)
        ax_right.set_title("Distorted + grid", fontsize=11, fontweight="bold")
        ax_right.axis("off")

        p_str = ", ".join(f"{k}={v}" for k, v in params.items()
                          if not k.startswith("_"))
        print(f"mean={mag.mean():.1f}px, max={mag.max():.1f}px  | {p_str}")

    fig.suptitle(
        "Fingerprint Geometric Distortion Preview v2\n"
        f"Source: {sample_rel}   |   Seed: {SEED}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    sheet_path = os.path.join(DST_ROOT, "comparison_sheet.png")
    fig.savefig(sheet_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # -- Standalone warped-grid images ----------------------------------------
    for name, fn in distortions:
        _, dx, dy, map_x, map_y, _ = fn(
            img, np.random.default_rng(SEED))   # same seed for reproducibility

        fig_g, ax_g = plt.subplots(1, 1, figsize=(5, 5))
        ax_g.set_facecolor("black")
        draw_warped_grid(ax_g, map_x, map_y, step=10, color="lime", lw=0.6)
        ax_g.set_xlim(0, img.shape[1])
        ax_g.set_ylim(img.shape[0], 0)
        ax_g.set_aspect("equal")
        ax_g.set_title(f"Warped Grid - {name}",
                       fontsize=11, fontweight="bold", color="white")
        ax_g.axis("off")
        grid_path = os.path.join(DST_ROOT, f"{name}_grid.png")
        fig_g.savefig(grid_path, dpi=150, bbox_inches="tight",
                      facecolor="black")
        plt.close(fig_g)

    # -- Save distortion_params.json ------------------------------------------
    json_path = os.path.join(DST_ROOT, "distortion_params.json")
    json_data = {
        "source_image": sample_rel.replace("\\", "/"),
        "seed": SEED,
        "image_size": IMG_SIZE,
        "distortions": {}
    }
    for name, p in all_params.items():
        json_data["distortions"][name] = {
            k: v for k, v in p.items()
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # -- Summary --------------------------------------------------------------
    sep = "=" * 68
    print(f"\n{sep}")
    print("  DISTORTION PREVIEW v2 -- SUMMARY")
    print(sep)
    print(f"  Source image : {sample_rel}")
    print(f"  Seed         : {SEED}")
    print(f"  Output dir   : {os.path.relpath(DST_ROOT, BASE_DIR)}/\n")

    print(f"  {'Type':<12} {'Mean disp':>10} {'Max disp':>10} {'Std disp':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, p in all_params.items():
        s = p["_stats"]
        print(f"  {name:<12} {s['mean_disp_px']:>8.1f}px "
              f"{s['max_disp_px']:>8.1f}px {s['std_disp_px']:>8.1f}px")

    print(f"\n  Files saved:")
    print(f"    comparison_sheet.png")
    print(f"    distortion_params.json")
    for name in ["elastic", "local", "bending"]:
        print(f"    {name}_original.png  {name}_distorted.png  {name}_grid.png")
    print(f"\n{sep}")
    print(f"  Original data/processed/ images are UNCHANGED.")
    print(sep)


if __name__ == "__main__":
    main()
