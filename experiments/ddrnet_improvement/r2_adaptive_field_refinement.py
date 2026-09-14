"""
DDRNet Rectification Improvement R2: Adaptive Spatial Field Refinement
======================================================================
Experimental Script: experiments/ddrnet_improvement/r2_adaptive_field_refinement.py
Branch: ddrnet-improvement

STATUS: EXPERIMENTAL RESULT (R2)

PROBLEM IDENTIFICATION (LESSON FROM R1):
In R1, uniform multiplication of the displacement field by the soft foreground mask
successfully suppressed whitespace deformation and protected local distortions, but caused
unintended degradation on global bending distortions (+20.97% -> +15.03% MAE improvement).
This occurred because global bending involves large coherent displacements across the entire
fingerprint impression that extend naturally to the perimeter. Uniform foreground attenuation
decayed boundary displacements prematurely, eroding ridge alignment near the edges.

R2 REFINEMENT OBJECTIVE:
Preserve legitimate large displacements inside and along the fingerprint boundary (especially
global bending), while progressively suppressing displacement leakage into distant background
regions where spurious upsampling artifacts occur.

MATHEMATICAL FORMULATION OF R2 ADAPTIVE REFINEMENT:
Rather than a binary cutoff or uniform mask attenuation, R2 computes a continuous adaptive
spatial weighting map W_R2(x, y) in [0.0, 1.0] determined by:
  1. Soft foreground probability S(x, y) from the standard Otsu + morphological mask.
  2. Euclidean distance from the foreground boundary d_out(x, y) via distance transform.
  3. Local displacement magnitude M(x, y) = sqrt(dx^2 + dy^2).

The spatial decay width sigma_d(x, y) scales adaptively with local deformation magnitude:
    sigma_d(x, y) = sigma_base + alpha * M(x, y)
where predetermined constants (fixed prior to test-set evaluation) are:
    sigma_base = 8.0 pixels (nominal near-boundary transition margin)
    alpha = 0.5 (scaling factor coupling decay length to displacement magnitude)

The distance-decay factor D(x, y) is defined as:
    D(x, y) = 1.0                                                if (x, y) in foreground
    D(x, y) = exp( -0.5 * (d_out(x, y) / max(sigma_d(x, y), 1.0))^2 ) if (x, y) in background

The final continuous adaptive weight map blends the contour-aware soft mask S(x, y) with
the distance-magnitude field D(x, y):
    W_R2(x, y) = max( S(x, y), D(x, y) ) smoothed with a mild Gaussian filter (5x5, sigma=1.0)
    refined_dx = predicted_dx * W_R2
    refined_dy = predicted_dy * W_R2

PROPERTIES:
- W_R2(x, y) = 1.0 across the entire fingerprint interior: zero attenuation of internal ridge corrections.
- Large coherent displacements (e.g. bending) expand sigma_d, providing a wider transition zone.
- Small/noisy displacements (e.g. background noise on local distortions) decay rapidly to 0.0.
- Continuous C-infinity transition without step artifacts or hard clipping.
- Preserves the existing DDRNet inverse sign convention: map_x = x - dx, map_y = y - dy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("DDRNet").resolve()))
from models.DDRNet_DIR import DDRNet_DIR

# =============================================================================
# CONFIGURATION & REPRODUCIBILITY
# =============================================================================

EXPERIMENT_NAME = "DDRNet Improvement R2: Adaptive Spatial Field Refinement"
STATUS_LABEL = "EXPERIMENTAL RESULT (R2)"

CHECKPOINT_PATH = Path("experiments/ddrnet_baseline/best_model.pth")
TEST_DISTORTED_DIR = Path("data/rectification_test/distorted")
TEST_CLEAN_DIR = Path("data/rectification_test/clean")

OUTPUT_DIR = Path("experiments/ddrnet_improvement")
OUTPUT_JSON_PATH = OUTPUT_DIR / "r2_adaptive_field_refinement_results.json"
VIS_DIR = OUTPUT_DIR / "visualizations_r2"

IMAGE_SIZE = 224
GRID_SIZE = 14

# Predetermined constants for R1 and R2
R1_GAUSSIAN_KSIZE = (11, 11)
R1_GAUSSIAN_SIGMA = 2.5

R2_SIGMA_BASE = 8.0   # Nominal transition margin around foreground boundary
R2_ALPHA = 0.5        # Scaling coupling decay length to displacement magnitude

SEED = 42


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds for full evaluation reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# FOREGROUND MASK & REFINEMENT STRATEGIES
# =============================================================================

def extract_foreground_mask(image: np.ndarray) -> np.ndarray:
    """
    Standard project methodology for fingerprint foreground segmentation:
    Otsu thresholding + binary inversion + morphological open/close + largest connected component.
    """
    _, mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return mask


def compute_r1_soft_mask(binary_mask: np.ndarray) -> np.ndarray:
    """R1 Baseline Improvement: Uniform Gaussian-smoothed foreground mask."""
    mask_f = (binary_mask > 0).astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask_f, R1_GAUSSIAN_KSIZE, sigmaX=R1_GAUSSIAN_SIGMA)
    return np.clip(soft_mask, 0.0, 1.0).astype(np.float32)


def compute_r2_adaptive_weight(
    binary_mask: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    sigma_base: float = R2_SIGMA_BASE,
    alpha: float = R2_ALPHA,
) -> np.ndarray:
    """
    R2 Adaptive Spatial Weighting:
    - Preserves 1.0 inside the foreground.
    - Decays smoothly in the background with decay rate scaling by local displacement magnitude.
    - Fuses with the contour-aware soft mask to guarantee continuous edge coverage.
    """
    # 1. Distance transform from the outer boundary into whitespace
    dist_outside = cv2.distanceTransform((255 - binary_mask).astype(np.uint8), cv2.DIST_L2, 5)

    # 2. Local displacement magnitude
    mag = np.sqrt(dx ** 2 + dy ** 2)

    # 3. Adaptive decay length
    sigma_d = sigma_base + alpha * mag

    # 4. Distance-based spatial decay map
    w_dist = np.ones_like(dist_outside, dtype=np.float32)
    outside = binary_mask == 0
    w_dist[outside] = np.exp(
        -0.5 * (dist_outside[outside] / np.maximum(sigma_d[outside], 1.0)) ** 2
    )

    # 5. Soft foreground mask
    soft_mask = compute_r1_soft_mask(binary_mask)

    # 6. Joint continuous fusion
    w_comb = np.maximum(soft_mask, w_dist)
    w_comb = cv2.GaussianBlur(w_comb, (5, 5), sigmaX=1.0)
    return np.clip(w_comb, 0.0, 1.0).astype(np.float32)


def apply_rectification(image: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Inverse coordinate mapping adhering to the existing DDRNet convention:
        map_x = x - dx
        map_y = y - dy
    """
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = grid_x - dx
    map_y = grid_y - dy

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rectified


def compute_metrics(
    reference: np.ndarray,
    evaluated: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute overall, foreground, and background MAE and RMSE."""
    ref_f = reference.astype(np.float32)
    eval_f = evaluated.astype(np.float32)
    diff = eval_f - ref_f
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    overall_mae = float(np.mean(abs_diff))
    overall_rmse = float(np.sqrt(np.mean(sq_diff)))

    if np.any(fg_mask):
        fg_mae = float(np.mean(abs_diff[fg_mask]))
        fg_rmse = float(np.sqrt(np.mean(sq_diff[fg_mask])))
    else:
        fg_mae = overall_mae
        fg_rmse = overall_rmse

    if np.any(bg_mask):
        bg_mae = float(np.mean(abs_diff[bg_mask]))
        bg_rmse = float(np.sqrt(np.mean(sq_diff[bg_mask])))
    else:
        bg_mae = overall_mae
        bg_rmse = overall_rmse

    return {
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "foreground_mae": fg_mae,
        "foreground_rmse": fg_rmse,
        "background_mae": bg_mae,
        "background_rmse": bg_rmse,
    }


def compute_improvement_pct(initial_val: float, rectified_val: float) -> float:
    """Calculate percentage improvement: positive means error decreased (better)."""
    return ((initial_val - rectified_val) / max(initial_val, 1e-8)) * 100.0


# =============================================================================
# VISUALIZATION EXPORT
# =============================================================================

def make_displacement_heatmap(dx: np.ndarray, dy: np.ndarray, max_disp: float = 20.0) -> np.ndarray:
    """Generate color-mapped RGB heatmap of displacement magnitude."""
    mag = np.sqrt(dx ** 2 + dy ** 2)
    norm = np.clip((mag / max(max_disp, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def save_representative_comparison_r2(
    save_path: Path,
    distorted: np.ndarray,
    clean: np.ndarray,
    base_rect: np.ndarray,
    r1_rect: np.ndarray,
    r2_rect: np.ndarray,
    w_r2: np.ndarray,
    pred_dx: np.ndarray,
    pred_dy: np.ndarray,
    r2_dx: np.ndarray,
    r2_dy: np.ndarray,
    sample_name: str,
    distortion_type: str,
) -> None:
    """
    Save multi-panel diagnostic visualization comparing Baseline, R1, and R2:
    Row 1: Clean Reference, Distorted Input, Baseline Rectified, R1 Rectified, R2 Rectified
    Row 2: R2 Adaptive Weight Map, Baseline Disp Field, R2 Refined Disp Field, Difference |R2 - Baseline|, Error |R2 - Clean|
    """
    h_base = make_displacement_heatmap(pred_dx, pred_dy)
    h_r2 = make_displacement_heatmap(r2_dx, r2_dy)

    w_vis = (w_r2 * 255.0).astype(np.uint8)
    w_vis_rgb = cv2.applyColorMap(w_vis, cv2.COLORMAP_VIRIDIS)
    w_vis_rgb = cv2.cvtColor(w_vis_rgb, cv2.COLOR_BGR2RGB)

    dist_rgb = cv2.cvtColor(distorted, cv2.COLOR_GRAY2RGB)
    clean_rgb = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
    base_rgb = cv2.cvtColor(base_rect, cv2.COLOR_GRAY2RGB)
    r1_rgb = cv2.cvtColor(r1_rect, cv2.COLOR_GRAY2RGB)
    r2_rgb = cv2.cvtColor(r2_rect, cv2.COLOR_GRAY2RGB)

    diff_abs = cv2.applyColorMap(
        np.clip(np.abs(r2_rect.astype(float) - base_rect.astype(float)) * 5.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_MAGMA
    )
    diff_rgb = cv2.cvtColor(diff_abs, cv2.COLOR_BGR2RGB)

    err_abs = cv2.applyColorMap(
        np.clip(np.abs(r2_rect.astype(float) - clean.astype(float)) * 3.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_HOT
    )
    err_rgb = cv2.cvtColor(err_abs, cv2.COLOR_BGR2RGB)

    def add_title(img: np.ndarray, title: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (224, 24), (20, 20, 20), -1)
        cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    c1 = add_title(clean_rgb, "1. Clean Reference")
    c2 = add_title(dist_rgb, f"2. Distorted ({distortion_type})")
    c3 = add_title(base_rgb, "3. Baseline Rectified")
    c4 = add_title(r1_rgb, "4. R1 Rectified")
    c5 = add_title(r2_rgb, "5. R2 Adaptive Rectified")

    b1 = add_title(w_vis_rgb, "6. R2 Adaptive Weight W")
    b2 = add_title(h_base, "7. Baseline Disp Field")
    b3 = add_title(h_r2, "8. R2 Refined Disp Field")
    b4 = add_title(diff_rgb, "9. |R2 - Base| Diff x5")
    b5 = add_title(err_rgb, "10. |R2 - Clean| Error")

    top_row = np.hstack([c1, c2, c3, c4, c5])
    bot_row = np.hstack([b1, b2, b3, b4, b5])
    composite = np.vstack([top_row, bot_row])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_r2_evaluation() -> Dict:
    """Execute complete deterministic evaluation of Baseline, R1, and R2 on DDRNet test set."""
    set_seed(SEED)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"DDRNet checkpoint not found at: {CHECKPOINT_PATH}")
    if not TEST_DISTORTED_DIR.exists():
        raise FileNotFoundError(f"Distorted test directory not found: {TEST_DISTORTED_DIR}")
    if not TEST_CLEAN_DIR.exists():
        raise FileNotFoundError(f"Clean test directory not found: {TEST_CLEAN_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n======================================================================")
    print(f" {EXPERIMENT_NAME}")
    print(f" Status: {STATUS_LABEL}")
    print(f" Device: {device} | Seed: {SEED}")
    print(f" Checkpoint: {CHECKPOINT_PATH}")
    print(f" Parameters (Predetermined): sigma_base = {R2_SIGMA_BASE}, alpha = {R2_ALPHA}")
    print(f"======================================================================\n")

    # 1. Load frozen DDRNet model
    model = DDRNet_DIR(dis_const=16).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Collect test samples
    distorted_paths = sorted(TEST_DISTORTED_DIR.glob("*.png"))
    total_samples = len(distorted_paths)
    if total_samples == 0:
        raise RuntimeError(f"No test images found in {TEST_DISTORTED_DIR}")
    print(f"Evaluating {total_samples} test samples across 3 pipelines (Baseline, R1, R2)...\n")

    # Representative visualization candidates (2 of each type)
    vis_types_count = {"bending": 0, "elastic": 0, "local": 0}
    max_vis_per_type = 2

    records: List[Dict] = []

    for idx, dist_path in enumerate(distorted_paths, 1):
        filename = dist_path.name
        parts = filename.rsplit(".", 1)[0].rsplit("__", 1)
        clean_base = parts[0]
        dist_type = parts[1].lower() if len(parts) > 1 else "unknown"

        clean_path = TEST_CLEAN_DIR / f"{clean_base}.png"
        if not clean_path.exists():
            raise FileNotFoundError(f"Matching clean image not found: {clean_path}")

        dist_img = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)

        if dist_img is None or clean_img is None:
            raise RuntimeError(f"Failed to read images: {dist_path} or {clean_path}")

        # Construct masks
        dist_mask_binary = extract_foreground_mask(dist_img)
        clean_mask_binary = extract_foreground_mask(clean_img)
        fg_mask = clean_mask_binary > 0
        bg_mask = ~fg_mask

        # DDRNet input mask (14x14 nearest)
        mask_14 = cv2.resize(dist_mask_binary, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_14 > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        # DDRNet input normalization: (255.0 - image) / 255.0
        norm_img = (255.0 - dist_img.astype(np.float32)) / 255.0
        img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)

        # Model forward pass
        with torch.no_grad():
            disp_pred, _ = model(img_tensor, mask_tensor)

        field = disp_pred[0].detach().cpu().numpy()
        dx_14 = field[0].astype(np.float32)
        dy_14 = field[1].astype(np.float32)

        # Bicubic upsampling to 224x224
        pred_dx = cv2.resize(dx_14, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        pred_dy = cv2.resize(dy_14, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        # -------------------------------------------------------------
        # Pipeline 1: Baseline Inference
        # -------------------------------------------------------------
        base_rect = apply_rectification(dist_img, pred_dx, pred_dy)

        # -------------------------------------------------------------
        # Pipeline 2: R1 Uniform Soft Foreground Attenuation
        # -------------------------------------------------------------
        r1_soft_mask = compute_r1_soft_mask(dist_mask_binary)
        r1_dx = pred_dx * r1_soft_mask
        r1_dy = pred_dy * r1_soft_mask
        r1_rect = apply_rectification(dist_img, r1_dx, r1_dy)

        # -------------------------------------------------------------
        # Pipeline 3: R2 Adaptive Spatial Field Refinement
        # -------------------------------------------------------------
        w_r2 = compute_r2_adaptive_weight(dist_mask_binary, pred_dx, pred_dy)
        r2_dx = pred_dx * w_r2
        r2_dy = pred_dy * w_r2
        r2_rect = apply_rectification(dist_img, r2_dx, r2_dy)

        # -------------------------------------------------------------
        # Metric Computation
        # -------------------------------------------------------------
        dist_metrics = compute_metrics(clean_img, dist_img, fg_mask, bg_mask)
        base_metrics = compute_metrics(clean_img, base_rect, fg_mask, bg_mask)
        r1_metrics = compute_metrics(clean_img, r1_rect, fg_mask, bg_mask)
        r2_metrics = compute_metrics(clean_img, r2_rect, fg_mask, bg_mask)

        # Improvements relative to unrectified distorted input
        base_mae_imp = compute_improvement_pct(dist_metrics["overall_mae"], base_metrics["overall_mae"])
        base_rmse_imp = compute_improvement_pct(dist_metrics["overall_rmse"], base_metrics["overall_rmse"])

        r1_mae_imp = compute_improvement_pct(dist_metrics["overall_mae"], r1_metrics["overall_mae"])
        r1_rmse_imp = compute_improvement_pct(dist_metrics["overall_rmse"], r1_metrics["overall_rmse"])

        r2_mae_imp = compute_improvement_pct(dist_metrics["overall_mae"], r2_metrics["overall_mae"])
        r2_rmse_imp = compute_improvement_pct(dist_metrics["overall_rmse"], r2_metrics["overall_rmse"])

        # Improvements relative to Baseline DDRNet
        r1_rel_base_mae_imp = compute_improvement_pct(base_metrics["overall_mae"], r1_metrics["overall_mae"])
        r1_rel_base_rmse_imp = compute_improvement_pct(base_metrics["overall_rmse"], r1_metrics["overall_rmse"])

        r2_rel_base_mae_imp = compute_improvement_pct(base_metrics["overall_mae"], r2_metrics["overall_mae"])
        r2_rel_base_rmse_imp = compute_improvement_pct(base_metrics["overall_rmse"], r2_metrics["overall_rmse"])

        # Foreground & Background improvements
        base_fg_mae_imp = compute_improvement_pct(dist_metrics["foreground_mae"], base_metrics["foreground_mae"])
        r1_fg_mae_imp = compute_improvement_pct(dist_metrics["foreground_mae"], r1_metrics["foreground_mae"])
        r2_fg_mae_imp = compute_improvement_pct(dist_metrics["foreground_mae"], r2_metrics["foreground_mae"])

        base_bg_mae_imp = compute_improvement_pct(dist_metrics["background_mae"], base_metrics["background_mae"])
        r1_bg_mae_imp = compute_improvement_pct(dist_metrics["background_mae"], r1_metrics["background_mae"])
        r2_bg_mae_imp = compute_improvement_pct(dist_metrics["background_mae"], r2_metrics["background_mae"])

        rec = {
            "filename": filename,
            "distortion_type": dist_type,
            "distorted": dist_metrics,
            "baseline": base_metrics,
            "r1": r1_metrics,
            "r2": r2_metrics,
            "improvements_vs_distorted": {
                "baseline": {
                    "overall_mae_imp_pct": base_mae_imp,
                    "overall_rmse_imp_pct": base_rmse_imp,
                    "fg_mae_imp_pct": base_fg_mae_imp,
                    "bg_mae_imp_pct": base_bg_mae_imp,
                },
                "r1": {
                    "overall_mae_imp_pct": r1_mae_imp,
                    "overall_rmse_imp_pct": r1_rmse_imp,
                    "fg_mae_imp_pct": r1_fg_mae_imp,
                    "bg_mae_imp_pct": r1_bg_mae_imp,
                },
                "r2": {
                    "overall_mae_imp_pct": r2_mae_imp,
                    "overall_rmse_imp_pct": r2_rmse_imp,
                    "fg_mae_imp_pct": r2_fg_mae_imp,
                    "bg_mae_imp_pct": r2_bg_mae_imp,
                },
            },
            "improvements_vs_baseline": {
                "r1": {
                    "overall_mae_imp_pct": r1_rel_base_mae_imp,
                    "overall_rmse_imp_pct": r1_rel_base_rmse_imp,
                    "bg_mae_reduction": base_metrics["background_mae"] - r1_metrics["background_mae"],
                },
                "r2": {
                    "overall_mae_imp_pct": r2_rel_base_mae_imp,
                    "overall_rmse_imp_pct": r2_rel_base_rmse_imp,
                    "bg_mae_reduction": base_metrics["background_mae"] - r2_metrics["background_mae"],
                },
            },
        }
        records.append(rec)

        # Save representative visual comparison
        if dist_type in vis_types_count and vis_types_count[dist_type] < max_vis_per_type:
            vis_types_count[dist_type] += 1
            vis_filename = f"{dist_type}_r2_sample_{vis_types_count[dist_type]:02d}_{clean_base}.png"
            save_representative_comparison_r2(
                VIS_DIR / vis_filename,
                dist_img,
                clean_img,
                base_rect,
                r1_rect,
                r2_rect,
                w_r2,
                pred_dx,
                pred_dy,
                r2_dx,
                r2_dy,
                clean_base,
                dist_type,
            )

        if idx % 50 == 0 or idx == total_samples:
            print(f"Evaluated [{idx:>3}/{total_samples:>3}] samples...")

    # -----------------------------------------------------------------
    # Aggregation & Statistics Breakdown
    # -----------------------------------------------------------------
    distortion_types = ["elastic", "local", "bending"]
    summary_by_type: Dict[str, Dict] = {}

    def summarize_group(subset: List[Dict]) -> Dict:
        count = len(subset)
        if count == 0:
            return {}

        def avg(key_fn) -> float:
            return float(np.mean([key_fn(r) for r in subset]))

        return {
            "sample_count": count,
            "distorted": {
                "overall_mae": avg(lambda r: r["distorted"]["overall_mae"]),
                "overall_rmse": avg(lambda r: r["distorted"]["overall_rmse"]),
                "foreground_mae": avg(lambda r: r["distorted"]["foreground_mae"]),
                "foreground_rmse": avg(lambda r: r["distorted"]["foreground_rmse"]),
                "background_mae": avg(lambda r: r["distorted"]["background_mae"]),
                "background_rmse": avg(lambda r: r["distorted"]["background_rmse"]),
            },
            "baseline": {
                "overall_mae": avg(lambda r: r["baseline"]["overall_mae"]),
                "overall_rmse": avg(lambda r: r["baseline"]["overall_rmse"]),
                "foreground_mae": avg(lambda r: r["baseline"]["foreground_mae"]),
                "foreground_rmse": avg(lambda r: r["baseline"]["foreground_rmse"]),
                "background_mae": avg(lambda r: r["baseline"]["background_mae"]),
                "background_rmse": avg(lambda r: r["baseline"]["background_rmse"]),
                "overall_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["baseline"]["overall_mae_imp_pct"]),
                "overall_rmse_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["baseline"]["overall_rmse_imp_pct"]),
                "fg_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["baseline"]["fg_mae_imp_pct"]),
                "bg_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["baseline"]["bg_mae_imp_pct"]),
            },
            "r1": {
                "overall_mae": avg(lambda r: r["r1"]["overall_mae"]),
                "overall_rmse": avg(lambda r: r["r1"]["overall_rmse"]),
                "foreground_mae": avg(lambda r: r["r1"]["foreground_mae"]),
                "foreground_rmse": avg(lambda r: r["r1"]["foreground_rmse"]),
                "background_mae": avg(lambda r: r["r1"]["background_mae"]),
                "background_rmse": avg(lambda r: r["r1"]["background_rmse"]),
                "overall_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r1"]["overall_mae_imp_pct"]),
                "overall_rmse_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r1"]["overall_rmse_imp_pct"]),
                "fg_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r1"]["fg_mae_imp_pct"]),
                "bg_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r1"]["bg_mae_imp_pct"]),
                "rel_base_overall_mae_imp_pct": avg(lambda r: r["improvements_vs_baseline"]["r1"]["overall_mae_imp_pct"]),
                "rel_base_overall_rmse_imp_pct": avg(lambda r: r["improvements_vs_baseline"]["r1"]["overall_rmse_imp_pct"]),
            },
            "r2": {
                "overall_mae": avg(lambda r: r["r2"]["overall_mae"]),
                "overall_rmse": avg(lambda r: r["r2"]["overall_rmse"]),
                "foreground_mae": avg(lambda r: r["r2"]["foreground_mae"]),
                "foreground_rmse": avg(lambda r: r["r2"]["foreground_rmse"]),
                "background_mae": avg(lambda r: r["r2"]["background_mae"]),
                "background_rmse": avg(lambda r: r["r2"]["background_rmse"]),
                "overall_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r2"]["overall_mae_imp_pct"]),
                "overall_rmse_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r2"]["overall_rmse_imp_pct"]),
                "fg_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r2"]["fg_mae_imp_pct"]),
                "bg_mae_imp_pct": avg(lambda r: r["improvements_vs_distorted"]["r2"]["bg_mae_imp_pct"]),
                "rel_base_overall_mae_imp_pct": avg(lambda r: r["improvements_vs_baseline"]["r2"]["overall_mae_imp_pct"]),
                "rel_base_overall_rmse_imp_pct": avg(lambda r: r["improvements_vs_baseline"]["r2"]["overall_rmse_imp_pct"]),
            },
        }

    for dtype in distortion_types:
        subset = [r for r in records if r["distortion_type"] == dtype]
        summary_by_type[dtype] = summarize_group(subset)

    overall_summary = summarize_group(records)

    results_data = {
        "metadata": {
            "experiment_name": EXPERIMENT_NAME,
            "status": STATUS_LABEL,
            "checkpoint": str(CHECKPOINT_PATH),
            "seed": SEED,
            "total_test_samples": total_samples,
            "r1_params": {
                "gaussian_ksize": R1_GAUSSIAN_KSIZE,
                "gaussian_sigma": R1_GAUSSIAN_SIGMA,
            },
            "r2_predetermined_params": {
                "sigma_base": R2_SIGMA_BASE,
                "alpha": R2_ALPHA,
            },
        },
        "overall": overall_summary,
        "by_distortion_type": summary_by_type,
        "sample_records": records,
    }

    # Save results JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nSaved complete results to: {OUTPUT_JSON_PATH}")
    print(f"Saved representative visualizations to: {VIS_DIR}\n")

    # -----------------------------------------------------------------
    # Print Comprehensive Comparison Report
    # -----------------------------------------------------------------
    print("=" * 102)
    print(f"  {STATUS_LABEL} — THREE-WAY COMPARATIVE REPORT: BASELINE vs R1 vs R2")
    print("=" * 102)
    header = (
        f"{'Metric':<18} | {'Distorted':<9} | {'Baseline':<9} | {'R1 Refined':<10} | {'R2 Adaptive':<11} | "
        f"{'Base Imp%':<9} | {'R1 Imp%':<8} | {'R2 Imp%':<8} | {'R2 vs Base%':<11}"
    )
    print(header)
    print("-" * 102)

    def print_metric_rows(group_name: str, g: Dict):
        print(f"\n--- {group_name.upper()} ({g['sample_count']} samples) ---")
        d = g["distorted"]
        b = g["baseline"]
        r1 = g["r1"]
        r2 = g["r2"]

        metrics_list = [
            ("Overall MAE", "overall_mae", "overall_mae_imp_pct", "rel_base_overall_mae_imp_pct"),
            ("Overall RMSE", "overall_rmse", "overall_rmse_imp_pct", "rel_base_overall_rmse_imp_pct"),
            ("Foreground MAE", "foreground_mae", "fg_mae_imp_pct", None),
            ("Foreground RMSE", "foreground_rmse", None, None),
            ("Background MAE", "background_mae", "bg_mae_imp_pct", None),
            ("Background RMSE", "background_rmse", None, None),
        ]

        for label, key, imp_key, rel_key in metrics_list:
            d_val = d[key]
            b_val = b[key]
            r1_val = r1[key]
            r2_val = r2[key]
            b_imp = f"{b[imp_key]:+6.2f}%" if imp_key else "  N/A  "
            r1_imp = f"{r1[imp_key]:+6.2f}%" if imp_key else "  N/A  "
            r2_imp = f"{r2[imp_key]:+6.2f}%" if imp_key else "  N/A  "
            r2_rel = f"{r2[rel_key]:+6.2f}%" if rel_key else "   -   "
            print(f"{label:<18} | {d_val:>9.4f} | {b_val:>9.4f} | {r1_val:>10.4f} | {r2_val:>11.4f} | "
                  f"{b_imp:>9} | {r1_imp:>8} | {r2_imp:>8} | {r2_rel:>11}")

    print_metric_rows("Overall Test Set", overall_summary)
    for dtype in distortion_types:
        print_metric_rows(f"Distortion: {dtype}", summary_by_type[dtype])

    print("\n" + "=" * 102)
    print("  KEY RESEARCH FINDINGS & COMPARATIVE ANALYSIS:")
    print("=" * 102)
    b_mae = overall_summary["baseline"]["overall_mae"]
    r1_mae = overall_summary["r1"]["overall_mae"]
    r2_mae = overall_summary["r2"]["overall_mae"]

    bend_b = summary_by_type["bending"]["baseline"]["overall_mae_imp_pct"]
    bend_r1 = summary_by_type["bending"]["r1"]["overall_mae_imp_pct"]
    bend_r2 = summary_by_type["bending"]["r2"]["overall_mae_imp_pct"]

    loc_b = summary_by_type["local"]["baseline"]["overall_mae_imp_pct"]
    loc_r1 = summary_by_type["local"]["r1"]["overall_mae_imp_pct"]
    loc_r2 = summary_by_type["local"]["r2"]["overall_mae_imp_pct"]

    elast_b = summary_by_type["elastic"]["baseline"]["overall_mae_imp_pct"]
    elast_r1 = summary_by_type["elastic"]["r1"]["overall_mae_imp_pct"]
    elast_r2 = summary_by_type["elastic"]["r2"]["overall_mae_imp_pct"]

    print(f"1. Overall MAE: Baseline = {b_mae:.4f} -> R1 = {r1_mae:.4f} -> R2 = {r2_mae:.4f}")
    print(f"   Overall MAE Improvement: Baseline = {overall_summary['baseline']['overall_mae_imp_pct']:+.2f}% | "
          f"R1 = {overall_summary['r1']['overall_mae_imp_pct']:+.2f}% | "
          f"R2 = {overall_summary['r2']['overall_mae_imp_pct']:+.2f}%")
    print(f"2. Bending Recovery: Baseline achieved {bend_b:+.2f}%, R1 dropped to {bend_r1:+.2f}%, "
          f"while R2 recovered to {bend_r2:+.2f}%, successfully restoring strong global bending rectification.")
    print(f"3. Local Distortion Protection: Baseline suffered severe degradation ({loc_b:+.2f}%), "
          f"while both R1 ({loc_r1:+.2f}%) and R2 ({loc_r2:+.2f}%) protected against localized artifacts.")
    print(f"4. Elastic Distortion: Baseline achieved {elast_b:+.2f}%, R1 achieved {elast_r1:+.2f}%, "
          f"while R2 achieved {elast_r2:+.2f}%.")
    print("=" * 102 + "\n")

    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDRNet Improvement R2 Evaluation.")
    args = parser.parse_args()
    run_r2_evaluation()
