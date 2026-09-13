"""
V4 Classifier Dataset Generator
=================================
Implements the approved V4 training-data design matrix from:
    experiments/classifier_v4/v4_training_plan.txt

Goal: Eliminate the 22.22% false-positive rate on real clean fingerprints
by decoupling image quality variation from geometric distortion.

Core Principle: "Image quality variation is NOT geometric distortion."

V4 TRAIN  (2,100 images, strict 1:1 balanced)
-------------------------------------------------
Class 0 - CLEAN      (1,050 images)
  525  Standard clean  (from data/split/train/)
  175  Quality-aug: Faint / Low-Contrast
  175  Quality-aug: Over-inked / Saturated
  175  Quality-aug: Partial / Peripheral + Grain

Class 1 - DISTORTED  (1,050 images)
  525  Standard V3 distortions (175 elastic + 175 local + 175 bending)
  525  Quality-augmented V3 distortions (175 each, augmented)

V4 VAL / TEST: frozen from V3 (225 clean + 225 distorted each)
-----------------------------------------------------------------
NOTE: Val and Test sets are NOT regenerated; they are copied
      from data/classifier_v3/val/ and data/classifier_v3/test/.

Deterministic: all operations use fixed numpy seeds.
Does NOT modify V1, V2, V3, DDRNet, or app.py.
"""

import os
import sys
import json
import shutil
import time
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR   = Path(".")
SPLIT_TRAIN = BASE_DIR / "data" / "split" / "train"
V3_TRAIN_CLEAN     = BASE_DIR / "data" / "classifier_v3" / "train" / "clean"
V3_TRAIN_DISTORTED = BASE_DIR / "data" / "classifier_v3" / "train" / "distorted"
V3_VAL_DIR  = BASE_DIR / "data" / "classifier_v3" / "val"
V3_TEST_DIR = BASE_DIR / "data" / "classifier_v3" / "test"

OUTPUT_ROOT  = BASE_DIR / "data" / "classifier_v4"
AUDIT_ROOT   = BASE_DIR / "experiments" / "classifier_v4"

IMAGE_SIZE = 224

# Reproducibility seeds
SEED_STANDARD_CLEAN     = 40000
SEED_FAINT_AUG          = 41000
SEED_SATURATION_AUG     = 42000
SEED_PARTIAL_AUG        = 43000
SEED_DIST_STANDARD_SEL  = 44000
SEED_DIST_QUAL_SEL      = 45000
SEED_DIST_QUAL_AUG      = 46000


# =============================================================================
# FOREGROUND MASK
# =============================================================================

def get_foreground_mask(img_gray):
    _, mask = cv2.threshold(img_gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


# =============================================================================
# QUALITY AUGMENTATION FUNCTIONS
# =============================================================================

def aug_faint(img, rng):
    img_f = img.astype(np.float32) / 255.0
    gamma = rng.uniform(1.6, 2.6)
    img_f = np.power(img_f, gamma)
    boost = rng.uniform(0.04, 0.12)
    img_f = img_f + boost
    sigma = rng.uniform(0.4, 0.9)
    img_f = gaussian_filter(img_f, sigma=sigma)
    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8)


def aug_overinked(img, rng):
    h, w = img.shape
    kernel_size = int(rng.integers(3, 6)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    img_inv = 255 - img
    dilated_inv = cv2.dilate(img_inv, kernel, iterations=1)
    img_dilated = 255 - dilated_inv
    cx = rng.uniform(w * 0.3, w * 0.7)
    cy = rng.uniform(h * 0.3, h * 0.7)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    radius = rng.uniform(30.0, 55.0)
    pool_mask = np.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) /
                       (2.0 * radius**2))
    pool_depth = rng.uniform(0.10, 0.22)
    img_f = img_dilated.astype(np.float32) / 255.0
    img_f = img_f - pool_depth * pool_mask
    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8)


def aug_partial_grain(img, rng):
    h, w = img.shape
    img_f = img.astype(np.float32)
    side = rng.integers(0, 4)
    cut_frac = rng.uniform(0.15, 0.45)
    mask = np.ones((h, w), dtype=np.float32)
    if side == 0:
        mask[:int(h * cut_frac), :] = 0.0
    elif side == 1:
        mask[int(h * (1.0 - cut_frac)):, :] = 0.0
    elif side == 2:
        mask[:, :int(w * cut_frac)] = 0.0
    else:
        mask[:, int(w * (1.0 - cut_frac)):] = 0.0
    mask = gaussian_filter(mask, sigma=rng.uniform(6.0, 14.0))
    img_f = img_f * mask + 255.0 * (1.0 - mask)
    grain_std = rng.uniform(6.0, 18.0)
    noise = rng.normal(0.0, grain_std, (h, w)).astype(np.float32)
    bg_weight = img_f / 255.0
    img_f = img_f + noise * bg_weight
    img_f = np.clip(img_f, 0.0, 255.0)
    return img_f.astype(np.uint8)


def apply_quality_aug(img, aug_type, rng):
    if aug_type == "faint":
        return aug_faint(img, rng)
    elif aug_type == "overinked":
        return aug_overinked(img, rng)
    elif aug_type == "partial":
        return aug_partial_grain(img, rng)
    else:
        raise ValueError(f"Unknown aug_type: {aug_type}")


# =============================================================================
# HELPERS
# =============================================================================

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read: {path}")
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA)
    return img


def save_gray(img, path):
    cv2.imwrite(str(path), img)


# =============================================================================
# GENERATE V4 TRAIN
# =============================================================================

def generate_v4_train():
    print("=" * 70)
    print("V4 TRAIN: Generating 2,100 training samples")
    print("=" * 70)

    train_clean_out = OUTPUT_ROOT / "train" / "clean"
    train_dist_out  = OUTPUT_ROOT / "train" / "distorted"
    train_clean_out.mkdir(parents=True, exist_ok=True)
    train_dist_out.mkdir(parents=True, exist_ok=True)

    metadata = []

    # ------------------------------------------------------------------
    # BLOCK 1: Standard clean (525)
    # ------------------------------------------------------------------
    print("\n[BLOCK 1] Standard clean images (525)")
    all_split_train = sorted(list(SPLIT_TRAIN.glob("**/*.png")))
    assert len(all_split_train) >= 1050, \
        f"Expected 1050 in split/train, found {len(all_split_train)}"

    rng_sel = np.random.default_rng(SEED_STANDARD_CLEAN)
    idxs_standard = sorted(rng_sel.choice(len(all_split_train), size=525,
                                          replace=False).tolist())
    idxs_standard_set = set(idxs_standard)

    for rank, idx in enumerate(idxs_standard):
        src = all_split_train[idx]
        rel = src.relative_to(SPLIT_TRAIN)
        stem = "__".join(rel.with_suffix("").parts)
        dst = train_clean_out / f"std_clean_{rank:04d}__{stem}.png"
        save_gray(load_gray(src), dst)
        metadata.append({
            "split": "train", "label": 0, "label_name": "clean",
            "subtype": "standard_clean", "aug_type": "none",
            "distortion_type": "",
            "source_file": str(src), "output_file": str(dst),
        })
    print(f"  Saved {len(idxs_standard)} standard clean images.")

    # ------------------------------------------------------------------
    # BLOCK 2: Quality-augmented clean hard negatives (525 = 175 x 3)
    # ------------------------------------------------------------------
    print("\n[BLOCK 2] Quality-augmented clean hard negatives (525)")

    idxs_remaining = [i for i in range(len(all_split_train))
                      if i not in idxs_standard_set]
    assert len(idxs_remaining) >= 525, \
        f"Need >= 525 remaining, found {len(idxs_remaining)}"

    rng_part = np.random.default_rng(SEED_FAINT_AUG - 1)
    perm = rng_part.permutation(len(idxs_remaining)).tolist()

    aug_configs = [
        ("faint",     SEED_FAINT_AUG),
        ("overinked", SEED_SATURATION_AUG),
        ("partial",   SEED_PARTIAL_AUG),
    ]

    aug_count = 0
    for aug_slot, (aug_type, aug_seed) in enumerate(aug_configs):
        subset_perm = perm[aug_slot * 175: (aug_slot + 1) * 175]
        chosen_idxs = [idxs_remaining[p] for p in subset_perm]
        rng_aug = np.random.default_rng(aug_seed)
        for global_idx in chosen_idxs:
            src = all_split_train[global_idx]
            rel = src.relative_to(SPLIT_TRAIN)
            stem = "__".join(rel.with_suffix("").parts)
            dst = train_clean_out / f"aug_{aug_type}_{aug_count:04d}__{stem}.png"
            img = load_gray(src)
            save_gray(apply_quality_aug(img, aug_type, rng_aug), dst)
            metadata.append({
                "split": "train", "label": 0, "label_name": "clean",
                "subtype": "quality_augmented_clean", "aug_type": aug_type,
                "distortion_type": "",
                "source_file": str(src), "output_file": str(dst),
            })
            aug_count += 1
        print(f"  [{aug_type}] Saved 175 augmented clean images.")

    print(f"  Total aug clean: {aug_count}")
    n_clean = len(list(train_clean_out.glob("*.png")))
    assert n_clean == 1050, f"Expected 1050 clean, got {n_clean}"
    print(f"  [PASS] Clean total: {n_clean}")

    # ------------------------------------------------------------------
    # BLOCK 3: Standard V3 distortions (525 = 175 x 3 types)
    # ------------------------------------------------------------------
    print("\n[BLOCK 3] Standard V3 distortions (525)")

    v3_all = sorted(list(V3_TRAIN_DISTORTED.glob("*.png")))
    by_type = {
        "elastic": [f for f in v3_all if f.name.endswith("__elastic.png")],
        "local":   [f for f in v3_all if f.name.endswith("__local.png")],
        "bending": [f for f in v3_all if f.name.endswith("__bending.png")],
    }
    for dt, pool in by_type.items():
        assert len(pool) >= 350, f"Need >= 350 {dt}, found {len(pool)}"

    rng_b3 = np.random.default_rng(SEED_DIST_STANDARD_SEL)
    std_sel = {}
    std_count = 0
    for dtype in ["elastic", "local", "bending"]:
        pool = by_type[dtype]
        sel = sorted(rng_b3.choice(len(pool), size=175, replace=False).tolist())
        std_sel[dtype] = set(sel)
        for pidx in sel:
            src = pool[pidx]
            dst = train_dist_out / f"std_dist_{std_count:04d}__{dtype}__{src.name}"
            shutil.copy2(str(src), str(dst))
            metadata.append({
                "split": "train", "label": 1, "label_name": "distorted",
                "subtype": "standard_v3_distorted", "aug_type": "none",
                "distortion_type": dtype,
                "source_file": str(src), "output_file": str(dst),
            })
            std_count += 1
        print(f"  [{dtype}] Saved 175 standard distorted images.")

    print(f"  Total standard distorted: {std_count}")

    # ------------------------------------------------------------------
    # BLOCK 4: Quality-augmented V3 distortions (525 = 175 x 3 types)
    # ------------------------------------------------------------------
    print("\n[BLOCK 4] Quality-augmented V3 distortions (525)")

    rng_b4_sel = np.random.default_rng(SEED_DIST_QUAL_SEL)
    rng_b4_aug = np.random.default_rng(SEED_DIST_QUAL_AUG)
    aug_types_cycle = ["faint", "overinked", "partial"]

    qaug_count = 0
    for dtype in ["elastic", "local", "bending"]:
        pool = by_type[dtype]
        remaining_idxs = [i for i in range(len(pool))
                          if i not in std_sel[dtype]]
        sel = sorted(rng_b4_sel.choice(len(remaining_idxs), size=175,
                                       replace=False).tolist())
        chosen = [remaining_idxs[s] for s in sel]
        for local_rank, pidx in enumerate(chosen):
            src = pool[pidx]
            aug_type = aug_types_cycle[rng_b4_aug.integers(0, 3)]
            dst = train_dist_out / (
                f"qaug_dist_{qaug_count:04d}__{aug_type}__{dtype}__{src.name}")
            img = load_gray(src)
            save_gray(apply_quality_aug(img, aug_type, rng_b4_aug), dst)
            metadata.append({
                "split": "train", "label": 1, "label_name": "distorted",
                "subtype": "quality_augmented_v3_distorted",
                "aug_type": aug_type, "distortion_type": dtype,
                "source_file": str(src), "output_file": str(dst),
            })
            qaug_count += 1
        print(f"  [{dtype}] Saved 175 quality-augmented distorted images.")

    print(f"  Total qaug distorted: {qaug_count}")
    n_dist = len(list(train_dist_out.glob("*.png")))
    assert n_dist == 1050, f"Expected 1050 distorted, got {n_dist}"
    print(f"  [PASS] Distorted total: {n_dist}")

    total = n_clean + n_dist
    assert total == 2100, f"Expected 2100 total, got {total}"
    print(f"\n  [PASS] Train grand total: {total}")

    return metadata


# =============================================================================
# COPY FROZEN VAL/TEST
# =============================================================================

def copy_frozen_val_test():
    print("\n" + "=" * 70)
    print("VAL/TEST: Copying frozen V3 splits into V4")
    print("=" * 70)
    for split_name, v3_dir in [("val", V3_VAL_DIR), ("test", V3_TEST_DIR)]:
        dest = OUTPUT_ROOT / split_name
        if dest.exists():
            clean_n = len(list((dest / "clean").glob("*.png")))
            dist_n  = len(list((dest / "distorted").glob("*.png")))
            print(f"  [{split_name}] Already exists: {clean_n} clean + "
                  f"{dist_n} distorted. Skipping.")
            continue
        shutil.copytree(str(v3_dir), str(dest))
        clean_n = len(list((dest / "clean").glob("*.png")))
        dist_n  = len(list((dest / "distorted").glob("*.png")))
        print(f"  [{split_name}] Copied: {clean_n} clean + {dist_n} distorted "
              f"= {clean_n + dist_n}")


# =============================================================================
# METADATA + AUDIT
# =============================================================================

def save_metadata(metadata):
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["split", "label", "label_name", "subtype", "aug_type",
              "distortion_type", "source_file", "output_file"]
    meta_csv = AUDIT_ROOT / "v4_train_metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    print(f"\n  Metadata CSV: {meta_csv}")

    summary = {
        "total_train":            len(metadata),
        "clean_total":            sum(1 for r in metadata if r["label"] == 0),
        "distorted_total":        sum(1 for r in metadata if r["label"] == 1),
        "clean_standard":         sum(1 for r in metadata if r["subtype"] == "standard_clean"),
        "clean_aug_faint":        sum(1 for r in metadata if r["label"] == 0 and r["aug_type"] == "faint"),
        "clean_aug_overinked":    sum(1 for r in metadata if r["label"] == 0 and r["aug_type"] == "overinked"),
        "clean_aug_partial":      sum(1 for r in metadata if r["label"] == 0 and r["aug_type"] == "partial"),
        "distorted_std_elastic":  sum(1 for r in metadata if r["label"] == 1 and r["subtype"] == "standard_v3_distorted"            and r["distortion_type"] == "elastic"),
        "distorted_std_local":    sum(1 for r in metadata if r["label"] == 1 and r["subtype"] == "standard_v3_distorted"            and r["distortion_type"] == "local"),
        "distorted_std_bending":  sum(1 for r in metadata if r["label"] == 1 and r["subtype"] == "standard_v3_distorted"            and r["distortion_type"] == "bending"),
        "distorted_qaug_elastic": sum(1 for r in metadata if r["label"] == 1 and r["subtype"] == "quality_augmented_v3_distorted"   and r["distortion_type"] == "elastic"),
        "distorted_qaug_local":   sum(1 for r in metadata if r["label"] == 1 and r["subtype"] == "quality_augmented_v3_distorted"   and r["distortion_type"] == "local"),
        "distorted_qaug_bending": sum(1 for r in metadata if r["label"] == 1 and r["subtype"] == "quality_augmented_v3_distorted"   and r["distortion_type"] == "bending"),
    }
    meta_json = AUDIT_ROOT / "v4_generation_summary.json"
    with open(meta_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {meta_json}")
    return summary


def print_audit(summary):
    print("\n" + "=" * 70)
    print("POST-GENERATION AUDIT")
    print("=" * 70)
    rows = [
        ("Class 0 - CLEAN",                summary["clean_total"],            1050),
        ("  Standard clean",               summary["clean_standard"],          525),
        ("  Aug: Faint",                   summary["clean_aug_faint"],         175),
        ("  Aug: Over-inked",              summary["clean_aug_overinked"],      175),
        ("  Aug: Partial+Grain",           summary["clean_aug_partial"],        175),
        ("Class 1 - DISTORTED",            summary["distorted_total"],         1050),
        ("  Std elastic",                  summary["distorted_std_elastic"],    175),
        ("  Std local",                    summary["distorted_std_local"],      175),
        ("  Std bending",                  summary["distorted_std_bending"],    175),
        ("  Qaug elastic",                 summary["distorted_qaug_elastic"],   175),
        ("  Qaug local",                   summary["distorted_qaug_local"],     175),
        ("  Qaug bending",                 summary["distorted_qaug_bending"],   175),
        ("TOTAL",                          summary["total_train"],             2100),
    ]
    print(f"  {'Category':<35} {'Count':>6}  {'Target':>6}  Status")
    print("  " + "-" * 58)
    errors = []
    for label, count, target in rows:
        status = "[PASS]" if count == target else "[FAIL]"
        print(f"  {label:<35} {count:>6}  {target:>6}  {status}")
        if count != target:
            errors.append(f"{label}: got {count}, expected {target}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("\n  [PASS] All counts match the V4 design matrix exactly.")


def check_val_test():
    print("\n" + "=" * 70)
    print("VAL / TEST VERIFICATION")
    print("=" * 70)
    errors = []
    for split, exp_c, exp_d in [("val", 225, 675), ("test", 225, 675)]:
        d = OUTPUT_ROOT / split
        cn = len(list((d / "clean").glob("*.png"))) if (d / "clean").exists() else 0
        dn = len(list((d / "distorted").glob("*.png"))) if (d / "distorted").exists() else 0
        print(f"  [{split}] clean={cn}, distorted={dn}, total={cn+dn}")
        if cn != exp_c: errors.append(f"{split} clean={cn} (expected {exp_c})")
        if dn != exp_d: errors.append(f"{split} distorted={dn} (expected {exp_d})")
    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("  [PASS] Val and test counts verified.")


def generate_visual_samples():
    print("\n[VISUAL] Saving augmentation preview...")
    vis_dir = AUDIT_ROOT / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    all_split = sorted(list(SPLIT_TRAIN.glob("**/*.png")))
    aug_types = ["faint", "overinked", "partial"]
    rng_vis = np.random.default_rng(99999)

    for demo_i, src_path in enumerate(all_split[10:12]):
        img_orig = load_gray(src_path)
        row_imgs   = [img_orig]
        row_labels = ["Original"]
        for at in aug_types:
            row_imgs.append(apply_quality_aug(img_orig.copy(), at, rng_vis))
            row_labels.append(at)

        pad = 4
        grid_w = len(row_imgs) * (IMAGE_SIZE + pad) + pad
        grid_h = IMAGE_SIZE + pad * 2 + 20
        grid = np.full((grid_h, grid_w), 220, dtype=np.uint8)
        for ci, (rim, lbl) in enumerate(zip(row_imgs, row_labels)):
            x0 = pad + ci * (IMAGE_SIZE + pad)
            grid[pad:pad + IMAGE_SIZE, x0:x0 + IMAGE_SIZE] = rim
            cv2.putText(grid, lbl, (x0, pad + IMAGE_SIZE + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1)
        out = vis_dir / f"aug_preview_sample{demo_i}.png"
        cv2.imwrite(str(out), grid)
        print(f"  Saved: {out}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "=" * 70)
    print("V4 CLASSIFIER DATASET GENERATOR")
    print("Ref: experiments/classifier_v4/v4_training_plan.txt")
    print("=" * 70)

    for p in [SPLIT_TRAIN, V3_TRAIN_CLEAN, V3_TRAIN_DISTORTED,
              V3_VAL_DIR, V3_TEST_DIR]:
        assert p.exists(), f"Missing required directory: {p}"

    metadata = generate_v4_train()
    copy_frozen_val_test()
    summary = save_metadata(metadata)
    print_audit(summary)
    check_val_test()
    generate_visual_samples()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("V4 DATASET GENERATION COMPLETE")
    print(f"  Output:  data/classifier_v4/")
    print(f"  Audit:   experiments/classifier_v4/v4_generation_summary.json")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)
    print("\nNext step: retrain the classifier with:")
    print("  python train_classifier_v4.py")
    print("=" * 70)
