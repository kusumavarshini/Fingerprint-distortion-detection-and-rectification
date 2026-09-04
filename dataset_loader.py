"""
Dataset Loader -- Fingerprint Binary Classification
=====================================================
Loads the training dataset from data/train_distorted/ for binary
classification:  CLEAN (0) vs DISTORTED (1).

All three distortion types (elastic, local, bending) map to class 1.

Each sample carries metadata (file path, family, member, distortion type)
so predictions can later be analysed per distortion type.

Usage:
    from dataset_loader import load_train_dataset

    dataset = load_train_dataset()
    # dataset["images"]   -> np.ndarray  (N, 224, 224)  float32 [0, 1]
    # dataset["labels"]   -> np.ndarray  (N,)           int      {0, 1}
    # dataset["metadata"] -> list[dict]  per-sample info

Or run directly:
    python dataset_loader.py
to print a summary report and verify integrity.
"""

import os
import sys
import json
import cv2
import numpy as np

# -- Configuration -----------------------------------------------------------
SEED     = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "train_distorted")
CLEAN_DIR = os.path.join(DATA_ROOT, "clean")
DIST_DIR  = os.path.join(DATA_ROOT, "distorted")

LABEL_CLEAN     = 0
LABEL_DISTORTED = 1
CLASS_NAMES      = {0: "CLEAN", 1: "DISTORTED"}

IMG_SIZE = 224


# ============================================================================
#  CORE LOADER
# ============================================================================

def _parse_distortion_type(filename):
    """Extract distortion type from filename like FM001_F1_elastic.png."""
    stem = os.path.splitext(filename)[0]       # FM001_F1_elastic
    suffix = stem.rsplit("_", 1)[-1]           # elastic
    if suffix in ("elastic", "local", "bending"):
        return suffix
    return "clean"


def _collect_entries(root_dir, label):
    """
    Walk a clean/ or distorted/ directory and return a list of dicts:
      {path, label, family, member, distortion_type, filename}
    """
    entries = []
    for family in sorted(os.listdir(root_dir)):
        fam_path = os.path.join(root_dir, family)
        if not os.path.isdir(fam_path) or not family.startswith("FAMILY-"):
            continue
        for member in sorted(os.listdir(fam_path)):
            mem_path = os.path.join(fam_path, member)
            if not os.path.isdir(mem_path):
                continue
            for fname in sorted(os.listdir(mem_path)):
                if not fname.lower().endswith(".png"):
                    continue
                entries.append({
                    "path": os.path.join(mem_path, fname),
                    "label": label,
                    "family": family,
                    "member": member,
                    "distortion_type": _parse_distortion_type(fname),
                    "filename": fname,
                })
    return entries


def load_train_dataset(verbose=False):
    """
    Load the full training dataset.

    Returns
    -------
    dict with keys:
        images   : np.ndarray, shape (N, 224, 224), dtype float32, range [0,1]
        labels   : np.ndarray, shape (N,), dtype int32, values {0, 1}
        metadata : list[dict], one dict per sample with keys:
                   path, family, member, distortion_type, filename, label
    """
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Training data not found at {DATA_ROOT}")

    # -- Collect all entries --------------------------------------------------
    clean_entries = _collect_entries(CLEAN_DIR, LABEL_CLEAN)
    dist_entries  = _collect_entries(DIST_DIR, LABEL_DISTORTED)
    all_entries   = clean_entries + dist_entries

    if verbose:
        print(f"Collected {len(clean_entries)} clean + "
              f"{len(dist_entries)} distorted = {len(all_entries)} total")

    # -- Reproducible shuffle -------------------------------------------------
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(all_entries))
    rng.shuffle(indices)
    all_entries = [all_entries[i] for i in indices]

    # -- Load images ----------------------------------------------------------
    n = len(all_entries)
    images = np.empty((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    labels = np.empty(n, dtype=np.int32)
    errors = []

    for i, entry in enumerate(all_entries):
        img = cv2.imread(entry["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors.append(f"UNREADABLE: {entry['path']}")
            images[i] = 0.0
            labels[i] = entry["label"]
            continue
        if img.shape != (IMG_SIZE, IMG_SIZE):
            errors.append(f"BAD SHAPE {img.shape}: {entry['path']}")
        # Normalise to [0, 1]
        images[i] = img.astype(np.float32) / 255.0
        labels[i] = entry["label"]

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{n} ...")

    # -- Build metadata (without numpy arrays) --------------------------------
    metadata = []
    for entry in all_entries:
        metadata.append({
            "path": os.path.relpath(entry["path"], BASE_DIR).replace("\\", "/"),
            "family": entry["family"],
            "member": entry["member"],
            "distortion_type": entry["distortion_type"],
            "filename": entry["filename"],
            "label": int(entry["label"]),
        })

    dataset = {
        "images": images,
        "labels": labels,
        "metadata": metadata,
        "errors": errors,
    }
    return dataset


# ============================================================================
#  SUMMARY & VERIFICATION (standalone execution)
# ============================================================================

def _print_summary(dataset):
    """Print and save a dataset summary report."""
    images   = dataset["images"]
    labels   = dataset["labels"]
    metadata = dataset["metadata"]
    errors   = dataset["errors"]

    n = len(labels)
    n_clean = int((labels == LABEL_CLEAN).sum())
    n_dist  = int((labels == LABEL_DISTORTED).sum())

    # Count per distortion type
    type_counts = {}
    for m in metadata:
        dt = m["distortion_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1

    sep = "=" * 56
    lines = [
        sep,
        "  TRAINING DATASET SUMMARY",
        sep,
        f"  Source       : {os.path.relpath(DATA_ROOT, BASE_DIR)}/",
        f"  Shuffle seed : {SEED}",
        "",
        f"  {'Metric':<28} {'Value':>12}",
        f"  {'-'*28} {'-'*12}",
        f"  {'Total images':<28} {n:>12}",
        f"  {'Clean (class 0)':<28} {n_clean:>12}",
        f"  {'Distorted (class 1)':<28} {n_dist:>12}",
    ]
    for dt in ["elastic", "local", "bending"]:
        c = type_counts.get(dt, 0)
        lines.append(f"    {'- ' + dt:<26} {c:>12}")

    lines.extend([
        "",
        f"  {'Image dimensions':<28} {'224 x 224':>12}",
        f"  {'Channels':<28} {'1 (gray)':>12}",
        f"  {'Dtype':<28} {'float32':>12}",
        f"  {'Pixel range':<28} {'[0, 1]':>12}",
        "",
        f"  {'Class distribution':<28}",
        f"    {'CLEAN  (0)':<26} {n_clean/n*100:>10.1f}%",
        f"    {'DISTORTED (1)':<26} {n_dist/n*100:>10.1f}%",
        "",
    ])

    # Verification
    checks = []
    checks.append(("Total == 4200",     n == 4200))
    checks.append(("Clean == 1050",     n_clean == 1050))
    checks.append(("Distorted == 3150", n_dist == 3150))
    checks.append(("Elastic == 1050",   type_counts.get("elastic", 0) == 1050))
    checks.append(("Local == 1050",     type_counts.get("local", 0) == 1050))
    checks.append(("Bending == 1050",   type_counts.get("bending", 0) == 1050))
    checks.append(("No read errors",    len(errors) == 0))
    checks.append(("All float32 [0,1]",
                    images.dtype == np.float32
                    and float(images.min()) >= 0.0
                    and float(images.max()) <= 1.0))
    checks.append(("Shape (N,224,224)",  images.shape == (n, 224, 224)))

    lines.append("  VERIFICATION")
    lines.append(f"  {'-'*40}")
    all_pass = True
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        lines.append(f"    [{status}] {desc}")

    lines.extend(["", sep])
    if all_pass:
        lines.append("  ALL CHECKS PASSED")
    else:
        lines.append("  SOME CHECKS FAILED")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)

    # Save report
    report_path = os.path.join(DATA_ROOT, "dataset_loader_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n  Report saved -> {os.path.relpath(report_path, BASE_DIR)}")

    return all_pass


def main():
    print("Loading training dataset ...\n")
    dataset = load_train_dataset(verbose=True)
    print()
    ok = _print_summary(dataset)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
