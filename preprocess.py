"""
Preprocessing Script — Family Fingerprint Dataset
===================================================
• Reads every PNG recursively from the raw dataset.
• Converts LA / RGBA / L images → single-channel grayscale.
• Resizes to 224 × 224 (CNN input size).
• Normalises pixel values to [0, 1] (float32).
• Saves processed grayscale images as uint8 PNGs in data/processed/,
  preserving the FAMILY-*/MEMBER/ folder hierarchy.
• Prints a summary report at the end.

Nothing in the original dataset is modified.
"""

import os
import sys
import time
import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(
    BASE_DIR, "data",
    "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET"
)
DST_ROOT = os.path.join(BASE_DIR, "data", "processed")

TARGET_SIZE = (224, 224)          # width, height


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_as_grayscale(path: str) -> np.ndarray:
    """Load any PNG (L / LA / RGBA) and return a single-channel uint8 array."""
    # cv2.IMREAD_UNCHANGED preserves all channels including alpha
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read: {path}")

    # Determine channel layout and collapse to grayscale
    if img.ndim == 2:
        # Already single-channel (L mode)
        gray = img
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 2:
            # LA — take the luminance channel, ignore alpha
            gray = img[:, :, 0]
        elif channels == 3:
            # BGR (unlikely for this dataset, but handle it)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            # BGRA / RGBA — convert to grayscale, ignore alpha
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unexpected channel count ({channels}): {path}")
    else:
        raise ValueError(f"Unexpected dimensions ({img.ndim}): {path}")

    return gray   # uint8, single channel


def preprocess(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize to TARGET_SIZE and normalise to [0, 1].

    Returns
    -------
    resized_uint8 : np.ndarray   – 224×224 uint8  (for saving as PNG)
    normalised    : np.ndarray   – 224×224 float32 in [0, 1]
    """
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return resized, normalised


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: source dataset not found at\n  {SRC_ROOT}")
        sys.exit(1)

    print(f"Source : {SRC_ROOT}")
    print(f"Target : {DST_ROOT}")
    print(f"Resize : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}  •  Grayscale  •  Normalised [0,1]\n")

    t0 = time.time()
    processed = 0
    skipped = 0
    errors = []
    norm_min_global = 1.0
    norm_max_global = 0.0

    for family in sorted(os.listdir(SRC_ROOT)):
        family_src = os.path.join(SRC_ROOT, family)
        if not os.path.isdir(family_src):
            continue

        for member in sorted(os.listdir(family_src)):
            member_src = os.path.join(family_src, member)
            if not os.path.isdir(member_src):
                continue

            # Mirror the folder in the destination
            member_dst = os.path.join(DST_ROOT, family, member)
            os.makedirs(member_dst, exist_ok=True)

            for fname in sorted(os.listdir(member_src)):
                if not fname.lower().endswith(".png"):
                    skipped += 1
                    continue

                src_path = os.path.join(member_src, fname)
                dst_path = os.path.join(member_dst, fname)

                try:
                    gray = load_as_grayscale(src_path)
                    resized_uint8, normalised = preprocess(gray)

                    # Track normalisation stats
                    norm_min_global = min(norm_min_global, float(normalised.min()))
                    norm_max_global = max(norm_max_global, float(normalised.max()))

                    # Save as single-channel grayscale PNG
                    cv2.imwrite(dst_path, resized_uint8)
                    processed += 1
                except Exception as exc:
                    errors.append((src_path, str(exc)))

    elapsed = time.time() - t0

    # ── Verify a saved file to confirm dimensions / channels ──────────────
    sample_path = None
    for root, _, files in os.walk(DST_ROOT):
        for f in files:
            if f.lower().endswith(".png"):
                sample_path = os.path.join(root, f)
                break
        if sample_path:
            break

    sample_info = ""
    if sample_path:
        check = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
        h, w = check.shape[:2]
        ch = 1 if check.ndim == 2 else check.shape[2]
        sample_info = (
            f"  Sample file    : {os.path.relpath(sample_path, BASE_DIR)}\n"
            f"  Dimensions     : {w}×{h}\n"
            f"  Channels       : {ch} (grayscale)\n"
            f"  Dtype on disk  : {check.dtype}\n"
            f"  Norm. range    : [{norm_min_global:.4f}, {norm_max_global:.4f}]  (verified in-memory)"
        )

    # ── Report ────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("  PREPROCESSING REPORT")
    print(sep)
    print(f"  Images processed : {processed}")
    print(f"  Non-PNG skipped  : {skipped}")
    print(f"  Errors           : {len(errors)}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print()
    if sample_info:
        print(sample_info)
    if errors:
        print("\n  Errors:")
        for path, msg in errors:
            print(f"    ✗ {os.path.relpath(path, BASE_DIR)} — {msg}")
    print(f"\n  Output saved to  : {os.path.relpath(DST_ROOT, BASE_DIR)}/")
    print(sep)


if __name__ == "__main__":
    main()
