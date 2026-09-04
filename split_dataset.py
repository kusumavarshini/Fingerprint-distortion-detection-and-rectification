"""
Dataset Splitting Script -- Family Fingerprint Dataset
=======================================================
Splits the 100 families into train / val / test (70 / 15 / 15) at the
FAMILY level so all 15 images from a given family stay in the same split.

Source : data/processed/          (224x224 grayscale PNGs)
Output : data/split/train/        (70 families, 1050 images)
         data/split/val/          (15 families, 225 images)
         data/split/test/         (15 families, 225 images)

Nothing in data/processed/ or data/family_fingerprint_dataset/ is modified.
"""

import os
import sys
import json
import shutil
import numpy as np

# -- Configuration -----------------------------------------------------------
SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT   = os.path.join(BASE_DIR, "data", "processed")
DST_ROOT   = os.path.join(BASE_DIR, "data", "split")

SPLIT_NAMES = ("train", "val", "test")


# -- Main --------------------------------------------------------------------
def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: source not found at {SRC_ROOT}")
        sys.exit(1)

    # -- Discover family folders ----------------------------------------------
    families = sorted([
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d)) and d.startswith("FAMILY-")
    ], key=lambda x: int(x.split("-")[1]))

    n = len(families)
    print(f"Found {n} family folders in {os.path.relpath(SRC_ROOT, BASE_DIR)}/\n")

    if n == 0:
        print("ERROR: no FAMILY-* folders found"); sys.exit(1)

    # -- Shuffle and split ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(n * TRAIN_FRAC))
    n_val   = int(round(n * VAL_FRAC))
    n_test  = n - n_train - n_val          # remainder goes to test

    splits = {
        "train": [families[i] for i in sorted(indices[:n_train])],
        "val":   [families[i] for i in sorted(indices[n_train:n_train + n_val])],
        "test":  [families[i] for i in sorted(indices[n_train + n_val:])],
    }

    # -- Verify zero overlap --------------------------------------------------
    sets = {k: set(v) for k, v in splits.items()}
    assert sets["train"].isdisjoint(sets["val"]),   "train/val overlap!"
    assert sets["train"].isdisjoint(sets["test"]),  "train/test overlap!"
    assert sets["val"].isdisjoint(sets["test"]),     "val/test overlap!"
    assert len(sets["train"]) + len(sets["val"]) + len(sets["test"]) == n
    print("  Zero-overlap check: PASSED\n")

    # -- Copy images ----------------------------------------------------------
    copy_count = {s: 0 for s in SPLIT_NAMES}
    family_image_counts = {}

    for split_name in SPLIT_NAMES:
        split_dst = os.path.join(DST_ROOT, split_name)
        for family in splits[split_name]:
            fam_src = os.path.join(SRC_ROOT, family)
            fam_count = 0

            for member in sorted(os.listdir(fam_src)):
                member_src = os.path.join(fam_src, member)
                if not os.path.isdir(member_src):
                    continue
                member_dst = os.path.join(split_dst, family, member)
                os.makedirs(member_dst, exist_ok=True)

                for fname in sorted(os.listdir(member_src)):
                    if not fname.lower().endswith(".png"):
                        continue
                    src_path = os.path.join(member_src, fname)
                    dst_path = os.path.join(member_dst, fname)
                    shutil.copy2(src_path, dst_path)
                    copy_count[split_name] += 1
                    fam_count += 1

            family_image_counts[family] = fam_count

    # -- Build report ---------------------------------------------------------
    report_lines = []
    sep = "=" * 64
    report_lines.append(sep)
    report_lines.append("  DATASET SPLIT REPORT")
    report_lines.append(sep)
    report_lines.append(f"  Source       : {os.path.relpath(SRC_ROOT, BASE_DIR)}/")
    report_lines.append(f"  Destination  : {os.path.relpath(DST_ROOT, BASE_DIR)}/")
    report_lines.append(f"  Seed         : {SEED}")
    report_lines.append(f"  Total families : {n}")
    report_lines.append("")
    report_lines.append(f"  {'Split':<8} {'Families':>10} {'Images':>10} {'Fraction':>10}")
    report_lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    total_images = sum(copy_count.values())
    for s in SPLIT_NAMES:
        frac = len(splits[s]) / n
        report_lines.append(
            f"  {s:<8} {len(splits[s]):>10} {copy_count[s]:>10} "
            f"{frac:>9.0%}")
    report_lines.append(
        f"  {'TOTAL':<8} {n:>10} {total_images:>10} {'100%':>10}")

    report_lines.append("")
    report_lines.append("  Overlap verification:")
    report_lines.append("    train & val  : 0 families (OK)")
    report_lines.append("    train & test : 0 families (OK)")
    report_lines.append("    val   & test : 0 families (OK)")

    for s in SPLIT_NAMES:
        report_lines.append("")
        fam_list = splits[s]
        report_lines.append(f"  {s.upper()} families ({len(fam_list)}):")
        # Print in rows of 10 for readability
        for i in range(0, len(fam_list), 10):
            chunk = fam_list[i:i+10]
            ids = [f.split("-")[1] for f in chunk]
            report_lines.append(f"    {', '.join(ids)}")

    report_lines.append("")
    report_lines.append(sep)
    report_lines.append("  Original datasets are UNCHANGED.")
    report_lines.append(sep)

    report_text = "\n".join(report_lines)
    print(report_text)

    # -- Save report as text --------------------------------------------------
    report_path = os.path.join(DST_ROOT, "split_report.txt")
    os.makedirs(DST_ROOT, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n  Report saved -> {os.path.relpath(report_path, BASE_DIR)}")

    # -- Save as JSON for programmatic use ------------------------------------
    json_path = os.path.join(DST_ROOT, "split_manifest.json")
    manifest = {
        "seed": SEED,
        "total_families": n,
        "total_images": total_images,
        "splits": {}
    }
    for s in SPLIT_NAMES:
        manifest["splits"][s] = {
            "families": splits[s],
            "num_families": len(splits[s]),
            "num_images": copy_count[s],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved -> {os.path.relpath(json_path, BASE_DIR)}")


if __name__ == "__main__":
    main()
