"""Comprehensive verification of evaluation distortion datasets."""
import os
import cv2
import json

BASE = os.path.dirname(os.path.abspath(__file__))

# Load manifest
with open(os.path.join(BASE, "data", "split", "split_manifest.json")) as f:
    manifest = json.load(f)

print("=" * 70)
print("  COMPREHENSIVE VERIFICATION REPORT")
print("=" * 70)

passes = 0
fails = 0


def check(cond, desc):
    global passes, fails
    tag = "PASS" if cond else "FAIL"
    if cond:
        passes += 1
    else:
        fails += 1
    print(f"  [{tag}] {desc}")


for split in ["val", "test"]:
    dist_dir = os.path.join(BASE, "data", f"{split}_distorted")
    src_dir = os.path.join(BASE, "data", "split", split)

    print(f"\n  --- {split.upper()} SPLIT ---")

    # Count clean files
    clean_files = []
    for r, d, files in os.walk(os.path.join(dist_dir, "clean")):
        for fn in files:
            if fn.endswith(".png"):
                clean_files.append(os.path.join(r, fn))
    check(len(clean_files) == 225, f"Clean count = {len(clean_files)} (expected 225)")

    # Count distorted files
    dist_files = []
    for r, d, files in os.walk(os.path.join(dist_dir, "distorted")):
        for fn in files:
            if fn.endswith(".png"):
                dist_files.append(os.path.join(r, fn))
    check(len(dist_files) == 225, f"Distorted count = {len(dist_files)} (expected 225)")

    # Count by type using metadata records (type info is in the metadata)
    meta_path = os.path.join(dist_dir, "distortion_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Handle both formats: list or dict with "records" key
    if isinstance(meta, list):
        records = meta
    else:
        records = meta.get("records", [])

    elastic_count = sum(1 for r in records if r.get("distortion_type") == "elastic")
    local_count = sum(1 for r in records if r.get("distortion_type") == "local")
    bending_count = sum(1 for r in records if r.get("distortion_type") == "bending")

    check(elastic_count == 75, f"Elastic = {elastic_count} (expected 75)")
    check(local_count == 75, f"Local = {local_count} (expected 75)")
    check(bending_count == 75, f"Bending = {bending_count} (expected 75)")
    check(elastic_count + local_count + bending_count == 225,
          f"Type total = {elastic_count + local_count + bending_count} (expected 225)")

    # Check all clean images are 224x224 grayscale
    bad = 0
    for fp in clean_files:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape != (224, 224):
            bad += 1
    ok_count = len(clean_files) - bad
    check(bad == 0, f"All clean images 224x224 grayscale ({ok_count}/{len(clean_files)})")

    # Check all distorted images are 224x224 grayscale
    bad = 0
    for fp in dist_files:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape != (224, 224):
            bad += 1
    ok_count = len(dist_files) - bad
    check(bad == 0, f"All distorted images 224x224 grayscale ({ok_count}/{len(dist_files)})")

    # Check families match source
    src_fams = set(d for d in os.listdir(src_dir) if d.startswith("FAMILY-"))
    clean_fams = set(
        d for d in os.listdir(os.path.join(dist_dir, "clean"))
        if d.startswith("FAMILY-")
    )
    dist_fams = set(
        d for d in os.listdir(os.path.join(dist_dir, "distorted"))
        if d.startswith("FAMILY-")
    )
    check(clean_fams == src_fams,
          f"Clean families = source families ({len(clean_fams)})")
    check(dist_fams == src_fams,
          f"Distorted families = source families ({len(dist_fams)})")

    # Verify metadata record count
    check(len(records) == 225, f"Metadata records = {len(records)} (expected 225)")

    # Verify all distorted output files referenced in metadata actually exist
    missing = 0
    for rec in records:
        # Try both possible path keys
        dpath = rec.get("distorted_output_path") or rec.get("distorted_path", "")
        # Handle both absolute and relative paths
        if not os.path.isabs(dpath):
            dpath = os.path.join(BASE, dpath)
        if not os.path.isfile(dpath):
            missing += 1
    check(missing == 0,
          f"All metadata-referenced distorted files exist ({len(records) - missing}/{len(records)})")


# Cross-split checks
print("\n  --- CROSS-SPLIT CHECKS ---")
val_fams = set(manifest["splits"]["val"]["families"])
test_fams = set(manifest["splits"]["test"]["families"])
train_fams = set(manifest["splits"]["train"]["families"])
check(len(val_fams & test_fams) == 0,
      f"Val/Test family overlap: {len(val_fams & test_fams)}")
check(len(val_fams & train_fams) == 0,
      f"Val/Train family overlap: {len(val_fams & train_fams)}")
check(len(test_fams & train_fams) == 0,
      f"Test/Train family overlap: {len(test_fams & train_fams)}")

# Source integrity
print("\n  --- SOURCE INTEGRITY ---")
for split in ["val", "test", "train"]:
    src = os.path.join(BASE, "data", "split", split)
    cnt = 0
    for r, d, files in os.walk(src):
        cnt += sum(1 for f in files if f.endswith(".png"))
    expected = manifest["splits"][split]["num_images"]
    check(cnt == expected,
          f"{split} source intact: {cnt} images (expected {expected})")

# Training distortions untouched
print("\n  --- TRAINING DISTORTIONS INTEGRITY ---")
train_dist = os.path.join(BASE, "data", "train_distorted")
cnt = 0
for r, d, files in os.walk(os.path.join(train_dist, "clean")):
    cnt += sum(1 for f in files if f.endswith(".png"))
check(cnt == 1050, f"Training clean images untouched: {cnt} (expected 1050)")

cnt = 0
for r, d, files in os.walk(os.path.join(train_dist, "distorted")):
    cnt += sum(1 for f in files if f.endswith(".png"))
check(cnt == 3150, f"Training distorted images untouched: {cnt} (expected 3150)")

# Seed independence
print("\n  --- SEED INDEPENDENCE ---")
with open(os.path.join(BASE, "data", "val_distorted",
                       "distortion_metadata.json")) as f:
    val_meta_raw = json.load(f)
with open(os.path.join(BASE, "data", "test_distorted",
                       "distortion_metadata.json")) as f:
    test_meta_raw = json.load(f)

val_records = val_meta_raw if isinstance(val_meta_raw, list) else val_meta_raw.get("records", [])
test_records = test_meta_raw if isinstance(test_meta_raw, list) else test_meta_raw.get("records", [])

val_seeds = set(r["seed"] for r in val_records)
test_seeds = set(r["seed"] for r in test_records)
overlap = len(val_seeds & test_seeds)
check(overlap == 0, f"Val/Test seed overlap: {overlap} (expected 0)")

# Check master seeds are different from training
# Determine master seed from the records
val_master = val_records[0]["seed"] if val_records else 0
test_master = test_records[0]["seed"] if test_records else 0
# All val seeds should be different from all test seeds
check(val_seeds.isdisjoint(test_seeds),
      "Val and test per-image seeds are fully disjoint")

# Check no seed matches training pattern (42 * 100000 = 4200000 base)
train_seed_base = 42 * 100000
val_from_train = sum(1 for s in val_seeds if s // 100000 == 42)
test_from_train = sum(1 for s in test_seeds if s // 100000 == 42)
check(val_from_train == 0,
      f"Val seeds not in training seed space: {val_from_train} matches (expected 0)")
check(test_from_train == 0,
      f"Test seeds not in training seed space: {test_from_train} matches (expected 0)")

# Final summary
print(f"\n{'=' * 70}")
print("  FINAL SUMMARY")
print("=" * 70)

# Dataset overview table
print("\n  Dataset Overview:")
print(f"  {'Split':<12} {'Clean':>8} {'Distorted':>10} {'Total':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
print(f"  {'Train':<12} {'1050':>8} {'3150':>10} {'4200':>8}")
print(f"  {'Validation':<12} {'225':>8} {'225':>10} {'450':>8}")
print(f"  {'Test':<12} {'225':>8} {'225':>10} {'450':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
print(f"  {'TOTAL':<12} {'1500':>8} {'3600':>10} {'5100':>8}")

print(f"\n  Total checks: {passes + fails}")
print(f"  Passed: {passes}")
print(f"  Failed: {fails}")
if fails == 0:
    print("  ALL CHECKS PASSED")
else:
    print("  SOME CHECKS FAILED — review above")
print("=" * 70)
