"""
Dataset Audit Script
====================
Recursively scans the FAMILY FINGERPRINT DATASET and reports statistics.
This script is READ-ONLY — it does not modify, move, rename, or delete any files.
"""

import os
import sys
from collections import Counter, defaultdict
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET"
)

def main():
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERROR: Dataset root not found at:\n  {DATASET_ROOT}")
        sys.exit(1)

    # ── Counters & accumulators ────────────────────────────────────────────
    family_folders = []
    member_counter = Counter()          # FATHER / MOTHER / CHILD counts
    total_png = 0
    images_per_family_member = {}       # (family, member) -> count
    dimension_counter = Counter()       # (W, H) -> count
    mode_counter = Counter()            # image mode -> count
    corrupted_files = []
    all_filenames = []                  # for duplicate check
    non_png_files = []                  # unexpected files

    # ── Walk the dataset ───────────────────────────────────────────────────
    for family_name in sorted(os.listdir(DATASET_ROOT)):
        family_path = os.path.join(DATASET_ROOT, family_name)
        if not os.path.isdir(family_path):
            continue
        family_folders.append(family_name)

        for member_name in sorted(os.listdir(family_path)):
            member_path = os.path.join(family_path, member_name)
            if not os.path.isdir(member_path):
                continue
            member_upper = member_name.upper()
            member_counter[member_upper] += 1

            png_count = 0
            for fname in sorted(os.listdir(member_path)):
                fpath = os.path.join(member_path, fname)
                if not os.path.isfile(fpath):
                    continue

                all_filenames.append(fname)

                if not fname.lower().endswith(".png"):
                    non_png_files.append(fpath)
                    continue

                png_count += 1
                total_png += 1

                # Try to open and inspect the image
                try:
                    with Image.open(fpath) as img:
                        img.verify()  # lightweight integrity check
                    # Re-open after verify (verify can leave file in odd state)
                    with Image.open(fpath) as img:
                        dimension_counter[(img.width, img.height)] += 1
                        mode_counter[img.mode] += 1
                except Exception as exc:
                    corrupted_files.append((fpath, str(exc)))

            images_per_family_member[(family_name, member_name)] = png_count

    # ── Duplicate filename detection ──────────────────────────────────────
    filename_counts = Counter(all_filenames)
    duplicates = {k: v for k, v in filename_counts.items() if v > 1}

    # ══════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════
    sep = "=" * 70
    print(sep)
    print("  FAMILY FINGERPRINT DATASET — AUDIT REPORT")
    print(sep)

    # 1. Family folders
    print(f"\n1) Number of family folders : {len(family_folders)}")

    # 2. Member folders
    print(f"\n2) Member-role folders:")
    for role in ("FATHER", "MOTHER", "CHILD"):
        print(f"   {role:8s} : {member_counter.get(role, 0)}")
    other_roles = {k: v for k, v in member_counter.items()
                   if k not in ("FATHER", "MOTHER", "CHILD")}
    if other_roles:
        for role, cnt in sorted(other_roles.items()):
            print(f"   {role:8s} : {cnt}  (unexpected role)")

    # 3. Total PNGs
    print(f"\n3) Total PNG images : {total_png}")

    # 4. Images per family / member  (summary table)
    print(f"\n4) Images per family/member (showing first 10 families, then summary):")
    print(f"   {'Family':<12} {'FATHER':>8} {'MOTHER':>8} {'CHILD':>8} {'Total':>8}")
    print(f"   {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    per_family_totals = defaultdict(int)
    for (fam, mem), cnt in images_per_family_member.items():
        per_family_totals[fam] += cnt

    shown = 0
    for fam in sorted(family_folders, key=lambda x: int(x.split("-")[1])):
        f_cnt = images_per_family_member.get((fam, "FATHER"), 0)
        m_cnt = images_per_family_member.get((fam, "MOTHER"), 0)
        c_cnt = images_per_family_member.get((fam, "CHILD"), 0)
        t_cnt = per_family_totals[fam]
        if shown < 10:
            print(f"   {fam:<12} {f_cnt:>8} {m_cnt:>8} {c_cnt:>8} {t_cnt:>8}")
            shown += 1
    if len(family_folders) > 10:
        print(f"   ... ({len(family_folders) - 10} more families omitted for brevity)")

    counts_list = list(per_family_totals.values())
    print(f"\n   Min images per family : {min(counts_list)}")
    print(f"   Max images per family : {max(counts_list)}")
    print(f"   Avg images per family : {sum(counts_list)/len(counts_list):.1f}")

    per_member_counts = list(images_per_family_member.values())
    print(f"   Min images per member : {min(per_member_counts)}")
    print(f"   Max images per member : {max(per_member_counts)}")

    # 5. Image dimensions
    print(f"\n5) Image dimensions (W x H):")
    for (w, h), cnt in dimension_counter.most_common():
        print(f"   {w} x {h}  — {cnt} images")

    # 6. Channels / mode
    print(f"\n6) Image modes (channels):")
    for mode, cnt in mode_counter.most_common():
        channels = len(Image.new(mode, (1, 1)).getbands())
        print(f"   {mode} ({channels} channel{'s' if channels != 1 else ''}) — {cnt} images")

    # 7. Corrupted / unreadable files
    print(f"\n7) Corrupted / unreadable PNG files : {len(corrupted_files)}")
    if corrupted_files:
        for fpath, err in corrupted_files:
            rel = os.path.relpath(fpath, DATASET_ROOT)
            print(f"   ✗ {rel}  —  {err}")

    # 8. Duplicate filenames
    print(f"\n8) Duplicate filenames : {len(duplicates)}")
    if duplicates:
        for fname, cnt in sorted(duplicates.items(), key=lambda x: -x[1])[:20]:
            print(f"   '{fname}' appears {cnt} times")
        if len(duplicates) > 20:
            print(f"   ... ({len(duplicates) - 20} more duplicates omitted)")
    else:
        print("   No duplicate filenames found.")

    # 9. Non-PNG files (bonus)
    if non_png_files:
        print(f"\n9) Non-PNG files found : {len(non_png_files)}")
        for f in non_png_files[:10]:
            print(f"   {os.path.relpath(f, DATASET_ROOT)}")

    print(f"\n{sep}")
    print("  Audit complete. No files were modified.")
    print(sep)


if __name__ == "__main__":
    main()
