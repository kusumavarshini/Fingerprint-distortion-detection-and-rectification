from pathlib import Path
import csv
import itertools

import cv2
from afis import MindtctExtractor


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path("data/split/test")

OUTPUT_DIR = Path("experiments/matching_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "matching_scores.csv"

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# ============================================================
# Collect test images
# ============================================================

def collect_images():
    images = []

    for family_dir in sorted(TEST_DIR.glob("FAMILY-*")):

        for member_dir in sorted(family_dir.iterdir()):

            if not member_dir.is_dir():
                continue

            for image_path in sorted(member_dir.glob("*.png")):

                images.append({
                    "path": image_path,
                    "family": family_dir.name,
                    "member": member_dir.name,
                    "name": image_path.stem,
                })

    return images


# ============================================================
# Extract MINDTCT templates
# ============================================================

def extract_templates(images):

    extractor = MindtctExtractor()

    templates = {}

    total = len(images)

    # Create CLAHE once and reuse it
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    for i, item in enumerate(images, 1):

        image = cv2.imread(
            str(item["path"]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {item['path']}"
            )

        # ----------------------------------------------------
        # CLAHE preprocessing
        # ----------------------------------------------------
        image = clahe.apply(image)

        # ----------------------------------------------------
        # MINDTCT minutiae extraction
        # ----------------------------------------------------
        template = extractor.extract_minutiae(image)

        templates[str(item["path"])] = template

        if i % 25 == 0 or i == total:
            print(
                f"Extracted {i}/{total} templates"
            )

    return extractor, templates


# ============================================================
# Build genuine and impostor pairs
# ============================================================

def build_pairs(images):

    genuine = []
    impostor = []

    # --------------------------------------------------------
    # Genuine pairs
    #
    # Same family + same member
    # Different impressions
    #
    # Example:
    # FAMILY-13 / CHILD / C1
    # FAMILY-13 / CHILD / C2
    # --------------------------------------------------------

    groups = {}

    for item in images:

        key = (
            item["family"],
            item["member"]
        )

        groups.setdefault(key, []).append(item)

    for group in groups.values():

        for a, b in itertools.combinations(group, 2):

            genuine.append(
                (a, b, 1)
            )

    # --------------------------------------------------------
    # Impostor pairs
    #
    # Same member type but different families
    # One pair per family combination
    # --------------------------------------------------------

    member_groups = {}

    for item in images:

        member_groups.setdefault(
            item["member"],
            []
        ).append(item)

    for member, group in member_groups.items():

        by_family = {}

        for item in group:

            by_family.setdefault(
                item["family"],
                []
            ).append(item)

        families = sorted(by_family)

        for family_a, family_b in itertools.combinations(
            families,
            2
        ):

            # Use the first impression from each family
            a = by_family[family_a][0]
            b = by_family[family_b][0]

            impostor.append(
                (a, b, 0)
            )

    return genuine, impostor


# ============================================================
# Match pairs
# ============================================================

def match_pairs(
    extractor,
    templates,
    pairs,
    label
):

    results = []

    total = len(pairs)

    for i, (a, b, pair_label) in enumerate(
        pairs,
        1
    ):

        template_a = templates[
            str(a["path"])
        ]

        template_b = templates[
            str(b["path"])
        ]

        # ----------------------------------------------------
        # Bozorth3 matching
        # ----------------------------------------------------

        result = extractor.match(
            template_a,
            template_b
        )

        results.append({

            "type": label,

            "label": pair_label,

            "image_a": a["path"].as_posix(),

            "image_b": b["path"].as_posix(),

            "family_a": a["family"],

            "family_b": b["family"],

            "member": a["member"],

            "score": result.score,

            "raw_score": result.raw_score,

            "matched_pairs": result.n_matched_pairs,

            "decision": result.decision,

            "minutiae_a": template_a.n_minutiae,

            "minutiae_b": template_b.n_minutiae,
        })

        if i % 100 == 0 or i == total:

            print(
                f"{label}: matched {i}/{total}"
            )

    return results


# ============================================================
# Print summary
# ============================================================

def summarize(results):

    genuine = [
        r for r in results
        if r["label"] == 1
    ]

    impostor = [
        r for r in results
        if r["label"] == 0
    ]

    print()
    print("=" * 60)
    print("MATCHING BASELINE SUMMARY - CLAHE + MINDTCT + BOZORTH3")
    print("=" * 60)

    print(
        f"Total pairs:      {len(results)}"
    )

    print(
        f"Genuine pairs:    {len(genuine)}"
    )

    print(
        f"Impostor pairs:   {len(impostor)}"
    )

    for name, group in [
        ("Genuine", genuine),
        ("Impostor", impostor),
    ]:

        if not group:
            continue

        scores = [
            r["score"]
            for r in group
        ]

        matched = [
            r["matched_pairs"]
            for r in group
        ]

        print()
        print(f"{name}:")

        print(
            f"  Mean score:       "
            f"{sum(scores) / len(scores):.4f}"
        )

        print(
            f"  Min score:        "
            f"{min(scores):.4f}"
        )

        print(
            f"  Max score:        "
            f"{max(scores):.4f}"
        )

        print(
            f"  Mean matched:     "
            f"{sum(matched) / len(matched):.2f}"
        )

        print(
            f"  Pairs > 0 match:  "
            f"{sum(x > 0 for x in matched)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Collect images
    # --------------------------------------------------------

    print("Collecting test images...")

    images = collect_images()

    print(
        f"Found {len(images)} test images"
    )

    if len(images) == 0:

        raise RuntimeError(
            f"No PNG images found in {TEST_DIR}"
        )

    # --------------------------------------------------------
    # 2. Extract templates
    # --------------------------------------------------------

    print()
    print(
        "Extracting MINDTCT templates with CLAHE..."
    )

    extractor, templates = extract_templates(
        images
    )

    # --------------------------------------------------------
    # 3. Build pairs
    # --------------------------------------------------------

    print()
    print(
        "Building genuine/impostor pairs..."
    )

    genuine, impostor = build_pairs(
        images
    )

    print(
        f"Genuine pairs:  {len(genuine)}"
    )

    print(
        f"Impostor pairs: {len(impostor)}"
    )

    # --------------------------------------------------------
    # 4. Genuine matching
    # --------------------------------------------------------

    print()
    print(
        "Running genuine matching..."
    )

    genuine_results = match_pairs(
        extractor,
        templates,
        genuine,
        "genuine"
    )

    # --------------------------------------------------------
    # 5. Impostor matching
    # --------------------------------------------------------

    print()
    print(
        "Running impostor matching..."
    )

    impostor_results = match_pairs(
        extractor,
        templates,
        impostor,
        "impostor"
    )

    # --------------------------------------------------------
    # 6. Combine results
    # --------------------------------------------------------

    results = (
        genuine_results +
        impostor_results
    )

    # --------------------------------------------------------
    # 7. Save CSV
    # --------------------------------------------------------

    fieldnames = [
        "type",
        "label",
        "image_a",
        "image_b",
        "family_a",
        "family_b",
        "member",
        "score",
        "raw_score",
        "matched_pairs",
        "decision",
        "minutiae_a",
        "minutiae_b",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # --------------------------------------------------------
    # 8. Print summary
    # --------------------------------------------------------

    summarize(results)

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()