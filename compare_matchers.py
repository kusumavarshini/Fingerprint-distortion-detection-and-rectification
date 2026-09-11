from pathlib import Path
import itertools

import cv2
import numpy as np
from afis import MindtctExtractor
from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path("data/split/test")

MATCHERS = [
    "bozorth3",
    "geometric",
    "mcc",
    "nbis_bozorth3",
    "safis",
]


# ============================================================
# Collect images
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
                })

    return images


# ============================================================
# Extract MINDTCT templates
# ============================================================

def extract_templates(images):

    extractor = MindtctExtractor()

    templates = {}

    total = len(images)

    for i, item in enumerate(images, 1):

        image = cv2.imread(
            str(item["path"]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {item['path']}"
            )

        template = extractor.extract_minutiae(image)

        templates[str(item["path"])] = template

        if i % 25 == 0 or i == total:
            print(
                f"Extracted {i}/{total} templates"
            )

    return extractor, templates


# ============================================================
# Build genuine pairs
# ============================================================

def build_genuine_pairs(images):

    groups = {}

    for item in images:

        key = (
            item["family"],
            item["member"]
        )

        groups.setdefault(key, []).append(item)

    pairs = []

    for group in groups.values():

        for a, b in itertools.combinations(group, 2):

            pairs.append((a, b, 1))

    return pairs


# ============================================================
# Build impostor pairs
# ============================================================

def build_impostor_pairs(images):

    member_groups = {}

    for item in images:

        member_groups.setdefault(
            item["member"],
            []
        ).append(item)

    pairs = []

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

            a = by_family[family_a][0]
            b = by_family[family_b][0]

            pairs.append((a, b, 0))

    return pairs


# ============================================================
# Evaluate one matcher
# ============================================================

def evaluate_matcher(
    extractor,
    templates,
    genuine_pairs,
    impostor_pairs,
    matcher
):

    pairs = genuine_pairs + impostor_pairs

    labels = []
    scores = []

    total = len(pairs)

    print()
    print(f"Testing matcher: {matcher}")

    for i, (a, b, label) in enumerate(
        pairs,
        1
    ):

        template_a = templates[
            str(a["path"])
        ]

        template_b = templates[
            str(b["path"])
        ]

        try:

            result = extractor.match(
                template_a,
                template_b,
                method=matcher
            )

            score = float(result.score)

        except Exception as e:

            print()
            print(
                f"ERROR with matcher '{matcher}'"
            )
            print(
                f"Pair: {a['path']} <-> {b['path']}"
            )
            print(e)

            return None

        labels.append(label)
        scores.append(score)

        if i % 100 == 0 or i == total:

            print(
                f"  Matched {i}/{total}"
            )

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    genuine_scores = scores[
        labels == 1
    ]

    impostor_scores = scores[
        labels == 0
    ]

    auc = roc_auc_score(
        labels,
        scores
    )

    return {
        "matcher": matcher,
        "auc": auc,
        "genuine_mean": genuine_scores.mean(),
        "genuine_min": genuine_scores.min(),
        "genuine_max": genuine_scores.max(),
        "impostor_mean": impostor_scores.mean(),
        "impostor_min": impostor_scores.min(),
        "impostor_max": impostor_scores.max(),
        "genuine_above_05": int(
            np.sum(genuine_scores >= 0.5)
        ),
        "impostor_above_05": int(
            np.sum(impostor_scores >= 0.5)
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("AFIS MATCHER COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    images = collect_images()

    print(
        f"Test images: {len(images)}"
    )

    if len(images) != 225:

        raise RuntimeError(
            f"Expected 225 test images, found {len(images)}"
        )

    # --------------------------------------------------------
    # Extract templates once
    # --------------------------------------------------------

    print()
    print("Extracting MINDTCT templates...")

    extractor, templates = extract_templates(
        images
    )

    # --------------------------------------------------------
    # Build pairs
    # --------------------------------------------------------

    genuine_pairs = build_genuine_pairs(
        images
    )

    impostor_pairs = build_impostor_pairs(
        images
    )

    print()
    print(
        f"Genuine pairs:  {len(genuine_pairs)}"
    )

    print(
        f"Impostor pairs: {len(impostor_pairs)}"
    )

    # --------------------------------------------------------
    # Test all matchers
    # --------------------------------------------------------

    results = []

    for matcher in MATCHERS:

        result = evaluate_matcher(
            extractor,
            templates,
            genuine_pairs,
            impostor_pairs,
            matcher
        )

        if result is not None:

            results.append(result)

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("MATCHER COMPARISON")
    print("=" * 90)

    print(
        f"{'Matcher':<18}"
        f"{'ROC-AUC':>10}"
        f"{'Gen Mean':>12}"
        f"{'Imp Mean':>12}"
        f"{'Gen>=0.5':>12}"
        f"{'Imp>=0.5':>12}"
    )

    print("-" * 90)

    for r in results:

        print(
            f"{r['matcher']:<18}"
            f"{r['auc']:>10.4f}"
            f"{r['genuine_mean']:>12.4f}"
            f"{r['impostor_mean']:>12.4f}"
            f"{r['genuine_above_05']:>12}"
            f"{r['impostor_above_05']:>12}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()