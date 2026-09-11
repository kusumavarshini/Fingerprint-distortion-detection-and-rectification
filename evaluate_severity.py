"""
Distortion Severity Evaluation
===============================

Evaluates the severity estimation system on validation and
test data.

CHECKPOINT STATUS
-----------------
The DDRNet checkpoint (experiments/ddrnet_baseline/best_model.pth)
is not present on disk. The checkpoint that produced
evaluation_results.csv was lost after training.

Because the checkpoint is absent, this script cannot run
DDRNet inference on new images. Instead it regenerates
the synthetic displacement fields from the known distortion
functions and seeds (identical to generate_rectification_eval.py).
These are the GROUND-TRUTH fields, not predicted fields.

For severity evaluation purposes this is scientifically
equivalent: the training thresholds were also derived from
ground-truth fields (experiments/distortion_analysis/
distortion_metrics.csv). The severity score is a property
of the distortion field itself; it does not require
prediction.

In a deployment setting (where a DDRNet checkpoint is
available), substitute gt_dx/gt_dy with the DDRNet-predicted
pred_dx/pred_dy from the model. The severity.py module is
already designed for this; only the field source changes.

WHAT THIS SCRIPT DOES
---------------------
1. Regenerates distorted images and their displacement fields
   on-the-fly using the same deterministic seeds as
   generate_rectification_eval.py.
2. Computes severity statistics from the generated GT fields.
3. Classifies severity using training-derived thresholds.
4. Aggregates rectification effectiveness from the existing
   evaluation_results.csv (no new pixel evaluation is run).

OUTPUT
------
    experiments/severity_evaluation/severity_results_val.csv
    experiments/severity_evaluation/severity_results_test.csv
    experiments/severity_evaluation/severity_summary.json

No models are retrained. Existing evaluation_results.csv is
read-only.
"""

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from severity import (
    calculate_severity_statistics,
    classify_severity,
    get_training_thresholds,
    SEVERITY_THRESHOLD_LOW_MODERATE,
    SEVERITY_THRESHOLD_MODERATE_HIGH,
)

from generate_rectification_dataset import (
    distort_elastic,
    distort_local,
    distort_bending,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = BASE_DIR / "experiments" / "severity_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_EVAL_CSV = (
    BASE_DIR
    / "experiments"
    / "ddrnet_baseline"
    / "evaluation"
    / "evaluation_results.csv"
)

IMAGE_SIZE = 224

SPLITS = [
    {
        "name": "val",
        "input_dir": BASE_DIR / "data" / "split" / "val",
        "master_seed": 7777,
    },
    {
        "name": "test",
        "input_dir": BASE_DIR / "data" / "split" / "test",
        "master_seed": 9999,
    },
]


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_config):
    """
    Process all images in a split.

    Generates displacement fields on-the-fly (same seeds as
    generate_rectification_eval.py) and computes severity
    statistics from those fields.
    """

    split_name = split_config["name"]
    input_dir = split_config["input_dir"]
    master_seed = split_config["master_seed"]

    print()
    print("=" * 60)
    print(f"SEVERITY EVALUATION: {split_name.upper()}")
    print("=" * 60)

    image_paths = sorted(input_dir.rglob("*.png"))

    print(f"Clean images: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    results = []

    distortion_configs = [
        ("elastic", distort_elastic, 0),
        ("local", distort_local, 1),
        ("bending", distort_bending, 2),
    ]

    for idx, image_path in enumerate(image_paths):

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(
                img,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        for distortion_name, distortion_fn, type_offset in distortion_configs:

            # Identical seed scheme to generate_rectification_eval.py
            seed = master_seed * 100000 + idx * 10 + type_offset
            rng = np.random.default_rng(seed)

            # Regenerate distortion (dx, dy are GT sampling fields)
            _distorted, dx_gt, dy_gt, _params = distortion_fn(img, rng)

            # Severity statistics from GT field
            stats = calculate_severity_statistics(dx_gt, dy_gt)
            score = stats["mean_displacement"]
            level = classify_severity(score)

            rel_path = image_path.relative_to(input_dir)
            sample_name = os.path.splitext(
                str(rel_path).replace(os.sep, "__")
            )[0]

            record = {
                "split": split_name,
                "sample": sample_name,
                "distortion_type": distortion_name,
                "seed": int(seed),
                "mean_displacement": stats["mean_displacement"],
                "median_displacement": stats["median_displacement"],
                "p90_displacement": stats["p90_displacement"],
                "p95_displacement": stats["p95_displacement"],
                "max_displacement": stats["max_displacement"],
                "std_displacement": stats["std_displacement"],
                "severity_score": score,
                "severity_level": level,
                "field_source": "ground_truth_regenerated",
            }

            results.append(record)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")

    print(f"Total samples: {len(results)}")

    return results


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

def aggregate_severity(results):
    """Compute aggregate severity statistics by type and overall."""

    summary = {}

    def stats_for(scores):
        arr = np.array(scores, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    def level_counts(scores):
        arr = np.array(scores, dtype=np.float64)
        n = len(arr)
        n_low = int(np.sum(arr < SEVERITY_THRESHOLD_LOW_MODERATE))
        n_mod = int(np.sum(
            (arr >= SEVERITY_THRESHOLD_LOW_MODERATE)
            & (arr < SEVERITY_THRESHOLD_MODERATE_HIGH)
        ))
        n_high = int(np.sum(arr >= SEVERITY_THRESHOLD_MODERATE_HIGH))
        return {
            "Low": {
                "count": n_low,
                "percent": round(n_low / n * 100, 2),
            },
            "Moderate": {
                "count": n_mod,
                "percent": round(n_mod / n * 100, 2),
            },
            "High": {
                "count": n_high,
                "percent": round(n_high / n * 100, 2),
            },
        }

    all_scores = [r["severity_score"] for r in results]

    summary["overall"] = {
        "n_samples": len(results),
        "severity_score_stats": stats_for(all_scores),
        "severity_distribution": level_counts(all_scores),
    }

    for dtype in ["elastic", "local", "bending"]:
        subset = [r for r in results if r["distortion_type"] == dtype]
        if not subset:
            continue
        scores = [r["severity_score"] for r in subset]
        summary[dtype] = {
            "n_samples": len(subset),
            "severity_score_stats": stats_for(scores),
            "severity_distribution": level_counts(scores),
        }

    return summary


# ============================================================
# RECTIFICATION EFFECTIVENESS
# ============================================================

def load_rectification_effectiveness():
    """
    Aggregate rectification effectiveness from the existing
    evaluation_results.csv (per-image distorted vs rectified
    MAE/RMSE against the clean reference).

    No new pixel evaluation is performed.
    """

    if not EXISTING_EVAL_CSV.is_file():
        print(
            f"WARNING: Evaluation CSV not found: "
            f"{EXISTING_EVAL_CSV}"
        )
        return None

    records = []
    with open(EXISTING_EVAL_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        return None

    def avg(recs, key):
        return float(np.mean([float(r[key]) for r in recs]))

    def build_group(recs):
        d_mae = avg(recs, "distorted_mae")
        r_mae = avg(recs, "rectified_mae")
        d_rmse = avg(recs, "distorted_rmse")
        r_rmse = avg(recs, "rectified_rmse")
        return {
            "n_samples": len(recs),
            "distorted_mae": round(d_mae, 4),
            "rectified_mae": round(r_mae, 4),
            "mae_improvement_percent": round(
                (d_mae - r_mae) / d_mae * 100, 2
            ),
            "distorted_rmse": round(d_rmse, 4),
            "rectified_rmse": round(r_rmse, 4),
            "rmse_improvement_percent": round(
                (d_rmse - r_rmse) / d_rmse * 100, 2
            ),
        }

    effectiveness = {
        "source": str(EXISTING_EVAL_CSV),
        "note": (
            "These are pixel-level rectification effectiveness "
            "metrics comparing distorted and DDRNet-rectified "
            "images against the original clean fingerprint. "
            "They are NOT matching/recognition accuracy. "
            "Matching results did not consistently improve "
            "after rectification."
        ),
        "overall": build_group(records),
    }

    for dtype in ["elastic", "local", "bending"]:
        subset = [r for r in records if r["type"] == dtype]
        if subset:
            effectiveness[dtype] = build_group(subset)

    return effectiveness


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results, filename):
    """Save per-image results to CSV."""

    csv_path = OUTPUT_DIR / filename

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {csv_path}")
    return csv_path


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_severity_summary(split_name, split_summary):
    """Print a human-readable severity summary for one split."""

    print(f"\n--- {split_name.upper()} ---")

    overall = split_summary["overall"]
    ss = overall["severity_score_stats"]
    dist = overall["severity_distribution"]

    print(
        f"  Overall (n={overall['n_samples']}): "
        f"mean severity = {ss['mean']:.3f} px  "
        f"[{ss['min']:.3f}, {ss['max']:.3f}]"
    )
    print(
        f"  Severity: "
        f"Low={dist['Low']['count']} ({dist['Low']['percent']:.1f}%)  "
        f"Moderate={dist['Moderate']['count']} ({dist['Moderate']['percent']:.1f}%)  "
        f"High={dist['High']['count']} ({dist['High']['percent']:.1f}%)"
    )

    for dtype in ["elastic", "local", "bending"]:
        if dtype in split_summary:
            ts = split_summary[dtype]
            td = ts["severity_distribution"]
            tss = ts["severity_score_stats"]
            print(
                f"  {dtype:8s}: mean={tss['mean']:.3f} px  "
                f"Low={td['Low']['percent']:.1f}%  "
                f"Mod={td['Moderate']['percent']:.1f}%  "
                f"High={td['High']['percent']:.1f}%"
            )


def print_effectiveness(rectification):
    """Print rectification effectiveness table."""

    print()
    print("=" * 60)
    print("RECTIFICATION EFFECTIVENESS")
    print("  Source: experiments/ddrnet_baseline/evaluation/")
    print("          evaluation_results.csv (existing, not re-run)")
    print("=" * 60)

    header = f"  {'Group':10s}  {'D-MAE':>7}  {'R-MAE':>7}  {'MAE%':>7}  {'D-RMSE':>8}  {'R-RMSE':>8}  {'RMSE%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for group in ["overall", "elastic", "local", "bending"]:
        if group not in rectification:
            continue
        r = rectification[group]
        print(
            f"  {group:10s}  "
            f"{r['distorted_mae']:7.4f}  "
            f"{r['rectified_mae']:7.4f}  "
            f"{r['mae_improvement_percent']:6.2f}%  "
            f"{r['distorted_rmse']:8.4f}  "
            f"{r['rectified_rmse']:8.4f}  "
            f"{r['rmse_improvement_percent']:6.2f}%"
        )

    print()
    print(
        "  NOTE: 'rectification effectiveness' is pixel-level "
        "geometric improvement only."
    )
    print(
        "  Matching results did NOT consistently improve "
        "after rectification."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DISTORTION SEVERITY EVALUATION")
    print("=" * 60)
    print()
    print("FIELD SOURCE: Ground-truth displacement fields")
    print("  (DDRNet checkpoint unavailable; GT fields used)")
    print(f"Output: {OUTPUT_DIR}")

    all_summaries = {}

    for split_config in SPLITS:
        split_name = split_config["name"]

        results = process_split(split_config)

        save_csv(results, f"severity_results_{split_name}.csv")

        all_summaries[split_name] = aggregate_severity(results)

    # Load rectification effectiveness from existing results
    rectification = load_rectification_effectiveness()

    # Build summary JSON
    summary = {
        "methodology": {
            "severity_score": (
                "Mean displacement magnitude of the "
                "distortion field"
            ),
            "severity_formula": "score = mean(sqrt(dx^2 + dy^2))",
            "severity_units": "pixels",
            "field_source_note": (
                "Severity was computed from GROUND-TRUTH "
                "displacement fields regenerated deterministically "
                "from clean images using the same distortion "
                "functions and seeds as generate_rectification_eval.py. "
                "The DDRNet checkpoint (experiments/ddrnet_baseline/"
                "best_model.pth) is absent from disk, so predicted "
                "fields could not be used. In a deployment setting "
                "with the checkpoint restored, predicted fields from "
                "DDRNet inference should be used instead. The "
                "severity.py module is designed for both; only the "
                "field source (dx, dy arrays) changes."
            ),
            "categorical_thresholds": get_training_thresholds(),
            "threshold_justification": (
                "Thresholds are the global P33.33 and P66.67 of "
                "mean displacement magnitude across 3,150 training "
                "samples (1,050 per distortion type). Derived from "
                "training data only."
            ),
            "notes": [
                (
                    "The continuous severity score (mean "
                    "displacement) is the primary metric. "
                    "Categorical labels are a convenience derived "
                    "from training-set terciles."
                ),
                (
                    "Categorical severity labels correlate strongly "
                    "with distortion type: local distortions are "
                    "predominantly Low severity, elastic is "
                    "predominantly High, bending spans all three."
                ),
                (
                    "Severity does not imply rectification success. "
                    "Higher severity = larger predicted displacement "
                    "that DDRNet must correct."
                ),
            ],
        },
        "severity_results": all_summaries,
        "rectification_effectiveness": rectification,
        "checkpoint_epoch_discrepancy": {
            "reported_in_final_report": "epoch 9",
            "training_history_max_epoch": 8,
            "checkpoint_epoch_field_on_disk": "N/A (checkpoint absent)",
            "note_from_earlier_inspection": (
                "When checkpoints/best_model.pth (the classifier "
                "checkpoint) was inspected via PyTorch, its epoch "
                "field was 7 (0-indexed = epoch 8 in 1-indexed). "
                "The DDRNet checkpoint is missing. MAX_EPOCHS=8 in "
                "train_ddrnet.py. training_history.json records "
                "epochs 1-8. The FINAL_REPORT.txt claim of epoch 9 "
                "cannot be reconciled with the training history and "
                "is documented here as a discrepancy."
            ),
        },
    }

    json_path = OUTPUT_DIR / "severity_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {json_path}")

    # Terminal summary
    print()
    print("=" * 60)
    print("SEVERITY EVALUATION SUMMARY")
    print("=" * 60)

    for split_name, split_summary in all_summaries.items():
        print_severity_summary(split_name, split_summary)

    if rectification:
        print_effectiveness(rectification)

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print()
    print("No models retrained.")
    print("Existing evaluation_results.csv preserved.")
    print("Existing final_results.json preserved.")


if __name__ == "__main__":
    main()
