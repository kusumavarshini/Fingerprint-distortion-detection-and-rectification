"""Final leakage-safe MINDTCT + Bozorth3 matching evaluation.

Thresholds are fitted only on the family-disjoint validation split (EER
operating point) and then held fixed on test.  The script records every score
so ROC and verification statistics are reproducible.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from afis import MindtctExtractor
from sklearn.metrics import roc_auc_score, roc_curve


OUTPUT = Path("experiments/final_results")
OUTPUT.mkdir(parents=True, exist_ok=True)
DISTORTIONS = ("elastic", "local", "bending")
MEMBERS = ("FATHER", "MOTHER", "CHILD")
SPLITS = {
    "validation": {
        "clean": Path("data/split/val"),
        "distorted": Path("data/rectification_val/distorted"),
        "rectified": Path("data/rectified_ddrnet_val"),
    },
    "test": {
        "clean": Path("data/split/test"),
        "distorted": Path("data/rectification_test/distorted"),
        "rectified": Path("data/rectified_ddrnet_test"),
    },
}


def collect_identities(root: Path) -> dict[tuple[str, str], list[Path]]:
    identities: dict[tuple[str, str], list[Path]] = {}
    for family_dir in sorted(root.glob("FAMILY-*")):
        for member_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            images = sorted(member_dir.glob("*.png"))
            if images:
                identities[(family_dir.name, member_dir.name)] = images
    return identities


def genuine_pairs(identities):
    return [(family, member, a, b) for (family, member), images in sorted(identities.items())
            for a, b in combinations(images, 2)]


def impostor_pairs(identities):
    families = sorted({family for family, _ in identities})
    return [(a_family, member, identities[(a_family, member)][0], b_family, member,
             identities[(b_family, member)][0])
            for member in MEMBERS for a_family, b_family in combinations(families, 2)]


def transformed(clean: Path, root: Path, distortion: str) -> Path:
    return root / f"{clean.parent.parent.name}__{clean.parent.name}__{clean.stem}__{distortion}.png"


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def extract_templates(paths: list[Path], extractor: MindtctExtractor, name: str):
    templates = {}
    for index, path in enumerate(paths, 1):
        templates[path] = extractor.extract_minutiae(read_image(path))
        if index % 100 == 0 or index == len(paths):
            print(f"  {name}: templates {index}/{len(paths)}")
    return templates


def score(extractor, first, second) -> float:
    return float(extractor.match(first, second, method="bozorth3").score)


def evaluate_condition(split: str, condition: str, distortion: str | None, pairs_g, pairs_i,
                       clean_templates, probe_templates, transformed_map, extractor):
    records = []
    for pair_type, pairs in (("genuine", pairs_g), ("impostor", pairs_i)):
        for item in pairs:
            if pair_type == "genuine":
                family, member, probe_clean, reference = item
                reference_family = family
            else:
                family, member, probe_clean, reference_family, _, reference = item
            probe = probe_clean if transformed_map is None else transformed_map[probe_clean]
            records.append({
                "split": split, "condition": condition, "distortion": distortion or "clean",
                "pair_type": pair_type, "label": int(pair_type == "genuine"),
                "probe_clean": probe_clean.as_posix(), "probe": probe.as_posix(),
                "reference_clean": reference.as_posix(), "probe_family": family,
                "reference_family": reference_family, "member": member,
                "score": score(extractor, probe_templates[probe], clean_templates[reference]),
            })
    print(f"  {split} {condition}: {len(records)} scores")
    return records


def eer_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    index = int(np.nanargmin(np.abs(fpr - (1.0 - tpr))))
    return float(thresholds[index]), float((fpr[index] + 1.0 - tpr[index]) / 2.0)


def metrics(frame: pd.DataFrame, threshold: float | None = None):
    labels = frame.label.to_numpy(dtype=int)
    scores = frame.score.to_numpy(dtype=float)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    output = {
        "n_pairs": int(len(frame)), "genuine_count": int(len(genuine)), "impostor_count": int(len(impostor)),
        "genuine_mean": float(genuine.mean()), "genuine_median": float(np.median(genuine)),
        "impostor_mean": float(impostor.mean()), "impostor_median": float(np.median(impostor)),
        "score_separation": float(genuine.mean() - impostor.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    _, output["eer"] = eer_threshold(labels, scores)
    if threshold is not None:
        predicted = scores >= threshold
        tp = int(np.sum((predicted == 1) & (labels == 1)))
        tn = int(np.sum((predicted == 0) & (labels == 0)))
        fp = int(np.sum((predicted == 1) & (labels == 0)))
        fn = int(np.sum((predicted == 0) & (labels == 1)))
        output.update({
            "threshold_from_validation": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": float((tp + tn) / len(labels)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall_tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "far": float(fp / (fp + tn)) if fp + tn else 0.0,
            "frr": float(fn / (fn + tp)) if fn + tp else 0.0,
        })
    return output


def plot_roc(scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"clean": "black", "elastic": "#1f77b4", "local": "#ff7f0e", "bending": "#2ca02c"}
    for condition, part in scores[scores.split == "test"].groupby("condition", sort=False):
        fpr, tpr, _ = roc_curve(part.label, part.score)
        kind = "clean" if condition == "clean_to_clean" else condition.split("_", 1)[1]
        style = "--" if condition.startswith("rectified") else "-"
        ax.plot(fpr, tpr, style, color=colors[kind], label=f"{condition} (AUC={roc_auc_score(part.label, part.score):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set(xlabel="False acceptance rate", ylabel="True acceptance rate", title="Test ROC curves: MINDTCT + Bozorth3", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUTPUT / "roc_curves.png", dpi=200); plt.close(fig)


def plot_distributions(scores: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, distortion in zip(axes, DISTORTIONS):
        for condition, color in ((f"distorted_{distortion}", "#d95f02"), (f"rectified_{distortion}", "#1b9e77")):
            part = scores[(scores.split == "test") & (scores.condition == condition)]
            for label, style, text in ((1, "-", "genuine"), (0, "--", "impostor")):
                ax.hist(part[part.label == label].score, bins=24, range=(0, 1), density=True,
                        histtype="step", linewidth=1.5, linestyle=style, color=color,
                        label=f"{condition.split('_')[0]} {text}")
        ax.set(title=distortion.capitalize(), xlabel="Bozorth3 score", xlim=(0, 1))
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .86)); fig.savefig(OUTPUT / "score_distributions.png", dpi=200); plt.close(fig)


def plot_comparison(summary):
    rows = [row for row in summary if row["split"] == "test" and row["condition"] != "clean_to_clean"]
    labels = [f"{row['distortion']}\n{row['variant']}" for row in rows]
    values = [row["roc_auc"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(rows)), values, color=["#d95f02" if r["variant"] == "distorted" else "#1b9e77" for r in rows])
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1); ax.set_xticks(range(len(rows)), labels); ax.set_ylim(0.45, 0.60)
    ax.set_ylabel("ROC-AUC"); ax.set_title("Test matching performance after DDRNet rectification")
    for bar, value in zip(bars, values): ax.text(bar.get_x() + bar.get_width()/2, value + .002, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT / "matching_performance_comparison.png", dpi=200); plt.close(fig)


def plot_examples():
    clean_root, distorted_root, rectified_root = SPLITS["test"].values()
    clean = sorted(clean_root.glob("FAMILY-*/*/*.png"))[0]
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for row, distortion in enumerate(DISTORTIONS):
        paths = [clean, transformed(clean, distorted_root, distortion), transformed(clean, rectified_root, distortion)]
        for col, path in enumerate(paths):
            axes[row, col].imshow(read_image(path), cmap="gray"); axes[row, col].axis("off")
            if row == 0: axes[row, col].set_title(("Clean", "Distorted", "DDRNet rectified")[col])
        axes[row, 0].set_ylabel(distortion.capitalize())
    fig.tight_layout(); fig.savefig(OUTPUT / "clean_distorted_rectified_examples.png", dpi=200); plt.close(fig)


def write_report(final, minutiae):
    test_rows = [r for r in final["summary"] if r["split"] == "test"]
    lines = [
        "FINGERPRINT DISTORTION DETECTION AND RECTIFICATION — FINAL REPORT", "=" * 72, "",
        "1. Project objective", "Assess whether DDRNet_DIR rectification improves MINDTCT + Bozorth3 verification of synthetically distorted fingerprints against clean fingerprints.", "",
        "2. Dataset and split", "Family Fingerprint Dataset: 100 families, 1,500 images (512x512 source images, processed to 224x224). Family-disjoint split: 70 train / 15 validation / 15 test families (1,050 / 225 / 225 images).", "",
        "3. Frozen distortion classifier", "Experiment 5 test: accuracy 0.8889, precision 0.9534, recall 0.8178, F1 0.8804, ROC-AUC 0.9432 (TN=216, FP=9, FN=41, TP=184).", "",
        "4. DDRNet_DIR rectification", "Frozen best checkpoint: experiments/ddrnet_baseline/best_model.pth (epoch 9). Existing pixel evaluation: MAE improvement 9.54%, RMSE improvement 11.06%; bending RMSE improvement 30.56%.", "",
        "5. Matching methodology", "MINDTCT minutiae extraction and Bozorth3 scoring. Every split has 45 identities, 450 cross-impression genuine pairs and 315 cross-family, same-member-type impostor pairs. Test identities are absent from train and validation. Thresholds use validation EER only; test labels were never used to select a threshold.", "",
        "6. Test matching / verification results", "Condition, AUC, genuine mean, impostor mean, separation, EER, threshold, accuracy, FAR, FRR",
    ]
    for row in test_rows:
        lines.append("{condition}, {roc_auc:.4f}, {genuine_mean:.4f}, {impostor_mean:.4f}, {score_separation:.4f}, {eer:.4f}, {threshold_from_validation:.4f}, {accuracy:.4f}, {far:.4f}, {frr:.4f}".format(**row))
    lines += ["", "7. Distorted versus DDRNet-rectified AUC", "Distortion, distorted AUC, rectified AUC, change, improved?"]
    for item in final["comparisons"]:
        lines.append(f"{item['distortion']}, {item['distorted_auc']:.4f}, {item['rectified_auc']:.4f}, {item['auc_change']:+.4f}, {item['improved']}")
    lines += ["", "8. Minutiae preservation findings"]
    for item in minutiae:
        lines.append(f"{item['distortion_type']}: count clean/distorted/rectified = {item['clean_count']:.2f}/{item['distorted_count']:.2f}/{item['rectified_count']:.2f}; spatial match rate distorted/rectified = {item['distorted_match_rate']:.4f}/{item['rectified_match_rate']:.4f}.")
    lines += ["", "9. Conclusion", "The result is reported as observed. Pixel-level geometric improvement does not by itself establish improved MINDTCT/Bozorth3 recognition. AUC and the validation-calibrated verification measures are the primary recognition evidence.", "", "10. Limitations and future improvements", "Scores are weakly separated and the test protocol has many zero/low Bozorth3 scores, so threshold accuracy is unstable; ROC-AUC is more threshold-independent. Future work should improve minutiae preservation/orientation consistency, train rectification with matcher-aware losses, evaluate additional real distortions, and use a stronger but independently validated matcher."]
    (OUTPUT / "FINAL_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    all_records = []
    integrity = {}
    for split, roots in SPLITS.items():
        identities = collect_identities(roots["clean"])
        clean_paths = sorted(path for images in identities.values() for path in images)
        g_pairs, i_pairs = genuine_pairs(identities), impostor_pairs(identities)
        if not (len(identities) == 45 and len(clean_paths) == 225 and len(g_pairs) == 450 and len(i_pairs) == 315):
            raise RuntimeError(f"Unexpected {split} protocol counts")
        extractor = MindtctExtractor()
        clean_templates = extract_templates(clean_paths, extractor, f"{split} clean")
        all_records += evaluate_condition(split, "clean_to_clean", None, g_pairs, i_pairs, clean_templates, clean_templates, None, extractor)
        integrity[split] = {"identities": len(identities), "clean_images": len(clean_paths), "genuine_pairs": len(g_pairs), "impostor_pairs": len(i_pairs)}
        for distortion in DISTORTIONS:
            distorted_paths = [transformed(path, roots["distorted"], distortion) for path in clean_paths]
            rectified_paths = [transformed(path, roots["rectified"], distortion) for path in clean_paths]
            missing = [str(path) for path in distorted_paths + rectified_paths if not path.is_file()]
            if missing: raise FileNotFoundError(f"Missing mapped files ({len(missing)}): {missing[0]}")
            distorted_templates = extract_templates(distorted_paths, extractor, f"{split} distorted {distortion}")
            rectified_templates = extract_templates(rectified_paths, extractor, f"{split} rectified {distortion}")
            d_map, r_map = dict(zip(clean_paths, distorted_paths)), dict(zip(clean_paths, rectified_paths))
            all_records += evaluate_condition(split, f"distorted_{distortion}", distortion, g_pairs, i_pairs, clean_templates, distorted_templates, d_map, extractor)
            all_records += evaluate_condition(split, f"rectified_{distortion}", distortion, g_pairs, i_pairs, clean_templates, rectified_templates, r_map, extractor)

    scores = pd.DataFrame(all_records)
    scores.to_csv(OUTPUT / "matching_pair_scores.csv", index=False)
    thresholds = {}
    summary = []
    for condition, validation in scores[scores.split == "validation"].groupby("condition", sort=False):
        threshold, val_eer = eer_threshold(validation.label, validation.score)
        thresholds[condition] = {"threshold": threshold, "validation_eer": val_eer}
        summary.append({"split": "validation", "condition": condition, **metrics(validation, threshold)})
        test = scores[(scores.split == "test") & (scores.condition == condition)]
        row = {"split": "test", "condition": condition, **metrics(test, threshold)}
        if condition == "clean_to_clean": row.update({"distortion": "clean", "variant": "clean"})
        else:
            variant, distortion = condition.split("_", 1); row.update({"distortion": distortion, "variant": variant})
        summary.append(row)
    for variant in ("distorted", "rectified"):
        validation = scores[(scores.split == "validation") & scores.condition.str.startswith(variant)]
        threshold, val_eer = eer_threshold(validation.label, validation.score)
        thresholds[f"{variant}_overall"] = {"threshold": threshold, "validation_eer": val_eer}

    minute = pd.read_csv("experiments/minutiae_rectification/minutiae_comparison.csv")
    minute_summary = minute.groupby("distortion_type").mean(numeric_only=True).reset_index().to_dict(orient="records")
    comparisons = []
    test_by_condition = {r["condition"]: r for r in summary if r["split"] == "test"}
    for distortion in DISTORTIONS:
        distorted, rectified = test_by_condition[f"distorted_{distortion}"], test_by_condition[f"rectified_{distortion}"]
        comparisons.append({"distortion": distortion, "distorted_auc": distorted["roc_auc"], "rectified_auc": rectified["roc_auc"], "auc_change": rectified["roc_auc"] - distorted["roc_auc"], "auc_percent_change": 100 * (rectified["roc_auc"] - distorted["roc_auc"]) / distorted["roc_auc"], "improved": bool(rectified["roc_auc"] > distorted["roc_auc"]), "distorted_separation": distorted["score_separation"], "rectified_separation": rectified["score_separation"]})
    final = {"methodology": {"matcher": "MINDTCT + Bozorth3", "threshold_policy": "EER threshold selected on validation only", "integrity_audit": integrity}, "thresholds": thresholds, "summary": summary, "comparisons": comparisons, "minutiae_summary": minute_summary}
    (OUTPUT / "final_results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(OUTPUT / "final_results.csv", index=False)
    write_report(final, minute_summary)
    plot_roc(scores); plot_distributions(scores); plot_comparison(summary); plot_examples()
    print(f"Final outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
