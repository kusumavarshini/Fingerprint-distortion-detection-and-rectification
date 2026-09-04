"""
Evaluation Utilities -- Fingerprint Distortion Classifier
==========================================================
Metrics calculation for binary classification with per-distortion-type
breakdowns.

Calculates:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
  - Confusion matrix
  - Per-distortion-type performance (elastic, local, bending, clean)
"""

import numpy as np
from collections import defaultdict


def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)

    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    negative = ~positive
    exp_x = np.exp(x[negative])
    result[negative] = exp_x / (1.0 + exp_x)

    return result


def compute_metrics(labels, logits, threshold=0.5):
    """
    Compute binary classification metrics.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Ground truth labels {0, 1}.
    logits : np.ndarray, shape (N,)
        Raw model logits (before sigmoid).
    threshold : float
        Classification threshold on probabilities.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
                    tp, fp, tn, fn, confusion_matrix
    """
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC-AUC (manual implementation to avoid sklearn dependency)
    roc_auc = _compute_roc_auc(labels, probs)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_samples": len(labels),
    }


def _compute_roc_auc(labels, probs):
    """
    Compute ROC-AUC using the trapezoidal rule.

    Handles the case where all labels are the same class by returning 0.0.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by probability descending
    desc_order = np.argsort(-probs)
    sorted_labels = labels[desc_order]

    # Accumulate TPR/FPR
    tpr_values = []
    fpr_values = []
    tp_cum = 0
    fp_cum = 0

    for label in sorted_labels:
        if label == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        tpr_values.append(tp_cum / n_pos)
        fpr_values.append(fp_cum / n_neg)

    # Prepend (0,0) and compute AUC via trapezoidal rule
    fpr_values = [0.0] + fpr_values
    tpr_values = [0.0] + tpr_values

    auc = 0.0
    for i in range(1, len(fpr_values)):
        auc += (fpr_values[i] - fpr_values[i - 1]) * (tpr_values[i] + tpr_values[i - 1]) / 2

    return auc


def compute_per_type_metrics(labels, logits, distortion_types, threshold=0.5):
    """
    Compute metrics broken down by distortion type.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
    logits : np.ndarray, shape (N,)
    distortion_types : list[str], length N
        e.g. ["clean", "elastic", "local", "bending"]
    threshold : float

    Returns
    -------
    dict : { type_name: metrics_dict }
    """
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    # Group indices by distortion type
    groups = defaultdict(list)
    for i, dt in enumerate(distortion_types):
        groups[dt].append(i)

    results = {}
    for dtype in ["clean", "elastic", "local", "bending"]:
        indices = groups.get(dtype, [])
        if not indices:
            continue

        idx = np.array(indices)
        g_labels = labels[idx]
        g_preds = preds[idx]
        g_probs = probs[idx]

        n = len(g_labels)
        correct = int((g_preds == g_labels).sum())
        accuracy = correct / n if n > 0 else 0.0

        # For clean: correct means predicted 0
        # For distorted types: correct means predicted 1
        if dtype == "clean":
            tp = int(((g_preds == 0) & (g_labels == 0)).sum())  # true negative (correctly clean)
            fp = int(((g_preds == 1) & (g_labels == 0)).sum())  # false positive
            recall_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            results[dtype] = {
                "n_samples": n,
                "accuracy": round(accuracy, 4),
                "correctly_classified": correct,
                "misclassified_as_distorted": int((g_preds == 1).sum()),
                "true_negative_rate": round(recall_val, 4),
            }
        else:
            tp = int(((g_preds == 1) & (g_labels == 1)).sum())  # true positive
            fn = int(((g_preds == 0) & (g_labels == 1)).sum())  # false negative
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results[dtype] = {
                "n_samples": n,
                "accuracy": round(accuracy, 4),
                "correctly_classified": correct,
                "misclassified_as_clean": int((g_preds == 0).sum()),
                "detection_rate": round(recall_val, 4),
            }

    return results


def format_evaluation_report(overall, per_type, split_name=""):
    """
    Format a complete evaluation report as a string.

    Parameters
    ----------
    overall : dict from compute_metrics()
    per_type : dict from compute_per_type_metrics()
    split_name : str, e.g. "Validation" or "Test"

    Returns
    -------
    str
    """
    sep = "=" * 60
    lines = [
        sep,
        f"  EVALUATION REPORT — {split_name.upper()}" if split_name else "  EVALUATION REPORT",
        sep,
        "",
        "  Overall Metrics:",
        f"    Accuracy   : {overall['accuracy']:.4f}",
        f"    Precision  : {overall['precision']:.4f}",
        f"    Recall     : {overall['recall']:.4f}",
        f"    F1-score   : {overall['f1']:.4f}",
        f"    ROC-AUC    : {overall['roc_auc']:.4f}",
        "",
        "  Confusion Matrix:",
        f"                    Predicted",
        f"                  CLEAN  DISTORTED",
        f"    Actual CLEAN   {overall['tn']:>5}  {overall['fp']:>5}",
        f"    Actual DIST    {overall['fn']:>5}  {overall['tp']:>5}",
        "",
        f"    Total samples: {overall['n_samples']}",
        "",
    ]

    # Per-type breakdown
    lines.append("  Per-Distortion-Type Performance:")
    lines.append(f"  {'-'*55}")

    for dtype in ["clean", "elastic", "local", "bending"]:
        if dtype not in per_type:
            continue
        m = per_type[dtype]
        lines.append(f"    {dtype.upper()}")
        lines.append(f"      Samples            : {m['n_samples']}")
        lines.append(f"      Accuracy           : {m['accuracy']:.4f}")
        lines.append(f"      Correct            : {m['correctly_classified']}")
        if dtype == "clean":
            lines.append(f"      Misclassified→DIST : {m['misclassified_as_distorted']}")
            lines.append(f"      True Negative Rate : {m['true_negative_rate']:.4f}")
        else:
            lines.append(f"      Misclassified→CLEAN: {m['misclassified_as_clean']}")
            lines.append(f"      Detection Rate     : {m['detection_rate']:.4f}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)
