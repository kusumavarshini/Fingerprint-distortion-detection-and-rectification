"""
Distortion Severity Estimation Module
======================================

Estimates the severity of fingerprint distortion from a
DDRNet-predicted displacement field.

Severity is measured as the mean displacement magnitude
across all pixels in the predicted field:

    magnitude(x, y) = sqrt(dx^2 + dy^2)
    severity_score  = mean(magnitude)

Categorical severity labels (Low / Moderate / High) are
derived from the training-set displacement distribution
using tercile boundaries (P33.33 and P66.67 of the global
mean_disp across all 3,150 training samples).

Training-set analysis source:
    experiments/distortion_analysis/distortion_metrics.csv
    experiments/distortion_analysis/distortion_summary.json

Training data: 3,150 samples (1,050 each: elastic, local, bending)

    Global mean_disp P33.33 = 4.2982 px
    Global mean_disp P66.67 = 5.9275 px

These thresholds were computed exclusively from training
data. No validation or test labels were used.

IMPORTANT: The severity score is computed from the
*predicted* displacement field. At inference time, no clean
reference is needed.

Usage:
    # As a module
    from severity import (
        calculate_severity_statistics,
        calculate_severity_score,
        classify_severity,
    )

    stats = calculate_severity_statistics(dx, dy)
    score = calculate_severity_score(dx, dy)
    level = classify_severity(score)

    # From the command line
    python severity.py --field path/to/displacement.npz
    python severity.py --field path/to/displacement.npz --dx-key dx --dy-key dy
"""

import argparse
import json
import os
import sys

import numpy as np


# ============================================================
# TRAINING-DERIVED SEVERITY THRESHOLDS
# ============================================================
#
# Source: experiments/distortion_analysis/distortion_metrics.csv
#
# Computed from 3,150 training samples (1,050 per type).
# Method: global tercile boundaries of per-sample mean
# displacement magnitude.
#
#   P33.33 = 4.2982 px  (boundary: Low / Moderate)
#   P66.67 = 5.9275 px  (boundary: Moderate / High)
#
# Distribution within each type at these thresholds:
#
#   Elastic: 1.1% Low, 30.7% Moderate, 68.2% High
#   Local:   73.3% Low, 26.7% Moderate, 0.0% High
#   Bending: 25.5% Low, 42.7% Moderate, 31.8% High
#
# Note: categorical labels correlate strongly with distortion
# type because different synthetic distortion functions
# produce different displacement magnitudes by design.

SEVERITY_THRESHOLD_LOW_MODERATE = 4.2982
SEVERITY_THRESHOLD_MODERATE_HIGH = 5.9275

SEVERITY_THRESHOLDS = {
    "low_moderate_boundary": SEVERITY_THRESHOLD_LOW_MODERATE,
    "moderate_high_boundary": SEVERITY_THRESHOLD_MODERATE_HIGH,
    "derivation": "training-set terciles (P33.33, P66.67)",
    "training_samples": 3150,
    "training_source": (
        "experiments/distortion_analysis/distortion_metrics.csv"
    ),
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def calculate_displacement_magnitude(dx, dy):
    """
    Calculate per-pixel displacement magnitude.

    Parameters
    ----------
    dx : np.ndarray
        Horizontal displacement field (H x W).
    dy : np.ndarray
        Vertical displacement field (H x W).

    Returns
    -------
    np.ndarray
        Per-pixel magnitude: sqrt(dx^2 + dy^2).
    """

    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)

    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    return magnitude


def calculate_severity_statistics(dx, dy):
    """
    Calculate severity statistics from a displacement field.

    Parameters
    ----------
    dx : np.ndarray
        Horizontal displacement field (H x W).
    dy : np.ndarray
        Vertical displacement field (H x W).

    Returns
    -------
    dict
        Dictionary with keys:
            mean_displacement
            median_displacement
            p95_displacement
            max_displacement
            std_displacement
            p90_displacement
    """

    magnitude = calculate_displacement_magnitude(dx, dy)

    stats = {
        "mean_displacement": float(np.mean(magnitude)),
        "median_displacement": float(np.median(magnitude)),
        "p90_displacement": float(np.percentile(magnitude, 90)),
        "p95_displacement": float(np.percentile(magnitude, 95)),
        "max_displacement": float(np.max(magnitude)),
        "std_displacement": float(np.std(magnitude)),
    }

    return stats


def calculate_severity_score(dx, dy):
    """
    Calculate the continuous severity score.

    The severity score is the mean displacement magnitude
    across all pixels in the field.

    Parameters
    ----------
    dx : np.ndarray
        Horizontal displacement field (H x W).
    dy : np.ndarray
        Vertical displacement field (H x W).

    Returns
    -------
    float
        Mean displacement magnitude in pixels.
    """

    magnitude = calculate_displacement_magnitude(dx, dy)

    return float(np.mean(magnitude))


def classify_severity(
    score,
    low_moderate=None,
    moderate_high=None,
):
    """
    Classify the severity score into a categorical label.

    Parameters
    ----------
    score : float
        Continuous severity score (mean displacement).
    low_moderate : float, optional
        Boundary between Low and Moderate.
        Defaults to training-set P33.33.
    moderate_high : float, optional
        Boundary between Moderate and High.
        Defaults to training-set P66.67.

    Returns
    -------
    str
        "Low", "Moderate", or "High".
    """

    if low_moderate is None:
        low_moderate = SEVERITY_THRESHOLD_LOW_MODERATE

    if moderate_high is None:
        moderate_high = SEVERITY_THRESHOLD_MODERATE_HIGH

    if score < low_moderate:
        return "Low"
    elif score < moderate_high:
        return "Moderate"
    else:
        return "High"


def get_training_thresholds():
    """
    Return the training-derived severity thresholds.

    Returns
    -------
    dict
        Dictionary containing threshold values and
        derivation metadata.
    """

    return dict(SEVERITY_THRESHOLDS)


def format_severity_report(dx, dy):
    """
    Generate a formatted text report for a single
    displacement field.

    Parameters
    ----------
    dx : np.ndarray
        Horizontal displacement field (H x W).
    dy : np.ndarray
        Vertical displacement field (H x W).

    Returns
    -------
    str
        Multi-line formatted report.
    """

    stats = calculate_severity_statistics(dx, dy)
    score = stats["mean_displacement"]
    level = classify_severity(score)

    lines = [
        "Distortion detected",
        f"Mean displacement:    {stats['mean_displacement']:.2f} px",
        f"Median displacement:  {stats['median_displacement']:.2f} px",
        f"P95 displacement:     {stats['p95_displacement']:.2f} px",
        f"Maximum displacement: {stats['max_displacement']:.2f} px",
        f"Severity score:       {score:.2f}",
        f"Severity level:       {level}",
    ]

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate distortion severity from a "
            "DDRNet-predicted displacement field."
        ),
    )

    parser.add_argument(
        "--field",
        required=True,
        type=str,
        help=(
            "Path to an .npz file containing the "
            "displacement field (dx, dy arrays)."
        ),
    )

    parser.add_argument(
        "--dx-key",
        default="dx",
        type=str,
        help=(
            "Key for the dx array in the .npz file. "
            "Default: 'dx'."
        ),
    )

    parser.add_argument(
        "--dy-key",
        default="dy",
        type=str,
        help=(
            "Key for the dy array in the .npz file. "
            "Default: 'dy'."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of text.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.field):
        print(f"ERROR: File not found: {args.field}")
        sys.exit(1)

    data = np.load(args.field)

    if args.dx_key not in data:
        print(
            f"ERROR: Key '{args.dx_key}' not found in "
            f"{args.field}. "
            f"Available keys: {list(data.keys())}"
        )
        sys.exit(1)

    if args.dy_key not in data:
        print(
            f"ERROR: Key '{args.dy_key}' not found in "
            f"{args.field}. "
            f"Available keys: {list(data.keys())}"
        )
        sys.exit(1)

    dx = data[args.dx_key]
    dy = data[args.dy_key]

    if args.json:
        stats = calculate_severity_statistics(dx, dy)
        score = stats["mean_displacement"]
        level = classify_severity(score)

        result = {
            "field_path": args.field,
            "field_shape": list(dx.shape),
            **stats,
            "severity_score": score,
            "severity_level": level,
            "thresholds": get_training_thresholds(),
        }

        print(json.dumps(result, indent=2))

    else:
        report = format_severity_report(dx, dy)
        print(report)


if __name__ == "__main__":
    main()
