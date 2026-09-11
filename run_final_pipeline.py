"""Run the frozen classifier → DDRNet_DIR → MINDTCT/Bozorth3 demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from afis import MindtctExtractor

from model import FingerprintCNN
from generate_rectified_test import create_mask, rectify_image

sys.path.insert(0, str(Path("DDRNet").resolve()))
from models.DDRNet_DIR import DDRNet_DIR


IMAGE_SIZE = 224
CLASSIFIER_CHECKPOINT = Path("experiments/experiment_5/best_model_exp5.pth")
DDRNET_CHECKPOINT = Path("experiments/ddrnet_baseline/best_model.pth")
FINAL_RESULTS = Path("experiments/final_results/final_results.json")
# This is the Experiment 5 validation-selected classifier threshold, recorded
# in its frozen evaluation file. It is not adjusted by this demonstration.
CLASSIFIER_THRESHOLD = 0.18


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def state_dict(checkpoint):
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def load_classifier(device):
    if not CLASSIFIER_CHECKPOINT.is_file():
        raise FileNotFoundError(CLASSIFIER_CHECKPOINT)
    model = FingerprintCNN().to(device)
    model.load_state_dict(state_dict(torch.load(CLASSIFIER_CHECKPOINT, map_location=device, weights_only=False)))
    return model.eval()


def classify(image, model, device):
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor).squeeze()).item())
    return probability, probability >= CLASSIFIER_THRESHOLD


def rectify(image, device):
    if not DDRNET_CHECKPOINT.is_file():
        raise FileNotFoundError(DDRNET_CHECKPOINT)
    model = DDRNet_DIR(dis_const=16).to(device)
    checkpoint = torch.load(DDRNET_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_tensor = torch.from_numpy((255.0 - image.astype(np.float32)) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    mask = cv2.resize(create_mask(image), (14, 14), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        field, _ = model(image_tensor, mask_tensor)
    field = field[0].cpu().numpy()
    dx = cv2.resize(field[0], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(field[1], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    return rectify_image(image, dx, dy)


def matching_threshold(rectified: bool) -> float:
    if not FINAL_RESULTS.is_file():
        raise FileNotFoundError(f"Run evaluate_matching_rectification.py first: {FINAL_RESULTS}")
    values = json.loads(FINAL_RESULTS.read_text(encoding="utf-8"))["thresholds"]
    key = "rectified_overall" if rectified else "clean_to_clean"
    return float(values[key]["threshold"])


def main():
    parser = argparse.ArgumentParser(description="Fingerprint distortion detection, DDRNet rectification, and matching demo.")
    parser.add_argument("--input", required=True, type=Path, help="Fingerprint to assess.")
    parser.add_argument("--reference", required=True, type=Path, help="Clean/original reference fingerprint.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/final_results/demo_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image, reference = load_gray(args.input), load_gray(args.reference)
    probability, is_distorted = classify(input_image, load_classifier(device), device)
    if is_distorted:
        candidate = rectify(input_image, device)
        rectification_message = "Distorted fingerprint - DDRNet rectification applied"
        output_image = args.output_dir / "rectified_fingerprint.png"
    else:
        candidate = input_image
        rectification_message = "Clean fingerprint - rectification not required"
        output_image = args.output_dir / "clean_fingerprint_used_for_matching.png"
    if not cv2.imwrite(str(output_image), candidate):
        raise RuntimeError(f"Could not save {output_image}")

    extractor = MindtctExtractor()
    result = extractor.match(extractor.extract_minutiae(candidate), extractor.extract_minutiae(reference), method="bozorth3")
    threshold = matching_threshold(is_distorted)
    report = {
        "input": str(args.input), "reference": str(args.reference), "device": str(device),
        "distortion_prediction": "distorted" if is_distorted else "clean",
        "distortion_probability": probability, "classifier_threshold": CLASSIFIER_THRESHOLD,
        "rectification": rectification_message, "saved_image": str(output_image),
        "matching_method": "MINDTCT + Bozorth3", "matching_score": float(result.score),
        "decision_threshold_from_validation": threshold,
        "match_decision": "match" if result.score >= threshold else "non-match",
    }
    (args.output_dir / "pipeline_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
