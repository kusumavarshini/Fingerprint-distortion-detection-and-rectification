"""Generate DDRNet_DIR rectified fingerprint images.

Defaults write a new DDRNet result directory and deliberately leave the
earlier ``data/rectified_test`` images untouched. Those images were created
with the previous FingerprintRectificationNet checkpoint and must not be
reported as DDRNet_DIR output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("DDRNet").resolve()))
from models.DDRNet_DIR import DDRNet_DIR


DEFAULT_CHECKPOINT = Path("experiments/ddrnet_baseline/best_model.pth")
DEFAULT_INPUT = Path("data/rectification_test/distorted")
DEFAULT_OUTPUT = Path("data/rectified_ddrnet_test")
IMAGE_SIZE = 224
GRID_SIZE = 14


def create_mask(image: np.ndarray) -> np.ndarray:
    """Reproduce the mask construction used to prepare DDRNet training data."""
    _, mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def rectify_image(image: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    return cv2.remap(
        image,
        grid_x - dx,
        grid_y - dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def load_mask(image: np.ndarray, image_path: Path, mask_dir: Path | None) -> np.ndarray:
    if mask_dir is not None:
        mask_path = mask_dir / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read DDRNet mask: {mask_path}")
    else:
        mask = create_mask(image)
    mask = cv2.resize(mask, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DDRNet_DIR rectified fingerprints.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files (sanity check).")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    files = sorted(args.input_dir.glob("*.png"))
    if not files:
        raise RuntimeError(f"No PNG files found in {args.input_dir}")
    if args.limit is not None:
        files = files[: args.limit]
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDRNet_DIR(dis_const=16).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input images: {len(files)}")

    with torch.no_grad():
        for start in range(0, len(files), args.batch_size):
            batch_paths = files[start : start + args.batch_size]
            images = []
            masks = []
            for image_path in batch_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError(f"Could not read image: {image_path}")
                if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
                    raise RuntimeError(f"Expected 224x224 image, got {image.shape}: {image_path}")
                images.append(image)
                masks.append(load_mask(image, image_path, args.mask_dir))

            image_tensor = torch.from_numpy(
                np.stack([(255.0 - image.astype(np.float32)) / 255.0 for image in images])
            ).unsqueeze(1).to(device)
            mask_tensor = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(device)
            displacement, _ = model(image_tensor, mask_tensor)
            fields = displacement.cpu().numpy()

            for image_path, image, field in zip(batch_paths, images, fields):
                dx = cv2.resize(field[0], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                dy = cv2.resize(field[1], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                rectified = rectify_image(image, dx, dy)
                if not cv2.imwrite(str(args.output_dir / image_path.name), rectified):
                    raise RuntimeError(f"Could not write output for {image_path.name}")

            completed = min(start + len(batch_paths), len(files))
            if completed % 25 == 0 or completed == len(files):
                print(f"Processed {completed}/{len(files)}")

    print(f"DDRNet rectification complete: {args.output_dir}")


if __name__ == "__main__":
    main()
