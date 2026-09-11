from pathlib import Path
import cv2
import numpy as np
import shutil

# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path("data/rectification_val")
OUTPUT_DIR = Path("data/ddrnet_val")

IMAGE_SIZE = 224
GRID_SIZE = 14

DISTORTED_DIR = SOURCE_DIR / "distorted"
FIELDS_DIR = SOURCE_DIR / "fields"

OUT_IMAGES = OUTPUT_DIR / "images"
OUT_FIELDS = OUTPUT_DIR / "fields"
OUT_MASKS = OUTPUT_DIR / "masks"
OUT_ORIENTATION = OUTPUT_DIR / "orientation"

for directory in [OUT_IMAGES, OUT_FIELDS, OUT_MASKS, OUT_ORIENTATION]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Create DDRNet 14x14 displacement target
# ============================================================

def downsample_field(field):
    """
    Convert the existing 224x224 displacement field
    into the 14x14 representation expected by DDRNet.

    Average pooling is used because each DDRNet pixel
    represents a 16x16 region of the original field.
    """
    field = field.astype(np.float32)

    field_14 = cv2.resize(
        field,
        (GRID_SIZE, GRID_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return field_14


# ============================================================
# Generate orientation target
# ============================================================

def compute_orientation(image):
    """
    Estimate fingerprint ridge orientation from the
    distorted fingerprint.

    Output:
        orientation: 14x14 integer values [0,179]
    """

    image = image.astype(np.float32)

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Fingerprint orientation
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)

    orientation = 0.5 * np.arctan2(
        2.0 * gxy,
        gxx - gyy
    )

    # Convert radians to degrees.
    # DDRNet uses 180 orientation classes.
    orientation = np.degrees(orientation)

    # Convert [-90, 90] → [0, 180)
    orientation = orientation + 90.0

    orientation = np.mod(orientation, 180.0)

    orientation = cv2.resize(
        orientation.astype(np.float32),
        (GRID_SIZE, GRID_SIZE),
        interpolation=cv2.INTER_AREA
    )

    orientation = np.clip(
        np.round(orientation),
        0,
        179
    ).astype(np.uint8)

    return orientation


# ============================================================
# Generate segmentation mask
# ============================================================

def create_mask(image):
    """
    Generate a simple fingerprint foreground mask.

    The actual DDRNet inference pipeline uses
    segmentation_coherence(). We will later replace/
    validate this mask against the reference implementation.
    """

    # Otsu gives a rough foreground estimate.
    _, mask = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise.
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest connected component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return mask


# ============================================================
# Process one sample
# ============================================================

samples = sorted(DISTORTED_DIR.glob("*.png"))

if not samples:
    raise RuntimeError(
        f"No distorted images found in {DISTORTED_DIR}"
    )

processed = 0
errors = 0

for image_path in samples:

    try:
        name = image_path.stem

        field_path = FIELDS_DIR / f"{name}.npz"

        if not field_path.exists():
            print(f"Missing field: {field_path}")
            errors += 1
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"Could not read: {image_path}")
            errors += 1
            continue

        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )

        # ----------------------------------------------------
        # Load existing ground-truth displacement
        # ----------------------------------------------------

        data = np.load(field_path)

        dx = data["dx"].astype(np.float32)
        dy = data["dy"].astype(np.float32)

        if dx.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Unexpected dx shape {dx.shape}"
            )

        if dy.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Unexpected dy shape {dy.shape}"
            )

        # ----------------------------------------------------
        # Convert 224x224 → 14x14
        # ----------------------------------------------------

        dx16 = downsample_field(dx)
        dy16 = downsample_field(dy)

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        orientation = compute_orientation(image)

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        mask = create_mask(image)

        mask16 = cv2.resize(
            mask,
            (GRID_SIZE, GRID_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        mask16 = (mask16 > 0).astype(np.uint8)

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        cv2.imwrite(
            str(OUT_IMAGES / f"{name}.png"),
            image
        )

        # ----------------------------------------------------
        # Save displacement
        # ----------------------------------------------------

        np.savez_compressed(
            OUT_FIELDS / f"{name}.npz",
            dx_rect16=dx16,
            dy_rect16=dy16
        )

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        cv2.imwrite(
            str(OUT_MASKS / f"{name}.png"),
            mask16 * 255
        )

        # ----------------------------------------------------
        # Save orientation
        # ----------------------------------------------------

        np.save(
            OUT_ORIENTATION / f"{name}.npy",
            orientation
        )

        processed += 1

    except Exception as exc:
        print(f"ERROR {image_path.name}: {exc}")
        errors += 1


# ============================================================
# Report
# ============================================================

print()
print("=" * 60)
print("DDRNet DATASET PREPARATION COMPLETE")
print("=" * 60)
print(f"Source samples : {len(samples)}")
print(f"Processed       : {processed}")
print(f"Errors          : {errors}")
print(f"Output          : {OUTPUT_DIR}")
print()
print("Expected per sample:")
print("  Image        : 224 x 224")
print("  Displacement : 14 x 14 x 2")
print("  Mask         : 14 x 14")
print("  Orientation  : 14 x 14, classes 0-179")
print("=" * 60)
