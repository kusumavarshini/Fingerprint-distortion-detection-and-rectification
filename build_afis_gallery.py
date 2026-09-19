from pathlib import Path
import pickle
import time
import os
import cv2

from afis import MindtctExtractor


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

GALLERY_DIR = Path(
    "data/FAMILY FINGERPRINT DATASET/FAMILY FINGERPRINT DATASET"
)

OUTPUT_DIR = Path("experiments/afis_gallery")
OUTPUT_FILE = OUTPUT_DIR / "mindtct_gallery.pkl"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = (512, 512)

# Save progress every 25 images.
CHECKPOINT_SIZE = 25


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_grayscale_512(path: Path):
    """Load an image as grayscale and ensure 512x512 resolution."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    if image.shape != IMAGE_SIZE:
        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_CUBIC,
        )

    return image


# ---------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------

def save_checkpoint(
    checkpoint_number,
    records,
    processed_paths,
    errors,
):
    """
    Save an exact checkpoint atomically.

    processed_paths contains every image successfully processed OR
    attempted in this checkpoint history.

    This allows exact resume after a restart.
    """

    final_path = (
        CHECKPOINT_DIR /
        f"checkpoint_{checkpoint_number:04d}.pkl"
    )

    temp_path = final_path.with_suffix(".tmp")

    checkpoint = {
        "version": 1,
        "records": records,
        "processed_paths": processed_paths,
        "errors": errors,
    }

    print()
    print(
        f"Saving checkpoint "
        f"{checkpoint_number}..."
    )

    with open(temp_path, "wb") as f:
        pickle.dump(
            checkpoint,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Replace only after successful write.
    os.replace(temp_path, final_path)

    print(
        f"CHECKPOINT SAVED: "
        f"{final_path.name}"
    )

    print(
        f"Total completed/attempted: "
        f"{len(processed_paths)}"
    )

    print()


# ---------------------------------------------------------------------
# Load checkpoints
# ---------------------------------------------------------------------

def load_checkpoints():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob("checkpoint_*.pkl")
    )

    all_records = []
    processed_paths = set()
    all_errors = []

    if not checkpoint_files:
        print("No previous checkpoints found.")
        return (
            all_records,
            processed_paths,
            all_errors,
        )

    print(
        f"Found {len(checkpoint_files)} "
        f"existing checkpoint(s)."
    )

    for checkpoint_file in checkpoint_files:

        try:

            with open(
                checkpoint_file,
                "rb",
            ) as f:

                checkpoint = pickle.load(f)

            records = checkpoint.get(
                "records",
                [],
            )

            paths = checkpoint.get(
                "processed_paths",
                [],
            )

            errors = checkpoint.get(
                "errors",
                [],
            )

            all_records.extend(records)

            processed_paths.update(paths)

            all_errors.extend(errors)

            print(
                f"Loaded {checkpoint_file.name}: "
                f"{len(records)} records, "
                f"{len(paths)} processed paths"
            )

        except Exception as exc:

            print(
                f"WARNING: Could not load "
                f"{checkpoint_file.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print()
    print(
        f"Previously processed: "
        f"{len(processed_paths)}"
    )

    print(
        f"Previously successful: "
        f"{len(all_records)}"
    )

    print(
        f"Previously failed: "
        f"{len(all_errors)}"
    )

    print()

    return (
        all_records,
        processed_paths,
        all_errors,
    )


# ---------------------------------------------------------------------
# Final gallery saving
# ---------------------------------------------------------------------

def save_final_gallery(
    gallery,
    success_count,
    error_count,
    elapsed,
):

    output = {
        "version": 1,
        "gallery_dir": str(GALLERY_DIR),
        "image_size": IMAGE_SIZE,
        "extractor": "MindtctExtractor.extract_minutiae",
        "matcher": "bozorth3",
        "records": gallery,
        "num_images": len(gallery),
        "success_count": success_count,
        "error_count": error_count,
    }

    temp_path = OUTPUT_FILE.with_suffix(".tmp")

    print()
    print("=" * 70)
    print("SAVING FINAL GALLERY")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE.resolve()}"
    )

    with open(temp_path, "wb") as f:

        pickle.dump(
            output,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Atomic final replacement.
    os.replace(
        temp_path,
        OUTPUT_FILE,
    )

    file_size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("FINAL GALLERY SAVED")
    print(
        f"File size: {file_size_mb:.2f} MB"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # Validate source dataset
    # -------------------------------------------------------------

    if not GALLERY_DIR.exists():

        raise FileNotFoundError(
            f"Gallery directory does not exist:\n"
            f"{GALLERY_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Find all images
    # -------------------------------------------------------------

    image_paths = sorted(
        GALLERY_DIR.glob(
            "FAMILY-*/*/*.png"
        )
    )

    if not image_paths:

        raise RuntimeError(
            f"No PNG images found under:\n"
            f"{GALLERY_DIR.resolve()}"
        )

    print("=" * 70)
    print("AFIS GALLERY PRECOMPUTATION")
    print("=" * 70)

    print(
        f"Gallery directory : "
        f"{GALLERY_DIR.resolve()}"
    )

    print(
        f"Images found      : "
        f"{len(image_paths)}"
    )

    print(
        f"Output file       : "
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"Checkpoint dir    : "
        f"{CHECKPOINT_DIR.resolve()}"
    )

    print(
        f"Checkpoint size   : "
        f"{CHECKPOINT_SIZE} images"
    )

    print()

    # -------------------------------------------------------------
    # Safety check
    # -------------------------------------------------------------

    if len(image_paths) != 1500:

        raise RuntimeError(
            f"Expected 1500 images, "
            f"but found {len(image_paths)}.\n"
            f"Check the dataset before continuing."
        )

    # -------------------------------------------------------------
    # Load previous progress
    # -------------------------------------------------------------

    (
        gallery,
        processed_paths,
        errors,
    ) = load_checkpoints()

    # -------------------------------------------------------------
    # Remove duplicate records if any
    # -------------------------------------------------------------

    unique_gallery = {}

    for record in gallery:

        path = record["path"]

        unique_gallery[path] = record

    gallery = list(
        unique_gallery.values()
    )

    # -------------------------------------------------------------
    # Start extractor
    # -------------------------------------------------------------

    extractor = MindtctExtractor()

    start_time = time.time()

    current_records = []
    current_processed_paths = []
    current_errors = []

    checkpoint_number = (
        len(list(
            CHECKPOINT_DIR.glob(
                "checkpoint_*.pkl"
            )
        ))
    )

    # -------------------------------------------------------------
    # Process images
    # -------------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        relative_path = image_path.relative_to(
            GALLERY_DIR
        )

        relative_string = str(
            relative_path
        )

        # ---------------------------------------------------------
        # Exact resume check
        # ---------------------------------------------------------

        if relative_string in processed_paths:

            print(
                f"[{index:4d}/{len(image_paths)}] "
                f"{relative_path} "
                f"-> SKIP (already processed)"
            )

            continue

        print(
            f"[{index:4d}/{len(image_paths)}] "
            f"{relative_path}",
            end="",
            flush=True,
        )

        try:

            image = load_grayscale_512(
                image_path
            )

            # Correct afis 1.2.7 API.
            template = (
                extractor.extract_minutiae(
                    image
                )
            )

            # Identity = FAMILY-XX / MEMBER
            identity = "/".join(
                relative_path.parts[:-1]
            )

            record = {
                "path": relative_string,
                "identity": identity,
                "template": template,
                "height": 512,
                "width": 512,
            }

            gallery.append(record)
            current_records.append(record)

            print(
                " -> OK",
                end="",
            )

            try:

                minutiae_count = len(
                    template.minutiae
                )

                print(
                    f" | minutiae="
                    f"{minutiae_count}"
                )

            except Exception:

                print(
                    " | minutiae=unknown"
                )

        except Exception as exc:

            error_record = {
                "path": relative_string,
                "error_type": type(
                    exc
                ).__name__,
                "error": str(exc),
            }

            errors.append(
                error_record
            )

            current_errors.append(
                error_record
            )

            print(
                f" -> ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        # ---------------------------------------------------------
        # Mark this exact image as attempted.
        # ---------------------------------------------------------

        processed_paths.add(
            relative_string
        )

        current_processed_paths.append(
            relative_string
        )

        # ---------------------------------------------------------
        # Checkpoint every CHECKPOINT_SIZE images.
        # ---------------------------------------------------------

        if (
            len(current_processed_paths)
            >= CHECKPOINT_SIZE
        ):

            checkpoint_number += 1

            save_checkpoint(
                checkpoint_number,
                current_records,
                current_processed_paths,
                current_errors,
            )

            current_records = []
            current_processed_paths = []
            current_errors = []

    # -------------------------------------------------------------
    # Save final partial checkpoint
    # -------------------------------------------------------------

    if current_processed_paths:

        checkpoint_number += 1

        save_checkpoint(
            checkpoint_number,
            current_records,
            current_processed_paths,
            current_errors,
        )

    # -------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------

    elapsed = (
        time.time() - start_time
    )

    successful_count = len(gallery)
    error_count = len(errors)

    print()
    print("=" * 70)
    print("EXTRACTION FINISHED")
    print("=" * 70)

    print(
        f"Images found       : "
        f"{len(image_paths)}"
    )

    print(
        f"Images attempted   : "
        f"{len(processed_paths)}"
    )

    print(
        f"Successful         : "
        f"{successful_count}"
    )

    print(
        f"Errors             : "
        f"{error_count}"
    )

    print(
        f"Elapsed            : "
        f"{elapsed / 60:.2f} minutes"
    )

    # -------------------------------------------------------------
    # Require all 1500 images to have been attempted.
    # -------------------------------------------------------------

    if len(processed_paths) != len(image_paths):

        missing = (
            set(
                str(p.relative_to(GALLERY_DIR))
                for p in image_paths
            )
            - processed_paths
        )

        print()
        print(
            "WARNING: Not all images were processed."
        )

        print(
            f"Missing images: "
            f"{len(missing)}"
        )

        return

    # -------------------------------------------------------------
    # Save final gallery
    # -------------------------------------------------------------

    save_final_gallery(
        gallery,
        successful_count,
        error_count,
        elapsed,
    )

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 70)

    if error_count == 0:

        print(
            "SUCCESS: All 1,500 gallery images "
            "were successfully converted "
            "to MINDTCT templates."
        )

    else:

        print(
            f"WARNING: {error_count} image(s) "
            f"failed MINDTCT extraction."
        )

        print(
            "The successful templates were "
            "still saved in the final gallery."
        )


if __name__ == "__main__":
    main()