import os
import cv2
import numpy as np


DATASET_DIR = r"data\rectification_test"
OUTPUT_DIR = r"experiments\warp_convention_analysis"

NUM_SAMPLES = 10


def warp(image, dx, dy, sign):
    h, w = image.shape

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = grid_x + sign * dx
    map_y = grid_y + sign * dy

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )


def metrics(clean, result):
    diff = result.astype(np.float32) - clean.astype(np.float32)

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    return mae, rmse


os.makedirs(OUTPUT_DIR, exist_ok=True)

distorted_dir = os.path.join(DATASET_DIR, "distorted")
clean_dir = os.path.join(DATASET_DIR, "clean")
fields_dir = os.path.join(DATASET_DIR, "fields")

images = sorted(
    [
        f for f in os.listdir(distorted_dir)
        if f.endswith(".png") and "__elastic.png" in f
    ]
)

images = images[:NUM_SAMPLES]

results = []

print("=" * 90)
print("GROUND-TRUTH WARP CONVENTION TEST")
print("=" * 90)

for index, filename in enumerate(images):

    distorted_path = os.path.join(
        distorted_dir,
        filename
    )

    field_path = os.path.join(
        fields_dir,
        os.path.splitext(filename)[0] + ".npz"
    )

    # Remove distortion suffix to locate clean image.
    clean_filename = filename.rsplit("__", 1)[0] + ".png"

    clean_path = os.path.join(
        clean_dir,
        clean_filename
    )

    distorted = cv2.imread(
        distorted_path,
        cv2.IMREAD_GRAYSCALE
    )

    clean = cv2.imread(
        clean_path,
        cv2.IMREAD_GRAYSCALE
    )

    field = np.load(field_path)

    dx = field["dx"].astype(np.float32)
    dy = field["dy"].astype(np.float32)

    if distorted is None or clean is None:
        print(f"Skipping: {filename}")
        continue

    # Baseline distorted vs clean
    distorted_mae, distorted_rmse = metrics(
        clean,
        distorted
    )

    # Candidate A: + field
    rect_plus = warp(
        distorted,
        dx,
        dy,
        sign=1
    )

    plus_mae, plus_rmse = metrics(
        clean,
        rect_plus
    )

    # Candidate B: - field
    rect_minus = warp(
        distorted,
        dx,
        dy,
        sign=-1
    )

    minus_mae, minus_rmse = metrics(
        clean,
        rect_minus
    )

    results.append(
        {
            "filename": filename,
            "distorted_mae": distorted_mae,
            "distorted_rmse": distorted_rmse,
            "plus_mae": plus_mae,
            "plus_rmse": plus_rmse,
            "minus_mae": minus_mae,
            "minus_rmse": minus_rmse
        }
    )

    # Save first few visualizations
    if index < 3:

        combined = np.hstack(
            [
                clean,
                distorted,
                rect_plus,
                rect_minus
            ]
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{index + 1:02d}_warp_test.png"
        )

        cv2.imwrite(
            output_path,
            combined
        )

    print(
        f"{index + 1:02d}/{len(images)} "
        f"{filename}"
    )


print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

if results:

    distorted_mae = np.mean(
        [r["distorted_mae"] for r in results]
    )

    distorted_rmse = np.mean(
        [r["distorted_rmse"] for r in results]
    )

    plus_mae = np.mean(
        [r["plus_mae"] for r in results]
    )

    plus_rmse = np.mean(
        [r["plus_rmse"] for r in results]
    )

    minus_mae = np.mean(
        [r["minus_mae"] for r in results]
    )

    minus_rmse = np.mean(
        [r["minus_rmse"] for r in results]
    )

    print(f"\nDistorted vs Clean")
    print(f"  MAE:  {distorted_mae:.4f}")
    print(f"  RMSE: {distorted_rmse:.4f}")

    print(f"\nGT field with + sign")
    print(f"  MAE:  {plus_mae:.4f}")
    print(f"  RMSE: {plus_rmse:.4f}")

    print(f"\nGT field with - sign")
    print(f"  MAE:  {minus_mae:.4f}")
    print(f"  RMSE: {minus_rmse:.4f}")

    print("\nImprovement over distorted:")

    print(
        f"  + sign MAE:  "
        f"{(distorted_mae - plus_mae) / distorted_mae * 100:.2f}%"
    )

    print(
        f"  - sign MAE:  "
        f"{(distorted_mae - minus_mae) / distorted_mae * 100:.2f}%"
    )

    print(
        f"  + sign RMSE: "
        f"{(distorted_rmse - plus_rmse) / distorted_rmse * 100:.2f}%"
    )

    print(
        f"  - sign RMSE: "
        f"{(distorted_rmse - minus_rmse) / distorted_rmse * 100:.2f}%"
    )

    if plus_rmse < minus_rmse:
        print("\nBEST CONVENTION: + field")
    else:
        print("\nBEST CONVENTION: - field")

else:
    print("No valid samples found.")


print("\nResults saved to:")
print(OUTPUT_DIR)