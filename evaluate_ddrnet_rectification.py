import os
import glob
import cv2
import numpy as np
import torch
from scipy.ndimage import zoom

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


CHECKPOINT_PATH = r"experiments\ddrnet_baseline\best_model.pth"

DATASETS = {
    "validation": r"data\rectification_val",
    "test": r"data\rectification_test",
}

BATCH_SIZE = 8

torch.set_num_threads(16)
torch.set_num_interop_threads(2)


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = DDRNet_DIR(dis_const=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

    return model


def prepare_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Failed to read: {path}")

    if img.shape != (224, 224):
        img = cv2.resize(
            img,
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )

    # Exact DDRNet preprocessing
    mask = segmentation_coherence(
        img,
        win_size=16,
        stride=8,
    )

    mask16 = zoom(
        mask,
        1 / 16,
        order=0,
    )

    x = ((255 - img) / 255.0).astype(np.float32)

    return img, x, mask16.astype(np.float32)


def rectify_batch(model, images, masks):
    x = torch.from_numpy(
        np.stack(images)
    ).unsqueeze(1)

    m = torch.from_numpy(
        np.stack(masks)
    ).unsqueeze(1)

    with torch.no_grad():
        field, _ = model(x, m)

    field = field.numpy()

    dx16 = field[:, 0]
    dy16 = field[:, 1]

    rectified = []

    h, w = 224, 224

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    for i in range(len(images)):

        dx = cv2.resize(
            dx16[i],
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        dy = cv2.resize(
            dy16[i],
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Established negative-field convention
        map_x = grid_x - dx
        map_y = grid_y - dy

        result = cv2.remap(
            cv2.convertScaleAbs(
                images[i] * 255.0
            ),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        rectified.append(result)

    return rectified


def calculate_metrics(clean, result):
    clean = clean.astype(np.float32)
    result = result.astype(np.float32)

    diff = clean - result

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return mae, rmse


def evaluate_dataset(model, dataset_name, root):

    distorted_dir = os.path.join(root, "distorted")
    clean_dir = os.path.join(root, "clean")

    files = sorted(
        glob.glob(
            os.path.join(
                distorted_dir,
                "*.png",
            )
        )
    )

    print()
    print("=" * 70)
    print(dataset_name.upper())
    print("=" * 70)
    print(f"Samples: {len(files)}")

    results = {
        "all": [],
        "elastic": [],
        "local": [],
        "bending": [],
    }

    for start in range(0, len(files), BATCH_SIZE):

        batch_files = files[
            start:start + BATCH_SIZE
        ]

        images = []
        inputs = []
        masks = []
        cleans = []
        types = []

        for path in batch_files:

            filename = os.path.basename(path)
            stem = os.path.splitext(filename)[0]

            distortion_type = None

            for dtype in [
                "elastic",
                "local",
                "bending",
            ]:
                if stem.endswith("__" + dtype):
                    distortion_type = dtype
                    break

            if distortion_type is None:
                raise RuntimeError(
                    f"Unknown distortion type: {filename}"
                )

            clean_name = (
                stem.rsplit("__", 1)[0]
                + ".png"
            )

            clean_path = os.path.join(
                clean_dir,
                clean_name,
            )

            clean = cv2.imread(
                clean_path,
                cv2.IMREAD_GRAYSCALE,
            )

            if clean is None:
                raise RuntimeError(
                    f"Failed to read clean image: {clean_path}"
                )

            img, x, mask = prepare_image(path)

            images.append(img)
            inputs.append(x)
            masks.append(mask)
            cleans.append(clean)
            types.append(distortion_type)

        rectified_images = rectify_batch(
            model,
            inputs,
            masks,
        )

        for i in range(len(batch_files)):

            clean = cleans[i]
            distorted = images[i]
            rectified = rectified_images[i]

            d_mae, d_rmse = calculate_metrics(
                clean,
                distorted,
            )

            r_mae, r_rmse = calculate_metrics(
                clean,
                rectified,
            )

            value = (
                d_mae,
                d_rmse,
                r_mae,
                r_rmse,
            )

            results["all"].append(value)
            results[types[i]].append(value)

        done = min(
            start + BATCH_SIZE,
            len(files),
        )

        if done % 80 == 0 or done == len(files):
            print(
                f"Processed {done}/{len(files)}"
            )

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    for group in [
        "all",
        "elastic",
        "local",
        "bending",
    ]:

        values = np.asarray(
            results[group],
            dtype=np.float64,
        )

        d_mae = values[:, 0]
        d_rmse = values[:, 1]
        r_mae = values[:, 2]
        r_rmse = values[:, 3]

        mae_change = (
            (d_mae.mean() - r_mae.mean())
            / d_mae.mean()
            * 100
        )

        rmse_change = (
            (d_rmse.mean() - r_rmse.mean())
            / d_rmse.mean()
            * 100
        )

        print()
        print(group.upper())
        print(f"Samples              : {len(values)}")
        print(f"Distorted MAE        : {d_mae.mean():.4f}")
        print(f"Rectified MAE        : {r_mae.mean():.4f}")
        print(f"MAE improvement      : {mae_change:.2f}%")
        print(f"Distorted RMSE       : {d_rmse.mean():.4f}")
        print(f"Rectified RMSE       : {r_rmse.mean():.4f}")
        print(f"RMSE improvement     : {rmse_change:.2f}%")


def main():

    print("=" * 70)
    print("DDRNET BATCH RECTIFICATION EVALUATION")
    print("=" * 70)
    print(f"Batch size: {BATCH_SIZE}")
    print("CPU threads: 16")

    model = load_model()

    for name, root in DATASETS.items():
        evaluate_dataset(
            model,
            name,
            root,
        )


if __name__ == "__main__":
    main()
