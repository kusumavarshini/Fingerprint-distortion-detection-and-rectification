from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path("DDRNet").resolve()))

from models.DDRNet_DIR import DDRNet_DIR


# ============================================================
# Configuration
# ============================================================

TRAIN_DIR = Path("data/ddrnet_train")
VAL_DIR = Path("data/ddrnet_val")

OUTPUT_DIR = Path("experiments/ddrnet_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"

SEED = 42

IMAGE_SIZE = 224
GRID_SIZE = 14

BATCH_SIZE = 4
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 8
PATIENCE = 3 

# Loss weights
DISPLACEMENT_WEIGHT = 1.0
ORIENTATION_WEIGHT = 0.1


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Dataset
# ============================================================

class DDRNetDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.image_dir = self.root_dir / "images"
        self.field_dir = self.root_dir / "fields"
        self.mask_dir = self.root_dir / "masks"
        self.orientation_dir = self.root_dir / "orientation"

        self.files = sorted(self.image_dir.glob("*.png"))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        # Verify corresponding files exist
        valid_files = []

        for image_path in self.files:
            name = image_path.name

            field_path = self.field_dir / image_path.with_suffix(".npz").name
            mask_path = self.mask_dir / name
            orientation_path = self.orientation_dir / image_path.with_suffix(".npy").name

            if (
                field_path.exists()
                and mask_path.exists()
                and orientation_path.exists()
            ):
                valid_files.append(image_path)

        self.files = valid_files

        print(
            f"{root_dir}: {len(self.files)} complete samples"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image_path = self.files[index]

        name = image_path.name

        field_path = self.field_dir / image_path.with_suffix(".npz").name
        mask_path = self.mask_dir / name
        orientation_path = (
            self.orientation_dir /
            image_path.with_suffix(".npy").name
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = image.astype(np.float32)

        # Same basic convention as DDRNet reference:
        # (255 - image) / 255
        image = (255.0 - image) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)

        # ----------------------------------------------------
        # Displacement field
        # ----------------------------------------------------

        field = np.load(field_path)

        dx = field["dx_rect16"].astype(np.float32)
        dy = field["dy_rect16"].astype(np.float32)

        displacement = np.stack(
            [dx, dy],
            axis=0
        )

        displacement = torch.from_numpy(displacement)

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        mask = (mask > 0).astype(np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        orientation = np.load(
            orientation_path
        ).astype(np.int64)

        orientation = torch.from_numpy(orientation)

        return {
            "image": image,
            "displacement": displacement,
            "mask": mask,
            "orientation": orientation,
        }


# ============================================================
# Loss
# ============================================================

class DDRNetLoss(nn.Module):

    def __init__(
        self,
        displacement_weight=1.0,
        orientation_weight=0.1
    ):
        super().__init__()

        self.displacement_weight = displacement_weight
        self.orientation_weight = orientation_weight

        self.displacement_loss = nn.SmoothL1Loss(
            reduction="none"
        )

        self.orientation_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        displacement_pred,
        orientation_pred,
        displacement_target,
        orientation_target,
        mask
    ):

        # ----------------------------------------------------
        # Displacement loss
        # ----------------------------------------------------

        disp_error = self.displacement_loss(
            displacement_pred,
            displacement_target
        )

        # Mask has shape:
        # B x 1 x 14 x 14
        #
        # Expand it to:
        # B x 2 x 14 x 14

        mask_2 = mask.expand_as(disp_error)

        masked_error = disp_error * mask_2

        valid_pixels = mask_2.sum().clamp_min(1.0)

        displacement_loss = (
            masked_error.sum() / valid_pixels
        )

       

        # ----------------------------------------------------
        # Orientation loss — fingerprint region only
        # ----------------------------------------------------

        # CrossEntropyLoss per spatial location
        orientation_logits = torch.log(
            orientation_pred.clamp_min(1e-8)
        )

        orientation_error = F.nll_loss(
            orientation_logits,
            orientation_target,
            reduction="none"
        )

        # Mask background pixels
        orientation_mask = mask.squeeze(1)

        valid_orientation = orientation_mask.sum().clamp_min(1.0)

        orientation_loss = (
            (orientation_error * orientation_mask).sum()
            /valid_orientation
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            self.displacement_weight *
            displacement_loss
            +
            self.orientation_weight *
            orientation_loss
        )

        return (
            total_loss,
            displacement_loss.detach(),
            orientation_loss.detach()
        )


# ============================================================
# One epoch
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training=True
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_disp_loss = 0.0
    total_ori_loss = 0.0

    num_samples = 0

    for batch in loader:

        images = batch["image"].to(device)
        displacement = batch["displacement"].to(device)
        mask = batch["mask"].to(device)
        orientation = batch["orientation"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            displacement_pred, orientation_pred = model(
                images,
                mask
            )

            loss, disp_loss, ori_loss = criterion(
                displacement_pred,
                orientation_pred,
                displacement,
                orientation,
                mask
            )

            if training:
                loss.backward()

                # Prevent unusually large gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_disp_loss += disp_loss.item() * batch_size
        total_ori_loss += ori_loss.item() * batch_size

        num_samples += batch_size

    return {
        "loss": total_loss / num_samples,
        "displacement_loss": total_disp_loss / num_samples,
        "orientation_loss": total_ori_loss / num_samples,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DDRNet DISTORTION RECTIFICATION TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cpu":
        print("WARNING: DDRNet is being trained on CPU.")
        print("This may take a long time.")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = DDRNetDataset(TRAIN_DIR)
    val_dataset = DDRNetDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DDRNet_DIR(
        dis_const=16
    )

    model = model.to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_parameters:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = DDRNetLoss(
        displacement_weight=DISPLACEMENT_WEIGHT,
        orientation_weight=ORIENTATION_WEIGHT
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        print()
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False
        )

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Disp: {train_metrics['displacement_loss']:.6f} | "
            f"Ori: {train_metrics['orientation_loss']:.6f}"
        )

        print(
            f"Val   Loss: {val_metrics['loss']:.6f} | "
            f"Disp: {val_metrics['displacement_loss']:.6f} | "
            f"Ori: {val_metrics['orientation_loss']:.6f}"
        )

        print(
            f"Learning Rate: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(epoch_record)

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        "displacement_weight":
                            DISPLACEMENT_WEIGHT,
                        "orientation_weight":
                            ORIENTATION_WEIGHT,
                        "learning_rate":
                            LEARNING_RATE,
                        "weight_decay":
                            WEIGHT_DECAY,
                        "batch_size":
                            BATCH_SIZE,
                        "seed":
                            SEED,
                    }
                },
                CHECKPOINT_PATH
            )

            print(
                f"✓ Best model saved "
                f"(Val Loss: {best_val_loss:.6f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DDRNet TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"History: "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()