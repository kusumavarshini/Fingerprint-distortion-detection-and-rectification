"""
Rectification Model Training
============================

Trains FingerprintRectificationNet to predict dense dx/dy
displacement fields from distorted fingerprint images.

Training:
    data/rectification_train/

Validation:
    data/rectification_val/

Target:
    2-channel displacement field
    channel 0 -> dx
    channel 1 -> dy

Loss:
    Smooth L1 field regression loss
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from rectification_dataset import FingerprintRectificationDataset
from rectification_model import (
    FingerprintRectificationNet,
    count_parameters
)
VAL_DATA_ROOT = "data/rectification_val"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "rectification_train"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments",
    "rectification_baseline"
)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "best_model.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

BATCH_SIZE = 8
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 12
PATIENCE = 5

VAL_RATIO = 0.15


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOSS
# ============================================================

class FieldLoss(nn.Module):
    """
    Smooth L1 loss between predicted and ground-truth
    displacement fields.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.SmoothL1Loss(
            beta=1.0
        )

    def forward(self, predicted, target):
        return self.loss(
            predicted,
            target
        )


# ============================================================
# ONE EPOCH
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
    total_samples = 0

    for batch in loader:

        distorted = batch["distorted"].to(
            device,
            non_blocking=True
        )

        target_field = batch["field"].to(
            device,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            predicted_field = model(
                distorted
            )

            loss = criterion(
                predicted_field,
                target_field
            )

            if training:
                loss.backward()

                optimizer.step()

        batch_size = distorted.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("FINGERPRINT RECTIFICATION MODEL TRAINING")
    print("=" * 70)

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading rectification dataset...")

    train_dataset = FingerprintRectificationDataset(
    data_root=TRAIN_DATA_ROOT
    )

    val_dataset = FingerprintRectificationDataset(
    data_root=VAL_DATA_ROOT
    )

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    print(
    f"Train samples : {train_size}"
    )

    print(
    f"Val samples   : {val_size}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FingerprintRectificationNet().to(
        device
    )

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = FieldLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss)
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "seed": SEED
                },
                CHECKPOINT_PATH
            )

            print(
                f"  -> Best model saved "
                f"(val loss: {val_loss:.6f})"
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest epoch     : {best_epoch}"
    )

    print(
        f"Best val loss  : {best_val_loss:.6f}"
    )

    print(
        f"\nCheckpoint     : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"History        : "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()