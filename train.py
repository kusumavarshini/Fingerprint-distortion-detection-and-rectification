"""
Training Module -- Fingerprint Distortion Binary Classifier
=============================================================
PyTorch Dataset/DataLoader, loss setup, and training infrastructure
for binary classification of fingerprint distortion.

Datasets:
    data/train_distorted/   (4,200 images: 1,050 clean + 3,150 distorted)
    data/val_distorted/     (  450 images:   225 clean +   225 distorted)
    data/test_distorted/    (  450 images:   225 clean +   225 distorted)

Labels:
    CLEAN     = 0
    DISTORTED = 1   (elastic, local, bending all map to class 1)

When run as __main__, performs environment verification and a single-batch
forward pass to confirm the entire pipeline works before starting training.
"""

import os
import sys
import json
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from model import FingerprintCNN, count_parameters, model_summary

# ============================================================================
#  CONFIGURATION
# ============================================================================

SEED       = 42
IMG_SIZE   = 224
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

DATASET_PATHS = {
    "train": os.path.join(BASE_DIR, "data", "train_distorted"),
    "val":   os.path.join(BASE_DIR, "data", "val_distorted"),
    "test":  os.path.join(BASE_DIR, "data", "test_distorted"),
}

LABEL_CLEAN     = 0
LABEL_DISTORTED = 1

# ============================================================================
#  REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker seed function for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
#  DATASET
# ============================================================================

def _parse_distortion_type_from_filename(filename: str) -> str:
    """
    Extract distortion type from filename.

    Training files: FM001_F1_elastic.png → 'elastic'
                    FM001_F1_clean.png   → 'clean'
    Eval files:     FM012_C1_distorted.png → 'distorted' (need metadata)
                    FM012_C1.png           → 'clean'
    """
    stem = os.path.splitext(filename)[0]
    suffix = stem.rsplit("_", 1)[-1]
    if suffix in ("elastic", "local", "bending", "clean"):
        return suffix
    return "unknown"


def _load_eval_metadata(data_root: str) -> dict:
    """
    Load distortion metadata for eval splits (val/test).
    Returns a dict mapping distorted filename → distortion_type.
    """
    meta_path = os.path.join(data_root, "distortion_metadata.json")
    if not os.path.isfile(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Handle both formats: list or dict with "records" key
    records = meta if isinstance(meta, list) else meta.get("records", [])

    lookup = {}
    for rec in records:
        # Get the distorted output path and extract filename
        dpath = rec.get("distorted_output_path") or rec.get("distorted_path", "")
        fname = os.path.basename(dpath)
        dtype = rec.get("distortion_type", "unknown")
        if fname:
            lookup[fname] = dtype

    return lookup


class FingerprintDataset(Dataset):
    """
    PyTorch Dataset for fingerprint distortion binary classification.

    Scans clean/ and distorted/ subdirectories under a given root.
    Each sample is a dict:
        image           : torch.Tensor, shape (1, 224, 224), float32, [0,1]
        label           : torch.Tensor, shape (), float32, {0.0, 1.0}
        path            : str, relative file path
        family          : str, e.g. "FAMILY-12"
        member          : str, e.g. "FATHER"
        distortion_type : str, one of "clean", "elastic", "local", "bending"
    """

    def __init__(self, data_root: str, split_name: str = ""):
        super().__init__()
        self.data_root = data_root
        self.split_name = split_name
        self.samples = []

        clean_dir = os.path.join(data_root, "clean")
        dist_dir  = os.path.join(data_root, "distorted")

        # Load eval metadata lookup for distortion type resolution
        meta_lookup = _load_eval_metadata(data_root)

        # -- Collect clean images ---------------------------------------------
        if os.path.isdir(clean_dir):
            for family in sorted(os.listdir(clean_dir)):
                fam_path = os.path.join(clean_dir, family)
                if not os.path.isdir(fam_path) or not family.startswith("FAMILY-"):
                    continue
                for member in sorted(os.listdir(fam_path)):
                    mem_path = os.path.join(fam_path, member)
                    if not os.path.isdir(mem_path):
                        continue
                    for fname in sorted(os.listdir(mem_path)):
                        if not fname.lower().endswith(".png"):
                            continue
                        self.samples.append({
                            "path": os.path.join(mem_path, fname),
                            "label": LABEL_CLEAN,
                            "family": family,
                            "member": member,
                            "distortion_type": "clean",
                        })

        # -- Collect distorted images -----------------------------------------
        if os.path.isdir(dist_dir):
            for family in sorted(os.listdir(dist_dir)):
                fam_path = os.path.join(dist_dir, family)
                if not os.path.isdir(fam_path) or not family.startswith("FAMILY-"):
                    continue
                for member in sorted(os.listdir(fam_path)):
                    mem_path = os.path.join(fam_path, member)
                    if not os.path.isdir(mem_path):
                        continue
                    for fname in sorted(os.listdir(mem_path)):
                        if not fname.lower().endswith(".png"):
                            continue

                        # Determine distortion type
                        dtype = _parse_distortion_type_from_filename(fname)
                        if dtype in ("unknown", "distorted"):
                            # Fall back to metadata lookup
                            dtype = meta_lookup.get(fname, "unknown")

                        self.samples.append({
                            "path": os.path.join(mem_path, fname),
                            "label": LABEL_DISTORTED,
                            "family": family,
                            "member": member,
                            "distortion_type": dtype,
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load grayscale image
        img = cv2.imread(sample["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {sample['path']}")

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0
        # Convert to tensor: (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)

        return {
            "image": tensor,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "path": sample["path"],
            "family": sample["family"],
            "member": sample["member"],
            "distortion_type": sample["distortion_type"],
        }

    def get_class_counts(self):
        """Return (n_clean, n_distorted) counts."""
        n_clean = sum(1 for s in self.samples if s["label"] == LABEL_CLEAN)
        n_dist  = sum(1 for s in self.samples if s["label"] == LABEL_DISTORTED)
        return n_clean, n_dist

    def get_type_counts(self):
        """Return dict of distortion_type → count."""
        counts = {}
        for s in self.samples:
            dt = s["distortion_type"]
            counts[dt] = counts.get(dt, 0) + 1
        return counts

    def summary(self):
        """Print a summary of the dataset."""
        n_clean, n_dist = self.get_class_counts()
        type_counts = self.get_type_counts()
        name = self.split_name.upper() if self.split_name else "DATASET"

        lines = [
            f"  {name}:",
            f"    Root     : {os.path.relpath(self.data_root, BASE_DIR)}/",
            f"    Total    : {len(self)}",
            f"    Clean    : {n_clean}",
            f"    Distorted: {n_dist}",
        ]
        for dt in ["elastic", "local", "bending"]:
            if dt in type_counts:
                lines.append(f"      {dt:<10}: {type_counts[dt]}")
        return "\n".join(lines)


# ============================================================================
#  DATALOADER FACTORY
# ============================================================================

def create_dataloaders(batch_size: int = 32, num_workers: int = 0):
    """
    Create DataLoaders for train, val, and test splits.

    Parameters
    ----------
    batch_size : int
    num_workers : int
        Number of worker processes.  Use 0 for Windows compatibility.

    Returns
    -------
    dict of DataLoaders: {"train": ..., "val": ..., "test": ...}
    dict of Datasets:    {"train": ..., "val": ..., "test": ...}
    """
    g = torch.Generator()
    g.manual_seed(SEED)

    datasets = {}
    loaders = {}

    for split_name, data_root in DATASET_PATHS.items():
        ds = FingerprintDataset(data_root, split_name=split_name)
        datasets[split_name] = ds

        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return loaders, datasets


# ============================================================================
#  LOSS WITH CLASS WEIGHTS
# ============================================================================

def create_weighted_loss(n_clean: int, n_distorted: int, device: torch.device):
    """
    Create BCEWithLogitsLoss with pos_weight to handle class imbalance.

    pos_weight = n_negative / n_positive = n_clean / n_distorted

    This down-weights the majority DISTORTED class so that each clean
    sample's contribution equals that of each distorted sample in aggregate.

    Parameters
    ----------
    n_clean : int
        Number of CLEAN (class 0 / negative) samples.
    n_distorted : int
        Number of DISTORTED (class 1 / positive) samples.
    device : torch.device

    Returns
    -------
    nn.BCEWithLogitsLoss
    """
    pos_weight = torch.tensor([n_clean / n_distorted], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion, pos_weight


# ============================================================================
#  TRAINING LOOP
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average batch loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_split(model, loader, criterion, device):
    """
    Evaluate model on a data split.

    Returns
    -------
    avg_loss : float
    metrics  : dict from evaluate.compute_metrics()
    labels   : np.ndarray
    logits   : np.ndarray
    dtypes   : list[str]
    """
    from evaluate import compute_metrics

    model.eval()
    all_logits = []
    all_labels = []
    all_dtypes = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_dtypes.extend(batch["distortion_type"])

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / n_batches

    metrics = compute_metrics(all_labels, all_logits)

    return avg_loss, metrics, all_labels, all_logits, all_dtypes

def find_best_threshold(labels, logits):
    """
    Find the probability threshold that gives the best F1
    on the validation set.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = None

    from evaluate import compute_metrics

    for threshold in thresholds:
        metrics = compute_metrics(
            labels,
            logits,
            threshold=float(threshold)
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics

def run_training():
    """
    Full baseline training run.

    - Adam optimizer, lr=0.0001, weight_decay=0.0001
    - Early stopping on validation F1 (patience=4)
    - Best checkpoint saved to checkpoints/best_model.pth
    - Final evaluation on the test set using best checkpoint
    """
    from evaluate import compute_metrics

    # --- Configuration -------------------------------------------------------
    BATCH_SIZE   = 32
    LR           = 0.0001
    WEIGHT_DECAY = 0.0001
    MAX_EPOCHS   = 12
    PATIENCE     = 4

    sep = "=" * 70

    print(sep)
    print("  BASELINE TRAINING RUN")
    print(sep)

    set_seed(SEED)
    device = torch.device("cpu")

    print(f"\n  Configuration:")
    print(f"    Device           : {device}")
    print(f"    Seed             : {SEED}")
    print(f"    Batch size       : {BATCH_SIZE}")
    print(f"    Optimizer        : Adam (lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"    Max epochs       : {MAX_EPOCHS}")
    print(f"    Early stopping   : patience={PATIENCE} (on val F1)")
    print(f"    Loss             : BCEWithLogitsLoss (class-weighted)")

    # --- Data ----------------------------------------------------------------
    print(f"\n  Loading datasets ...")
    loaders, datasets = create_dataloaders(batch_size=BATCH_SIZE, num_workers=0)

    print()
    for split_name in ["train", "val", "test"]:
        print(datasets[split_name].summary())
        print()

    # --- Class weights -------------------------------------------------------
    n_clean, n_dist = datasets["train"].get_class_counts()
    criterion, pos_weight = create_weighted_loss(n_clean, n_dist, device)
    print(f"  pos_weight = {n_clean}/{n_dist} = {pos_weight.item():.6f}")

    # --- Model ---------------------------------------------------------------
    model = FingerprintCNN(dropout=0.5).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    n_params = count_parameters(model)
    print(f"  Model parameters   : {n_params:,}\n")

    # --- Checkpoint directory ------------------------------------------------
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")

    # --- Training loop -------------------------------------------------------
    print(sep)
    print("  TRAINING")
    print(sep)

    header = (f"  {'Ep':>3} | {'Tr Loss':>8} | {'Va Loss':>8} | "
              f"{'Acc':>6} | {'Prec':>6} | {'Rec':>6} | "
              f"{'F1':>6} | {'AUC':>6} | Status")
    print(f"\n{header}")
    print(f"  {'---':>3}-+-{'--------':>8}-+-{'--------':>8}-+-"
          f"{'------':>6}-+-{'------':>6}-+-{'------':>6}-+-"
          f"{'------':>6}-+-{'------':>6}-+-{'-' * 24}")

    history = []
    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_m, _, _, _ = evaluate_split(
            model, loaders["val"], criterion, device
        )

        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"],
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(record)

        # Checkpointing on best val F1
        val_f1 = val_m["f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1,
                "val_metrics": val_m,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_ckpt_path)
            status = f"* Best (F1={best_f1:.4f})"
        else:
            patience_counter += 1
            status = f"No improve ({patience_counter}/{PATIENCE})"

        print(f"  {epoch:3d} | {train_loss:8.4f} | {val_loss:8.4f} | "
              f"{val_m['accuracy']:6.4f} | {val_m['precision']:6.4f} | "
              f"{val_m['recall']:6.4f} | {val_f1:6.4f} | "
              f"{val_m['roc_auc']:6.4f} | {status}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no F1 improvement for {PATIENCE} epochs)")
            break

    total_time = time.time() - train_start
    print(f"\n  Training completed in {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")
    print(f"  Best epoch: {best_epoch} (val F1 = {best_f1:.4f})")

    # --- Save training history -----------------------------------------------
    history_path = os.path.join(BASE_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved -> {os.path.relpath(history_path, BASE_DIR)}")

    # =========================================================================
    #  TEST SET EVALUATION
    # =========================================================================
    print(f"\n{sep}")
    print(f"  TEST SET EVALUATION  (best checkpoint: epoch {best_epoch})")
    print(sep)

    checkpoint = torch.load(best_ckpt_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")

    # =========================================================================
    #  THRESHOLD TUNING ON VALIDATION SET
    # =========================================================================
    #
    # The model is already trained. We use the validation set to select the
    # probability threshold that maximizes F1. The test set is not used for
    # threshold selection.
    val_loss, val_m, val_labels, val_logits, _ = evaluate_split(
        model, loaders["val"], criterion, device
    )

    best_threshold, best_val_threshold_metrics = find_best_threshold(
        val_labels, val_logits
    )

    print(f"\n  Threshold tuning:")
    print(f"    Default threshold : 0.50")
    print(f"    Best threshold    : {best_threshold:.2f}")
    print(f"    Validation F1     : {best_val_threshold_metrics['f1']:.4f}")

    # =========================================================================
    #  TEST SET EVALUATION USING VALIDATION-SELECTED THRESHOLD
    # =========================================================================

    test_loss, _, test_labels, test_logits, test_dtypes = evaluate_split(
        model, loaders["test"], criterion, device
    )

    # Recalculate test classification metrics using the threshold selected
    # from the validation set.
    test_m = compute_metrics(
        test_labels,
        test_logits,
        threshold=best_threshold
    )

    # Overall test metrics
    print(f"\n  Overall Test Metrics:")
    print(f"    Accuracy    : {test_m['accuracy']:.4f}")
    print(f"    Precision   : {test_m['precision']:.4f}")
    print(f"    Recall      : {test_m['recall']:.4f}")
    print(f"    F1          : {test_m['f1']:.4f}")
    print(f"    ROC-AUC     : {test_m['roc_auc']:.4f}")
    print(f"    Loss        : {test_loss:.6f}")

    # Confusion matrix
    cm = test_m["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"                      Predicted")
    print(f"                    CLEAN  DISTORTED")
    print(f"    Actual CLEAN   {cm['tn']:>5}  {cm['fp']:>5}")
    print(f"    Actual DIST    {cm['fn']:>5}  {cm['tp']:>5}")

    # Per-distortion-type evaluation
    print(f"\n  Per-Distortion-Type Performance (test set):")
    print(f"  {'-' * 60}")

    test_dtypes_arr = np.array(test_dtypes)
    per_type_results = {}

    for dtype in ["clean", "elastic", "local", "bending"]:
        mask = test_dtypes_arr == dtype
        n_type = int(mask.sum())
        if n_type == 0:
            continue

        dt_labels = test_labels[mask]
        dt_logits = test_logits[mask]
        dt_m = compute_metrics(
            dt_labels, dt_logits, threshold=best_threshold
        )
        per_type_results[dtype] = dt_m

        print(f"\n    {dtype.upper()} ({n_type} samples):")
        print(f"      Accuracy    : {dt_m['accuracy']:.4f}")
        print(f"      Precision   : {dt_m['precision']:.4f}")
        print(f"      Recall      : {dt_m['recall']:.4f}")
        print(f"      F1          : {dt_m['f1']:.4f}")

    # --- Save evaluation results ---------------------------------------------
    eval_results = {
        "experiment": "baseline_v1",
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "training_epochs_completed": len(history),
        "total_training_time_s": round(total_time, 1),
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "pos_weight": round(pos_weight.item(), 6),
            "classification_threshold": round(best_threshold, 2),
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "device": str(device),
            "seed": SEED,
        },
        "test_loss": round(test_loss, 6),
        "test_metrics": test_m,
        "test_per_type": per_type_results,
        "checkpoint_path": os.path.relpath(best_ckpt_path, BASE_DIR),
    }
    eval_path = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Evaluation results saved -> "
          f"{os.path.relpath(eval_path, BASE_DIR)}")

    print(f"\n{sep}")
    print("  BASELINE EXPERIMENT COMPLETE")
    print(sep)

    return history, test_m, per_type_results


# ============================================================================
#  SINGLE BATCH VERIFICATION
# ============================================================================

def verify_single_batch():
    """
    End-to-end verification: load one batch, run through model, compute loss.
    Prints a complete diagnostic report and exits.
    """
    set_seed(SEED)

    sep = "=" * 65
    print(sep)
    print("  SINGLE BATCH VERIFICATION")
    print(sep)

    # -- Environment ----------------------------------------------------------
    print("\n  Environment:")
    print(f"    Python        : {sys.version.split()[0]}")
    print(f"    PyTorch       : {torch.__version__}")
    try:
        import torchvision
        print(f"    torchvision   : {torchvision.__version__}")
    except (ImportError, RuntimeError):
        print(f"    torchvision   : not available (not required)")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device        : {device}")
    print(f"    Random seed   : {SEED}")

    # -- Datasets -------------------------------------------------------------
    print("\n  Loading datasets ...")
    loaders, datasets = create_dataloaders(batch_size=32, num_workers=0)

    print()
    for split_name in ["train", "val", "test"]:
        print(datasets[split_name].summary())
        print()

    # -- Class weights --------------------------------------------------------
    n_clean, n_dist = datasets["train"].get_class_counts()
    criterion, pos_weight = create_weighted_loss(n_clean, n_dist, device)
    print(f"  Class Weight Calculation:")
    print(f"    Training clean (negative) : {n_clean}")
    print(f"    Training distorted (pos)  : {n_dist}")
    print(f"    pos_weight = {n_clean}/{n_dist} = {pos_weight.item():.6f}")
    print(f"    Effect: each DISTORTED sample loss is scaled by {pos_weight.item():.4f}")

    # -- Model ----------------------------------------------------------------
    print()
    model = FingerprintCNN(dropout=0.5).to(device)
    print(model_summary(model))

    n_params = count_parameters(model)
    print(f"\n  Total trainable parameters: {n_params:,}")

    # -- Single batch forward pass --------------------------------------------
    print(f"\n{sep}")
    print("  FORWARD PASS -- Training batch")
    print(sep)

    model.train()
    train_loader = loaders["train"]
    batch = next(iter(train_loader))

    images = batch["image"].to(device)           # (B, 1, 224, 224)
    labels = batch["label"].to(device)            # (B,)
    families = batch["family"]                     # list of str
    dtypes = batch["distortion_type"]              # list of str

    print(f"\n  Batch image shape    : {images.shape}")
    print(f"  Batch image dtype    : {images.dtype}")
    print(f"  Batch image range    : [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"  Batch label shape    : {labels.shape}")
    print(f"  Batch label dtype    : {labels.dtype}")
    print(f"  Batch label values   : {labels.cpu().numpy().tolist()}")

    # Count labels in batch
    n_clean_batch = int((labels == 0).sum().item())
    n_dist_batch  = int((labels == 1).sum().item())
    print(f"  Labels in batch      : {n_clean_batch} clean, {n_dist_batch} distorted")

    # Distortion types in batch
    from collections import Counter
    type_counts = Counter(dtypes)
    print(f"  Distortion types     : {dict(type_counts)}")

    # Unique families in batch
    unique_fams = set(families)
    print(f"  Unique families      : {len(unique_fams)}")

    # Forward pass
    logits = model(images)                        # (B, 1)
    print(f"\n  Model output shape   : {logits.shape}")
    print(f"  Model output values  : {logits.detach().cpu().squeeze().numpy().tolist()[:8]}{'...' if len(logits) > 8 else ''}")

    # Compute loss
    loss = criterion(logits.squeeze(1), labels)
    print(f"\n  Loss function        : BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
    print(f"  Loss value           : {loss.item():.6f}")

    # Backward pass test
    loss.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"  Backward pass        : OK")
    print(f"  Gradient norms       : min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")

    # -- Validation batch test ------------------------------------------------
    print(f"\n{sep}")
    print("  FORWARD PASS -- Validation batch")
    print(sep)

    model.eval()
    val_batch = next(iter(loaders["val"]))
    val_images = val_batch["image"].to(device)
    val_labels = val_batch["label"].to(device)
    val_dtypes = val_batch["distortion_type"]

    with torch.no_grad():
        val_logits = model(val_images)
        val_loss = criterion(val_logits.squeeze(1), val_labels)

    val_probs = torch.sigmoid(val_logits.squeeze(1))
    val_preds = (val_probs >= 0.5).long()
    val_correct = int((val_preds == val_labels.long()).sum().item())

    print(f"\n  Batch image shape    : {val_images.shape}")
    print(f"  Batch label shape    : {val_labels.shape}")
    print(f"  Label values         : {val_labels.cpu().numpy().tolist()}")
    print(f"  Distortion types     : {dict(Counter(val_dtypes))}")
    print(f"  Model output shape   : {val_logits.shape}")
    print(f"  Loss value           : {val_loss.item():.6f}")
    print(f"  Predictions          : {val_preds.cpu().numpy().tolist()}")
    print(f"  Probabilities        : {[round(p, 4) for p in val_probs.cpu().numpy().tolist()[:8]]}{'...' if len(val_probs) > 8 else ''}")
    print(f"  Batch accuracy       : {val_correct}/{len(val_labels)} = {val_correct/len(val_labels):.4f}")

    # -- Test batch -----------------------------------------------------------
    print(f"\n{sep}")
    print("  FORWARD PASS -- Test batch")
    print(sep)

    test_batch = next(iter(loaders["test"]))
    test_images = test_batch["image"].to(device)
    test_labels = test_batch["label"].to(device)

    with torch.no_grad():
        test_logits = model(test_images)
        test_loss = criterion(test_logits.squeeze(1), test_labels)

    print(f"\n  Batch image shape    : {test_images.shape}")
    print(f"  Batch label shape    : {test_labels.shape}")
    print(f"  Model output shape   : {test_logits.shape}")
    print(f"  Loss value           : {test_loss.item():.6f}")

    # -- Final summary --------------------------------------------------------
    total_images = sum(len(ds) for ds in datasets.values())
    print(f"\n\n{sep}")
    print("  VERIFICATION SUMMARY")
    print(sep)
    print(f"  Environment          : Python {sys.version.split()[0]}, "
          f"PyTorch {torch.__version__}, Device: {device}")
    print(f"  Random seed          : {SEED}")
    print(f"  Total dataset images : {total_images}")
    print(f"    Train              : {len(datasets['train'])} "
          f"({datasets['train'].get_class_counts()[0]} clean + "
          f"{datasets['train'].get_class_counts()[1]} distorted)")
    print(f"    Validation         : {len(datasets['val'])} "
          f"({datasets['val'].get_class_counts()[0]} clean + "
          f"{datasets['val'].get_class_counts()[1]} distorted)")
    print(f"    Test               : {len(datasets['test'])} "
          f"({datasets['test'].get_class_counts()[0]} clean + "
          f"{datasets['test'].get_class_counts()[1]} distorted)")
    print(f"  Model                : FingerprintCNN (4-layer)")
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  Loss                 : BCEWithLogitsLoss "
          f"(pos_weight={pos_weight.item():.4f})")
    print(f"  Batch size           : 32")
    print(f"  Image tensor shape   : [1, 224, 224]")
    print(f"  Output shape         : [1] (single logit)")
    print()
    print(f"  [OK] Dataset loading    : OK")
    print(f"  [OK] Model forward pass : OK")
    print(f"  [OK] Loss computation   : OK")
    print(f"  [OK] Backward pass      : OK")
    print(f"  [OK] All dataloaders    : OK")
    print(sep)
    print("  READY FOR TRAINING")
    print(sep)


# ============================================================
# ERROR ANALYSIS - SAVE MISCLASSIFIED TEST IMAGES
# ============================================================

def save_error_images(model, test_dataset, device, threshold, output_dir):
    """
    Save false-positive and false-negative test images using
    the trained best checkpoint and validation-selected threshold.
    """

    import shutil

    fp_dir = os.path.join(output_dir, "false_positive_clean")
    fn_dir = os.path.join(output_dir, "false_negative_distorted")

    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    model.eval()

    fp_count = 0
    fn_count = 0
    fp_by_type = {}
    fn_by_type = {}

    with torch.no_grad():
        for idx in range(len(test_dataset)):

            sample = test_dataset[idx]

            image = sample["image"].unsqueeze(0).to(device)
            label = int(sample["label"].item())

            logit = model(image).squeeze().item()

            # Stable sigmoid
            probability = float(
                1.0 / (1.0 + np.exp(-logit))
                if logit >= 0
                else np.exp(logit) / (1.0 + np.exp(logit))
            )

            prediction = 1 if probability >= threshold else 0

            image_path = sample["path"]
            dtype = sample["distortion_type"]
            family = sample["family"]
            member = sample["member"]

            # ------------------------------------------------
            # FALSE POSITIVE
            # Actual = CLEAN
            # Predicted = DISTORTED
            # ------------------------------------------------
            if label == LABEL_CLEAN and prediction == LABEL_DISTORTED:

                fp_count += 1
                fp_by_type[dtype] = fp_by_type.get(dtype, 0) + 1

                filename = (
                    f"FP_{fp_count:03d}_"
                    f"{family}_{member}_"
                    f"prob_{probability:.4f}.png"
                )

                shutil.copy2(
                    image_path,
                    os.path.join(fp_dir, filename)
                )

            # ------------------------------------------------
            # FALSE NEGATIVE
            # Actual = DISTORTED
            # Predicted = CLEAN
            # ------------------------------------------------
            elif label == LABEL_DISTORTED and prediction == LABEL_CLEAN:

                fn_count += 1
                fn_by_type[dtype] = fn_by_type.get(dtype, 0) + 1

                filename = (
                    f"FN_{fn_count:03d}_"
                    f"{family}_{member}_"
                    f"{dtype}_"
                    f"prob_{probability:.4f}.png"
                )

                shutil.copy2(
                    image_path,
                    os.path.join(fn_dir, filename)
                )

    print("\n" + "=" * 70)
    print("  ERROR ANALYSIS")
    print("=" * 70)

    print(f"\n  Threshold used : {threshold:.2f}")
    print(f"  False positives: {fp_count}")
    print(f"  False negatives: {fn_count}")

    print("\n  False positives by type:")
    for dtype, count in sorted(fp_by_type.items()):
        print(f"    {dtype:<10}: {count}")

    print("\n  False negatives by distortion type:")
    for dtype, count in sorted(fn_by_type.items()):
        print(f"    {dtype:<10}: {count}")

    print("\n  Saved to:")
    print(f"    {os.path.abspath(output_dir)}")

    print("=" * 70)


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_single_batch()

    elif len(sys.argv) > 1 and sys.argv[1] == "--error-analysis":
        # -----------------------------------------------------------------
        # ERROR ANALYSIS ONLY — no retraining
        # Load existing best checkpoint and save misclassified test images.
        # -----------------------------------------------------------------
        device = torch.device("cpu")

        best_ckpt_path = os.path.join(
            BASE_DIR, "checkpoints", "best_model.pth"
        )

        if not os.path.isfile(best_ckpt_path):
            print(f"  [ERROR] Checkpoint not found: {best_ckpt_path}")
            sys.exit(1)

        checkpoint = torch.load(
            best_ckpt_path,
            map_location=device,
            weights_only=False
        )

        model = FingerprintCNN(dropout=0.5).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")

        # Load test dataset
        test_dataset = FingerprintDataset(
            DATASET_PATHS["test"],
            split_name="test"
        )
        print(f"  Test dataset: {len(test_dataset)} images")

        # Read threshold from the completed experiment's saved results
        eval_results_path = os.path.join(BASE_DIR, "evaluation_results.json")
        if not os.path.isfile(eval_results_path):
            print(f"  [ERROR] evaluation_results.json not found: {eval_results_path}")
            sys.exit(1)
        with open(eval_results_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)
        threshold = eval_results["config"]["classification_threshold"]
        print(f"  Threshold (from evaluation_results.json): {threshold}")

        # Save misclassified test images
        save_error_images(
            model=model,
            test_dataset=test_dataset,
            device=device,
            threshold=threshold,
            output_dir=os.path.join(BASE_DIR, "error_analysis"),
        )

    else:
        run_training()