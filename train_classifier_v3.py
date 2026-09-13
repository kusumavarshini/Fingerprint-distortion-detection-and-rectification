"""
Retrain and Evaluate Custom CNN Distortion Classifier on Approved V3 Dataset
=============================================================================
- Dataset: data/classifier_v3/
- Balanced 1:1 binary classification:
    Train: 1050 clean + 1050 distorted (350 elastic + 350 local + 350 bending) = 2100 images
    Val:    225 clean +  225 distorted ( 75 elastic +  75 local +  75 bending) =  450 images
    Test:   225 clean +  225 distorted ( 75 elastic +  75 local +  75 bending) =  450 images
- Custom CNN: FingerprintCNN (exactly 389,057 parameters)
- Preprocessing: Grayscale 224x224 normalized to [0, 1] float32
- Threshold tuning on validation set only across [0.05, 0.95] (maximizing F1)
- Held-out test evaluation with locked threshold
- Distortion breakdown: elastic, local, bending
- Comparison table vs frozen V1 baseline
- Real unseen clean fingerprint check on data/split/test/ (225 images)
- Output artifacts saved to: experiments/classifier_v3/
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import FingerprintCNN, count_parameters
from evaluate import compute_metrics, sigmoid

# =============================================================================
# CONFIGURATION & REPRODUCIBILITY
# =============================================================================

SEED = 42
BATCH_SIZE = 32
LR = 0.0001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 15
PATIENCE = 5

BASE_DIR = Path(".")
DATA_V3_ROOT = BASE_DIR / "data" / "classifier_v3"
RAW_TEST_ROOT = BASE_DIR / "data" / "split" / "test"
EXP_DIR = BASE_DIR / "experiments" / "classifier_v3"

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =============================================================================
# BALANCED DATASET LOADER
# =============================================================================

class BalancedV3Dataset(Dataset):
    """
    Balanced dataset loader for V3 classifier.
    Strictly balances classes (1:1 clean:distorted) with equal distortion types.
    """
    def __init__(self, split_dir: Path, split_name: str, seed: int = SEED):
        super().__init__()
        self.split_name = split_name
        self.samples = []

        clean_dir = split_dir / "clean"
        dist_dir = split_dir / "distorted"

        clean_files = sorted(list(clean_dir.glob("*.png")))
        dist_files = sorted(list(dist_dir.glob("*.png")))

        # Group distorted files by type
        dist_by_type = {"elastic": [], "local": [], "bending": []}
        for f in dist_files:
            dtype = f.stem.rsplit("__", 1)[-1]
            if dtype in dist_by_type:
                dist_by_type[dtype].append(f)

        # Number of samples to take per type
        n_clean = len(clean_files)
        per_type_target = n_clean // 3

        rng = random.Random(seed + (100 if split_name == "train" else (200 if split_name == "val" else 300)))

        sampled_dist = []
        for dtype in ["elastic", "local", "bending"]:
            pool = sorted(dist_by_type[dtype])
            chosen = rng.sample(pool, min(per_type_target, len(pool)))
            for f in chosen:
                sampled_dist.append((f, dtype))

        # Add clean samples
        for f in clean_files:
            parts = f.stem.split("__")
            family = parts[0] if len(parts) > 0 else "unknown"
            member = parts[1] if len(parts) > 1 else "unknown"
            self.samples.append({
                "path": str(f),
                "filename": f.name,
                "label": 0.0,
                "distortion_type": "clean",
                "family": family,
                "member": member,
            })

        # Add distorted samples
        for f, dtype in sampled_dist:
            parts = f.stem.split("__")
            family = parts[0] if len(parts) > 0 else "unknown"
            member = parts[1] if len(parts) > 1 else "unknown"
            self.samples.append({
                "path": str(f),
                "filename": f.name,
                "label": 1.0,
                "distortion_type": dtype,
                "family": family,
                "member": member,
            })

        # Deterministic shuffle
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {item['path']}")
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] float32
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        label = torch.tensor(item["label"], dtype=torch.float32)

        return {
            "image": tensor,
            "label": label,
            "path": item["path"],
            "filename": item["filename"],
            "distortion_type": item["distortion_type"],
            "family": item["family"],
            "member": item["member"],
        }

    def get_breakdown(self):
        counts = {}
        for s in self.samples:
            dt = s["distortion_type"]
            counts[dt] = counts.get(dt, 0) + 1
        return counts


class RealCleanTestDataset(Dataset):
    """Loads all 225 raw clean test fingerprints from data/split/test/."""
    def __init__(self, raw_test_dir: Path):
        super().__init__()
        self.files = sorted(list(raw_test_dir.glob("**/*.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {p}")
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            "image": tensor,
            "label": torch.tensor(0.0, dtype=torch.float32),
            "path": str(p),
            "filename": p.name,
        }


# =============================================================================
# EVALUATION & METRICS HELPERS
# =============================================================================

def compute_full_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1e-8) if (tp + fp) > 0 else 0.0
    rec = tp / max(tp + fn, 1e-8) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0
    spec = tn / max(tn + fp, 1e-8) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2.0

    # ROC AUC using project's _compute_roc_auc
    from evaluate import _compute_roc_auc
    roc_auc = _compute_roc_auc(labels, probs)

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "specificity": round(float(spec), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_model(model, loader, criterion, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_dtypes = []
    all_paths = []
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
            if "distortion_type" in batch:
                all_dtypes.extend(batch["distortion_type"])
            if "path" in batch:
                all_paths.extend(batch["path"])

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_probs = sigmoid(all_logits)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, all_labels, all_logits, all_probs, all_dtypes, all_paths


# =============================================================================
# MAIN TRAINING & EVALUATION PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("CUSTOM CNN DISTORTION CLASSIFIER TRAINING & EVALUATION (V3 DATASET)")
    print("=" * 70)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build Datasets
    print("\nLoading Balanced Datasets...")
    train_ds = BalancedV3Dataset(DATA_V3_ROOT / "train", "train", seed=SEED)
    val_ds = BalancedV3Dataset(DATA_V3_ROOT / "val", "val", seed=SEED)
    test_ds = BalancedV3Dataset(DATA_V3_ROOT / "test", "test", seed=SEED)

    print("Train dataset breakdown:", train_ds.get_breakdown(), f"| Total: {len(train_ds)}")
    print("Val dataset breakdown:  ", val_ds.get_breakdown(), f"| Total: {len(val_ds)}")
    print("Test dataset breakdown: ", test_ds.get_breakdown(), f"| Total: {len(test_ds)}")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # 2. Build Model
    model = FingerprintCNN(dropout=0.5).to(device)
    param_count = count_parameters(model)
    print(f"\nModel: FingerprintCNN")
    print(f"Trainable Parameters: {param_count:,} (Expected baseline: 389,057)")
    assert param_count == 389057, f"Parameter mismatch: {param_count} != 389057"
    print("[PASS] Parameter count verified exactly matches custom CNN baseline.")

    # 3. Loss & Optimizer (balanced pos_weight=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_ckpt_path = EXP_DIR / "best_model.pth"
    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Training Loop...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Val Prec':>8} | {'Val Rec':>8} | {'Val F1':>8} | {'Val AUC':>8} | {'Status'}")
    print("-" * 88)

    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # Validation at default 0.5 for epoch tracking
        val_loss, val_labels, _, val_probs, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_m = compute_full_metrics(val_labels, val_probs, threshold=0.5)

        val_f1 = val_m["f1"]
        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_f1,
            "val_roc_auc": val_m["roc_auc"],
            "epoch_time_s": round(time.time() - ep_t0, 1),
        }
        history.append(rec)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": val_m,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_ckpt_path)
            status = f"* Best (F1={best_val_f1:.4f})"
        else:
            patience_counter += 1
            status = f"No improve ({patience_counter}/{PATIENCE})"

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_m['accuracy']:8.4f} | {val_m['precision']:8.4f} | {val_m['recall']:8.4f} | {val_f1:8.4f} | {val_m['roc_auc']:8.4f} | {status}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\nTraining completed in {time.time() - t_start:.1f}s. Best Epoch: {best_epoch} (Val F1={best_val_f1:.4f})")

    # Save training history
    with open(EXP_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train Loss", marker="o")
    plt.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="Val Loss", marker="s")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([h["epoch"] for h in history], [h["val_accuracy"] for h in history], label="Val Accuracy", marker="o")
    plt.plot([h["epoch"] for h in history], [h["val_f1"] for h in history], label="Val F1", marker="s")
    plt.plot([h["epoch"] for h in history], [h["val_roc_auc"] for h in history], label="Val ROC-AUC", marker="^")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXP_DIR / "training_curves.png", dpi=150)
    plt.close()

    # 4. Load Best Model
    print(f"\nLoading best checkpoint from epoch {best_epoch}...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5. Threshold Selection on Validation Set Only
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION (VALIDATION SET ONLY)")
    print("=" * 70)

    val_loss, val_labels, _, val_probs, _, _ = evaluate_model(model, val_loader, criterion, device)
    thresholds = np.arange(0.05, 0.96, 0.01)
    thresh_records = []

    best_thresh = 0.5
    best_thresh_f1 = -1.0
    best_thresh_metrics = None

    for th in thresholds:
        m = compute_full_metrics(val_labels, val_probs, threshold=th)
        thresh_records.append(m)
        if m["f1"] > best_thresh_f1:
            best_thresh_f1 = m["f1"]
            best_thresh = round(float(th), 2)
            best_thresh_metrics = m

    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv(EXP_DIR / "threshold_analysis.csv", index=False)

    print(f"Selected Optimal Validation Threshold: {best_thresh:.2f}")
    print(f"Validation Metrics at Threshold {best_thresh:.2f}:")
    print(f"  F1:                 {best_thresh_metrics['f1']:.4f}")
    print(f"  Accuracy:           {best_thresh_metrics['accuracy']:.4f}")
    print(f"  Precision:          {best_thresh_metrics['precision']:.4f}")
    print(f"  Recall:             {best_thresh_metrics['recall']:.4f}")
    print(f"  Specificity:        {best_thresh_metrics['specificity']:.4f}")
    print(f"  Balanced Accuracy:  {best_thresh_metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC:            {best_thresh_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:   TP={best_thresh_metrics['tp']}, FP={best_thresh_metrics['fp']}, TN={best_thresh_metrics['tn']}, FN={best_thresh_metrics['fn']}")

    # 6. Held-out Test Set Evaluation with Locked Threshold
    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST EVALUATION (LOCKED THRESHOLD = {best_thresh:.2f})")
    print("=" * 70)

    test_loss, test_labels, test_logits, test_probs, test_dtypes, test_paths = evaluate_model(
        model, test_loader, criterion, device
    )
    test_metrics = compute_full_metrics(test_labels, test_probs, threshold=best_thresh)

    print(f"Test Set Overall Results (N = {len(test_labels)}):")
    print(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"  Precision:          {test_metrics['precision']:.4f}")
    print(f"  Recall:             {test_metrics['recall']:.4f}")
    print(f"  F1:                 {test_metrics['f1']:.4f}")
    print(f"  Specificity:        {test_metrics['specificity']:.4f}")
    print(f"  Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    print(f"  TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")

    # Save test results CSV
    test_res_df = pd.DataFrame([{
        "dataset": "V3 Held-out Test (Balanced)",
        "threshold": best_thresh,
        "n_samples": len(test_labels),
        **test_metrics
    }])
    test_res_df.to_csv(EXP_DIR / "test_results.csv", index=False)

    # Save detailed per-sample predictions
    test_samples_df = pd.DataFrame({
        "path": test_paths,
        "distortion_type": test_dtypes,
        "true_label": test_labels.astype(int),
        "logit": test_logits,
        "probability": test_probs,
        "pred_label": (test_probs >= best_thresh).astype(int),
    })
    test_samples_df.to_csv(EXP_DIR / "test_sample_predictions.csv", index=False)

    # 7. Distortion-Type Analysis on Test Set
    print("\n" + "=" * 70)
    print("PER-DISTORTION-TYPE BREAKDOWN (TEST SET)")
    print("=" * 70)

    per_dist_records = []
    # Distorted samples only for recall, precision, F1, probs
    for dtype in ["elastic", "local", "bending"]:
        mask_d = np.array(test_dtypes) == dtype
        sub_probs = test_probs[mask_d]
        sub_preds = (sub_probs >= best_thresh).astype(int)
        tp_d = int((sub_preds == 1).sum())
        fn_d = int((sub_preds == 0).sum())
        rec_d = tp_d / max(tp_d + fn_d, 1)

        # To calculate precision and F1 for this specific distortion against clean negatives:
        mask_clean = np.array(test_dtypes) == "clean"
        clean_probs = test_probs[mask_clean]
        clean_preds = (clean_probs >= best_thresh).astype(int)
        fp_clean = int((clean_preds == 1).sum())

        prec_d = tp_d / max(tp_d + fp_clean, 1) if (tp_d + fp_clean) > 0 else 0.0
        f1_d = 2 * prec_d * rec_d / max(prec_d + rec_d, 1e-8) if (prec_d + rec_d) > 0 else 0.0
        acc_d = (tp_d + int((clean_preds == 0).sum())) / (len(sub_probs) + len(clean_probs))

        rec = {
            "distortion_type": dtype,
            "n_samples": len(sub_probs),
            "accuracy": round(acc_d, 4),
            "precision": round(prec_d, 4),
            "recall": round(rec_d, 4),
            "f1": round(f1_d, 4),
            "false_negatives": fn_d,
            "mean_prob": round(float(np.mean(sub_probs)), 4),
            "median_prob": round(float(np.median(sub_probs)), 4),
            "min_prob": round(float(np.min(sub_probs)), 4),
            "max_prob": round(float(np.max(sub_probs)), 4),
        }
        per_dist_records.append(rec)
        print(f"--- {dtype.upper()} (N = {len(sub_probs)}) ---")
        print(f"  Recall:            {rec['recall']:.4f}")
        print(f"  Precision:         {rec['precision']:.4f}")
        print(f"  F1:                {rec['f1']:.4f}")
        print(f"  Accuracy:          {rec['accuracy']:.4f}")
        print(f"  False Negatives:   {rec['false_negatives']} / {len(sub_probs)}")
        print(f"  Prob Mean/Median:  {rec['mean_prob']:.4f} / {rec['median_prob']:.4f}")

    per_dist_df = pd.DataFrame(per_dist_records)
    per_dist_df.to_csv(EXP_DIR / "per_distortion_results.csv", index=False)

    # 8. Comparison vs Frozen V1 Baseline
    print("\n" + "=" * 70)
    print("COMPARISON: V1 BASELINE vs V3 CLASSIFIER")
    print("=" * 70)

    # V1 Baseline values from prompt
    # Accuracy = 0.8889, Precision = 0.9534, Recall = 0.8178, F1 = 0.8804, ROC-AUC = 0.9432, Bending Rec = 0.4533, Bending F1 = 0.6239
    bending_sub = per_dist_df[per_dist_df["distortion_type"] == "bending"].iloc[0]

    comp_rows = [
        {"metric": "Test Accuracy",        "v1_baseline": 0.8889, "v3_classifier": test_metrics["accuracy"]},
        {"metric": "Test Precision",       "v1_baseline": 0.9534, "v3_classifier": test_metrics["precision"]},
        {"metric": "Test Recall",          "v1_baseline": 0.8178, "v3_classifier": test_metrics["recall"]},
        {"metric": "Test F1",              "v1_baseline": 0.8804, "v3_classifier": test_metrics["f1"]},
        {"metric": "Test ROC-AUC",         "v1_baseline": 0.9432, "v3_classifier": test_metrics["roc_auc"]},
        {"metric": "Bending Recall",       "v1_baseline": 0.4533, "v3_classifier": bending_sub["recall"]},
        {"metric": "Bending F1",           "v1_baseline": 0.6239, "v3_classifier": bending_sub["f1"]},
    ]
    for r in comp_rows:
        diff = r["v3_classifier"] - r["v1_baseline"]
        pct = (diff / r["v1_baseline"]) * 100.0
        r["change_abs"] = round(diff, 4)
        r["change_pct"] = f"{pct:+.2f}%"

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(EXP_DIR / "comparison_v1_vs_v3.csv", index=False)
    print(comp_df.to_string(index=False))

    # 9. Real Unseen Fingerprint Check
    print("\n" + "=" * 70)
    print("REAL UNSEEN CLEAN FINGERPRINT CHECK (data/split/test/)")
    print("=" * 70)

    real_clean_ds = RealCleanTestDataset(RAW_TEST_ROOT)
    real_clean_loader = DataLoader(real_clean_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_real_probs = []
    all_real_paths = []
    with torch.no_grad():
        for batch in real_clean_loader:
            imgs = batch["image"].to(device)
            logits = model(imgs).squeeze(1)
            probs = sigmoid(logits.cpu().numpy())
            all_real_probs.extend(probs)
            all_real_paths.extend(batch["path"])

    all_real_probs = np.array(all_real_probs)
    real_preds = (all_real_probs >= best_thresh).astype(int)

    n_real = len(all_real_probs)
    n_pred_clean = int((real_preds == 0).sum())
    n_pred_dist = int((real_preds == 1).sum())
    fpr_real = n_pred_dist / max(n_real, 1)

    real_results_df = pd.DataFrame({
        "path": all_real_paths,
        "probability": all_real_probs,
        "predicted_label": real_preds,
        "predicted_class": ["DISTORTED" if p == 1 else "CLEAN" for p in real_preds]
    })
    real_results_df.to_csv(EXP_DIR / "real_clean_test_results.csv", index=False)

    print(f"Total Real Clean Test Images: {n_real}")
    print(f"Correctly Predicted CLEAN:     {n_pred_clean} ({n_pred_clean/n_real*100:.2f}%)")
    print(f"Falsely Predicted DISTORTED:   {n_pred_dist} ({fpr_real*100:.2f}%)")
    print(f"False Positive Rate on Real:   {fpr_real:.4f}")
    print(f"Probability Distribution on Real Clean:")
    print(f"  Min:    {all_real_probs.min():.4f}")
    print(f"  Mean:   {all_real_probs.mean():.4f}")
    print(f"  Median: {np.median(all_real_probs):.4f}")
    print(f"  Max:    {all_real_probs.max():.4f}")

    # Plot Confusion Matrix
    cm = np.array([[test_metrics["tn"], test_metrics["fp"]],
                   [test_metrics["fn"], test_metrics["tp"]]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix (Threshold = {best_thresh:.2f})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["CLEAN (0)", "DISTORTED (1)"])
    plt.yticks(tick_marks, ["CLEAN (0)", "DISTORTED (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(EXP_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # 10. Save Configuration
    config = {
        "model_architecture": "FingerprintCNN",
        "parameters": param_count,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_val_threshold": best_thresh,
        "dataset_v3_root": str(DATA_V3_ROOT),
        "raw_test_root": str(RAW_TEST_ROOT),
    }
    with open(EXP_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 11. Write Full Evaluation Report
    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("CUSTOM CNN DISTORTION CLASSIFIER EVALUATION REPORT (V3 DATASET)")
    report_lines.append("=" * 75)
    report_lines.append(f"Model Architecture:          FingerprintCNN (4 Conv blocks + Global Avg Pool + FC)")
    report_lines.append(f"Trainable Parameters:        {param_count:,}")
    report_lines.append(f"Training Epochs:             {len(history)} (Best epoch: {best_epoch})")
    report_lines.append(f"Optimal Validation Threshold:{best_thresh:.2f} (Tuned strictly on validation set)")
    report_lines.append("")
    report_lines.append("1. BALANCED DATASET SAMPLE SIZES")
    report_lines.append("-" * 75)
    report_lines.append(f"Train: {len(train_ds)} (1050 clean + 350 elastic + 350 local + 350 bending)")
    report_lines.append(f"Val:    {len(val_ds)} ( 225 clean +  75 elastic +  75 local +  75 bending)")
    report_lines.append(f"Test:   {len(test_ds)} ( 225 clean +  75 elastic +  75 local +  75 bending)")
    report_lines.append("")
    report_lines.append("2. OVERALL HELD-OUT TEST METRICS (THRESHOLD = {:.2f})".format(best_thresh))
    report_lines.append("-" * 75)
    report_lines.append(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    report_lines.append(f"Precision:          {test_metrics['precision']:.4f}")
    report_lines.append(f"Recall:             {test_metrics['recall']:.4f}")
    report_lines.append(f"F1-Score:           {test_metrics['f1']:.4f}")
    report_lines.append(f"Specificity:        {test_metrics['specificity']:.4f}")
    report_lines.append(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    report_lines.append(f"ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    report_lines.append(f"Confusion Matrix:   TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")
    report_lines.append("")
    report_lines.append("3. PER-DISTORTION BREAKDOWN ON TEST SET")
    report_lines.append("-" * 75)
    for _, r in per_dist_df.iterrows():
        report_lines.append(f"  {r['distortion_type'].upper():7s} (N={int(r['n_samples'])}): Recall={r['recall']:.4f} | Prec={r['precision']:.4f} | F1={r['f1']:.4f} | FN={int(r['false_negatives'])} | Mean Prob={r['mean_prob']:.4f}")
    report_lines.append("")
    report_lines.append("4. COMPARISON vs FROZEN V1 BASELINE")
    report_lines.append("-" * 75)
    for _, r in comp_df.iterrows():
        report_lines.append(f"  {r['metric']:20s} | V1: {r['v1_baseline']:.4f} | V3: {r['v3_classifier']:.4f} | Change: {r['change_pct']}")
    report_lines.append("")
    report_lines.append("5. REAL UNSEEN CLEAN FINGERPRINT AUDIT (data/split/test/)")
    report_lines.append("-" * 75)
    report_lines.append(f"Total Real Images:           {n_real}")
    report_lines.append(f"Correctly Classified Clean:  {n_pred_clean} ({n_pred_clean/n_real*100:.2f}%)")
    report_lines.append(f"Falsely Classified Distorted:{n_pred_dist} ({fpr_real*100:.2f}%)")
    report_lines.append(f"Real False Positive Rate:    {fpr_real:.4f}")
    report_lines.append(f"Probability Mean/Median:     {all_real_probs.mean():.4f} / {np.median(all_real_probs):.4f}")
    report_lines.append("")
    report_lines.append("6. DECISION SUMMARY")
    report_lines.append("-" * 75)
    report_lines.append(f"Did overall classifier improve?         {'YES' if test_metrics['f1'] >= 0.8804 else 'NO'}")
    report_lines.append(f"Did bending detection improve?          {'YES' if bending_sub['recall'] > 0.4533 else 'NO'} (Recall: {bending_sub['recall']:.4f} vs 0.4533)")
    report_lines.append(f"Did elastic detection remain strong?    {'YES' if per_dist_df[per_dist_df['distortion_type']=='elastic']['recall'].values[0] >= 0.85 else 'NO'}")
    report_lines.append(f"Did local detection remain strong?      {'YES' if per_dist_df[per_dist_df['distortion_type']=='local']['recall'].values[0] >= 0.85 else 'NO'}")
    report_lines.append(f"Final Held-out Test F1:                 {test_metrics['f1']:.4f}")
    report_lines.append(f"Distorted test classified as clean (FN):{test_metrics['fn']} / 225")
    report_lines.append(f"Real clean classified as distorted (FP):{n_pred_dist} / 225")
    report_lines.append("=" * 75)

    with open(EXP_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll results, checkpoints, and reports saved to {EXP_DIR}")


if __name__ == "__main__":
    main()
