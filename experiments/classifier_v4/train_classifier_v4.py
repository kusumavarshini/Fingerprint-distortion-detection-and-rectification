"""
V4 Classifier Training and Evaluation Script
==============================================
Train and evaluate the FingerprintCNN (EXACT same architecture as V1/V3,
389,057 parameters) on the V4 dataset.

V4 design: symmetric quality augmentation decouples image-quality variation
from geometric distortion, targeting real-clean FPR <= 5%.

Constraints
-----------
- Exact FingerprintCNN architecture (UNCHANGED from V1/V3)
- BCEWithLogitsLoss, Adam, LR=1e-4, WD=1e-4, max 15 epochs, patience=5
- Threshold selected ONLY from validation set:
    primary rule: maximize F1 subject to clean-FPR <= 5%
    fallback: lowest achievable FPR if constraint cannot be met
- Test set evaluated ONCE with locked threshold
- V1/V3 artifacts NOT modified, app.py NOT modified, DDRNet NOT modified

Outputs saved to experiments/classifier_v4/
  best_model.pth
  v4_training_report.txt
  v4_test_results.json
  v4_confusion_matrix.png
  training_history.json
  v4_threshold_analysis.csv
  v4_per_distortion_results.csv
  v4_comparison_table.csv
  v4_real_clean_audit.csv
  v4_training_curves.png
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

# Locate project root (fingerprint/) from experiments/classifier_v4/
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import FingerprintCNN, count_parameters
from evaluate import _compute_roc_auc, sigmoid

# =============================================================================
# CONFIG
# =============================================================================

SEED         = 42
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 15
PATIENCE     = 5

DATA_V4_ROOT  = ROOT_DIR / "data" / "classifier_v4"
RAW_TEST_ROOT = ROOT_DIR / "data" / "split" / "test"   # frozen real-clean audit
EXP_DIR       = SCRIPT_DIR

# Known baseline results (used ONLY for comparison table)
V1 = dict(
    accuracy=0.8889, precision=0.9534, recall=0.8178,
    f1=0.8804, roc_auc=0.9432, bending_recall=0.4533,
    real_clean_fpr=None
)
V3 = dict(
    accuracy=0.8556, precision=0.8077, recall=0.9333,
    f1=0.8660, roc_auc=0.9491, bending_recall=0.9333,
    real_clean_fpr=0.2222
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws)
    random.seed(ws)


# =============================================================================
# DATASET
# =============================================================================

class V4Dataset(Dataset):
    """
    Loads V4 train / val / test split.
    Label: 0 = clean, 1 = distorted.
    Parses distortion type from filename for per-distortion evaluation.
    """
    def __init__(self, split_dir: Path, split_name: str):
        super().__init__()
        self.split_name = split_name
        self.samples    = []

        clean_dir = split_dir / "clean"
        dist_dir  = split_dir / "distorted"

        for f in sorted(clean_dir.glob("*.png")):
            self.samples.append({
                "path": str(f), "filename": f.name,
                "label": 0.0, "distortion_type": "clean"
            })

        for f in sorted(dist_dir.glob("*.png")):
            # V4 naming: std_dist_NNNN__DTYPE__... or qaug_dist_NNNN__aug__DTYPE__...
            # V3 naming (val/test copied from V3): FAMILY__MEMBER__STEM__DTYPE.png
            name  = f.name
            dtype = "unknown"
            for dt in ["elastic", "local", "bending"]:
                if ("__" + dt + "__") in name or name.endswith("__" + dt + ".png"):
                    dtype = dt
                    break
            self.samples.append({
                "path": str(f), "filename": f.name,
                "label": 1.0, "distortion_type": dtype
            })

        rng_offset = {"train": 100, "val": 200, "test": 300}.get(split_name, 0)
        rng = random.Random(SEED + rng_offset)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {s['path']}")
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            "image":           tensor,
            "label":           torch.tensor(s["label"], dtype=torch.float32),
            "path":            s["path"],
            "filename":        s["filename"],
            "distortion_type": s["distortion_type"],
        }

    def get_breakdown(self):
        counts = {}
        for s in self.samples:
            dt = s["distortion_type"]
            counts[dt] = counts.get(dt, 0) + 1
        return counts


class RealCleanAuditDataset(Dataset):
    """Frozen real-clean fingerprints from data/split/test/ (audit only)."""
    def __init__(self, root: Path):
        self.files = sorted(root.glob("**/*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p   = self.files[idx]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {p}")
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return {"image": tensor, "path": str(p), "filename": p.name}


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n  = max(tp + fp + tn + fn, 1)
    acc  = (tp + tn) / n
    prec = tp / max(tp + fp, 1e-9)
    rec  = tp / max(tp + fn, 1e-9)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    spec = tn / max(tn + fp, 1e-9)
    bal  = (rec + spec) / 2.0
    auc  = _compute_roc_auc(labels, probs)
    return {
        "threshold":        round(float(threshold), 4),
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "specificity":      round(spec, 4),
        "balanced_accuracy":round(bal,  4),
        "roc_auc":          round(auc,  4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def eval_loader(model, loader, criterion, device):
    model.eval()
    logits_l, labels_l, dtypes_l, paths_l = [], [], [], []
    tot_loss, nb = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbs  = batch["label"].to(device)
            lg   = model(imgs).squeeze(1)
            loss = criterion(lg, lbs)
            tot_loss += loss.item()
            nb += 1
            logits_l.append(lg.cpu().numpy())
            labels_l.append(lbs.cpu().numpy())
            dtypes_l.extend(batch.get("distortion_type", [""] * len(imgs)))
            paths_l.extend(batch.get("path",             [""] * len(imgs)))
    la = np.concatenate(logits_l)
    lb = np.concatenate(labels_l)
    return tot_loss / max(nb, 1), lb, la, sigmoid(la), dtypes_l, paths_l


# =============================================================================
# THRESHOLD SELECTION  (validation-only, FPR-constrained)
# =============================================================================

def select_threshold(val_labels, val_probs, max_fpr=0.05):
    """
    Primary:  maximize F1 subject to clean-FPR <= max_fpr.
    Fallback: if no threshold satisfies constraint, use lowest achievable FPR.
    clean-FPR = fraction of clean samples (label==0) predicted DISTORTED.
    """
    thresholds  = np.arange(0.05, 0.96, 0.01)
    clean_mask  = val_labels == 0
    n_clean     = int(clean_mask.sum())
    records     = []
    candidates  = []

    for th in thresholds:
        m    = compute_metrics(val_labels, val_probs, th)
        cfpr = float((val_probs[clean_mask] >= th).sum()) / max(n_clean, 1)
        records.append({**m, "clean_fpr": round(cfpr, 4)})
        if cfpr <= max_fpr:
            candidates.append((float(th), m["f1"], cfpr, m))

    if candidates:
        best        = max(candidates, key=lambda x: x[1])
        chosen_th   = round(best[0], 2)
        chosen_m    = best[3]
        constraint_met = True
        print(f"  [PASS] Constraint (clean-FPR <= {max_fpr:.0%}) MET.")
        print(f"         Threshold={chosen_th:.2f}  F1={best[1]:.4f}  "
              f"clean-FPR={best[2]:.4f}")
    else:
        best        = min(records, key=lambda r: (r["clean_fpr"], -r["f1"]))
        chosen_th   = round(float(best["threshold"]), 2)
        chosen_m    = best
        constraint_met = False
        print(f"  [WARN] No threshold satisfies clean-FPR <= {max_fpr:.0%}.")
        print(f"         Fallback: threshold={chosen_th:.2f}  "
              f"clean-FPR={best['clean_fpr']:.4f}")

    return chosen_th, chosen_m, records, constraint_met


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 72)
    print("V4 CLASSIFIER  --  FingerprintCNN TRAINING & EVALUATION")
    print("Ref: experiments/classifier_v4/v4_training_plan.txt")
    print("=" * 72)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD DATASETS
    # -------------------------------------------------------------------------
    print("\nLoading V4 datasets ...")
    train_ds = V4Dataset(DATA_V4_ROOT / "train", "train")
    val_ds   = V4Dataset(DATA_V4_ROOT / "val",   "val")
    test_ds  = V4Dataset(DATA_V4_ROOT / "test",  "test")

    print(f"Train: {train_ds.get_breakdown()}  | Total: {len(train_ds)}")
    print(f"Val:   {val_ds.get_breakdown()}   | Total: {len(val_ds)}")
    print(f"Test:  {test_ds.get_breakdown()}   | Total: {len(test_ds)}")

    # Pre-training verification
    print("\n--- PRE-TRAINING COUNT VERIFICATION ---")
    assert len(train_ds) == 2100, f"Train={len(train_ds)} != 2100"
    assert len(val_ds)   ==  900, f"Val={len(val_ds)} != 900"
    assert len(test_ds)  ==  900, f"Test={len(test_ds)} != 900"
    print(f"  Train: {len(train_ds)}  [PASS]")
    print(f"  Val:   {len(val_ds)}    [PASS]")
    print(f"  Test:  {len(test_ds)}   [PASS]")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -------------------------------------------------------------------------
    # 2. MODEL VERIFICATION
    # -------------------------------------------------------------------------
    model      = FingerprintCNN(dropout=0.5).to(device)
    param_count = count_parameters(model)
    print(f"\nModel: FingerprintCNN")
    print(f"Trainable parameters: {param_count:,} (expected: 389,057)")
    assert param_count == 389057, f"[FAIL] Parameter mismatch: {param_count}"
    print(f"  [PASS] Parameter count verified: {param_count:,}")

    # -------------------------------------------------------------------------
    # 3. LOSS & OPTIMIZER
    # -------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_ckpt     = EXP_DIR / "best_model.pth"
    history       = []
    best_val_f1   = -1.0
    best_epoch    = 0
    patience_cnt  = 0

    # -------------------------------------------------------------------------
    # 4. TRAINING LOOP
    # -------------------------------------------------------------------------
    print("\nStarting training ...")
    hdr = (f"{'Ep':>3} | {'TrLoss':>8} | {'VaLoss':>8} | "
           f"{'VaAcc':>7} | {'VaPrec':>7} | {'VaRec':>7} | "
           f"{'VaF1':>7} | {'VaAUC':>7} | Status")
    print(hdr)
    print("-" * len(hdr))

    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_t0 = time.time()

        # Train
        model.train()
        tr_loss, n_tr = 0.0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            n_tr    += 1
        tr_loss /= max(n_tr, 1)

        # Validate
        va_loss, va_lbl, _, va_prb, _, _ = eval_loader(model, val_loader, criterion, device)
        va_m = compute_metrics(va_lbl, va_prb, threshold=0.5)

        rec = {
            "epoch":         epoch,
            "train_loss":    round(tr_loss, 5),
            "val_loss":      round(va_loss, 5),
            "val_accuracy":  va_m["accuracy"],
            "val_precision": va_m["precision"],
            "val_recall":    va_m["recall"],
            "val_f1":        va_m["f1"],
            "val_roc_auc":   va_m["roc_auc"],
            "epoch_time_s":  round(time.time() - ep_t0, 1),
        }
        history.append(rec)

        if va_m["f1"] > best_val_f1:
            best_val_f1  = va_m["f1"]
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               best_val_f1,
                "val_metrics":          va_m,
                "train_loss":           tr_loss,
                "val_loss":             va_loss,
            }, best_ckpt)
            status = f"* Best (F1={best_val_f1:.4f})"
        else:
            patience_cnt += 1
            status = f"No improve ({patience_cnt}/{PATIENCE})"

        print(f"{epoch:3d} | {tr_loss:8.4f} | {va_loss:8.4f} | "
              f"{va_m['accuracy']:7.4f} | {va_m['precision']:7.4f} | "
              f"{va_m['recall']:7.4f} | {va_m['f1']:7.4f} | "
              f"{va_m['roc_auc']:7.4f} | {status}")

        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE}).")
            break

    elapsed_train = time.time() - t_start
    print(f"\nTraining finished in {elapsed_train:.1f}s.  "
          f"Best epoch: {best_epoch}  (Val F1={best_val_f1:.4f})")

    # Save history
    with open(EXP_DIR / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # Training curves
    eps = [h["epoch"] for h in history]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(eps, [h["train_loss"] for h in history], "o-", label="Train Loss")
    plt.plot(eps, [h["val_loss"]   for h in history], "s-", label="Val Loss")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Loss Curves")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(eps, [h["val_accuracy"] for h in history], "o-", label="Val Acc")
    plt.plot(eps, [h["val_f1"]       for h in history], "s-", label="Val F1")
    plt.plot(eps, [h["val_roc_auc"]  for h in history], "^-", label="Val AUC")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation Metrics")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXP_DIR / "v4_training_curves.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # 5. LOAD BEST CHECKPOINT
    # -------------------------------------------------------------------------
    print(f"\nLoading best checkpoint (epoch {best_epoch}) ...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -------------------------------------------------------------------------
    # 6. THRESHOLD SELECTION ON VALIDATION SET ONLY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("THRESHOLD SELECTION  (validation set, clean-FPR constraint <= 5%)")
    print("=" * 72)

    _, va_lbl_f, _, va_prb_f, _, _ = eval_loader(model, val_loader, criterion, device)

    chosen_th, chosen_val_m, thresh_records, constraint_met = select_threshold(
        va_lbl_f, va_prb_f, max_fpr=0.05)

    pd.DataFrame(thresh_records).to_csv(EXP_DIR / "v4_threshold_analysis.csv", index=False)

    print(f"\n  Selected threshold : {chosen_th:.2f}")
    print(f"  Accuracy:          {chosen_val_m['accuracy']:.4f}")
    print(f"  Precision:         {chosen_val_m['precision']:.4f}")
    print(f"  Recall:            {chosen_val_m['recall']:.4f}")
    print(f"  F1:                {chosen_val_m['f1']:.4f}")
    print(f"  Specificity:       {chosen_val_m['specificity']:.4f}")
    print(f"  Balanced Accuracy: {chosen_val_m['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC:           {chosen_val_m['roc_auc']:.4f}")
    print(f"  TP={chosen_val_m['tp']} FP={chosen_val_m['fp']} "
          f"TN={chosen_val_m['tn']} FN={chosen_val_m['fn']}")

    # -------------------------------------------------------------------------
    # 7. HELD-OUT TEST EVALUATION  (one shot, locked threshold)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"HELD-OUT TEST EVALUATION  (locked threshold = {chosen_th:.2f})")
    print("=" * 72)

    te_loss, te_lbl, te_lg, te_prb, te_dt, te_paths = eval_loader(
        model, test_loader, criterion, device)
    test_m = compute_metrics(te_lbl, te_prb, threshold=chosen_th)

    print(f"Test N = {len(te_lbl)}")
    for k in ["accuracy", "precision", "recall", "f1",
               "specificity", "balanced_accuracy", "roc_auc"]:
        print(f"  {k:<22}: {test_m[k]:.4f}")
    print(f"  TP={test_m['tp']} FP={test_m['fp']} "
          f"TN={test_m['tn']} FN={test_m['fn']}")

    # -------------------------------------------------------------------------
    # 8. PER-DISTORTION TEST BREAKDOWN
    # -------------------------------------------------------------------------
    print("\n--- Per-Distortion Test Breakdown ---")
    te_dt_arr = np.array(te_dt)
    clean_mask_te  = te_dt_arr == "clean"
    fp_clean_te    = int((te_prb[clean_mask_te] >= chosen_th).sum())

    per_dist = []
    for dtype in ["elastic", "local", "bending"]:
        mask     = te_dt_arr == dtype
        sub_prb  = te_prb[mask]
        sub_pred = (sub_prb >= chosen_th).astype(int)
        tp_d = int((sub_pred == 1).sum())
        fn_d = int((sub_pred == 0).sum())
        n_d  = len(sub_prb)
        rec_d  = tp_d / max(tp_d + fn_d, 1)
        prec_d = tp_d / max(tp_d + fp_clean_te, 1)
        f1_d   = 2 * prec_d * rec_d / max(prec_d + rec_d, 1e-9)
        per_dist.append({
            "distortion_type": dtype, "n_samples": n_d,
            "recall": round(rec_d, 4), "precision": round(prec_d, 4),
            "f1": round(f1_d, 4), "false_negatives": fn_d,
            "mean_prob":   round(float(sub_prb.mean()), 4),
            "median_prob": round(float(np.median(sub_prb)), 4),
            "min_prob":    round(float(sub_prb.min()), 4),
            "max_prob":    round(float(sub_prb.max()), 4),
        })
        print(f"  {dtype.upper():8s} (N={n_d:4d}) | "
              f"Recall={rec_d:.4f} | Prec={prec_d:.4f} | "
              f"F1={f1_d:.4f} | FN={fn_d}")

    per_dist_df = pd.DataFrame(per_dist)
    per_dist_df.to_csv(EXP_DIR / "v4_per_distortion_results.csv", index=False)

    # Save per-sample test predictions
    pd.DataFrame({
        "path":            te_paths,
        "distortion_type": te_dt,
        "true_label":      te_lbl.astype(int),
        "logit":           te_lg,
        "probability":     te_prb,
        "pred_label":      (te_prb >= chosen_th).astype(int),
    }).to_csv(EXP_DIR / "v4_test_sample_predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # 9. CONFUSION MATRIX
    # -------------------------------------------------------------------------
    cm = np.array([[test_m["tn"], test_m["fp"]],
                   [test_m["fn"], test_m["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["CLEAN (0)", "DISTORTED (1)"])
    ax.set_yticklabels(["CLEAN (0)", "DISTORTED (1)"])
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title(f"V4 Test Confusion Matrix  (threshold={chosen_th:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(EXP_DIR / "v4_confusion_matrix.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # 10. REAL-CLEAN AUDIT  (frozen, NOT used for threshold or training)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("REAL-CLEAN FINGERPRINT AUDIT  (data/split/test/ -- frozen)")
    print("=" * 72)

    real_ds     = RealCleanAuditDataset(RAW_TEST_ROOT)
    real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    real_prb_list, real_path_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in real_loader:
            imgs   = batch["image"].to(device)
            logits = model(imgs).squeeze(1)
            real_prb_list.extend(sigmoid(logits.cpu().numpy()).tolist())
            real_path_list.extend(batch["path"])

    real_prb   = np.array(real_prb_list)
    real_pred  = (real_prb >= chosen_th).astype(int)
    n_real     = len(real_prb)
    n_real_ok  = int((real_pred == 0).sum())
    n_real_fp  = int((real_pred == 1).sum())
    real_fpr   = n_real_fp / max(n_real, 1)

    pd.DataFrame({
        "path":            real_path_list,
        "probability":     real_prb,
        "predicted_label": real_pred,
        "predicted_class": ["DISTORTED" if p == 1 else "CLEAN" for p in real_pred],
    }).to_csv(EXP_DIR / "v4_real_clean_audit.csv", index=False)

    print(f"  Total real clean images:         {n_real}")
    print(f"  Correctly predicted CLEAN:       {n_real_ok}  ({n_real_ok/n_real*100:.2f}%)")
    print(f"  Falsely predicted DISTORTED:     {n_real_fp}  ({real_fpr*100:.2f}%)")
    print(f"  Real-clean false-positive rate:  {real_fpr:.4f}")
    print(f"  Probability statistics:")
    print(f"    Min    : {real_prb.min():.4f}")
    print(f"    Mean   : {real_prb.mean():.4f}")
    print(f"    Median : {np.median(real_prb):.4f}")
    print(f"    Std    : {real_prb.std():.4f}")
    print(f"    Max    : {real_prb.max():.4f}")
    fpr_status = "[PASS] <= 5.0% target met" if real_fpr <= 0.05 else f"[ABOVE TARGET] {real_fpr*100:.2f}% > 5.0%"
    print(f"  FPR target (<=5%):               {fpr_status}")

    # -------------------------------------------------------------------------
    # 11. V1 vs V3 vs V4 COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("V1  vs  V3  vs  V4  COMPARISON")
    print("=" * 72)

    bending_row = per_dist_df[per_dist_df["distortion_type"] == "bending"].iloc[0]

    comp = [
        ("Test Accuracy",    V1["accuracy"],        V3["accuracy"],        test_m["accuracy"]),
        ("Test Precision",   V1["precision"],       V3["precision"],       test_m["precision"]),
        ("Test Recall",      V1["recall"],          V3["recall"],          test_m["recall"]),
        ("Test F1",          V1["f1"],              V3["f1"],              test_m["f1"]),
        ("Test ROC-AUC",     V1["roc_auc"],         V3["roc_auc"],         test_m["roc_auc"]),
        ("Bending Recall",   V1["bending_recall"],  V3["bending_recall"],  float(bending_row["recall"])),
        ("Real-Clean FPR",   V1["real_clean_fpr"], V3["real_clean_fpr"],  real_fpr),
    ]

    print(f"  {'Metric':<22} | {'V1':>8} | {'V3':>8} | {'V4':>8} | {'V3->V4':>10}")
    print("  " + "-" * 65)
    for name, v1v, v3v, v4v in comp:
        v1s = f"{v1v:.4f}" if v1v is not None else "  N/A  "
        v3s = f"{v3v:.4f}" if v3v is not None else "  N/A  "
        if v3v is not None:
            delta = v4v - v3v
            ds    = f"{delta:+.4f}"
        else:
            ds = "N/A"
        print(f"  {name:<22} | {v1s:>8} | {v3s:>8} | {v4v:>8.4f} | {ds:>10}")

    pd.DataFrame([
        {"metric": n, "v1": v1v, "v3": v3v, "v4": v4v}
        for n, v1v, v3v, v4v in comp
    ]).to_csv(EXP_DIR / "v4_comparison_table.csv", index=False)

    # -------------------------------------------------------------------------
    # 12. SAVE JSON
    # -------------------------------------------------------------------------
    out_json = {
        "dataset_version": "V4",
        "model": "FingerprintCNN", "parameters": param_count,
        "seed": SEED, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS, "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_f1_at_0.5": round(best_val_f1, 4),
        "selected_threshold": chosen_th,
        "threshold_constraint_met": constraint_met,
        "validation_metrics": chosen_val_m,
        "test_metrics": test_m,
        "per_distortion_test": per_dist,
        "real_clean_audit": {
            "n_total": n_real,
            "n_correctly_clean": n_real_ok,
            "n_false_distorted": n_real_fp,
            "false_positive_rate": round(real_fpr, 4),
            "prob_min":    round(float(real_prb.min()),    4),
            "prob_mean":   round(float(real_prb.mean()),   4),
            "prob_median": round(float(np.median(real_prb)), 4),
            "prob_std":    round(float(real_prb.std()),    4),
            "prob_max":    round(float(real_prb.max()),    4),
        },
        "comparison": {
            "v1_f1": V1["f1"], "v3_f1": V3["f1"], "v4_f1": test_m["f1"],
            "v1_roc_auc": V1["roc_auc"], "v3_roc_auc": V3["roc_auc"],
            "v4_roc_auc": test_m["roc_auc"],
            "v3_real_clean_fpr": V3["real_clean_fpr"],
            "v4_real_clean_fpr": round(real_fpr, 4),
        },
    }
    with open(EXP_DIR / "v4_test_results.json", "w") as fh:
        json.dump(out_json, fh, indent=2)

    # -------------------------------------------------------------------------
    # 13. TEXT REPORT
    # -------------------------------------------------------------------------
    improved_fpr = real_fpr < V3["real_clean_fpr"]
    improved_f1  = test_m["f1"] >= V3["f1"]
    recommend = (
        "V4 REPLACES V3 as production classifier"
        if (improved_fpr and test_m["roc_auc"] >= 0.90)
        else "Further investigation required"
    )

    R = [
        "=" * 75,
        "V4 CLASSIFIER TRAINING & EVALUATION REPORT",
        "=" * 75, "",
        f"Model:               FingerprintCNN (4-Conv + GAP + FC, unchanged)",
        f"Parameters:          {param_count:,}",
        f"Seed:                {SEED}",
        f"Batch size:          {BATCH_SIZE}",
        f"Learning rate:       {LR}",
        f"Weight decay:        {WEIGHT_DECAY}",
        f"Max epochs:          {MAX_EPOCHS}  (patience={PATIENCE})",
        f"Epochs trained:      {len(history)}",
        f"Best epoch:          {best_epoch}",
        f"Best val F1@0.5:     {best_val_f1:.4f}",
        f"Selected threshold:  {chosen_th:.2f}  (constraint met: {constraint_met})",
        "",
        "1. DATASET COUNTS",
        "-" * 75,
        f"  Train : {len(train_ds)}  (1050 clean + 1050 distorted, quality-augmented)",
        f"  Val   : {len(val_ds)}    ( 225 clean +  675 distorted, frozen from V3)",
        f"  Test  : {len(test_ds)}   ( 225 clean +  675 distorted, frozen from V3)",
        "",
        f"2. VALIDATION METRICS  (threshold={chosen_th:.2f})",
        "-" * 75,
        f"  Accuracy:          {chosen_val_m['accuracy']:.4f}",
        f"  Precision:         {chosen_val_m['precision']:.4f}",
        f"  Recall:            {chosen_val_m['recall']:.4f}",
        f"  F1:                {chosen_val_m['f1']:.4f}",
        f"  Specificity:       {chosen_val_m['specificity']:.4f}",
        f"  Balanced Accuracy: {chosen_val_m['balanced_accuracy']:.4f}",
        f"  ROC-AUC:           {chosen_val_m['roc_auc']:.4f}",
        f"  Confusion:         TP={chosen_val_m['tp']} FP={chosen_val_m['fp']} TN={chosen_val_m['tn']} FN={chosen_val_m['fn']}",
        "",
        f"3. TEST METRICS  (locked threshold={chosen_th:.2f})",
        "-" * 75,
        f"  Accuracy:          {test_m['accuracy']:.4f}",
        f"  Precision:         {test_m['precision']:.4f}",
        f"  Recall:            {test_m['recall']:.4f}",
        f"  F1:                {test_m['f1']:.4f}",
        f"  Specificity:       {test_m['specificity']:.4f}",
        f"  Balanced Accuracy: {test_m['balanced_accuracy']:.4f}",
        f"  ROC-AUC:           {test_m['roc_auc']:.4f}",
        f"  Confusion:         TP={test_m['tp']} FP={test_m['fp']} TN={test_m['tn']} FN={test_m['fn']}",
        "",
        "4. PER-DISTORTION TEST RESULTS",
        "-" * 75,
    ]
    for r in per_dist:
        R.append(
            f"  {r['distortion_type'].upper():8s} (N={r['n_samples']:4d}): "
            f"Recall={r['recall']:.4f}  Prec={r['precision']:.4f}  "
            f"F1={r['f1']:.4f}  FN={r['false_negatives']}  "
            f"MeanProb={r['mean_prob']:.4f}"
        )
    R += [
        "",
        "5. REAL-CLEAN FINGERPRINT AUDIT  (data/split/test/)",
        "-" * 75,
        f"  Total images:         {n_real}",
        f"  Correctly CLEAN:      {n_real_ok}  ({n_real_ok/n_real*100:.2f}%)",
        f"  Falsely DISTORTED:    {n_real_fp}  ({real_fpr*100:.2f}%)",
        f"  Real-clean FPR:       {real_fpr:.4f}",
        f"  Prob min/mean/med/max:{real_prb.min():.4f} / {real_prb.mean():.4f} / {np.median(real_prb):.4f} / {real_prb.max():.4f}",
        "",
        "6. V1  vs  V3  vs  V4  COMPARISON",
        "-" * 75,
    ]
    for name, v1v, v3v, v4v in comp:
        v1s = f"{v1v:.4f}" if v1v is not None else "N/A"
        v3s = f"{v3v:.4f}" if v3v is not None else "N/A"
        ds  = f"{v4v - v3v:+.4f}" if v3v is not None else "N/A"
        R.append(f"  {name:<22}: V1={v1s}  V3={v3s}  V4={v4v:.4f}  ({ds})")
    R += [
        "",
        "7. PRODUCTION DECISION",
        "-" * 75,
        f"  FPR improved vs V3?           {'YES (' + str(round(real_fpr*100,2)) + '% vs 22.22%)' if improved_fpr else 'NO'}",
        f"  F1 maintained/improved?       {'YES' if improved_f1 else 'NO'}  ({test_m['f1']:.4f} vs {V3['f1']:.4f})",
        f"  Bending recall maintained?    {'YES' if bending_row['recall'] >= 0.85 else 'NO'}  ({bending_row['recall']:.4f})",
        f"  RECOMMENDATION: {recommend}",
        "=" * 75,
    ]

    with open(EXP_DIR / "v4_training_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(R))

    print(f"\nAll artifacts saved to: {EXP_DIR}")
    print("  best_model.pth")
    print("  v4_training_report.txt")
    print("  v4_test_results.json")
    print("  v4_confusion_matrix.png")
    print("\n--- DONE ---")


if __name__ == "__main__":
    main()
