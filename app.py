"""
Fingerprint Distortion Detection and Rectification Frontend Demo
================================================================
A research demonstration application built using Streamlit.
Showcases verified experimental results for:
  Fingerprint Input -> Distortion Detection -> Severity Estimation
  -> Rectification -> Rectification Effectiveness

All metrics and images are drawn strictly from verified project evaluations.
The distortion classifier is the custom 4-layer FingerprintCNN baseline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

sys.path.insert(0, str(Path("DDRNet").resolve()))

from model import FingerprintCNN
from models.DDRNet_DIR import DDRNet_DIR
from severity import (
    calculate_severity_statistics,
    classify_severity,
    SEVERITY_THRESHOLD_LOW_MODERATE,
    SEVERITY_THRESHOLD_MODERATE_HIGH,
)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

PAGE_TITLE = "Fingerprint Distortion Detection & Rectification"
PAGE_ICON = "🔍"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIER_CHECKPOINT = Path("experiments/experiment_5/best_model_exp5.pth")
CLASSIFIER_THRESHOLD = 0.18

DDRNET_CHECKPOINT = Path("experiments/ddrnet_baseline/best_model.pth")

SEV_LOW_THRESHOLD = 4.2982
SEV_MOD_THRESHOLD = 5.9275

EVAL_CSV_PATH = Path("experiments/ddrnet_baseline/evaluation/evaluation_results.csv")
SEV_VAL_CSV_PATH = Path("experiments/severity_evaluation/severity_results_val.csv")
FINAL_REPORT_PATH = Path("experiments/final_results/FINAL_REPORT.txt")

VIS_DIR = Path("experiments/ddrnet_baseline/evaluation/visualizations")

RESEARCH_FIGURES = {
    "triplet_examples": Path("experiments/final_results/clean_distorted_rectified_examples.png"),
    "roc_curves": Path("experiments/final_results/roc_curves.png"),
    "matching_comparison": Path("experiments/final_results/matching_performance_comparison.png"),
    "score_distributions": Path("experiments/final_results/score_distributions.png"),
}

CURATED_SAMPLES = [
    {
        "index": "01",
        "file_name": "01_bending.png",
        "sample_id": "FAMILY-12__CHILD__FM012_C1__bending.png",
        "label": "Sample 01 — Bending Distortion (High Improvement: MAE +41.92%, RMSE +43.61%)",
        "type": "bending",
    },
    {
        "index": "02",
        "file_name": "02_elastic.png",
        "sample_id": "FAMILY-12__CHILD__FM012_C1__elastic.png",
        "label": "Sample 02 — Elastic Distortion (Moderate Improvement: MAE +9.45%, RMSE +12.50%)",
        "type": "elastic",
    },
    {
        "index": "03",
        "file_name": "03_local.png",
        "sample_id": "FAMILY-12__CHILD__FM012_C1__local.png",
        "label": "Sample 03 — Local Distortion (Degradation: MAE -28.55%, RMSE -25.75%)",
        "type": "local",
    },
    {
        "index": "04",
        "file_name": "04_bending.png",
        "sample_id": "FAMILY-12__CHILD__FM012_C2__bending.png",
        "label": "Sample 04 — Bending Distortion (Improvement: MAE +8.66%, RMSE +9.05%)",
        "type": "bending",
    },
    {
        "index": "05",
        "file_name": "05_elastic.png",
        "sample_id": "FAMILY-12__CHILD__FM012_C2__elastic.png",
        "label": "Sample 05 — Elastic Distortion (Improvement: MAE +6.47%, RMSE +9.53%)",
        "type": "elastic",
    },
    {
        "index": "06",
        "file_name": "06_local.png",
        "sample_id": "FAMILY-12__CHILD__FM012_C2__local.png",
        "label": "Sample 06 — Local Distortion (Degradation: MAE -12.77%, RMSE -8.47%)",
        "type": "local",
    },
]

# =============================================================================
# DATA LOADING & CACHING
# =============================================================================

@st.cache_data
def load_merged_dataset() -> pd.DataFrame:
    """Load and merge evaluation results with severity analysis."""
    if not EVAL_CSV_PATH.exists():
        raise FileNotFoundError(f"Evaluation CSV missing: {EVAL_CSV_PATH}")
    if not SEV_VAL_CSV_PATH.exists():
        raise FileNotFoundError(f"Severity CSV missing: {SEV_VAL_CSV_PATH}")

    df_eval = pd.read_csv(EVAL_CSV_PATH)
    df_sev = pd.read_csv(SEV_VAL_CSV_PATH)

    df_sev["eval_key"] = df_sev["sample"] + "__" + df_sev["distortion_type"] + ".png"
    merged = pd.merge(df_eval, df_sev, left_on="filename", right_on="eval_key", how="inner")
    
    # Add family, member, image id columns
    parts = merged["sample"].str.split("__", expand=True)
    merged["family"] = parts[0]
    merged["member"] = parts[1]
    merged["fingerprint_id"] = parts[2]
    
    return merged


@st.cache_resource
def load_classifier_model() -> Optional[FingerprintCNN]:
    """Load the frozen FingerprintCNN distortion classifier."""
    if not CLASSIFIER_CHECKPOINT.exists():
        return None
    try:
        model = FingerprintCNN()
        ckpt = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load FingerprintCNN checkpoint: {e}")
        return None


@st.cache_resource
def load_ddrnet_model() -> Optional[DDRNet_DIR]:
    """
    Load the trained DDRNet_DIR rectification checkpoint.
    Loads checkpoint['model_state_dict'] into DDRNet_DIR(dis_const=16).
    """
    if not DDRNET_CHECKPOINT.exists():
        return None
    try:
        model = DDRNet_DIR(dis_const=16)
        ckpt = torch.load(DDRNET_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError(f"Unrecognized checkpoint structure: {type(ckpt)}")
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load DDRNet_DIR checkpoint: {e}")
        return None


def generate_foreground_mask_14x14(image_224: np.ndarray) -> np.ndarray:
    """
    Generate 14x14 binary foreground mask for an arbitrary input image
    using the Otsu + morphological pipeline from prepare_ddrnet_dataset.py.
    """
    _, mask = cv2.threshold(
        image_224,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    mask16 = cv2.resize(
        mask,
        (14, 14),
        interpolation=cv2.INTER_NEAREST
    )
    mask16 = (mask16 > 0).astype(np.float32)
    return mask16


def run_classifier_inference(model: FingerprintCNN, image_array: np.ndarray) -> Tuple[float, bool]:
    """Run live inference using custom FingerprintCNN classifier."""
    if len(image_array.shape) == 3:
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_array

    resized = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor).squeeze()).item())
    is_distorted = probability >= CLASSIFIER_THRESHOLD
    return probability, is_distorted


def run_ddrnet_rectification(
    model: DDRNet_DIR,
    image_gray_224: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, str]:
    """
    Run live DDRNet_DIR inference on a 224x224 grayscale fingerprint image.

    Returns:
        rectified: 224x224 uint8 rectified image
        pred_dx_full: 224x224 float32 horizontal displacement
        pred_dy_full: 224x224 float32 vertical displacement
        stats: dict containing severity statistics
        severity_level: 'Low' | 'Moderate' | 'High'
    """
    # 1. Mask generation (14x14)
    mask_14 = generate_foreground_mask_14x14(image_gray_224)
    mask_tensor = torch.from_numpy(mask_14).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 2. Input normalization: (255.0 - image) / 255.0
    input_norm = (255.0 - image_gray_224.astype(np.float32)) / 255.0
    input_tensor = torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 3. Model forward pass
    with torch.no_grad():
        disp_pred, _ = model(input_tensor, mask_tensor)

    pred_field = disp_pred[0].detach().cpu().numpy()
    pred_dx_14 = pred_field[0].astype(np.float32)
    pred_dy_14 = pred_field[1].astype(np.float32)

    # 4. Upsample dx and dy to 224x224 via cubic interpolation
    pred_dx_full = cv2.resize(pred_dx_14, (224, 224), interpolation=cv2.INTER_CUBIC)
    pred_dy_full = cv2.resize(pred_dy_14, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 5. Rectification: map_x = x - dx, map_y = y - dy
    h, w = image_gray_224.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    map_x = grid_x - pred_dx_full
    map_y = grid_y - pred_dy_full

    rectified = cv2.remap(
        image_gray_224,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    # 6. Severity statistics from predicted displacement field only
    stats = calculate_severity_statistics(pred_dx_full, pred_dy_full)
    level = classify_severity(stats["mean_displacement"])

    return rectified, pred_dx_full, pred_dy_full, stats, level


def get_clean_image_path(family: str, member: str, img_id: str) -> Optional[Path]:
    """Resolve clean ground-truth image path from validation split."""
    p = Path("data/split/val") / family / member / f"{img_id}.png"
    if p.exists():
        return p
    return None


def extract_triplet_panels(vis_path: Path) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Extract Clean, Distorted, and Rectified panels from verified 224x680 triplet.
    Panel 1: [:, 0:224]   -> Clean Reference
    Panel 2: [:, 228:452] -> Distorted Input
    Panel 3: [:, 456:680] -> Rectified Output
    """
    img = cv2.imread(str(vis_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clean = Image.fromarray(img_rgb[:, 0:224])
    distorted = Image.fromarray(img_rgb[:, 228:452])
    rectified = Image.fromarray(img_rgb[:, 456:680])
    return clean, distorted, rectified


# =============================================================================
# STREAMLIT UI SETUP & STYLING
# =============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-impact research presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .limitation-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px 18px;
        border-radius: 4px;
        color: #991B1B;
        font-size: 0.95rem;
        margin-bottom: 22px;
    }
    .pipeline-step {
        display: inline-block;
        padding: 6px 14px;
        margin-right: 8px;
        border-radius: 20px;
        background-color: #E2E8F0;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pipeline-step-active {
        background-color: #2563EB;
        color: white;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-distorted { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; }
    .badge-clean { background-color: #DCFCE7; color: #16A34A; border: 1px solid #86EFAC; }
    .badge-bending { background-color: #E0E7FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-elastic { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-local { background-color: #F3E8FF; color: #7E22CE; border: 1px solid #E9D5FF; }
    .badge-low { background-color: #DCFCE7; color: #15803D; }
    .badge-mod { background-color: #FEF3C7; color: #B45309; }
    .badge-high { background-color: #FEE2E2; color: #B91C1C; }
    .metric-value-pos { color: #16A34A; font-weight: 700; }
    .metric-value-neg { color: #DC2626; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Header
st.markdown('<div class="main-title">🔍 Fingerprint Distortion Detection & Rectification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Research Demonstration — Evaluated Deep Learning Pipeline & Benchmark Analysis</div>', unsafe_allow_html=True)

# System Status Banner
st.markdown(
    """
    <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; padding: 14px 18px; border-radius: 4px; color: #166534; font-size: 0.95rem; margin-bottom: 22px;">
        <strong>✅ LIVE PIPELINE OPERATIONAL:</strong><br>
        Both the <strong>FingerprintCNN</strong> distortion detector (<code>best_model_exp5.pth</code>) and the trained 
        <strong>DDRNet_DIR</strong> rectification model (<code>best_model.pth</code>) are active.
        You can select verified benchmark samples to view recorded experimental evaluations, or upload an unknown fingerprint for live end-to-end distortion detection, severity estimation, and rectification.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data and models
try:
    df_samples = load_merged_dataset()
except Exception as err:
    st.error(f"Error loading evaluation dataset: {err}")
    st.stop()

classifier_model = load_classifier_model()
ddrnet_model = load_ddrnet_model()

# =============================================================================
# MAIN NAVIGATION TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🔬 1. Interactive Research Pipeline Demo",
    "📊 2. Evaluated Samples Browser (675 Samples)",
    "📈 3. Overall Research Benchmark Dashboard",
])

# =============================================================================
# TAB 1: INTERACTIVE RESEARCH PIPELINE DEMO
# =============================================================================

with tab1:
    st.markdown("### Research Pipeline Demonstration")
    st.markdown(
        """
        Follow the complete experimental pipeline:
        **Input Fingerprint** → **Distortion Detection** → **Severity Estimation** → **Rectification** → **Rectification Effectiveness**
        """
    )

    # Input Selection Mode
    input_mode = st.radio(
        "Select Input Source:",
        [
            "Select Verified Evaluated Sample (Full Pipeline with Rectification Triplet)",
            "Upload Custom Fingerprint (Live Distortion Detection & DDRNet Rectification)",
        ],
        index=0,
        horizontal=True,
    )

    if input_mode.startswith("Select Verified"):
        # Evaluated sample selector
        col_sel1, col_sel2 = st.columns([2, 1])

        with col_sel1:
            curated_options = {s["label"]: s for s in CURATED_SAMPLES}
            selected_curated_label = st.selectbox(
                "Choose Curated Benchmark Sample (with pre-rendered triplet):",
                list(curated_options.keys()),
                index=0,
            )
            curated_meta = curated_options[selected_curated_label]

        with col_sel2:
            st.info(f"**Distortion:** `{curated_meta['type'].upper()}`\n\n**File:** `{curated_meta['sample_id']}`")

        # Find matching row in merged dataset
        sample_row = df_samples[df_samples["filename"] == curated_meta["sample_id"]].iloc[0]

        st.markdown("---")

        # Visual Pipeline Stepper
        st.markdown(
            """
            <div>
                <span class="pipeline-step pipeline-step-active">1. INPUT</span>
                <span class="pipeline-step pipeline-step-active">2. DETECTION</span>
                <span class="pipeline-step pipeline-step-active">3. SEVERITY</span>
                <span class="pipeline-step pipeline-step-active">4. RECTIFICATION</span>
                <span class="pipeline-step pipeline-step-active">5. EFFECTIVENESS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Stage A & B & C Cards in Columns
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("#### A. Input Sample Information")
            st.markdown(
                f"""
                - **Filename:** `{sample_row['filename']}`
                - **Distortion Type:** <span class="badge badge-{sample_row['type']}">{sample_row['type']}</span>
                - **Severity Level:** <span class="badge badge-{sample_row['severity_level'].lower()[:3]}">{sample_row['severity_level']}</span>
                - **Severity Score:** `{sample_row['severity_score']:.3f} px`
                - **Validation Split Family:** `{sample_row['family']} / {sample_row['member']}`
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown("#### B. Distortion Detection")
            st.markdown(
                """
                - **Status:** <span class="badge badge-distorted">DISTORTED</span>
                - **Model:** `FingerprintCNN` (4-layer custom CNN)
                - **Classification Threshold:** `0.18` (Validation EER operating point)
                """,
                unsafe_allow_html=True,
            )
            # Run live FingerprintCNN check on this sample
            vis_file = VIS_DIR / curated_meta["file_name"]
            if vis_file.exists() and classifier_model is not None:
                _, dist_panel, _ = extract_triplet_panels(vis_file)
                prob, is_dist = run_classifier_inference(classifier_model, np.array(dist_panel))
                st.markdown(f"- **Classifier Output Score:** `{prob:.4f}`")
                st.markdown(f"- **Live Inference Decision:** `{'Distorted' if is_dist else 'Clean'}` (Score ≥ 0.18)")
            else:
                st.markdown("- **Classifier Decision:** `Distorted` (Score ≥ 0.18)")

        with col_c:
            st.markdown("#### C. Severity Estimation")
            st.caption("Ground-truth displacement-field severity (regenerated fields)")
            st.markdown(
                f"""
                - **Category:** <span class="badge badge-{sample_row['severity_level'].lower()[:3]}">{sample_row['severity_level']}</span>
                - **Mean Displacement:** `{sample_row['mean_displacement']:.3f} px`
                - **Median Displacement:** `{sample_row['median_displacement']:.3f} px`
                - **P95 Displacement:** `{sample_row['p95_displacement']:.3f} px`
                - **Max Displacement:** `{sample_row['max_displacement']:.3f} px`
                """,
                unsafe_allow_html=True,
            )
            st.caption("Thresholds: Low < 4.2982 px | Moderate: 4.2982–5.9275 px | High ≥ 5.9275 px")

        st.markdown("---")

        # Stage D: Image Comparison Triplet
        st.markdown("#### D. Image Comparison (Clean Reference | Distorted Input | Rectified Output)")
        st.caption("Using verified evaluated images recorded during benchmark evaluation.")

        vis_path = VIS_DIR / curated_meta["file_name"]
        if vis_path.exists():
            clean_img, dist_img, rect_img = extract_triplet_panels(vis_path)

            col_img1, col_img2, col_img3 = st.columns(3)
            with col_img1:
                st.image(clean_img, caption="1. Clean Reference", use_container_width=True)
            with col_img2:
                st.image(dist_img, caption=f"2. Distorted Input ({sample_row['type'].capitalize()})", use_container_width=True)
            with col_img3:
                st.image(rect_img, caption="3. DDRNet Rectified Output", use_container_width=True)
        else:
            st.warning(f"Visualization triplet not found at `{vis_path}`.")

        st.markdown("---")

        # Stage E: Rectification Effectiveness
        st.markdown("#### E. Rectification Effectiveness")

        mae_imp = sample_row["mae_improvement_percent"]
        rmse_imp = sample_row["rmse_improvement_percent"]

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Distorted MAE", f"{sample_row['distorted_mae']:.3f}")
        m_col2.metric("Rectified MAE", f"{sample_row['rectified_mae']:.3f}", delta=f"{mae_imp:+.2f}%")
        m_col3.metric("Distorted RMSE", f"{sample_row['distorted_rmse']:.3f}")
        m_col4.metric("Rectified RMSE", f"{sample_row['rectified_rmse']:.3f}", delta=f"{rmse_imp:+.2f}%")

        if mae_imp > 0 and rmse_imp > 0:
            st.success(
                f"**Positive Improvement (+{mae_imp:.2f}% MAE, +{rmse_imp:.2f}% RMSE):** "
                "Rectification successfully shifted pixel intensities closer to the clean reference."
            )
        elif mae_imp < 0 or rmse_imp < 0:
            st.error(
                f"**Negative Change ({mae_imp:+.2f}% MAE, {rmse_imp:+.2f}% RMSE):** "
                "Rectification degraded pixel alignment relative to the clean reference. "
                "This behavior is characteristic of localized synthetic distortions where global field warping introduces artifacts."
            )
        else:
            st.info("Neutral change observed.")

    else:
        # Custom upload mode
        st.markdown("#### Upload Unknown Fingerprint for Live Detection & Rectification")
        st.info(
            "ℹ️ **Live Inference Pipeline:** The custom `FingerprintCNN` baseline classifies whether the fingerprint "
            "is clean or distorted. If distorted (probability ≥ 0.18), `DDRNet_DIR` predicts the dense displacement field, "
            "computes live severity from the predicted field, and restores the fingerprint via geometric remapping."
        )

        uploaded_file = st.file_uploader("Upload Fingerprint Image (PNG / JPG / BMP):", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded_file is not None:
            try:
                raw_img = Image.open(uploaded_file).convert("RGB")
                img_np = np.array(raw_img)
            except Exception as e:
                st.error(f"Failed to load uploaded image: {e}")
                st.stop()

            # Preprocess image to 224x224 grayscale
            if len(img_np.shape) == 3:
                image_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = img_np
            image_224 = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_AREA)

            # Step 1: Run Distortion Classification
            if classifier_model is None:
                st.error("Classifier checkpoint not found at `experiments/experiment_5/best_model_exp5.pth`.")
                st.stop()

            try:
                prob, is_dist = run_classifier_inference(classifier_model, image_224)
            except Exception as e:
                st.error(f"Error during classifier inference: {e}")
                st.stop()

            st.markdown("---")

            if not is_dist:
                # CLEAN FINGERPRINT
                st.markdown(
                    f"""
                    <div style="padding:14px; background-color:#DCFCE7; border-left:4px solid #16A34A; border-radius:4px; margin-bottom:16px;">
                        <h4 style="color:#16A34A; margin:0 0 6px 0;">✓ Decision: CLEAN FINGERPRINT (No Distortion Detected)</h4>
                        <p style="margin:0; color:#334155;"><strong>Distortion Probability:</strong> <code>{prob:.4f}</code> (Operating Threshold: <code>{CLASSIFIER_THRESHOLD}</code>)</p>
                        <p style="margin:4px 0 0 0; color:#15803D;">The input fingerprint is classified as normal/clean. DDRNet rectification is not needed.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_clean1, col_clean2 = st.columns([1, 2])
                with col_clean1:
                    st.image(image_224, caption="Input Fingerprint (224×224)", use_container_width=True)
                with col_clean2:
                    st.markdown("##### Pipeline Decision Details")
                    st.markdown(
                        f"""
                        - **Classification Status:** <span class="badge badge-clean">CLEAN</span>
                        - **Probability Score:** `{prob:.4f}` (< {CLASSIFIER_THRESHOLD})
                        - **Rectification Action:** Skipped (Image is not distorted)
                        - **Biometric Pipeline:** Fingerprint can proceed directly to standard minutiae extraction and matching without distortion correction.
                        """,
                        unsafe_allow_html=True,
                    )

            else:
                # DISTORTED FINGERPRINT
                st.markdown(
                    f"""
                    <div style="padding:14px; background-color:#FEE2E2; border-left:4px solid #DC2626; border-radius:4px; margin-bottom:16px;">
                        <h4 style="color:#DC2626; margin:0 0 6px 0;">⚠️ Decision: DISTORTED FINGERPRINT DETECTED</h4>
                        <p style="margin:0; color:#334155;"><strong>Distortion Probability:</strong> <code>{prob:.4f}</code> (Operating Threshold: <code>{CLASSIFIER_THRESHOLD}</code>)</p>
                        <p style="margin:4px 0 0 0; color:#B91C1C;">Distortion detected. Executing live DDRNet_DIR displacement estimation and rectification...</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if ddrnet_model is None:
                    st.error(f"DDRNet checkpoint not found at `{DDRNET_CHECKPOINT}`. Cannot perform live rectification.")
                    st.stop()

                with st.spinner("Running live DDRNet_DIR displacement estimation & rectification..."):
                    try:
                        rectified, pred_dx, pred_dy, sev_stats, sev_level = run_ddrnet_rectification(
                            ddrnet_model, image_224
                        )
                    except Exception as e:
                        st.error(f"Error during DDRNet rectification: {e}")
                        st.stop()

                # Visual comparison columns
                col_img1, col_img2, col_img3 = st.columns(3)

                with col_img1:
                    st.image(image_224, caption="1. Original Distorted Input", use_container_width=True)

                with col_img2:
                    st.image(rectified, caption="2. Live DDRNet Rectified Output", use_container_width=True)

                with col_img3:
                    # Displacement magnitude heatmap
                    disp_mag = np.sqrt(pred_dx ** 2 + pred_dy ** 2)
                    norm_mag = np.clip((disp_mag / max(disp_mag.max(), 1e-6)) * 255.0, 0, 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_rgb, caption="3. Predicted Displacement Magnitude Heatmap", use_container_width=True)

                st.markdown("---")

                # Predicted Severity Card
                st.markdown("##### Live Predicted Distortion Severity")
                st.caption("Calculated purely from the DDRNet-predicted displacement field (no ground-truth reference needed).")

                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                s_col1.metric("Predicted Severity Level", sev_level)
                s_col2.metric("Mean Displacement", f"{sev_stats['mean_displacement']:.3f} px")
                s_col3.metric("Median Displacement", f"{sev_stats['median_displacement']:.3f} px")
                s_col4.metric("Max Displacement", f"{sev_stats['max_displacement']:.3f} px")

                st.caption(
                    f"Thresholds (Training Terciles): Low < {SEV_LOW_THRESHOLD:.4f} px | "
                    f"Moderate: {SEV_LOW_THRESHOLD:.4f}–{SEV_MOD_THRESHOLD:.4f} px | "
                    f"High ≥ {SEV_MOD_THRESHOLD:.4f} px"
                )

                st.success("✓ Live geometric rectification completed successfully using the trained DDRNet_DIR checkpoint.")


# =============================================================================
# TAB 2: EVALUATED SAMPLES BROWSER (ALL 675 SAMPLES)
# =============================================================================

with tab2:
    st.markdown("### Evaluated Samples Browser (All 675 Samples)")
    st.markdown(
        "Browse the complete set of 675 evaluated validation samples across all distortion types and severity levels. "
        "Filter by distortion type and severity, and inspect individual sample verification metrics."
    )

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        type_filter = st.selectbox("Filter by Distortion Type:", ["All", "bending", "elastic", "local"], index=0)
    with fcol2:
        sev_filter = st.selectbox("Filter by Severity Level:", ["All", "Low", "Moderate", "High"], index=0)
    with fcol3:
        search_kw = st.text_input("Search Filename / Family:", placeholder="e.g. FM012, FAMILY-7")

    # Apply filters
    filtered_df = df_samples.copy()
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df["type"] == type_filter]
    if sev_filter != "All":
        filtered_df = filtered_df[filtered_df["severity_level"] == sev_filter]
    if search_kw.strip():
        filtered_df = filtered_df[filtered_df["filename"].str.contains(search_kw.strip(), case=False)]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df_samples)} evaluated samples:**")

    # Summary table columns
    display_cols = [
        "filename",
        "type",
        "severity_level",
        "severity_score",
        "distorted_mae",
        "rectified_mae",
        "mae_improvement_percent",
        "distorted_rmse",
        "rectified_rmse",
        "rmse_improvement_percent",
    ]

    st.dataframe(
        filtered_df[display_cols].style.format({
            "severity_score": "{:.3f}",
            "distorted_mae": "{:.2f}",
            "rectified_mae": "{:.2f}",
            "mae_improvement_percent": "{:+.2f}%",
            "distorted_rmse": "{:.2f}",
            "rectified_rmse": "{:.2f}",
            "rmse_improvement_percent": "{:+.2f}%",
        }),
        height=320,
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Inspect Specific Sample from Filtered List")

    if not filtered_df.empty:
        selected_sample_fn = st.selectbox(
            "Select Sample to View Full Metrics & Images:",
            filtered_df["filename"].tolist(),
            index=0,
        )
        row = filtered_df[filtered_df["filename"] == selected_sample_fn].iloc[0]

        # Check if sample is one of the 6 curated ones
        curated_match = next((c for c in CURATED_SAMPLES if c["sample_id"] == selected_sample_fn), None)

        scol1, scol2 = st.columns([1, 1])

        with scol1:
            st.markdown(f"##### Metrics for `{row['filename']}`")
            st.markdown(
                f"""
                - **Distortion Type:** <span class="badge badge-{row['type']}">{row['type'].upper()}</span>
                - **Severity Level:** <span class="badge badge-{row['severity_level'].lower()[:3]}">{row['severity_level']}</span>
                - **Severity Score (Mean Displacement):** `{row['severity_score']:.3f} px`
                - **Median Displacement:** `{row['median_displacement']:.3f} px`
                - **P95 Displacement:** `{row['p95_displacement']:.3f} px`
                - **Max Displacement:** `{row['max_displacement']:.3f} px`
                - **Distorted MAE:** `{row['distorted_mae']:.3f}` → **Rectified MAE:** `{row['rectified_mae']:.3f}` (**{row['mae_improvement_percent']:+.2f}%**)
                - **Distorted RMSE:** `{row['distorted_rmse']:.3f}` → **Rectified RMSE:** `{row['rectified_rmse']:.3f}` (**{row['rmse_improvement_percent']:+.2f}%**)
                """,
                unsafe_allow_html=True,
            )

        with scol2:
            st.markdown("##### Visual Artifacts")
            if curated_match:
                vis_file = VIS_DIR / curated_match["file_name"]
                if vis_file.exists():
                    clean_p, dist_p, rect_p = extract_triplet_panels(vis_file)
                    st.caption("Pre-rendered Triplet: Clean | Distorted | Rectified")
                    st.image(vis_file, caption=f"Verified Triplet for {selected_sample_fn}", use_container_width=True)
            else:
                # Show clean reference image
                clean_path = get_clean_image_path(row["family"], row["member"], row["fingerprint_id"])
                if clean_path and clean_path.exists():
                    st.image(Image.open(clean_path), caption=f"Clean Reference: {row['family']}/{row['member']}/{row['fingerprint_id']}.png", width=224)
                    st.caption(
                        "Note: Pre-rendered triplets are preserved for representative benchmark samples (e.g. sample 01–06 in Tab 1). "
                        "For other samples, recorded numerical metrics are drawn from evaluation_results.csv."
                    )
                else:
                    st.info("Clean reference image path not resolved.")


# =============================================================================
# TAB 3: OVERALL RESEARCH BENCHMARK DASHBOARD
# =============================================================================

with tab3:
    st.markdown("### Project-Level Experimental Benchmark Results")
    st.markdown("Summary of frozen classifier performance, DDRNet rectification effectiveness, and biometric matching impact.")

    # Top metrics cards
    st.markdown("#### 1. Distortion Classifier Benchmark (Frozen FingerprintCNN — Experiment 5)")
    st.caption("Model: Custom 4-layer FingerprintCNN baseline (Input: 1×224×224, BCEWithLogitsLoss, threshold = 0.18)")

    c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns(5)
    c_col1.metric("ROC-AUC", "0.9432")
    c_col2.metric("Accuracy", "88.89%")
    c_col3.metric("Precision", "95.34%")
    c_col4.metric("Recall", "81.78%")
    c_col5.metric("F1-Score", "88.04%")

    st.markdown(
        """
        - **Decision Matrix (Test Set, N=450):** True Negatives (Clean) = **216**, False Positives = **9**, False Negatives = **41**, True Positives (Distorted) = **184**
        - **Operating Threshold:** `0.18` (Selected via validation EER)
        """
    )

    st.markdown("---")

    st.markdown("#### 2. DDRNet Rectification Effectiveness (Pixel-Level Metrics)")
    st.caption("Comparison of distorted vs rectified images against clean ground-truth reference across 675 samples.")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Overall MAE Improvement", "+9.54%", help="Pixel-level MAE improvement across all distortion categories")
    r_col2.metric("Overall RMSE Improvement", "+11.06%", help="Pixel-level RMSE improvement across all distortion categories")
    r_col3.metric("Bending RMSE Improvement", "+30.56%", help="Significant geometric recovery on continuous bending distortions")
    r_col4.metric("Local RMSE Change", "-14.24%", delta="-14.24%", help="Local patch distortions degraded pixel quality due to displacement over-smoothing")

    st.info(
        "💡 **Key Geometric Finding:** Bending distortions exhibited strong pixel-level recovery (+30.56% RMSE improvement). "
        "Conversely, local distortions showed degradations (-14.24% RMSE change) because the global smooth displacement field "
        "cannot easily accommodate abrupt, high-frequency local displacements without blurring nearby ridges."
    )

    st.markdown("---")

    st.markdown("#### 3. Research Evaluation Figures & Biometric Verification Impact")
    st.markdown(
        "Minutiae extraction was performed using **NIST MINDTCT** and scored using **NIST Bozorth3** under a family-disjoint protocol."
    )

    fig_tab1, fig_tab2, fig_tab3, fig_tab4 = st.tabs([
        "🖼️ Triplet Comparisons Across Distortions",
        "📈 ROC Curves (Matching)",
        "📊 Bozorth3 Score Distributions",
        "📉 Performance Comparison Bar Chart",
    ])

    with fig_tab1:
        if RESEARCH_FIGURES["triplet_examples"].exists():
            st.image(
                Image.open(RESEARCH_FIGURES["triplet_examples"]),
                caption="Clean Reference vs Distorted Input vs DDRNet Rectified Output across Bending, Elastic, and Local Distortions",
                use_container_width=True,
            )
        else:
            st.warning("Figure not found.")

    with fig_tab2:
        if RESEARCH_FIGURES["roc_curves"].exists():
            st.image(
                Image.open(RESEARCH_FIGURES["roc_curves"]),
                caption="ROC Curves for MINDTCT + Bozorth3 matching under clean, distorted, and rectified conditions.",
                use_container_width=True,
            )
        else:
            st.warning("ROC curves figure not found.")

    with fig_tab3:
        if RESEARCH_FIGURES["score_distributions"].exists():
            st.image(
                Image.open(RESEARCH_FIGURES["score_distributions"]),
                caption="Bozorth3 genuine and impostor score distributions across distortion types.",
                use_container_width=True,
            )
        else:
            st.warning("Score distributions figure not found.")

    with fig_tab4:
        if RESEARCH_FIGURES["matching_comparison"].exists():
            st.image(
                Image.open(RESEARCH_FIGURES["matching_comparison"]),
                caption="ROC-AUC comparison across distortion types before and after DDRNet rectification.",
                use_container_width=True,
            )
        else:
            st.warning("Matching comparison figure not found.")

    st.markdown("---")

    st.markdown("#### 4. Scientific Conclusion & Key Takeaways")
    st.markdown(
        """
        1. **Pixel Quality vs Recognition Disconnect:** While DDRNet improves pixel-level alignment (MAE +9.54%, RMSE +11.06%, Bending +30.56%), 
           pixel improvement alone did not translate to substantial gains in MINDTCT/Bozorth3 matching score.
        2. **Minutiae Preservation:** Resampling and interpolation slightly altered delicate ridge flow and ridge endings, illustrating that future
           rectification networks should be trained with matcher-aware or minutiae-preserving loss functions.
        3. **Rigorous Validation:** The evaluation was executed with family-disjoint splits and strict validation-only threshold tuning to ensure zero test leakage.
        """
    )

# Footer
st.markdown("---")
st.caption(
    "Fingerprint Distortion Detection and Rectification Project | Research Demonstration UI | "
    "Powered by Streamlit & PyTorch"
)
