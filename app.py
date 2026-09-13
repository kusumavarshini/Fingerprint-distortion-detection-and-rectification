"""
Fingerprint Distortion Detection & Identification Application
============================================================
A focused end-user application for fingerprint distortion detection,
geometric rectification via DDRNet_DIR, and biometric identification.

Workflow:
  1. Upload Fingerprint
  2. Display Uploaded Fingerprint Preview
  3. Distortion Classification (FingerprintCNN) -> CLEAN / DISTORTED
  4. Severity Estimation & Rectification (DDRNet_DIR) -> LOW / MODERATE / HIGH
  5. Fingerprint Matching / Identification (AFIS Backend)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
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

# Check for existing legitimate biometric matcher pipeline
try:
    from afis import MindtctExtractor
    AFIS_AVAILABLE = True
except ImportError:
    AFIS_AVAILABLE = False

# =============================================================================
# CONSTANTS & CONFIGURATION (PRESERVED RESEARCH ARTIFACTS)
# =============================================================================

PAGE_TITLE = "Fingerprint Distortion Detection & Identification"
PAGE_ICON = "🔍"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIER_CHECKPOINT = Path("experiments/classifier_hard_negative/best_model.pth")
CLASSIFIER_THRESHOLD = 0.60

DDRNET_CHECKPOINT = Path("experiments/ddrnet_baseline/best_model.pth")

SEV_LOW_THRESHOLD = 4.2982
SEV_MOD_THRESHOLD = 5.9275

TEST_DIR = Path("data/split/test")

# =============================================================================
# MODEL LOADERS (CACHED)
# =============================================================================

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
        st.sidebar.error(f"Error loading FingerprintCNN: {e}")
        return None


@st.cache_resource
def load_ddrnet_model() -> Optional[DDRNet_DIR]:
    """Load the trained DDRNet_DIR rectification checkpoint."""
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
        st.sidebar.error(f"Error loading DDRNet_DIR: {e}")
        return None


# =============================================================================
# INFERENCE & PROCESSING PIPELINE
# =============================================================================

def generate_foreground_mask_14x14(image_224: np.ndarray) -> np.ndarray:
    """Generate 14x14 binary foreground mask for arbitrary 224x224 input."""
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
    """Run live DDRNet_DIR inference on a 224x224 grayscale fingerprint image."""
    mask_14 = generate_foreground_mask_14x14(image_gray_224)
    mask_tensor = torch.from_numpy(mask_14).unsqueeze(0).unsqueeze(0).to(DEVICE)

    input_norm = (255.0 - image_gray_224.astype(np.float32)) / 255.0
    input_tensor = torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        disp_pred, _ = model(input_tensor, mask_tensor)

    pred_field = disp_pred[0].detach().cpu().numpy()
    pred_dx_14 = pred_field[0].astype(np.float32)
    pred_dy_14 = pred_field[1].astype(np.float32)

    pred_dx_full = cv2.resize(pred_dx_14, (224, 224), interpolation=cv2.INTER_CUBIC)
    pred_dy_full = cv2.resize(pred_dy_14, (224, 224), interpolation=cv2.INTER_CUBIC)

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

    stats = calculate_severity_statistics(pred_dx_full, pred_dy_full)
    level = classify_severity(stats["mean_displacement"])

    return rectified, pred_dx_full, pred_dy_full, stats, level


def enhance_fingerprint_for_display(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE enhancement strictly for display/visualization."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(image)
    return enhanced


def perform_fingerprint_matching(candidate_image_224: np.ndarray) -> dict:
    """
    Perform biometric fingerprint matching against reference gallery.
    Uses legitimate project AFIS / MindtctExtractor pipeline if available.
    Adheres strictly to research integrity: does NOT invent fake or synthetic matches.
    """
    if not AFIS_AVAILABLE:
        return {
            "status": "BACKEND_NOT_IMPLEMENTED",
            "message": "The AFIS biometric matching backend (MindtctExtractor / Bozorth3) is not installed in the local Python environment. In accordance with research integrity standards, simulated or synthetic match results are not fabricated.",
            "match_found": False,
            "matched_id": None,
            "score": None,
            "threshold": None,
        }

    if not TEST_DIR.exists():
        return {
            "status": "DATASET_NOT_FOUND",
            "message": f"Reference test gallery not found at {TEST_DIR}.",
            "match_found": False,
            "matched_id": None,
            "score": None,
            "threshold": None,
        }

    try:
        extractor = MindtctExtractor()
        probe_template = extractor.extract_minutiae(candidate_image_224)

        best_score = -1.0
        best_id = None
        threshold = 25.0  # Standard Bozorth3 operational decision threshold

        for family_dir in sorted(TEST_DIR.glob("FAMILY-*")):
            for member_dir in sorted(family_dir.iterdir()):
                if not member_dir.is_dir():
                    continue
                ref_images = sorted(member_dir.glob("*.png"))
                if not ref_images:
                    continue
                ref_img = cv2.imread(str(ref_images[0]), cv2.IMREAD_GRAYSCALE)
                if ref_img is None:
                    continue
                ref_template = extractor.extract_minutiae(ref_img)
                res = extractor.match(probe_template, ref_template, method="bozorth3")
                score = float(res.score if hasattr(res, "score") else res)
                if score > best_score:
                    best_score = score
                    best_id = f"{family_dir.name} / {member_dir.name} ({ref_images[0].name})"

        match_found = best_score >= threshold
        return {
            "status": "COMPLETED",
            "match_found": match_found,
            "matched_id": best_id if match_found else None,
            "score": best_score,
            "threshold": threshold,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Matcher error: {e}",
            "match_found": False,
            "matched_id": None,
            "score": None,
            "threshold": None,
        }


# =============================================================================
# STREAMLIT APPLICATION UI
# =============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Application Styling
st.markdown(
    """
    <style>
    .app-header {
        margin-bottom: 24px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 12px;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .badge-clean { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .badge-distorted { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; }
    .badge-low { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .badge-mod { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-high { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
    .badge-matched { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .badge-unmatched { background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }
    .badge-pending { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .metric-row {
        display: flex;
        gap: 20px;
        margin-top: 6px;
    }
    .metric-item {
        font-size: 0.88rem;
        color: #334155;
    }
    .metric-item strong {
        color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application Header
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">🔍 Fingerprint Distortion Detection & Identification</h1>
        <p class="app-subtitle">End-to-End Biometric Pipeline: Distortion Classification → DDRNet Rectification → Gallery Identification</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pre-load Models
classifier_model = load_classifier_model()
ddrnet_model = load_ddrnet_model()

# Sidebar: Upload Fingerprint
st.sidebar.markdown("### Upload Fingerprint")
st.sidebar.markdown("Provide a fingerprint image to evaluate distortion and perform identification.")

uploaded_file = st.sidebar.file_uploader(
    "Choose fingerprint image:",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Upload an uncompressed or standard format fingerprint scan.",
)

# Quick sample selector for convenience
sample_presets = {
    "-- None (Use Uploaded Image) --": None,
    "Sample: Distorted Print (Bending)": Path("data/split/test/FAMILY-14/FATHER/FM014_F1.png"),
    "Sample: Normal Print (Clean)": Path("data/split/test/FAMILY-13/MOTHER/FM013_M1.png"),
}
selected_preset = st.sidebar.selectbox("Or choose a reference test sample:", list(sample_presets.keys()))

raw_img: Optional[Image.Image] = None

if uploaded_file is not None:
    try:
        raw_img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")
elif sample_presets[selected_preset] is not None and sample_presets[selected_preset].exists():
    try:
        raw_img = Image.open(sample_presets[selected_preset]).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load sample image: {e}")

# If no image selected, show placeholder instructions
if raw_img is None:
    st.info(
        "👆 **Get Started:** Upload a fingerprint image using the sidebar file uploader, "
        "or select one of the reference test samples to execute the live pipeline."
    )
    st.stop()

# =============================================================================
# MAIN PROCESSING WORKFLOW
# =============================================================================

if raw_img is not None:
    # 1. Preprocessing to 224x224 grayscale (Research integrity: image_224 is the sole model input)
    img_np = np.array(raw_img, dtype=np.uint8)
    if len(img_np.shape) == 3:
        image_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = img_np
    image_224 = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_AREA)

    # 2. Display-only CLAHE enhancement (strictly for visualization, NEVER fed to models)
    enhanced_image = enhance_fingerprint_for_display(image_224)

    # 3. Run FingerprintCNN Distortion Classifier
    if classifier_model is None:
        st.error("Classifier checkpoint could not be loaded. Please ensure the model file exists.")
        st.stop()

    prob, is_distorted = run_classifier_inference(classifier_model, image_224)

    # 4. If Distorted, Run DDRNet Rectification
    rectified_image = None
    pred_dx = None
    pred_dy = None
    sev_stats = None
    sev_level = None

    if is_distorted:
        if ddrnet_model is None:
            st.error("DDRNet model checkpoint is unavailable for rectification.")
            st.stop()
        rectified_image, pred_dx, pred_dy, sev_stats, sev_level = run_ddrnet_rectification(
            ddrnet_model, image_224
        )
        candidate_image = rectified_image
    else:
        candidate_image = image_224

    # 5. Run Biometric Fingerprint Matching
    match_info = perform_fingerprint_matching(candidate_image)

    # =============================================================================
    # MAIN RESULT VIEW: TWO-COLUMN APPLICATION LAYOUT
    # =============================================================================

    col_left, col_right = st.columns([1, 1.4], gap="large")

    # LEFT COLUMN: Uploaded Fingerprint Preview
    with col_left:
        st.markdown("#### Uploaded Fingerprint Preview")
        st.image(raw_img, caption="Original Input Fingerprint", width="stretch")

    # RIGHT COLUMN: Primary Decision Cards
    with col_right:
        st.markdown("#### Processing & Identification Results")

        # Card 1: Distortion Classification Result
        status_text = "DISTORTED" if is_distorted else "CLEAN"
        status_cls = "badge-distorted" if is_distorted else "badge-clean"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">1. Distortion Classification</div>
                <span class="status-badge {status_cls}">{status_text}</span>
                <div class="metric-row">
                    <div class="metric-item"><strong>Distortion Probability:</strong> <code>{prob:.4f}</code></div>
                    <div class="metric-item"><strong>Operating Threshold:</strong> <code>{CLASSIFIER_THRESHOLD:.2f}</code></div>
                </div>
                <p style="font-size: 0.85rem; color: #64748B; margin: 8px 0 0 0;">
                    {"Distortion detected. Geometric DDRNet rectification triggered." if is_distorted else "Fingerprint ridge topology is normal. Rectification bypassed."}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Card 2: Distortion Severity (Only if Distorted)
        if is_distorted and sev_stats is not None and sev_level is not None:
            sev_badge_cls = f"badge-{sev_level.lower()[:3]}"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="card-title">2. Distortion Severity (DDRNet-Predicted Field)</div>
                    <span class="status-badge {sev_badge_cls}">{sev_level.upper()} SEVERITY</span>
                    <div class="metric-row">
                        <div class="metric-item"><strong>Mean Displacement:</strong> <code>{sev_stats['mean_displacement']:.3f} px</code></div>
                        <div class="metric-item"><strong>Max Displacement:</strong> <code>{sev_stats['max_displacement']:.3f} px</code></div>
                    </div>
                    <p style="font-size: 0.82rem; color: #64748B; margin: 6px 0 0 0;">
                        Thresholds: Low &lt; {SEV_LOW_THRESHOLD:.2f} px | Moderate {SEV_LOW_THRESHOLD:.2f}–{SEV_MOD_THRESHOLD:.2f} px | High &ge; {SEV_MOD_THRESHOLD:.2f} px
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Card 3: Matching / Identification Result
        if match_info["status"] == "COMPLETED":
            if match_info["match_found"]:
                match_badge = '<span class="status-badge badge-matched">MATCH FOUND</span>'
                match_details = f"""
                <div class="metric-row">
                    <div class="metric-item"><strong>Matched Identity:</strong> <code>{match_info['matched_id']}</code></div>
                    <div class="metric-item"><strong>Score:</strong> <code>{match_info['score']:.1f}</code> (Threshold: {match_info['threshold']})</div>
                </div>
                """
            else:
                match_badge = '<span class="status-badge badge-unmatched">NO MATCH FOUND</span>'
                match_details = f"""
                <div class="metric-row">
                    <div class="metric-item"><strong>Highest Gallery Score:</strong> <code>{match_info['score']:.1f}</code> (Threshold: {match_info['threshold']})</div>
                </div>
                <p style="font-size: 0.85rem; color: #64748B; margin: 6px 0 0 0;">No matching identity found in gallery above the required threshold.</p>
                """
        else:
            match_badge = '<span class="status-badge badge-pending">MATCHER BACKEND PENDING</span>'
            match_details = f"""
            <p style="font-size: 0.85rem; color: #92400E; margin: 6px 0 0 0;">
                {match_info['message']}
            </p>
            """

        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">3. Biometric Identification</div>
                {match_badge}
                {match_details}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =============================================================================
    # SECONDARY INSPECTION (OPTIONAL STAGE VISUALIZATION FOR DISTORTED PRINTS)
    # =============================================================================

    if is_distorted and rectified_image is not None:
        st.markdown("---")
        st.markdown("#### Diagnostic Visualization: Original → Enhanced → Rectified")
        st.caption("Enhanced stage is for visualization only. Both FingerprintCNN and DDRNet strictly receive the original preprocessed 224×224 image.")

        v_col1, v_col2, v_col3 = st.columns(3)

        with v_col1:
            st.image(image_224, caption="1. Original Distorted (224×224)", width="stretch")

        with v_col2:
            st.image(enhanced_image, caption="2. Enhanced (Visualization Only)", width="stretch")
            st.caption("CLAHE enhanced — visualization only")

        with v_col3:
            st.image(rectified_image, caption="3. DDRNet Rectified Output", width="stretch")

        with st.expander("Inspect Predicted Displacement Field Heatmap"):
            if pred_dx is not None and pred_dy is not None:
                disp_mag = np.sqrt(pred_dx ** 2 + pred_dy ** 2)
                norm_mag = np.clip((disp_mag / max(disp_mag.max(), 1e-6)) * 255.0, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, caption="Dense Displacement Magnitude Heatmap", width="stretch")
