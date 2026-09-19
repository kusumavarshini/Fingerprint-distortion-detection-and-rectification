from pathlib import Path
import os

import cv2
import numpy as np
import torch
from scipy.ndimage import zoom

from afis import MindtctExtractor
from model import FingerprintCNN
from mysql_auth import get_person_templates

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")


# Latest distortion classifier
CLASSIFIER_CHECKPOINT = Path(
    "experiments/classifier_hard_negative/best_model.pth"
)

CLASSIFIER_THRESHOLD = 0.60


# DDRNet rectification model
DDRNET_CHECKPOINT = Path(
    "experiments/ddrnet_baseline/best_model.pth"
)


# Bozorth3 authentication threshold
MATCH_THRESHOLD = 0.04


# Output directory for authentication images
AUTH_OUTPUT_DIR = Path(
    "experiments/authentication_outputs"
)

AUTH_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GLOBAL MODELS
# ============================================================

EXTRACTOR = MindtctExtractor()

CLASSIFIER = None
DDRNET = None


# ============================================================
# CLASSIFIER
# ============================================================

def load_classifier():

    global CLASSIFIER

    if CLASSIFIER is None:

        if not CLASSIFIER_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: "
                f"{CLASSIFIER_CHECKPOINT}"
            )

        CLASSIFIER = FingerprintCNN()

        checkpoint = torch.load(
            CLASSIFIER_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        CLASSIFIER.load_state_dict(state_dict)

        CLASSIFIER.to(DEVICE)
        CLASSIFIER.eval()

    return CLASSIFIER


def classify_distortion(image):
    """
    Classify fingerprint as clean or distorted.

    Returns:
        probability, is_distorted
    """

    if image is None:
        raise ValueError(
            "Fingerprint image is None."
        )

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Resize to classifier input size
    image_224 = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # Same preprocessing used by classifier
    image_float = (
        image_224.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        image_float
    ).unsqueeze(0).unsqueeze(0)

    model = load_classifier()

    with torch.no_grad():

        logits = model(
            tensor.to(DEVICE)
        )

        probability = torch.sigmoid(
            logits
        ).item()

    is_distorted = (
        probability >= CLASSIFIER_THRESHOLD
    )

    return (
        float(probability),
        bool(is_distorted)
    )


# ============================================================
# DDRNET
# ============================================================

def load_ddrnet():

    global DDRNET

    if DDRNET is None:

        if not DDRNET_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"DDRNet checkpoint not found: "
                f"{DDRNET_CHECKPOINT}"
            )

        DDRNET = DDRNet_DIR(
            dis_const=16
        )

        checkpoint = torch.load(
            DDRNET_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

        DDRNET.load_state_dict(
            state_dict
        )

        DDRNET.to(DEVICE)
        DDRNET.eval()

    return DDRNET


def rectify_with_ddrnet(image):

    if image is None:
        raise ValueError(
            "Fingerprint image is None."
        )

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # DDRNet operates at 224x224
    gray = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # Project's DDRNet segmentation mask
    mask = segmentation_coherence(
        gray,
        win_size=16,
        stride=8
    )

    mask16 = zoom(
        mask,
        1 / 16,
        order=0
    )

    # Same DDRNet image transformation:
    # (255 - image) / 255
    normalized = np.float32(
        (255.0 - gray) / 255.0
    )

    x = normalized[
        None,
        None,
        :,
        :
    ]

    mask_tensor = np.float32(
        mask16
    )[None, None, :, :]

    model = load_ddrnet()

    with torch.no_grad():

        field, _ = model(
            torch.from_numpy(x).to(DEVICE),
            torch.from_numpy(mask_tensor).to(DEVICE),
        )

    field = field[0].cpu().numpy()

    # 14x14 -> 224x224
    dx = cv2.resize(
        field[0],
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    dy = cv2.resize(
        field[1],
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(
            224,
            dtype=np.float32
        ),
        np.arange(
            224,
            dtype=np.float32
        ),
    )

    # Validated rectification convention:
    # map_x = x - dx
    # map_y = y - dy
    map_x = grid_x - dx
    map_y = grid_y - dy

    rectified = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return rectified


# ============================================================
# IMAGE SAVING
# ============================================================

def save_authentication_images(
    original_image,
    rectified_image,
    identity_code,
    is_distorted,
):
    """
    Save original uploaded fingerprint and,
    when applicable, the DDRNet-rectified fingerprint.
    """

    safe_identity = (
        identity_code
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    if is_distorted:

        original_path = (
            AUTH_OUTPUT_DIR
            / f"{safe_identity}_distorted.png"
        )

        rectified_path = (
            AUTH_OUTPUT_DIR
            / f"{safe_identity}_rectified.png"
        )

        cv2.imwrite(
            str(original_path),
            original_image
        )

        cv2.imwrite(
            str(rectified_path),
            rectified_image
        )

        return (
            str(original_path),
            str(rectified_path)
        )

    else:

        original_path = (
            AUTH_OUTPUT_DIR
            / f"{safe_identity}_clean.png"
        )

        cv2.imwrite(
            str(original_path),
            original_image
        )

        return (
            str(original_path),
            None
        )


# ============================================================
# MINDTCT + BOZORTH3
# ============================================================

def extract_mindtct_template(image):

    if image is None:
        raise ValueError(
            "Cannot extract template from empty image."
        )

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    return EXTRACTOR.extract_minutiae(
        image
    )


def match_against_enrolled_templates(
    probe_template,
    enrolled_templates,
):

    results = []

    for template_record in enrolled_templates:

        # IMPORTANT:
        # mysql_auth.py converts the database blob
        # into the key "template"
        stored_template = (
            template_record["template"]
        )

        impression_id = (
            template_record["impression_id"]
        )

        result = EXTRACTOR.match(
            probe_template,
            stored_template,
            method="bozorth3",
        )

        score = float(
            result.score
        )

        raw_score = float(
            result.raw_score
        )

        results.append(
            {
                "impression_id": impression_id,
                "score": score,
                "raw_score": raw_score,
                "decision": bool(
                    result.decision
                ),
            }
        )

    if not results:
        raise RuntimeError(
            "No enrolled fingerprint templates found."
        )

    best_match = max(
        results,
        key=lambda x: x["score"]
    )

    return (
        results,
        best_match
    )


# ============================================================
# MAIN AUTHENTICATION PIPELINE
# ============================================================

def authenticate(
    identity_code,
    fingerprint_image,
    mysql_password,
):

    if not identity_code:
        raise ValueError(
            "Identity code is required."
        )

    if fingerprint_image is None:
        raise ValueError(
            "Fingerprint image is required."
        )

    # --------------------------------------------------------
    # Preserve original uploaded image
    # --------------------------------------------------------

    original_image = (
        fingerprint_image.copy()
    )

    # --------------------------------------------------------
    # 1. Distortion classification
    # --------------------------------------------------------

    (
        distortion_probability,
        is_distorted,
    ) = classify_distortion(
        fingerprint_image
    )

    # --------------------------------------------------------
    # 2. Rectification
    # --------------------------------------------------------

    rectification_applied = False
    rectified_image = None

    if is_distorted:

        rectified_image = (
            rectify_with_ddrnet(
                fingerprint_image
            )
        )

        fingerprint_for_matching = (
            rectified_image
        )

        rectification_applied = True

    else:

        fingerprint_for_matching = (
            fingerprint_image
        )

    # --------------------------------------------------------
    # 3. Save images
    # --------------------------------------------------------

    (
        saved_original,
        saved_rectified,
    ) = save_authentication_images(
        original_image=original_image,
        rectified_image=rectified_image,
        identity_code=identity_code,
        is_distorted=is_distorted,
    )

    # --------------------------------------------------------
    # 4. Extract MINDTCT template
    # --------------------------------------------------------

    probe_template = (
        extract_mindtct_template(
            fingerprint_for_matching
        )
    )

    # --------------------------------------------------------
    # 5. Retrieve enrolled templates from MySQL
    # --------------------------------------------------------

    # mysql_auth.py returns an outer dictionary:
    #
    # {
    #     "person_id": ...,
    #     "identity_code": ...,
    #     "family_id": ...,
    #     "member_type": ...,
    #     "templates": [...]
    # }
    #
    # We therefore extract the actual template list.

    enrollment_data = (
        get_person_templates(
            identity_code=identity_code,
            mysql_password=mysql_password,
        )
    )

    enrolled_templates = (
        enrollment_data["templates"]
    )

    if not enrolled_templates:
        raise ValueError(
            f"No enrolled templates found for "
            f"{identity_code}"
        )

    # --------------------------------------------------------
    # 6. Bozorth3 matching against 5 impressions
    # --------------------------------------------------------

    scores, best_match = (
        match_against_enrolled_templates(
            probe_template,
            enrolled_templates,
        )
    )

    # --------------------------------------------------------
    # 7. Authentication decision
    # --------------------------------------------------------

    authenticated = (
        best_match["score"]
        >= MATCH_THRESHOLD
    )

    # --------------------------------------------------------
    # 8. Person metadata
    # --------------------------------------------------------

    person_id = (
        enrollment_data.get(
            "person_id"
        )
    )

    member_type = (
        enrollment_data.get(
            "member_type"
        )
    )

    # --------------------------------------------------------
    # 9. Return complete result
    # --------------------------------------------------------

    return {

        "authenticated": bool(
            authenticated
        ),

        "identity_code": identity_code,

        "person_id": person_id,

        "member_type": member_type,

        "distortion_probability": (
            distortion_probability
        ),

        "is_distorted": bool(
            is_distorted
        ),

        "rectification_applied": bool(
            rectification_applied
        ),

        "scores": scores,

        "best_score": (
            best_match["score"]
        ),

        "best_raw_score": (
            best_match["raw_score"]
        ),

        "best_impression_id": (
            best_match["impression_id"]
        ),

        "saved_original_path": (
            saved_original
        ),

        "saved_rectified_path": (
            saved_rectified
        ),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("FINGERPRINT AUTHENTICATION")
    print("=" * 70)

    identity_code = input(
        "Identity code: "
    ).strip()

    image_path = input(
        "Fingerprint image path: "
    ).strip()

    mysql_password = input(
        "MySQL password: "
    ).strip()

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    result = authenticate(
        identity_code=identity_code,
        fingerprint_image=image,
        mysql_password=mysql_password,
    )

    print()
    print("=" * 70)
    print("AUTHENTICATION RESULT")
    print("=" * 70)

    print(
        f"Identity       : "
        f"{result['identity_code']}"
    )

    print(
        f"Person ID      : "
        f"{result['person_id']}"
    )

    print(
        f"Member type    : "
        f"{result['member_type']}"
    )

    print(
        f"Distortion prob: "
        f"{result['distortion_probability']:.4f}"
    )

    print(
        f"Distorted      : "
        f"{result['is_distorted']}"
    )

    print(
        f"Rectified      : "
        f"{result['rectification_applied']}"
    )

    print()
    print("Bozorth3 scores:")

    for score in result["scores"]:

        print(
            f"  Impression "
            f"{score['impression_id']}: "
            f"score={score['score']:.4f}, "
            f"raw={score['raw_score']:.1f}, "
            f"decision={score['decision']}"
        )

    print()
    print(
        f"Best score     : "
        f"{result['best_score']:.4f}"
    )

    print(
        f"Best impression: "
        f"{result['best_impression_id']}"
    )

    print()
    print(
        f"AUTHENTICATED  : "
        f"{result['authenticated']}"
    )

    print()
    print("Saved images:")

    print(
        f"Original       : "
        f"{result['saved_original_path']}"
    )

    if result["saved_rectified_path"]:

        print(
            f"Rectified      : "
            f"{result['saved_rectified_path']}"
        )

    print("=" * 70)