from pathlib import Path

p = Path("app.py")
s = p.read_text(encoding="utf-8")

start = s.index("def perform_fingerprint_matching(")
end = s.index("\n\ndef ", start + 10)

new_func = '''def perform_fingerprint_matching(candidate_image: np.ndarray) -> dict:
    """
    Perform biometric fingerprint identification against the full enrolled gallery.

    AFIS operates on the original/high-resolution grayscale fingerprint.
    Each identity has five enrolled impressions; the strongest Bozorth3
    score across those impressions represents that identity.

    The research train/validation/test split is not modified by this
    application-level gallery search.
    """
    if not AFIS_AVAILABLE:
        return {
            "status": "BACKEND_NOT_IMPLEMENTED",
            "message": "The AFIS biometric matching backend is not available.",
            "match_found": False,
            "matched_id": None,
            "score": None,
            "threshold": None,
        }

    if not GALLERY_DIR.exists():
        return {
            "status": "DATASET_NOT_FOUND",
            "message": f"Enrolled fingerprint gallery not found at {GALLERY_DIR}.",
            "match_found": False,
            "matched_id": None,
            "score": None,
            "threshold": None,
        }

    try:
        extractor = MindtctExtractor()

        if candidate_image.ndim == 3:
            candidate_image = cv2.cvtColor(candidate_image, cv2.COLOR_RGB2GRAY)

        if candidate_image.shape != (512, 512):
            candidate_image = cv2.resize(
                candidate_image,
                (512, 512),
                interpolation=cv2.INTER_CUBIC,
            )

        probe_template = extractor.extract_minutiae(candidate_image)

        threshold = 0.04
        best_score = -1.0
        best_id = None
        best_reference = None
        identity_count = 0
        image_count = 0

        for family_dir in sorted(GALLERY_DIR.glob("FAMILY-*")):
            for member_dir in sorted(family_dir.iterdir()):
                if not member_dir.is_dir():
                    continue

                ref_images = sorted(member_dir.glob("*.png"))
                if not ref_images:
                    continue

                identity_count += 1
                identity_best_score = -1.0
                identity_best_reference = None

                for ref_path in ref_images:
                    ref_img = cv2.imread(
                        str(ref_path),
                        cv2.IMREAD_GRAYSCALE,
                    )

                    if ref_img is None:
                        continue

                    if ref_img.shape != (512, 512):
                        ref_img = cv2.resize(
                            ref_img,
                            (512, 512),
                            interpolation=cv2.INTER_CUBIC,
                        )

                    ref_template = extractor.extract_minutiae(ref_img)

                    res = extractor.match(
                        probe_template,
                        ref_template,
                        method="bozorth3",
                        height_a=512,
                        height_b=512,
                    )

                    score = float(
                        res.score if hasattr(res, "score") else res
                    )

                    image_count += 1

                    if score > identity_best_score:
                        identity_best_score = score
                        identity_best_reference = ref_path.name

                if identity_best_score > best_score:
                    best_score = identity_best_score
                    best_id = f"{family_dir.name} / {member_dir.name}"
                    best_reference = identity_best_reference

        match_found = best_score >= threshold

        return {
            "status": "COMPLETED",
            "match_found": match_found,
            "matched_id": (
                f"{best_id} ({best_reference})"
                if match_found and best_id is not None
                else None
            ),
            "score": best_score,
            "threshold": threshold,
            "gallery_identities": identity_count,
            "gallery_images": image_count,
            "probe_minutiae": (
                len(probe_template.minutiae)
                if hasattr(probe_template, "minutiae")
                else None
            ),
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"AFIS matching failed: {e}",
            "match_found": False,
            "matched_id": None,
            "score": None,
            "threshold": 0.04,
        }
'''

p.write_text(s[:start] + new_func + s[end:], encoding="utf-8")

print("AFIS matching function replaced successfully.")