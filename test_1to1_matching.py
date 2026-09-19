from pathlib import Path

from afis.pipeline import MindtctExtractor

from mysql_auth import get_person_templates


IDENTITY = "FAMILY-1/CHILD"

# Use one of the person's enrolled images as the probe.
# This is only a pipeline test.
PROBE_IMAGE = Path(
    "data/FAMILY FINGERPRINT DATASET/FAMILY FINGERPRINT DATASET/"
    "FAMILY-1/CHILD/FM001_C1.png"
)


password = input("MySQL password: ")

# --------------------------------------------------
# Retrieve ONLY this person's templates
# --------------------------------------------------

person = get_person_templates(
    IDENTITY,
    password
)

print()
print("Identity:", person["identity_code"])
print("Enrolled templates:", len(person["templates"]))


# --------------------------------------------------
# Extract probe minutiae
# --------------------------------------------------

import cv2

image = cv2.imread(
    str(PROBE_IMAGE),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not read probe image: {PROBE_IMAGE}"
    )

extractor = MindtctExtractor()

probe_template = extractor.extract_minutiae(image)

print("Probe image:", PROBE_IMAGE)
print("Probe minutiae:", probe_template.minutiae.shape)


# --------------------------------------------------
# 1:1 matching
# --------------------------------------------------

print()
print("Bozorth3 scores:")

scores = []

for item in person["templates"]:

    result = extractor.match(
        probe_template,
        item["template"]
    )

    scores.append(result)

    print(
        f"Impression {item['impression_id']}: "
        f"score={result.score:.4f}, "
        f"raw_score={result.raw_score:.2f}, "
        f"decision={result.decision}"
    )


# --------------------------------------------------
# Best score
# --------------------------------------------------

best = max(
    scores,
    key=lambda x: x.score
)

print()
print("Best score:", best.score)
print("Best raw score:", best.raw_score)
print("Best decision:", best.decision)