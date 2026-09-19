import pickle
from pathlib import Path

import numpy as np
from afis.pipeline import FingerprintTemplate, MindtctExtractor


GALLERY_PATH = Path(
    "experiments/afis_gallery/mindtct_gallery.pkl"
)


with GALLERY_PATH.open("rb") as f:
    gallery = pickle.load(f)

record_a = gallery["records"][0]
record_b = gallery["records"][1]

original_a = record_a["template"]
original_b = record_b["template"]


# Reconstruct compact templates
compact_a = FingerprintTemplate(
    minutiae=np.array(original_a.minutiae, copy=True),
    header=original_a.header,
    report=original_a.report,
)

compact_b = FingerprintTemplate(
    minutiae=np.array(original_b.minutiae, copy=True),
    header=original_b.header,
    report=original_b.report,
)


extractor = MindtctExtractor()


print("Image A:", record_a["path"])
print("Image B:", record_b["path"])
print("Identity A:", record_a["identity"])
print("Identity B:", record_b["identity"])

print()
print("Original A minutiae:", original_a.minutiae.shape)
print("Compact A minutiae:", compact_a.minutiae.shape)

print()
print("Testing original templates...")

original_score = extractor.match(
    original_a,
    original_b
)

print("Original Bozorth3 score:", original_score)

print()
print("Testing compact templates...")

compact_score = extractor.match(
    compact_a,
    compact_b
)

print("Compact Bozorth3 score:", compact_score)

print()

if original_score == compact_score:
    print("PASS: Scores are identical.")
else:
    print("WARNING: Scores are different.")
