import pickle
from pathlib import Path

import numpy as np
from afis.pipeline import FingerprintTemplate

gallery_path = Path(
    "experiments/afis_gallery/mindtct_gallery.pkl"
)

with gallery_path.open("rb") as f:
    gallery = pickle.load(f)

record = gallery["records"][0]
original = record["template"]

print("Original template:", type(original))
print("Original minutiae:", original.minutiae.shape)

# Reconstruct a minimal template containing only minutiae
compact = FingerprintTemplate(
    minutiae=np.array(original.minutiae, copy=True),
    header=original.header,
    report=original.report,
)
print("Compact template:", type(compact))
print("Compact minutiae:", compact.minutiae.shape)

compact_blob = pickle.dumps(
    compact,
    protocol=pickle.HIGHEST_PROTOCOL
)

print(
    "Compact serialized size:",
    len(compact_blob),
    "bytes"
)

print(
    "Size reduction:",
    round(
        (1 - len(compact_blob) / len(
            pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
        )) * 100,
        2
    ),
    "%"
)