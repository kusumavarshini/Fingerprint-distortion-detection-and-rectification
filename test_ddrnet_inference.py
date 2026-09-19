import torch
import cv2
import numpy as np
from scipy.ndimage import zoom

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence

image_path = r"data\split\test\FAMILY-13\CHILD\FM013_C1.png"
checkpoint_path = r"experiments\ddrnet_baseline\best_model.pth"

img = cv2.imread(image_path, 0)

mask = segmentation_coherence(
    img,
    win_size=16,
    stride=8,
)

mask16 = zoom(
    mask,
    1 / 16,
    order=0,
)

x = ((255 - img) / 255.0).astype(np.float32)

x = torch.from_numpy(x)[None, None]
m = torch.from_numpy(mask16.astype(np.float32))[None, None]

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model = DDRNet_DIR(dis_const=16)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    field, direction = model(x, m)

field = field[0].numpy()
direction = direction[0].numpy()

dx = field[0]
dy = field[1]

magnitude = np.sqrt(dx ** 2 + dy ** 2)

print("Image shape:", img.shape)
print("Mask16 shape:", mask16.shape)

print("Field shape:", field.shape)

print(
    "DX range:",
    float(dx.min()),
    float(dx.max()),
    "mean:",
    float(dx.mean()),
)

print(
    "DY range:",
    float(dy.min()),
    float(dy.max()),
    "mean:",
    float(dy.mean()),
)

print(
    "Field mean magnitude:",
    float(magnitude.mean()),
)

print(
    "Field max magnitude:",
    float(magnitude.max()),
)

print(
    "Direction shape:",
    direction.shape,
)
