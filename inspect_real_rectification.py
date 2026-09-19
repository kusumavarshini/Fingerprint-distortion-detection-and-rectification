import torch
import cv2
import numpy as np
from scipy.ndimage import zoom

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence

image_path = r"data\test_distorted\distorted\FAMILY-13\CHILD\FM013_C1_distorted.png"
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

dx = field[0]
dy = field[1]

magnitude = np.sqrt(dx ** 2 + dy ** 2)

inside = mask16.astype(bool)
outside = ~inside

print("Image shape:", img.shape)
print("Mask16 shape:", mask16.shape)

print("\n=== ALL ===")
print("Mean magnitude:", float(magnitude.mean()))
print("Max magnitude:", float(magnitude.max()))

print("\n=== INSIDE MASK ===")
print("Pixels:", int(inside.sum()))
print("DX min/max:", float(dx[inside].min()), float(dx[inside].max()))
print("DY min/max:", float(dy[inside].min()), float(dy[inside].max()))
print("Mean magnitude:", float(magnitude[inside].mean()))
print("Max magnitude:", float(magnitude[inside].max()))

print("\n=== OUTSIDE MASK ===")
print("Pixels:", int(outside.sum()))
print("DX min/max:", float(dx[outside].min()), float(dx[outside].max()))
print("DY min/max:", float(dy[outside].min()), float(dy[outside].max()))
print("Mean magnitude:", float(magnitude[outside].mean()))
print("Max magnitude:", float(magnitude[outside].max()))
