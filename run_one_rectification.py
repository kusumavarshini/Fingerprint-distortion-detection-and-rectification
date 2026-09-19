import torch
import cv2
import numpy as np
from scipy.ndimage import zoom

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence

distorted_path = r"data\test_distorted\distorted\FAMILY-13\CHILD\FM013_C1_distorted.png"
clean_path = r"data\test_distorted\clean\FAMILY-13\CHILD\FM013_C1.png"
checkpoint_path = r"experiments\ddrnet_baseline\best_model.pth"

# --------------------------------------------------
# Load distorted image
# --------------------------------------------------

distorted = cv2.imread(distorted_path, 0)
clean = cv2.imread(clean_path, 0)

if distorted is None:
    raise RuntimeError("Could not load distorted image")

if clean is None:
    raise RuntimeError("Could not load clean image")

# --------------------------------------------------
# DDRNet preprocessing
# --------------------------------------------------

mask = segmentation_coherence(
    distorted,
    win_size=16,
    stride=8,
)

mask16 = zoom(
    mask,
    1 / 16,
    order=0,
)

x = ((255 - distorted) / 255.0).astype(np.float32)

x = torch.from_numpy(x)[None, None]
m = torch.from_numpy(mask16.astype(np.float32))[None, None]

# --------------------------------------------------
# Load trained DDRNet
# --------------------------------------------------

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model = DDRNet_DIR(dis_const=16)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --------------------------------------------------
# Predict rectification field
# --------------------------------------------------

with torch.no_grad():
    field, _ = model(x, m)

field = field[0].numpy()

dx16 = field[0]
dy16 = field[1]

# --------------------------------------------------
# Upsample 14x14 -> 224x224
# --------------------------------------------------

dx = cv2.resize(
    dx16,
    (distorted.shape[1], distorted.shape[0]),
    interpolation=cv2.INTER_LINEAR,
)

dy = cv2.resize(
    dy16,
    (distorted.shape[1], distorted.shape[0]),
    interpolation=cv2.INTER_LINEAR,
)

# --------------------------------------------------
# Apply established negative-field convention
# --------------------------------------------------

h, w = distorted.shape

grid_x, grid_y = np.meshgrid(
    np.arange(w, dtype=np.float32),
    np.arange(h, dtype=np.float32),
)

map_x = grid_x - dx
map_y = grid_y - dy

rectified = cv2.remap(
    distorted,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=255,
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def rmse(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(diff ** 2)))


distorted_mae = mae(distorted, clean)
distorted_rmse = rmse(distorted, clean)

rectified_mae = mae(rectified, clean)
rectified_rmse = rmse(rectified, clean)

mae_improvement = (
    (distorted_mae - rectified_mae)
    / distorted_mae
    * 100
)

rmse_improvement = (
    (distorted_rmse - rectified_rmse)
    / distorted_rmse
    * 100
)

# --------------------------------------------------
# Save result
# --------------------------------------------------

output_path = r"rectified_FM013_C1.png"

cv2.imwrite(output_path, rectified)

print("=== RECTIFICATION RESULT ===")
print("Distorted MAE:", distorted_mae)
print("Rectified MAE:", rectified_mae)
print("MAE improvement (%):", mae_improvement)

print()

print("Distorted RMSE:", distorted_rmse)
print("Rectified RMSE:", rectified_rmse)
print("RMSE improvement (%):", rmse_improvement)

print()

print("Saved:", output_path)
print("Rectified shape:", rectified.shape)
