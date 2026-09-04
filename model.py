"""
CNN Model -- Fingerprint Distortion Binary Classifier
======================================================
Small custom CNN baseline for 224x224 single-channel fingerprint images.

Architecture:
    Input: 1 x 224 x 224
        ↓
    Conv2D(1→32)  → BN → ReLU → MaxPool2D   →  32 x 112 x 112
    Conv2D(32→64) → BN → ReLU → MaxPool2D   →  64 x 56 x 56
    Conv2D(64→128)→ BN → ReLU → MaxPool2D   → 128 x 28 x 28
    Conv2D(128→256)→ BN → ReLU              → 256 x 28 x 28
        ↓
    Adaptive Global Average Pooling           → 256 x 1 x 1
    Dropout(0.5)                              → 256
    Fully Connected                           → 1 logit

Binary classification:  CLEAN (0) vs DISTORTED (1)
Loss: BCEWithLogitsLoss (single output logit)
"""

import torch
import torch.nn as nn


class FingerprintCNN(nn.Module):
    """
    4-layer CNN baseline for binary fingerprint distortion classification.

    Parameters
    ----------
    dropout : float
        Dropout probability before the final FC layer.  Default 0.5.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # -- Convolutional feature extractor ----------------------------------
        self.features = nn.Sequential(
            # Block 1:  1 → 32, 224→112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 → 64, 112→56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 → 128, 56→28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 → 256, 28→28 (no pooling)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # -- Classifier head --------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)          # 256 x 1 x 1
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, 1)                   # single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, 1, 224, 224), float32, range [0, 1].

        Returns
        -------
        torch.Tensor
            Raw logits of shape (B, 1).  Apply sigmoid for probabilities.
        """
        x = self.features(x)         # (B, 256, 28, 28)
        x = self.pool(x)             # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)   # (B, 256)
        x = self.dropout(x)          # (B, 256)
        x = self.fc(x)               # (B, 1)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    """Return a formatted summary of the model architecture."""
    lines = []
    lines.append("=" * 65)
    lines.append("  FingerprintCNN -- Architecture Summary")
    lines.append("=" * 65)

    total_params = 0
    trainable_params = 0

    lines.append(f"\n  {'Layer':<40} {'Params':>12}")
    lines.append(f"  {'-'*40} {'-'*12}")

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n
        lines.append(f"  {name:<40} {n:>12,}")

    lines.append(f"  {'-'*40} {'-'*12}")
    lines.append(f"  {'Total parameters':<40} {total_params:>12,}")
    lines.append(f"  {'Trainable parameters':<40} {trainable_params:>12,}")
    lines.append(f"  {'Non-trainable parameters':<40} {total_params - trainable_params:>12,}")
    lines.append("=" * 65)

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test: create model and run a dummy batch through it
    model = FingerprintCNN(dropout=0.5)
    print(model)
    print()
    print(model_summary(model))

    # Dummy forward pass
    dummy = torch.randn(4, 1, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Dummy input shape  : {dummy.shape}")
    print(f"  Dummy output shape : {out.shape}")
    print(f"  Output values      : {out.squeeze().tolist()}")
