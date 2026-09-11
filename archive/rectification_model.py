"""
Rectification Model
===================

Custom CNN for dense fingerprint distortion-field regression.

Input:
    distorted fingerprint
    shape: (B, 1, 224, 224)

Output:
    displacement field
    shape: (B, 2, 224, 224)

Channel 0 -> dx
Channel 1 -> dy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONVOLUTIONAL BLOCK
# ============================================================

class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# RECTIFICATION NETWORK
# ============================================================

class FingerprintRectificationNet(nn.Module):
    """
    Encoder-decoder CNN for dense displacement prediction.

    The network preserves spatial information through skip
    connections and produces a full-resolution dx/dy field.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = ConvBlock(
            256,
            512
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.dec4 = ConvBlock(
            512 + 256,
            256
        )

        self.dec3 = ConvBlock(
            256 + 128,
            128
        )

        self.dec2 = ConvBlock(
            128 + 64,
            64
        )

        self.dec1 = ConvBlock(
            64 + 32,
            32
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            32,
            2,
            kernel_size=1
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = F.interpolate(
            b,
            size=e4.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(
            d4,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(
            d3,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Dense displacement field
        field = self.output(d1)

        return field


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(model):
    """Return number of trainable parameters."""

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# STANDALONE VERIFICATION
# ============================================================

def main():

    print("=" * 70)
    print("RECTIFICATION MODEL VERIFICATION")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    model = FingerprintRectificationNet().to(device)

    params = count_parameters(model)

    print(
        f"Trainable parameters: {params:,}"
    )

    # Dummy fingerprint batch
    x = torch.randn(
        2,
        1,
        224,
        224,
        device=device
    )

    with torch.no_grad():
        output = model(x)

    print(
        f"\nInput shape : {tuple(x.shape)}"
    )

    print(
        f"Output shape: {tuple(output.shape)}"
    )

    print(
        f"\nExpected output: "
        f"(2, 2, 224, 224)"
    )

    if output.shape != (
        2,
        2,
        224,
        224
    ):
        raise RuntimeError(
            f"Unexpected output shape: "
            f"{tuple(output.shape)}"
        )

    print(
        "\n[PASS] Rectification model is working."
    )


if __name__ == "__main__":
    main()