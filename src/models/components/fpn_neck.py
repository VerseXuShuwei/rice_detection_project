"""
FPN Neck - Simple Top-Down Feature Pyramid Network.

Recent Updates:
    - [2026-01-29] Initial: Created for multi-scale feature fusion

Key Features:
    - Top-down pathway: Stage 4 upsampled + Stage 3 fusion
    - Lateral connections via 1x1 conv for channel alignment
    - Single output scale (aligned with Stage 3 resolution)

Architecture:
    Stage 4 (C4, low res) ──→ 1x1 Conv ──→ Upsample ──┐
                                                       ├──→ Add ──→ 3x3 Conv ──→ Output
    Stage 3 (C3, high res) ──→ 1x1 Conv ──────────────┘

Usage:
    >>> fpn = FPNNeck(in_channels_s3=160, in_channels_s4=1280, out_channels=256)
    >>> c3 = torch.randn(4, 160, 24, 24)   # Stage 3 features
    >>> c4 = torch.randn(4, 1280, 12, 12)  # Stage 4 features
    >>> out = fpn(c3, c4)  # (4, 256, 24, 24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FPNNeck(nn.Module):
    """
    Simple Top-Down FPN for two-stage feature fusion.

    Fuses high-level semantic features (Stage 4) with finer spatial
    features (Stage 3) through top-down pathway.

    Args:
        in_channels_s3: Input channels from Stage 3 (e.g., 160 for EfficientNetV2-S)
        in_channels_s4: Input channels from Stage 4 (e.g., 1280 for EfficientNetV2-S)
        out_channels: Output channels for fused features

    Input:
        c3: (B, C3, H3, W3) Stage 3 feature maps
        c4: (B, C4, H4, W4) Stage 4 feature maps (H4 < H3)

    Output:
        out: (B, out_channels, H3, W3) Fused feature maps at Stage 3 resolution
    """

    def __init__(
        self,
        in_channels_s3: int = 160,
        in_channels_s4: int = 1280,
        out_channels: int = 256
    ):
        super().__init__()

        # Lateral connections (1x1 conv for channel alignment)
        self.lateral_s3 = nn.Conv2d(in_channels_s3, out_channels, kernel_size=1)
        self.lateral_s4 = nn.Conv2d(in_channels_s4, out_channels, kernel_size=1)

        # Output convolution (smooth the merged features)
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Top-down fusion.

        Args:
            c3: (B, C3, H3, W3) Stage 3 features (higher resolution)
            c4: (B, C4, H4, W4) Stage 4 features (lower resolution)

        Returns:
            out: (B, out_channels, H3, W3) Fused features
        """
        # Lateral connections
        p3 = self.lateral_s3(c3)  # (B, out_channels, H3, W3)
        p4 = self.lateral_s4(c4)  # (B, out_channels, H4, W4)

        # Top-down: Upsample p4 to match p3 resolution
        p4_upsampled = F.interpolate(
            p4,
            size=p3.shape[2:],  # Match H3, W3
            mode='bilinear',
            align_corners=False
        )

        # Merge via addition
        merged = p3 + p4_upsampled

        # Smooth output
        out = self.output_conv(merged)

        return out


__all__ = ['FPNNeck']
