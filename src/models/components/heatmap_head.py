"""
Heatmap Head - Conv1x1 based class-specific activation map generator.

Recent Updates:
    - [2026-01-29] Initial: Created for CAM-style pooling architecture

Key Features:
    - Generates per-class spatial heatmaps before pooling
    - Preserves spatial information for localization
    - Supports both GMP-pooled logits and raw heatmap output

Architecture:
    Input: (B, C, H, W)
    └─→ Conv1x1: (B, num_classes, H, W)  [Raw Heatmap]
    └─→ GMP: (B, num_classes)             [Pooled Logits]

Mathematical Formulation:
    $$
    \text{Heatmap}_c(i,j) = W_c^T \cdot F(i,j) + b_c
    $$
    $$
    \text{Logit}_c = \max_{i,j} \text{Heatmap}_c(i,j)
    $$

Usage:
    >>> head = HeatmapHead(in_channels=256, num_classes=10)
    >>> features = torch.randn(4, 256, 24, 24)
    >>> logits = head(features)  # (4, 12) - pooled logits
    >>> heatmap = head.get_heatmap(features)  # (4, 12, 24, 24) - spatial heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HeatmapHead(nn.Module):
    """
    Class-specific heatmap generator with configurable pooling.

    Generates spatial heatmaps via Conv1x1 and aggregates via
    Global Max Pooling (default) or Top-K Mean Pooling (optional).

    Args:
        in_channels: Input feature channels
        num_classes: Number of output classes (including Class 0 for healthy)
        pool_mode: "gmp" (Global Max Pool, default) | "topk_mean" (Top-K spatial mean)
        topk_pool_k: Number of spatial locations to average in topk_mean mode

    Methods:
        forward(x): Returns pooled logits (B, num_classes)
        get_heatmap(x): Returns raw spatial heatmap (B, num_classes, H, W)

    pool_mode rationale:
        "gmp": Picks single max-response location per class. Fast, but sensitive to
               background peaks on large feature maps.
        "topk_mean": Averages Top-K spatial responses per class. More robust to
                     spurious background activations; tested effective in inference
                     (D ablation, 2026-02-24). Default OFF to preserve training continuity.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_mode: str = "gmp",    # "gmp" | "topk_mean"
        topk_pool_k: int = 3,      # k for topk_mean mode
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool_mode = pool_mode
        self.topk_pool_k = topk_pool_k

        # Conv1x1 generates per-class activation maps
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True
        )

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def _pool(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Pool spatial heatmap to per-class logits.

        Args:
            heatmap: (B, num_classes, H, W)

        Returns:
            logits: (B, num_classes)
        """
        if self.pool_mode == "topk_mean":
            # Flatten spatial dims, take Top-K mean per class
            B, C, H, W = heatmap.shape
            flat = heatmap.view(B, C, -1)           # (B, C, H*W)
            k = min(self.topk_pool_k, flat.size(-1))
            topk_vals, _ = flat.topk(k, dim=-1)     # (B, C, k)
            return topk_vals.mean(dim=-1)            # (B, C)
        else:
            # Default: Global Max Pooling
            return F.adaptive_max_pool2d(heatmap, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Generate heatmap and pool to logits.

        Args:
            x: (B, C, H, W) input features

        Returns:
            logits: (B, num_classes) pooled classification logits
        """
        heatmap = self.conv(x)   # (B, num_classes, H, W)
        return self._pool(heatmap)

    def get_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw spatial heatmap (for visualization/localization).

        Args:
            x: (B, C, H, W) input features

        Returns:
            heatmap: (B, num_classes, H, W) class-specific activation maps
        """
        return self.conv(x)

    def forward_with_heatmap(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and heatmap.

        Args:
            x: (B, C, H, W) input features

        Returns:
            logits: (B, num_classes) pooled logits
            heatmap: (B, num_classes, H, W) spatial heatmap
        """
        heatmap = self.conv(x)
        logits = self._pool(heatmap)
        return logits, heatmap


__all__ = ['HeatmapHead']
