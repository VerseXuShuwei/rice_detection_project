"""
Model Components - Modular building blocks for MIL architectures.

Components:
    - ViTResidualBlock: Lightweight ViT with residual connection
    - FPNNeck: Top-down feature pyramid for multi-scale fusion
    - HeatmapHead: Conv1x1 -> GMP for class-specific activation maps
"""

from src.models.components.vit_block import ViTResidualBlock
from src.models.components.fpn_neck import FPNNeck
from src.models.components.heatmap_head import HeatmapHead

__all__ = [
    'ViTResidualBlock',
    'FPNNeck',
    'HeatmapHead'
]
