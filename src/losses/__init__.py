"""
Loss functions module for Asymmetric MIL training.

Recent Updates:
    - [2026-01-05] Refactor: Modularized losses.py into components and specialized modules

Structure:
    - components/: Building block losses (Focal, LabelSmoothing, BCE, Combined)
    - topk_anchored.py: TopKAnchoredMILLoss (Core innovation, 580 lines)
    - asymmetric.py: AsymmetricMILLoss (Ablation baseline)
    - builder.py: Loss factory for config-driven instantiation

Loss Types:
    MIL Losses (Tile-level):
        - TopKAnchoredMILLoss: Recommended MIL loss with dynamic weighting and ranking
        - AsymmetricMILLoss: Simpler MIL baseline for ablation

    Component Losses:
        - FocalLoss: Class imbalance handling
        - LabelSmoothingCrossEntropy: Regularization
        - CombinedLoss: Focal + Label Smoothing
        - BCEWithLogitsLossMultiLabel: Multi-label BCE
        - FocalBCEWithLogitsLoss: Focal-weighted multi-label BCE

    Factory:
        - create_loss_function: Config-driven loss instantiation

Usage:
    >>> # Direct instantiation
    >>> from src.losses import TopKAnchoredMILLoss
    >>> criterion = TopKAnchoredMILLoss(
    ...     top1_ce_weight=5.0,
    ...     enable_dynamic_weight=True,
    ...     enable_ranking=True
    ... )

    >>> # Config-driven instantiation (Recommended)
    >>> from src.losses import create_loss_function
    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.load('configs/algorithm/asymmetric_default.yaml')
    >>> criterion = create_loss_function(config.loss)
"""

# Component losses
from src.losses.components import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    CombinedLoss,
    BCEWithLogitsLossMultiLabel,
    FocalBCEWithLogitsLoss
)

# MIL losses
from src.losses.topk_anchored import TopKAnchoredMILLoss
from src.losses.asymmetric import AsymmetricMILLoss

# Loss factory
from src.losses.builder import create_loss_function


__all__ = [
    # Component losses
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'CombinedLoss',
    'BCEWithLogitsLossMultiLabel',
    'FocalBCEWithLogitsLoss',
    # MIL losses
    'TopKAnchoredMILLoss',
    'AsymmetricMILLoss',
    # Factory
    'create_loss_function'

]
