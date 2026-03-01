"""
Loss function components.

Recent Updates:
    - [2026-01-05] Refactor: Created components submodule for loss building blocks

Components:
    - FocalLoss: Focal Loss for class imbalance
    - LabelSmoothingCrossEntropy: Label smoothing regularization
    - CombinedLoss: Focal + Label Smoothing combination
    - BCEWithLogitsLossMultiLabel: Multi-label BCE loss
    - FocalBCEWithLogitsLoss: Focal-weighted BCE for multi-label

Usage:
    >>> from src.losses.components import FocalLoss, LabelSmoothingCrossEntropy
"""

from src.losses.components.focal import FocalLoss
from src.losses.components.smoothing import LabelSmoothingCrossEntropy
from src.losses.components.combined import (
    CombinedLoss,
    BCEWithLogitsLossMultiLabel,
    FocalBCEWithLogitsLoss
)

__all__ = [
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'CombinedLoss',
    'BCEWithLogitsLossMultiLabel',
    'FocalBCEWithLogitsLoss',
]
