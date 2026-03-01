"""
Combined loss functions for multi-label and multi-class classification.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from losses.py (no logic changes)

Components:
    - CombinedLoss: Focal + Label Smoothing combination (multi-class)
    - BCEWithLogitsLossMultiLabel: Standard BCE for multi-label tasks
    - FocalBCEWithLogitsLoss: Focal-weighted BCE for class imbalance

Reference:
    "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

Usage:
    >>> # Combined Loss (Focal + Label Smoothing)
    >>> criterion = CombinedLoss(
    ...     focal_alpha=1.0,
    ...     focal_gamma=2.0,
    ...     label_smoothing=0.1,
    ...     focal_weight=0.6,
    ...     ls_weight=0.4
    ... )

    >>> # Standard BCE
    >>> criterion = BCEWithLogitsLossMultiLabel(pos_weight=torch.tensor([2.0] * 19))
    >>> loss = criterion(logits, multi_hot_labels)

    >>> # Focal BCE (recommended for MIL)
    >>> criterion = FocalBCEWithLogitsLoss(
    ...     focal_alpha=0.25,
    ...     focal_gamma=1.5,
    ...     pos_weight=torch.tensor([2.0] * 19)
    ... )
    >>> loss = criterion(bag_logits, bag_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.losses.components.focal import FocalLoss
from src.losses.components.smoothing import LabelSmoothingCrossEntropy


class CombinedLoss(nn.Module):
    """
    Combined loss function (Focal Loss + Label Smoothing).

    Useful for handling both class imbalance and overfitting simultaneously.

    Args:
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        focal_weight: Weight for focal loss (default: 0.5)
        ls_weight: Weight for label smoothing loss (default: 0.5)

    Example:
        >>> criterion = CombinedLoss(
        ...     focal_alpha=1.0,
        ...     focal_gamma=2.0,
        ...     label_smoothing=0.1,
        ...     focal_weight=0.6,
        ...     ls_weight=0.4
        ... )
    """

    def __init__(
        self,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        focal_weight: float = 0.5,
        ls_weight: float = 0.5
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ls_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.focal_weight = focal_weight
        self.ls_weight = ls_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        focal = self.focal_loss(pred, target)
        ls = self.ls_loss(pred, target)

        return self.focal_weight * focal + self.ls_weight * ls


class BCEWithLogitsLossMultiLabel(nn.Module):
    """
    Binary Cross Entropy with Logits for multi-label classification.

    Suitable for MIL (Multiple Instance Learning) scenarios where each instance
    can have multiple positive labels simultaneously.

    Args:
        pos_weight: Weight for positive examples (shape: num_classes)
                   Useful for handling class imbalance
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> # For 19 disease classes
        >>> pos_weight = torch.tensor([2.0] * 19)  # Weight rare classes higher
        >>> criterion = BCEWithLogitsLossMultiLabel(pos_weight=pos_weight)
        >>> loss = criterion(logits, multi_hot_labels)
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE with logits loss for multi-label classification.

        Args:
            inputs: Predictions (logits) of shape (B, num_classes)
            targets: Multi-hot encoded labels of shape (B, num_classes)
                    Each element is 0 or 1

        Returns:
            BCE with logits loss
        """
        return F.binary_cross_entropy_with_logits(
            inputs,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal BCE with Logits Loss for multi-label classification.

    Combines Focal Loss weighting with Binary Cross Entropy to handle:
    1. Multi-label classification (unlike standard Focal Loss which is for multi-class)
    2. Class imbalance (through focal_alpha and pos_weight)
    3. Hard example mining (through focal_gamma)

    Reference: Adapted from "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * BCE(p, y)

    Args:
        focal_alpha: Weighting factor for positive class (default: 0.25)
                    Higher alpha = more weight on positive samples
        focal_gamma: Focusing parameter (default: 2.0)
                    Higher gamma = more focus on hard examples
                    Recommended: 1.5 for MIL (lower than standard 2.0)
        pos_weight: Per-class weight for positive examples (shape: num_classes)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> criterion = FocalBCEWithLogitsLoss(
        ...     focal_alpha=0.25,
        ...     focal_gamma=1.5,
        ...     pos_weight=torch.tensor([2.0] * 19)
        ... )
        >>> loss = criterion(bag_logits, bag_labels)
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal BCE with logits loss.

        Args:
            inputs: Predictions (logits) of shape (B, num_classes)
            targets: Multi-hot encoded labels of shape (B, num_classes)

        Returns:
            Focal BCE loss
        """
        # Compute BCE loss (element-wise, no reduction yet)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute p_t: probability of true class
        # If target = 1: p_t = prob
        # If target = 0: p_t = 1 - prob
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Compute alpha weight
        alpha_weight = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)

        # Combine: focal_loss = alpha * focal_weight * bce_loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


__all__ = ['CombinedLoss', 'BCEWithLogitsLossMultiLabel', 'FocalBCEWithLogitsLoss']
