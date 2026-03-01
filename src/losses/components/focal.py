"""
Focal Loss for addressing class imbalance.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from losses.py (no logic changes)

Reference:
    "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

Formula:
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    where pt is the probability of the true class.

Args:
    alpha: Weighting balance factor (default: 1.0)
           Gives more weight to hard-to-predict samples
    gamma: Focusing parameter (default: 2.0)
           Controls the intensity of balance factor
           Higher gamma = more focus on hard examples
    reduction: Reduction method ('mean', 'sum', or 'none')

Usage:
    >>> criterion = FocalLoss(alpha=1.0, gamma=2.0)
    >>> loss = criterion(outputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Based on Cross-Entropy Loss with a modulating factor (1 - pt)^gamma
    that down-weights easy examples and focuses training on hard examples.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss

        Implementation:
            1. Compute CE loss: ce = -log(pt)
            2. Compute modulating factor: (1 - pt)^gamma
            3. Apply weighting: FL = alpha * (1 - pt)^gamma * ce
        """
        # Compute cross entropy loss (reduction='none' to get per-sample loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute pt (probability of true class)
        # pt = exp(-ce_loss) because ce_loss = -log(pt)
        pt = torch.exp(-ce_loss)

        # Compute focal loss: alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


__all__ = ['FocalLoss']
