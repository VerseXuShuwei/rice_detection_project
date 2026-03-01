"""
Label Smoothing Cross Entropy Loss for regularization.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from losses.py (no logic changes)

Reference:
    "Rethinking the Inception Architecture for Computer Vision"
    https://arxiv.org/abs/1512.00567

Formula:
    Smooth Labels:
        y_smooth(k) = (1 - smoothing)  if k = true_class
        y_smooth(k) = smoothing / (C-1)  otherwise

    Loss = -sum(y_smooth * log(softmax(pred)))

Rationale:
    Label smoothing prevents overfitting by softening hard labels.
    Instead of forcing the model to predict [0, 0, 1, 0], we allow some
    uncertainty: [0.033, 0.033, 0.9, 0.033].

Args:
    smoothing: Label smoothing factor (default: 0.1)
               0.0 = no smoothing (standard CE)
               0.1 = 10% smoothing (recommended)

Usage:
    >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    >>> loss = criterion(outputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with label smoothing regularization.

    Prevents overfitting by softening hard labels.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.

        Args:
            pred: Predictions (logits) of shape (N, C)
            target: Ground truth labels of shape (N,)

        Returns:
            Label smoothing CE loss

        Implementation:
            1. Compute log_probs = log_softmax(pred)
            2. Create smoothed target distribution:
               - True class: 1 - smoothing
               - Other classes: smoothing / (C - 1)
            3. Compute KL divergence: -sum(smooth_target * log_probs)
        """
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # Create smoothed target distribution
        with torch.no_grad():
            # Initialize with uniform distribution for non-target classes
            # Each non-target class gets smoothing / (C - 1)
            smooth_target = torch.full_like(log_probs, self.smoothing / (n_classes - 1))

            # Set target class probability to (1 - smoothing)
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Compute KL divergence (equivalent to CE with smooth labels)
        loss = (-smooth_target * log_probs).sum(dim=-1)

        return loss.mean()


__all__ = ['LabelSmoothingCrossEntropy']
