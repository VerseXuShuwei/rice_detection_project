"""
Asymmetric MIL Loss with phase-aware strategy.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from losses.py (no logic changes)

Reference:
    Designed for ablation experiments comparing with TopKAnchoredMILLoss.
    Implements a simpler two-phase MIL strategy.

Formulas:
    Phase 1 (Warm-up): Negative Rejection Loss
        Positive tiles: L = -log(1 - p_0)
            where p_0 = P(Class 0 | positive tile)
        Negative tiles: L = CE(output, target=0)

    Phase 2 (Stable): Soft Bootstrapping
        Target = alpha × OneHot(c) + (1 - alpha) × p_model
        Loss = -sum(Target * log_softmax(pred))

Args:
    alpha: Soft bootstrapping weight (default: 0.9)
           Controls balance between hard label and model prediction
    epsilon: Numerical stability constant (default: 1e-8)

Usage:
    >>> criterion = AsymmetricMILLoss(alpha=0.9)
    >>> # Warm-up phase
    >>> loss = criterion(outputs, labels, is_warmup=True)
    >>> # Stable phase
    >>> loss = criterion(outputs, labels, is_warmup=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricMILLoss(nn.Module):
    """
    Asymmetric MIL Loss with phase-aware strategy.

    Implements three-phase loss evolution:
        Phase 1 (Warm-up): Negative Rejection Loss
            - Positive tiles: L = -log(1 - p_0) (only require "not background")
            - Negative tiles: L = CE(output, target=0)
            - Goal: Separate "background vs foreground" without forcing specific disease class

        Phase 2 (Stable): Soft Bootstrapping CE
            - Target = alpha × OneHot(c) + (1 - alpha) × p_model
            - Goal: Fine-grained classification while preventing over-confidence
            - Promotes heatmap region activation (not single-point)

    Key advantages:
        1. Warm-up: Allows uncertainty in positive tiles (Top-K may select backgrounds)
        2. Stable: Prevents over-confidence, promotes heatmap diffusion
        3. Consistent with evaluation metrics (Negative Recall, Negative Confidence)

    Args:
        alpha: Soft bootstrapping weight (default: 0.9)
               0.9 for hard label, 0.1 for model prediction
        epsilon: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> criterion = AsymmetricMILLoss(alpha=0.9)
        >>> # Warm-up phase
        >>> loss = criterion(outputs, labels, is_warmup=True)
        >>> # Stable phase
        >>> loss = criterion(outputs, labels, is_warmup=False)
    """

    def __init__(self, alpha: float = 0.9, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = alpha  # Soft bootstrapping weight
        self.epsilon = epsilon

        # Loss decomposition for logging (updated after each forward pass)
        self.last_pos_loss = 0.0
        self.last_neg_loss = 0.0

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_warmup: bool = True
    ) -> torch.Tensor:
        """
        Compute loss based on training phase.

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels (0 for negative, 1-C for positive)
            is_warmup: Whether in warm-up phase

        Returns:
            Loss value (scalar)
        """
        if is_warmup:
            return self.negative_rejection_loss(outputs, labels)
        else:
            return self.soft_bootstrapping_loss(outputs, labels)

    def negative_rejection_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative Rejection Loss for warm-up phase.

        For positive tiles (labels > 0):
            L = -log(1 - p_0)
            Meaning: Only require p_0 (background probability) to be low
                    Don't care about specific disease class (allow uncertainty)

        For negative tiles (labels == 0):
            L = CE(output, target=0)
            Meaning: Standard cross entropy, force to predict Class 0

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels

        Returns:
            Loss value
        """
        probs = F.softmax(outputs, dim=1)

        # Split positive and negative tiles
        neg_mask = labels == 0
        pos_mask = ~neg_mask

        # Negative tiles: standard CE, target Class 0
        if neg_mask.sum() > 0:
            neg_loss = F.cross_entropy(outputs[neg_mask], labels[neg_mask])
        else:
            neg_loss = torch.tensor(0.0, device=outputs.device)

        # Positive tiles: Negative Rejection
        # L = -log(1 - p_0) where p_0 = P(Class 0 | positive tile)
        if pos_mask.sum() > 0:
            p_0 = probs[pos_mask, 0]  # P(Class 0) for positive tiles
            pos_loss = -torch.log(1 - p_0 + self.epsilon).mean()
        else:
            pos_loss = torch.tensor(0.0, device=outputs.device)

        # Record loss decomposition (for logging)
        self.last_pos_loss = pos_loss.item()
        self.last_neg_loss = neg_loss.item()

        # Total loss (average of positive and negative components)
        total_loss = (pos_loss + neg_loss) / 2.0

        return total_loss

    def soft_bootstrapping_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft Bootstrapping CE for stable phase.

        Target = alpha × OneHot(c) + (1 - alpha) × p_model

        Advantages:
            1. Prevents over-confidence (allows soft distribution)
            2. Promotes heatmap region activation (not single-point)
            3. Self-adaptive learning (model's prediction influences target)

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels

        Returns:
            Loss value
        """
        num_classes = outputs.size(1)

        # Compute soft target
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)  # Model's current prediction
            one_hot = F.one_hot(labels, num_classes=num_classes).float()  # Hard label

            # Soft target: 90% hard label + 10% model prediction
            soft_target = self.alpha * one_hot + (1 - self.alpha) * probs

        # CE with soft target (KL divergence)
        log_probs = F.log_softmax(outputs, dim=1)
        per_sample_loss = -torch.sum(soft_target * log_probs, dim=1)

        # Calculate loss decomposition for logging (separate pos/neg)
        neg_mask = labels == 0
        pos_mask = ~neg_mask

        if neg_mask.sum() > 0:
            self.last_neg_loss = per_sample_loss[neg_mask].mean().item()
        else:
            self.last_neg_loss = 0.0

        if pos_mask.sum() > 0:
            self.last_pos_loss = per_sample_loss[pos_mask].mean().item()
        else:
            self.last_pos_loss = 0.0

        loss = per_sample_loss.mean()

        return loss


__all__ = ['AsymmetricMILLoss']
