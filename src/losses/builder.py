"""
Loss Function Factory for config-driven instantiation.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from losses.py with complete config mapping
    - [2026-01-05] Fix: Added missing TopKAnchoredMILLoss params (noise_drop_threshold, etc.)

Supported Loss Types:
    - 'focal': FocalLoss
    - 'label_smoothing_ce': LabelSmoothingCrossEntropy
    - 'combined': CombinedLoss (Focal + Label Smoothing)
    - 'topk_anchored_mil': TopKAnchoredMILLoss (Recommended for MIL)
    - 'asymmetric_mil': AsymmetricMILLoss (Ablation baseline)
    - 'bce_with_logits': BCEWithLogitsLossMultiLabel
    - 'focal_bce': FocalBCEWithLogitsLoss
    - 'ce': Standard CrossEntropyLoss (or LabelSmoothingCE if smoothing > 0)

Usage:
    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.load('configs/algorithm/asymmetric_default.yaml')
    >>> criterion = create_loss_function(config.loss)
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from src.losses.components.focal import FocalLoss
from src.losses.components.smoothing import LabelSmoothingCrossEntropy
from src.losses.components.combined import (
    CombinedLoss,
    BCEWithLogitsLossMultiLabel,
    FocalBCEWithLogitsLoss
)
from src.losses.topk_anchored import TopKAnchoredMILLoss
from src.losses.asymmetric import AsymmetricMILLoss


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        config: Loss configuration dict with structure:
            {
                'type': 'focal' | 'label_smoothing_ce' | 'combined' | 'ce' |
                        'bce_with_logits' | 'focal_bce' | 'topk_anchored_mil' | 'asymmetric_mil',
                # Type-specific parameters (see examples below)
            }

    Returns:
        Loss function module

    Examples:
        >>> # FocalLoss
        >>> config = {'type': 'focal', 'focal_alpha': 1.0, 'focal_gamma': 2.0}
        >>> criterion = create_loss_function(config)

        >>> # TopKAnchoredMILLoss (from YAML config)
        >>> config = OmegaConf.load('configs/algorithm/asymmetric_default.yaml').loss
        >>> criterion = create_loss_function(config)
    """
    # Support both 'type' and legacy 'loss_fn' keys
    loss_type = config.get('type', config.get('loss_fn', 'ce'))

    if loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('focal_alpha', 1.0),
            gamma=config.get('focal_gamma', 2.0)
        )

    elif loss_type == 'label_smoothing_ce':
        return LabelSmoothingCrossEntropy(
            smoothing=config.get('label_smoothing', 0.1)
        )

    elif loss_type == 'combined':
        return CombinedLoss(
            focal_alpha=config.get('focal_alpha', 1.0),
            focal_gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            focal_weight=config.get('focal_weight', 0.5),
            ls_weight=config.get('ls_weight', 0.5)
        )

    elif loss_type == 'bce_with_logits':
        # Multi-label BCE loss (for MIL)
        return BCEWithLogitsLossMultiLabel(
            pos_weight=config.get('pos_weight', None)
        )

    elif loss_type == 'focal_bce':
        # Focal BCE loss (recommended for MIL with class imbalance)
        return FocalBCEWithLogitsLoss(
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            pos_weight=config.get('pos_weight', None)
        )

    elif loss_type == 'topk_anchored_mil':
        # Top-K Anchored MIL Loss with dynamic weighting and ranking constraint
        # Warm-up: Top-1 CE anchor + Top-2~K Negative Rejection
        # Stable: Top-1 CE anchor + Top-2~K Progressive Soft Bootstrapping with Stricter Gate

        # ================================================================
        # [2026-02-05] CONFIG LOADING LOG - Prevent silent default fallback
        # ================================================================
        print("=" * 60)
        print("[LOSS-BUILDER] Creating TopKAnchoredMILLoss")
        print("=" * 60)

        # Base parameters
        top1_ce_weight = config.get('top1_ce_weight', 5.0)
        top2k_nr_weight = config.get('top2k_nr_weight', 0.3)
        top2k_soft_weight = config.get('top2k_soft_weight', 0.1)
        noise_drop_threshold = config.get('noise_drop_threshold', 0.6)
        stable_nr_weight = config.get('stable_nr_weight', 0.3)
        alpha = config.get('alpha', 0.95)
        epsilon = config.get('epsilon', 1e-8)

        print(f"[LOSS] Base Config:")
        print(f"  top1_ce_weight = {top1_ce_weight}")
        print(f"  top2k_nr_weight = {top2k_nr_weight}")
        print(f"  top2k_soft_weight = {top2k_soft_weight}")
        print(f"  noise_drop_threshold = {noise_drop_threshold}")
        print(f"  stable_nr_weight = {stable_nr_weight}")
        print(f"  alpha = {alpha}, epsilon = {epsilon}")

        # Read dynamic weight config
        dynamic_cfg = config.get('dynamic_weight', {})
        enable_dynamic = dynamic_cfg.get('enable', False)
        warmup_ce_weight = dynamic_cfg.get('warmup_weight', 0.1)
        stable_ce_weight = dynamic_cfg.get('stable_weight', 1.0)
        growth_epochs = dynamic_cfg.get('growth_epochs', 10)

        print(f"[LOSS] Dynamic Weight Config:")
        print(f"  enable = {enable_dynamic}")
        print(f"  warmup_weight = {warmup_ce_weight}")
        print(f"  stable_weight = {stable_ce_weight}")
        print(f"  growth_epochs = {growth_epochs}")

        # Read ranking config
        ranking_cfg = config.get('ranking', {})
        enable_ranking = ranking_cfg.get('enable', False)
        inter_weight = ranking_cfg.get('inter_weight', 1.0)
        margin = ranking_cfg.get('margin', 0.2)
        correction_margin = ranking_cfg.get('correction_margin', 0.4)
        correction_weight = ranking_cfg.get('correction_weight', 2.0)

        print(f"[LOSS] Ranking Config:")
        print(f"  enable = {enable_ranking}")
        print(f"  inter_weight = {inter_weight}, margin = {margin}")
        print(f"  correction_margin = {correction_margin}, correction_weight = {correction_weight}")

        # Read stable gate config
        stable_gate_cfg = config.get('stable_gate', {})
        stable_gate_enable = stable_gate_cfg.get('enable', False)
        stable_gate_conf = stable_gate_cfg.get('confidence_threshold', 0.5)
        stable_gate_correct = stable_gate_cfg.get('require_correct', True) if stable_gate_enable else False

        print(f"[LOSS] Stable Gate Config:")
        print(f"  enable = {stable_gate_enable}")
        print(f"  confidence_threshold = {stable_gate_conf}")
        print(f"  require_correct = {stable_gate_correct}")

        # Read Focal Loss config
        focal_cfg = config.get('focal_loss', {})
        enable_focal_loss = focal_cfg.get('enable', False)
        focal_alpha = focal_cfg.get('alpha', 1.0)
        focal_gamma = focal_cfg.get('gamma', 2.0)

        print(f"[LOSS] Focal Loss Config:")
        print(f"  enable = {enable_focal_loss}")
        print(f"  alpha = {focal_alpha}, gamma = {focal_gamma}")

        # [2026-02-02] Anti-Collapse config
        anti_collapse_cfg = config.get('anti_collapse', {})
        tier2_top2k_strategy = anti_collapse_cfg.get('tier2_top2k_strategy', 'weak_ce')
        tier2_weak_weight = anti_collapse_cfg.get('tier2_weak_weight', 1.0)  # [2026-02-26] default updated: 0.3→1.0
        tier3_fallback_top1_weight = anti_collapse_cfg.get('tier3_fallback_top1_weight', 0.3)
        tier3_fallback_top2k_weight = anti_collapse_cfg.get('tier3_fallback_top2k_weight', 0.1)
        tier2_top2k_weak_ce_weight = anti_collapse_cfg.get('tier2_top2k_weak_ce_weight', 0.1)
        stable_loss_mode = config.get('stable_loss_mode', 'tiered')

        print(f"[LOSS] Anti-Collapse Config:")
        print(f"  stable_loss_mode = {stable_loss_mode}")
        print(f"  tier2_top2k_strategy = {tier2_top2k_strategy}")
        print(f"  tier2_weak_weight = {tier2_weak_weight}")
        print(f"  tier3_fallback_top1_weight = {tier3_fallback_top1_weight}")
        print(f"  tier3_fallback_top2k_weight = {tier3_fallback_top2k_weight}")
        print(f"  tier2_top2k_weak_ce_weight = {tier2_top2k_weak_ce_weight}")
        print("=" * 60)

        return TopKAnchoredMILLoss(
            # Base parameters (use pre-read values)
            top1_ce_weight=top1_ce_weight,
            top2k_nr_weight=top2k_nr_weight,
            top2k_soft_weight=top2k_soft_weight,
            noise_drop_threshold=noise_drop_threshold,
            correction_margin=correction_margin,
            correction_weight=correction_weight,
            stable_nr_weight=stable_nr_weight,
            alpha=alpha,
            epsilon=epsilon,
            # Dynamic weighting
            enable_dynamic_weight=enable_dynamic,
            warmup_ce_weight=warmup_ce_weight,
            stable_ce_weight=stable_ce_weight,
            growth_epochs=growth_epochs,
            # Ranking constraint
            enable_ranking=enable_ranking,
            inter_weight=inter_weight,
            margin=margin,
            # Stricter gate
            stable_gate_conf=stable_gate_conf,
            stable_gate_correct=stable_gate_correct,
            # Focal Loss
            enable_focal_loss=enable_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            # [2026-02-02] Anti-Collapse
            tier2_top2k_strategy=tier2_top2k_strategy,
            tier2_weak_weight=tier2_weak_weight,
            tier3_fallback_top1_weight=tier3_fallback_top1_weight,
            tier3_fallback_top2k_weight=tier3_fallback_top2k_weight,
            tier2_top2k_weak_ce_weight=tier2_top2k_weak_ce_weight,
            # [2026-02-24] Ablation C
            stable_loss_mode=stable_loss_mode,
        )

    elif loss_type == 'asymmetric_mil':
        # Asymmetric MIL Loss (phase-aware)
        # Warm-up: Negative Rejection Loss
        # Stable: Soft Bootstrapping CE
        return AsymmetricMILLoss(
            alpha=config.get('alpha', 0.9),  # Soft bootstrapping weight
            epsilon=config.get('epsilon', 1e-8)
        )

    elif loss_type == 'ce':
        # Standard cross entropy
        label_smoothing = config.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss()

    else:
        raise ValueError(
            f"Unknown loss function: {loss_type}. "
            f"Supported: focal, label_smoothing_ce, combined, ce, "
            f"bce_with_logits, focal_bce, topk_anchored_mil, asymmetric_mil"
        )


__all__ = ['create_loss_function']
