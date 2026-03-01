"""
Optimizer and Scheduler Builder for Asymmetric MIL Training.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from train_topk_asymmetric.py (no logic changes)

Key Features:
    - Differential Learning Rates (backbone vs hybrid vs classifier)
    - Weight Decay Decoupling (bias/BN parameters excluded)
    - Trapezoidal LR Schedule (warmup → hold → decay)

Usage:
    >>> from src.utils.builder import create_optimizer, create_scheduler
    >>> optimizer = create_optimizer(model, config)
    >>> scheduler = create_scheduler(optimizer, config)
    >>> # Training loop
    >>> for epoch in range(1, num_epochs + 1):
    ...     train_one_epoch(...)
    ...     # Manual scheduler stepping (handles trapezoidal phases)
    ...     if isinstance(scheduler, dict):  # Trapezoidal/Manual Sequential
    ...         if epoch <= scheduler['warmup_epochs']:
    ...             scheduler['warmup_scheduler'].step()
    ...         elif 'hold_epochs' in scheduler and epoch <= scheduler['warmup_epochs'] + scheduler['hold_epochs']:
    ...             scheduler['hold_scheduler'].step()
    ...         else:
    ...             scheduler['cosine_scheduler'].step()
    ...     else:  # Standard scheduler
    ...         scheduler.step()

Configuration:
    # configs/trainer/default_schedule.yaml
    optimizer:
      name: "adamw"
      backbone_lr: 1.0e-5      # Pretrained backbone
      classifier_lr: 5.0e-4    # New classifier head
      weight_decay: 1.0e-2     # L2 regularization

    scheduler:
      name: "trapezoidal"
      warmup_epochs: 5
      hold_epochs: 5
      min_lr: 5.0e-7
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR
from typing import Dict, Union


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """
    Create optimizer with discriminative learning rates and parameter grouping.

    Implements:
        1. Differential Learning Rates: Backbone (1e-5) vs Hybrid (2e-4) vs Classifier (5e-4)
        2. Weight Decay Decoupling: Bias/BN parameters do NOT decay

    Rationale:
        - Pretrained backbone needs small LR to preserve learned features
        - Hybrid components (FPN/ViT) are randomly initialized, need intermediate LR
        - New classifier head needs larger LR to converge quickly
        - Bias/BN parameters control distribution statistics, not feature patterns
          → Should not be regularized by L2 penalty (weight_decay)

    Args:
        model: Model to optimize
        config: Configuration dict with structure:
            {
                'optimizer': {
                    'name': 'adamw',
                    'backbone_lr': 1e-5,
                    'classifier_lr': 5e-4,
                    'weight_decay': 1e-2,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8
                }
            }

    Returns:
        Optimizer instance with parameter groups

    Example:
        >>> optimizer = create_optimizer(model, config)
        >>> # Parameter groups: [backbone_decay, backbone_no_decay, classifier_decay, classifier_no_decay]
    """
    opt_cfg = config.get('optimizer', {})
    opt_name = opt_cfg.get('name', 'adamw').lower()

    # Differential learning rates (YAML: optimizer.backbone_lr, optimizer.classifier_lr, optimizer.hybrid_lr)
    backbone_lr = opt_cfg.get('backbone_lr', 1e-5)
    classifier_lr = opt_cfg.get('classifier_lr', 5e-5)
    hybrid_lr = opt_cfg.get('hybrid_lr', classifier_lr)  # fallback to classifier_lr
    weight_decay = opt_cfg.get('weight_decay', 1e-2)
    betas = opt_cfg.get('betas', [0.9, 0.999])
    eps = opt_cfg.get('eps', 1e-8)

    # [2026-02-05] CONFIG LOADING LOG
    print("=" * 60)
    print("[OPTIMIZER-BUILDER] Creating Optimizer")
    print("=" * 60)
    print(f"[OPTIMIZER] Config Loaded:")
    print(f"  name = {opt_name}")
    print(f"  backbone_lr = {backbone_lr}")
    print(f"  classifier_lr = {classifier_lr}")
    print(f"  hybrid_lr = {hybrid_lr}")
    print(f"  weight_decay = {weight_decay}")
    print(f"  betas = {betas}")
    print(f"  eps = {eps}")

    # Helper function: Split parameters into decay/no_decay groups
    def split_params(module, lr):
        """
        Split module parameters into weight_decay and no_decay groups.

        Args:
            module: nn.Module
            lr: Learning rate for this module

        Returns:
            List of 2 parameter groups: [decay_group, no_decay_group]
        """
        decay = []
        no_decay = []

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            # Core principle: Bias and BatchNorm/LayerNorm should NOT decay
            # - Bias: Affects output offset, not feature pattern
            # - BN/LN: Controls distribution statistics
            # Shape check: 1D tensors are typically bias/BN/LN parameters
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0.0}
        ]

    # Hybrid component names (randomly initialized, need higher LR than pretrained backbone)
    hybrid_module_names = ('fpn_neck', 'vit_block')

    # Build parameter groups by identifying backbone vs hybrid vs classifier
    param_groups = []

    for name, module in model.named_children():
        # Classifier modules: Use largest learning rate
        if 'classifier' in name or 'head' in name or 'fc' in name:
            param_groups.extend(split_params(module, classifier_lr))
            print(f"[OPTIMIZER] Classifier module '{name}': LR={classifier_lr:.2e}")
        # Hybrid modules (FPN/ViT): Use intermediate learning rate
        elif name in hybrid_module_names:
            param_groups.extend(split_params(module, hybrid_lr))
            print(f"[OPTIMIZER] Hybrid module '{name}': LR={hybrid_lr:.2e}")
        # Backbone modules: Use smallest learning rate
        else:
            param_groups.extend(split_params(module, backbone_lr))
            print(f"[OPTIMIZER] Backbone module '{name}': LR={backbone_lr:.2e}")

    # Create optimizer
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=betas,
            eps=eps
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            eps=opt_cfg.get('eps', 1e-8)
        )
    elif opt_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=opt_cfg.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Print summary
    total_params = sum(len(g['params']) for g in param_groups)
    print(f"[OPTIMIZER] Total parameter groups: {len(param_groups)}")
    print(f"[OPTIMIZER] Total parameters: {total_params}")
    print(f"[OPTIMIZER] Backbone LR: {backbone_lr:.2e}, Hybrid LR: {hybrid_lr:.2e}, Classifier LR: {classifier_lr:.2e}")
    print(f"[OPTIMIZER] LR Ratio (Classifier/Backbone): {classifier_lr/backbone_lr:.0f}:1")
    print(f"[OPTIMIZER] LR Ratio (Hybrid/Backbone): {hybrid_lr/backbone_lr:.0f}:1")

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict
) -> Union[Dict, optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler with Trapezoidal or Cosine strategy.

    Strategy Options:
        1. Trapezoidal (Recommended for Warmup Phase):
            Phase 1 (Epoch 1-5): Linear Warmup (0.01×lr → lr)
            Phase 2 (Epoch 6-10): Constant Hold (lr stays at peak, gives warmup energy)
            Phase 3 (Epoch 11-30): Cosine Decay (lr → min_lr)

        2. Cosine (Original):
            Phase 1 (Epoch 1-5): Linear Warmup (0.01×lr → lr)
            Phase 2 (Epoch 6-30): Cosine Decay (lr → min_lr)

    Args:
        optimizer: Optimizer instance with parameter groups
        config: Configuration dict with structure:
            {
                'scheduler': {
                    'name': 'trapezoidal',  # or 'cosine', 'step', 'plateau'
                    'warmup_epochs': 5,
                    'hold_epochs': 5,       # Only for trapezoidal
                    'min_lr': 5e-7
                },
                'training': {
                    'num_epochs': 30
                }
            }

    Returns:
        For trapezoidal/cosine: Dict with scheduler components (for manual switching)
            {
                'type': 'trapezoidal',
                'warmup_scheduler': LinearLR,
                'hold_scheduler': ConstantLR,  # Only for trapezoidal
                'cosine_scheduler': CosineAnnealingLR,
                'warmup_epochs': int,
                'hold_epochs': int,  # Only for trapezoidal
                'current_phase': 'warmup'
            }
        For other types: Single scheduler instance

    Example:
        >>> scheduler = create_scheduler(optimizer, config)
        >>> # Manual stepping (in training loop)
        >>> if isinstance(scheduler, dict):
        ...     if epoch <= scheduler['warmup_epochs']:
        ...         scheduler['warmup_scheduler'].step()
        ...     elif epoch <= scheduler['warmup_epochs'] + scheduler.get('hold_epochs', 0):
        ...         scheduler['hold_scheduler'].step()
        ...     else:
        ...         scheduler['cosine_scheduler'].step()
        ... else:
        ...     scheduler.step()
    """
    sched_cfg = config.get('scheduler', {})
    sched_name = sched_cfg.get('name', 'cosine').lower()
    num_epochs = config.get('training', {}).get('num_epochs', 30)

    # [2026-02-05] CONFIG LOADING LOG
    print("=" * 60)
    print("[SCHEDULER-BUILDER] Creating Scheduler")
    print("=" * 60)
    print(f"[SCHEDULER] Config Loaded:")
    print(f"  name = {sched_name}")
    print(f"  warmup_epochs = {sched_cfg.get('warmup_epochs', 5)}")
    print(f"  hold_epochs = {sched_cfg.get('hold_epochs', 0)}")
    print(f"  min_lr = {sched_cfg.get('min_lr', 1e-7)}")
    print(f"  total_epochs = {num_epochs}")

    if sched_name == 'trapezoidal':
        # Use custom TrapezoidalLRScheduler wrapper (encapsulates phase logic)
        from src.utils.scheduler import TrapezoidalLRScheduler

        warmup_epochs = sched_cfg.get('warmup_epochs', 5)
        hold_epochs = sched_cfg.get('hold_epochs', 5)
        min_lr = sched_cfg.get('min_lr', 1e-7)

        scheduler = TrapezoidalLRScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            hold_epochs=hold_epochs,
            total_epochs=num_epochs,
            min_lr=min_lr,
            verbose=True
        )

        print(f"[SCHEDULER] Trapezoidal LR Schedule:")
        print(f"  Phase 1 (Warmup): Epoch 1-{warmup_epochs}")
        print(f"  Phase 2 (Hold):   Epoch {warmup_epochs+1}-{warmup_epochs+hold_epochs}")
        print(f"  Phase 3 (Decay):  Epoch {warmup_epochs+hold_epochs+1}-{num_epochs} → {min_lr:.2e}")

    elif sched_name == 'cosine':
        # Use custom ManualSequentialScheduler wrapper
        from src.utils.scheduler import ManualSequentialScheduler

        warmup_epochs = sched_cfg.get('warmup_epochs', 5)
        min_lr = sched_cfg.get('min_lr', 1e-7)

        scheduler = ManualSequentialScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            min_lr=min_lr,
            verbose=True
        )

        print(f"[SCHEDULER] Warmup+Cosine Schedule:")
        print(f"  Phase 1 (Warmup): Epoch 1-{warmup_epochs}")
        print(f"  Phase 2 (Cosine): Epoch {warmup_epochs+1}-{num_epochs} → {min_lr:.2e}")
        print(f"[SCHEDULER] Using manual switching (avoids SequentialLR deprecation warning)")

    elif sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get('step_size', 10),
            gamma=sched_cfg.get('gamma', 0.1)
        )
        print(f"[SCHEDULER] StepLR: step_size={sched_cfg.get('step_size', 10)}")

    elif sched_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=sched_cfg.get('patience', 5),
            factor=sched_cfg.get('factor', 0.5)
        )
        print(f"[SCHEDULER] ReduceLROnPlateau: patience={sched_cfg.get('patience', 5)}")

    else:
        scheduler = None
        print(f"[SCHEDULER] No scheduler enabled")

    return scheduler


__all__ = ['create_optimizer', 'create_scheduler']
