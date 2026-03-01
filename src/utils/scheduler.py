"""
Custom Learning Rate Schedulers.

Recent Updates:
    - [2026-01-13] Refactor: Extracted from Trainer to decouple scheduler logic

Key Features:
    - TrapezoidalLRScheduler: Warmup → Hold → Cosine Decay with phase auto-detection
    - ManualSequentialScheduler: Warmup → Cosine with simple phase transition
    - Encapsulates internal phase management (Trainer no longer needs to know phases)

Usage:
    >>> from src.utils.scheduler import TrapezoidalLRScheduler
    >>> scheduler = TrapezoidalLRScheduler(optimizer, warmup_epochs=5, hold_epochs=10, ...)
    >>> for epoch in range(1, num_epochs+1):
    >>>     current_lr = scheduler.step(epoch)
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional


class TrapezoidalLRScheduler:
    """
    Trapezoidal Learning Rate Scheduler: Warmup → Hold → Cosine Decay.

    Encapsulates all phase logic internally. Trainer only needs to call step(epoch).

    Phases:
        1. Warmup (epoch 1 to warmup_epochs): Linear growth from 0 to base_lr
        2. Hold (epoch warmup_epochs+1 to warmup_epochs+hold_epochs): Constant base_lr
        3. Cosine Decay (remaining epochs): Cosine annealing to min_lr

    Args:
        optimizer: PyTorch optimizer instance
        warmup_epochs: Number of warmup epochs
        hold_epochs: Number of hold epochs (after warmup)
        total_epochs: Total training epochs
        min_lr: Minimum learning rate for cosine decay phase
        verbose: Print phase transitions

    Example:
        >>> scheduler = TrapezoidalLRScheduler(optimizer, warmup_epochs=5, hold_epochs=10, total_epochs=30)
        >>> for epoch in range(1, 31):
        >>>     lr = scheduler.step(epoch)
        >>>     # epoch 1-5: Warmup
        >>>     # epoch 6-15: Hold
        >>>     # epoch 16-30: Cosine Decay
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        hold_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.verbose = verbose

        # Store base learning rates for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Internal state
        self.current_phase = None
        self.last_epoch = 0

        # Backbone unfreeze warmup state
        # When backbone is unfrozen, these param groups get a mini warmup
        # (linear ramp from start_fraction to 1.0 over N epochs)
        self._backbone_unfreeze_epoch = None  # epoch when unfreeze happened
        self._backbone_unfreeze_duration = 0  # how many epochs to ramp
        self._backbone_unfreeze_start_fraction = 0.1  # starting fraction
        self._backbone_group_indices = []  # which param_group indices are backbone

        # Create sub-schedulers
        self._create_sub_schedulers()

    def _create_sub_schedulers(self):
        """Create internal schedulers for each phase."""
        # NOTE: DO NOT create sub-schedulers.
        # Manual LR update is safer to avoid accumulated step() calls
        self.warmup_scheduler = None
        self.hold_scheduler = None
        self.cosine_scheduler = None

    def step(self, epoch: int) -> float:
        """
        Update learning rate based on current epoch (manual calculation).

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            current_lr: Updated learning rate
        """
        import math

        self.last_epoch = epoch

        # Determine phase and calculate LR manually
        if epoch <= self.warmup_epochs:
            # Phase 1: Warmup (Linear: 0.01 * base_lr → base_lr)
            if self.current_phase != 'warmup':
                if self.verbose:
                    print(f"[SCHEDULER] Phase 1: Warmup (epochs 1-{self.warmup_epochs})")
                self.current_phase = 'warmup'

            # Linear interpolation from 0.01 to 1.0
            warmup_factor = 0.01 + (1.0 - 0.01) * (epoch / self.warmup_epochs)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor

        elif epoch <= self.warmup_epochs + self.hold_epochs:
            # Phase 2: Hold (Constant at base_lr)
            if self.current_phase != 'hold':
                if self.verbose:
                    print(f"[SCHEDULER] Phase 2: Hold at LR={self.base_lrs[0]:.2e} (epochs {self.warmup_epochs+1}-{self.warmup_epochs+self.hold_epochs})")
                self.current_phase = 'hold'

            # Keep base LR
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]

        else:
            # Phase 3: Cosine Decay
            if self.current_phase != 'cosine':
                if self.verbose:
                    print(f"[SCHEDULER] Phase 3: Cosine Decay (epochs {self.warmup_epochs+self.hold_epochs+1}-{self.total_epochs})")
                self.current_phase = 'cosine'

            # Cosine annealing
            cosine_epochs = self.total_epochs - self.warmup_epochs - self.hold_epochs
            current_cosine_epoch = epoch - self.warmup_epochs - self.hold_epochs

            for i, param_group in enumerate(self.optimizer.param_groups):
                # Cosine formula: min_lr + (base_lr - min_lr) * (1 + cos(pi * t / T)) / 2
                cosine_factor = (1 + math.cos(math.pi * current_cosine_epoch / cosine_epochs)) / 2
                param_group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * cosine_factor

        # Apply backbone unfreeze warmup modulation (if active)
        if self._backbone_unfreeze_epoch is not None and self._backbone_unfreeze_duration > 0:
            elapsed = epoch - self._backbone_unfreeze_epoch
            if elapsed < self._backbone_unfreeze_duration:
                # Linear ramp: start_fraction → 1.0 over duration epochs
                t = (elapsed + 1) / self._backbone_unfreeze_duration  # +1 because elapsed starts at 0
                fraction = self._backbone_unfreeze_start_fraction + \
                    (1.0 - self._backbone_unfreeze_start_fraction) * t
                for idx in self._backbone_group_indices:
                    self.optimizer.param_groups[idx]['lr'] *= fraction
                if self.verbose and elapsed == 0:
                    actual_lr = self.optimizer.param_groups[self._backbone_group_indices[0]]['lr'] \
                        if self._backbone_group_indices else 0
                    print(f"[SCHEDULER] Backbone unfreeze warmup: {fraction:.1%} of base_lr "
                          f"(epoch {epoch}, {self._backbone_unfreeze_duration - elapsed - 1} epochs remaining)")
            elif elapsed == self._backbone_unfreeze_duration:
                # Warmup complete, clear state
                if self.verbose:
                    print(f"[SCHEDULER] Backbone unfreeze warmup complete at epoch {epoch}")
                self._backbone_unfreeze_epoch = None

        # Return current LR (from first param group)
        return self.optimizer.param_groups[0]['lr']

    def notify_backbone_unfreeze(
        self,
        epoch: int,
        backbone_group_indices: list,
        warmup_epochs: int = 3,
        start_fraction: float = 0.1
    ):
        """
        Notify scheduler that backbone was just unfrozen.

        Activates a mini warmup that modulates backbone param_groups' LR
        from start_fraction × base_lr to 1.0 × base_lr over warmup_epochs.

        Args:
            epoch: Current epoch when unfreeze happened
            backbone_group_indices: List of optimizer param_group indices for backbone
            warmup_epochs: Number of epochs to ramp backbone LR
            start_fraction: Starting fraction of base_lr (e.g., 0.1 = 10%)
        """
        self._backbone_unfreeze_epoch = epoch
        self._backbone_unfreeze_duration = warmup_epochs
        self._backbone_unfreeze_start_fraction = start_fraction
        self._backbone_group_indices = backbone_group_indices

        print(f"[SCHEDULER] Backbone unfreeze warmup registered:")
        print(f"  Epoch {epoch} → {epoch + warmup_epochs - 1}: "
              f"{start_fraction:.0%} → 100% of backbone_lr")
        print(f"  Affected param_groups: {backbone_group_indices}")

    def state_dict(self) -> Dict[str, Any]:
        """Save scheduler state for checkpointing."""
        return {
            'last_epoch': self.last_epoch,
            'current_phase': self.current_phase,
            'base_lrs': self.base_lrs,
            'backbone_unfreeze_epoch': self._backbone_unfreeze_epoch,
            'backbone_unfreeze_duration': self._backbone_unfreeze_duration,
            'backbone_unfreeze_start_fraction': self._backbone_unfreeze_start_fraction,
            'backbone_group_indices': self._backbone_group_indices
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.last_epoch = state_dict.get('last_epoch', 0)
        self.current_phase = state_dict.get('current_phase', None)
        if 'base_lrs' in state_dict:
            self.base_lrs = state_dict['base_lrs']
        self._backbone_unfreeze_epoch = state_dict.get('backbone_unfreeze_epoch', None)
        self._backbone_unfreeze_duration = state_dict.get('backbone_unfreeze_duration', 0)
        self._backbone_unfreeze_start_fraction = state_dict.get('backbone_unfreeze_start_fraction', 0.1)
        self._backbone_group_indices = state_dict.get('backbone_group_indices', [])


class ManualSequentialScheduler:
    """
    Manual Sequential Scheduler: Warmup → Cosine Decay.

    Simpler version without hold phase.

    Args:
        optimizer: PyTorch optimizer instance
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate for cosine decay
        verbose: Print phase transitions
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.verbose = verbose

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_phase = None
        self.last_epoch = 0

        self._create_sub_schedulers()

    def _create_sub_schedulers(self):
        """Create internal schedulers."""
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )

        cosine_epochs = self.total_epochs - self.warmup_epochs
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=self.min_lr
        )

    def step(self, epoch: int) -> float:
        """Update learning rate."""
        self.last_epoch = epoch

        if epoch <= self.warmup_epochs:
            if self.current_phase != 'warmup':
                if self.verbose:
                    print(f"[SCHEDULER] Phase 1: Warmup (epochs 1-{self.warmup_epochs})")
                self.current_phase = 'warmup'
            self.warmup_scheduler.step()
        else:
            if self.current_phase != 'cosine':
                if self.verbose:
                    print(f"[SCHEDULER] Phase 2: Cosine Decay (epochs {self.warmup_epochs+1}-{self.total_epochs})")
                self.current_phase = 'cosine'
            self.cosine_scheduler.step()

        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict[str, Any]:
        """Save scheduler state."""
        return {
            'last_epoch': self.last_epoch,
            'current_phase': self.current_phase,
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'cosine_scheduler': self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state."""
        self.last_epoch = state_dict['last_epoch']
        self.current_phase = state_dict['current_phase']
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


__all__ = ['TrapezoidalLRScheduler', 'ManualSequentialScheduler']
