"""
Checkpoint Manager for model training with resume support.

Recent Updates:
    - [2025-12-18] Removed: Auto-cleanup logic (user request, was deleting best models)
    - [2025-01-19] New: Lightweight checkpoint manager for MIL training
    - [2025-01-19] Feature: Save/load model+optimizer+scheduler+scaler states

Key Features:
    - Save complete training state (model, optimizer, scheduler, scaler, config)
    - Load checkpoint for resume training
    - Config versioning: Save config.yaml with each checkpoint
    - Manual cleanup: All checkpoints preserved, user manages disk space

Usage:
    >>> from rice_detection.core.checkpoint_manager import CheckpointManager
    >>>
    >>> # Initialize
    >>> ckpt_manager = CheckpointManager('checkpoints/MIL_training')
    >>>
    >>> # Save checkpoint
    >>> ckpt_manager.save_checkpoint(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     scaler=scaler,
    ...     epoch=10,
    ...     metrics={'val_loss': 0.5, 'val_acc': 0.9},
    ...     config=config,
    ...     filename='best_model.pth'
    ... )
    >>>
    >>> # Load checkpoint
    >>> start_epoch, metrics = ckpt_manager.load_checkpoint(
    ...     checkpoint_path='checkpoints/MIL_training/best_model.pth',
    ...     model=model,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     scaler=scaler
    ... )

Configuration:
    - checkpoint_root: Root directory for checkpoints
    - keep_last_n: (Deprecated, no longer used) Previously controlled auto-cleanup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import shutil


class CheckpointManager:
    """
    Lightweight checkpoint manager for training with resume support.

    Handles saving/loading complete training state:
    - Model weights
    - Optimizer state
    - Scheduler state (if provided)
    - AMP scaler state (if provided)
    - Training metrics
    - Configuration
    """

    def __init__(self, checkpoint_root: str, keep_last_n: int = 3):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_root: Root directory for checkpoints
            keep_last_n: (Deprecated, no longer used) Previously controlled auto-cleanup
        """
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n  # Kept for backward compatibility, but not used

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        training_state: Optional[Dict[str, Any]] = None,
        filename: str = 'checkpoint.pth'
    ):
        """
        Save complete training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            metrics: Training metrics dict (e.g., {'val_loss': 0.5, 'val_acc': 0.9})
            config: Configuration dict
            scheduler: LR scheduler (optional)
            scaler: AMP scaler (optional)
            training_state: TrainingState state_dict (optional, for resume)
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_root / filename

        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }

        # Add optional states
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()

        if training_state is not None:
            checkpoint_data['training_state'] = training_state

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Save config separately for easy reference
        config_path = checkpoint_path.parent / f"{checkpoint_path.stem}_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        print(f"[CHECKPOINT] Saved checkpoint to {checkpoint_path}")

        # NEW (2025-12-18): Removed auto-cleanup logic (user request)
        # Reason: Auto-cleanup deleted best model checkpoints with names like
        #         'bestmodel_epoch_{N}_acc{val}_lift{val}.pth' since they weren't
        #         protected by the hardcoded 'best_model.pth' filename check.
        # User preference: Keep all checkpoints, let user manually manage disk space.

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        strict: bool = True
    ) -> Tuple[int, Dict[str, float]]:
        """
        Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (None for inference)
            scheduler: Scheduler to load state into (optional)
            scaler: Scaler to load state into (optional)
            strict: Strictly enforce state_dict keys match

        Returns:
            Tuple of (start_epoch, metrics)
            - start_epoch: Next epoch to continue from (epoch + 1)
            - metrics: Training metrics from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"[CHECKPOINT] Loaded model weights from {checkpoint_path}")

        # Load optimizer state (if provided and available)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"[CHECKPOINT] Loaded optimizer state")

        # Load scheduler state (if provided and available)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"[CHECKPOINT] Loaded scheduler state")

        # Load scaler state (if provided and available)
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"[CHECKPOINT] Loaded AMP scaler state")

        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        start_epoch = epoch + 1  # Continue from next epoch

        print(f"[CHECKPOINT] Resuming from epoch {start_epoch}, metrics: {metrics}")

        return start_epoch, metrics

    def _cleanup_old_checkpoints(self, keep_files: set):
        """
        Clean up old checkpoints, keeping only the most recent N.

        Args:
            keep_files: Set of filenames to always keep
        """
        # Get all checkpoint files
        checkpoint_files = sorted(
            [f for f in self.checkpoint_root.glob('*.pth') if f.name not in keep_files],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # Remove old checkpoints
        for old_file in checkpoint_files[self.keep_last_n:]:
            old_file.unlink()
            # Also remove associated config file
            config_file = old_file.parent / f"{old_file.stem}_config.yaml"
            if config_file.exists():
                config_file.unlink()
            print(f"[CHECKPOINT] Removed old checkpoint: {old_file.name}")


__all__ = ['CheckpointManager']
