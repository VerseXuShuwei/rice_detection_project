"""
Lightweight Callback System for Trainer.

Recent Updates:
    - [2026-01-13] Feature: Created callback system for modular training hooks

Key Features:
    - Base Callback class with lifecycle hooks
    - CheckpointCallback: Handle checkpoint saving logic
    - DriftMonitorCallback: Concept drift monitoring in stable phase
    - HeatmapCallback: Periodic heatmap generation

Usage:
    >>> from src.trainer.callbacks import CheckpointCallback, HeatmapCallback
    >>> trainer = AsymmetricMILTrainer(config)
    >>> trainer.callbacks = CallbackList([
    >>>     CheckpointCallback(),
    >>>     HeatmapCallback(frequency=10)
    >>> ])
    >>> trainer.fit()
"""

from typing import Dict, Any
from pathlib import Path


class Callback:
    """
    Base Callback class for training lifecycle hooks.

    Override methods to inject custom logic at specific training stages.
    """

    def on_train_start(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_epoch_start(self, trainer, epoch: int):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass


class CheckpointCallback(Callback):
    """
    Handle checkpoint saving logic.

    Saves best model and regular checkpoints based on configured intervals.
    """

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Save checkpoints if needed."""
        # Regular checkpoint
        save_every = trainer.train_cfg.get('save_every', 5)
        if epoch % save_every == 0:
            trainer._save_regular_checkpoint(epoch, metrics)

        # Best model checkpoint (Stable phase only)
        if not trainer.is_warmup and hasattr(trainer, 'topk_metrics'):
            trainer._save_best_checkpoint_if_improved(epoch, trainer.topk_metrics, metrics)


class DriftMonitorCallback(Callback):
    """
    Monitor concept drift in stable phase.

    Tracks validation accuracy changes and warns if significant drift detected.
    """

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Check for concept drift."""
        if not trainer.is_warmup:
            val_acc = metrics.get('val_acc', 0.0)
            trainer._monitor_concept_drift(val_acc)


class HeatmapCallback(Callback):
    """
    Generate heatmaps periodically during training.

    CRITICAL: Uses VALIDATION dataset to ensure:
        1. Same images across all epochs (fixed_samples=True)
        2. Images not used in training (unbiased visualization)
        3. Enables tracking model learning progress on consistent samples

    Fixed Sample Mode (default):
        - First call: Randomly select samples from VAL set and save to reference_samples.json
        - Subsequent calls: Load existing samples to track learning progress on same images

    Args:
        frequency: Generate heatmaps every N epochs
        phases: List of phases to generate heatmaps ('warmup_end', 'stable')
        fixed_samples: If True, reuse same samples across epochs (default: True)
        samples_per_class: Number of samples per class (default: 1)
        use_spatial_heatmap: If True, generate full spatial heatmaps (default: True)
    """

    def __init__(
        self,
        frequency: int = 10,
        phases: list = None,
        fixed_samples: bool = True,
        samples_per_class: int = 1,
        use_spatial_heatmap: bool = True
    ):
        self.frequency = frequency
        self.phases = phases or ['stable']
        self.fixed_samples = fixed_samples
        self.samples_per_class = samples_per_class
        self.use_spatial_heatmap = use_spatial_heatmap

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Generate heatmaps if conditions met."""
        # Stable phase periodic generation
        # Skip final epoch: FinalEvaluator generates heatmaps after training ends
        num_epochs = trainer.config.get('training', {}).get('num_epochs', 30)
        if not trainer.is_warmup and epoch % self.frequency == 0 and epoch != num_epochs:
            if 'stable' in self.phases:
                print(f"\n{'='*60}")
                print(f"[HEATMAP-VIS] Saving representative heatmaps (Epoch {epoch})")
                print(f"[HEATMAP-VIS] Using VALIDATION dataset for consistent tracking")
                print(f"{'='*60}")

                # CRITICAL: Use val_dataset, not train_dataset
                # This ensures:
                # 1. Same images across epochs (not affected by training shuffle)
                # 2. Unbiased visualization (images not seen during training)
                val_dataset = getattr(trainer, 'val_dataset', None)
                if val_dataset is None:
                    print("[HEATMAP-VIS] Warning: val_dataset not found, using train_dataset")
                    val_dataset = trainer.train_dataset

                # NOTE: Heatmap settings are configured at MILVisualizer construction time
                # in TrainerBuilder (from config). No config passed here - follows
                # "Single Source of Truth" principle.
                trainer.visualizer.generate_monitoring_heatmaps(
                    trainer.model,
                    val_dataset,
                    epoch,
                    phase='stable',
                    samples_per_class=self.samples_per_class,
                    fixed_samples=self.fixed_samples,
                    use_spatial_heatmap=self.use_spatial_heatmap
                )


class CallbackList:
    """
    Container for managing multiple callbacks.

    Executes callbacks in order for each lifecycle hook.
    """

    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_start(self, trainer):
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_epoch_start(self, trainer, epoch: int):
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)


__all__ = [
    'Callback',
    'CheckpointCallback',
    'DriftMonitorCallback',
    'HeatmapCallback',
    'CallbackList'
]
