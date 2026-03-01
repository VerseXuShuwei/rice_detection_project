"""
Core module for rice detection system.

This module contains base classes and core functionality that other modules depend on.

Components:
    - BaseModel: Abstract base class for detection models
    - BaseTrainer: Abstract base class for training logic, manages training loop and checkpoints
    - ExperimentTracker: Experiment management, tracks configs/metrics/artifacts for analysis
    - Config: Global configuration management (deprecated, use config/base.py)

Division of Responsibilities (BaseTrainer vs ExperimentTracker):

    BaseTrainer:
        - Training execution (train loop, validation, backprop)
        - Checkpoint management (model + optimizer + scheduler states for resume training)
        - Saves to: checkpoints/ablation_study/{experiment_name}/
        - Purpose: Enable resume training with full training state

    ExperimentTracker:
        - Experiment lifecycle tracking (start/end time, duration)
        - Metrics logging (config, per-epoch metrics with timestamps)
        - Artifact management (plots, confusion matrices, feature maps)
        - Saves to: results/ablation_study/{experiment_name}/
        - Purpose: Post-training analysis, experiment comparison

    Typical Usage:
        # Training with both components
        tracker = ExperimentTracker('exp_baseline_v2')
        tracker.log_config(config)

        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train()  # Saves checkpoint to checkpoints/

        tracker.log_metrics({'final_acc': history['val_acc'][-1]})
        tracker.save_artifact(plot, 'training_curves', 'png')
        tracker.finalize()  # Saves summary to results/
"""

from .base_model import BaseModel
from .base_trainer import BaseTrainer
from .checkpoint_manager import CheckpointManager

__all__ = [
    'BaseModel',
    'BaseTrainer',
    'CheckpointManager'
]
