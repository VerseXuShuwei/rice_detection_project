"""
Base trainer class for model training with complete checkpoint support.

This module provides the abstract base class for all training strategies.
Supports resume training from checkpoints with full state restoration.

Architecture:
    - BaseTrainer: Abstract base class with checkpoint management
    - Subclasses (Trainer): Implement train_epoch(), validate_epoch(), and checkpoint methods

Checkpoint System Design:
    1. BaseTrainer provides:
       - train() method with resume_from parameter
       - load_checkpoint() for loading model + basic training state
       - save_checkpoint() for saving complete checkpoints
       - _prepare_checkpoint_data() template method (for subclasses to override)

    2. Subclasses (Trainer) provide:
       - _prepare_checkpoint_data() override to include optimizer/scheduler/scaler
       - load_checkpoint() override to restore optimizer/scheduler/scaler

Resume Training Usage:
    # From scratch (with experiment name)
    config = {
        'experiment_name': 'exp_baseline_v2',  # Checkpoints saved to checkpoints/ablation_study/exp_baseline_v2/
        'epochs': 100,
        'lr': 0.001,
        ...
    }
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train()
    # → Checkpoint saved to: checkpoints/ablation_study/exp_baseline_v2/best_model.pth

    # Resume from checkpoint
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train(resume_from='checkpoints/ablation_study/exp_baseline_v2/best_model.pth')

    # Custom checkpoint directory (override default)
    config = {
        'save_dir': 'custom/path/my_experiment',  # Explicit checkpoint path
        ...
    }
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train()

Checkpoint Contents:
    - epoch: Current epoch number
    - model_state_dict: Model weights
    - optimizer_state_dict: Optimizer state (momentum, etc.)
    - scheduler_state_dict: LR scheduler state (if enabled)
    - scaler_state_dict: AMP scaler state (if using mixed precision)
    - best_metric: Best validation metric
    - history: Complete training history (train_loss, val_loss, etc.)
    - train_loss_history: Training loss history (optional, for plotting)
    - val_loss_history: Validation loss history (optional, for plotting)

Design Principles:
    - Template Method Pattern: BaseTrainer defines skeleton, subclasses fill in details
    - Open/Closed Principle: BaseTrainer is closed for modification, open for extension
    - Single Responsibility: BaseTrainer handles training loop, subclasses handle training logic
    - DRY: Checkpoint logic centralized in BaseTrainer, no duplication in subclasses
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.
    
    Provides common training functionality and defines the interface
    that all trainer implementations should follow.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Training parameters
        self.epochs = self.config.get('epochs', 100)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Default checkpoint directory: checkpoints/ablation_study/{experiment_name}
        # This is for saving training checkpoints (model weights + optimizer state for resume training)
        # For final results/metrics, use results/ablation_study/{experiment_name} in your training script
        experiment_name = self.config.get('experiment_name', 'default_experiment')
        default_checkpoint_dir = Path('checkpoints/ablation_study') / experiment_name
        self.save_dir = Path(self.config.get('save_dir', default_checkpoint_dir))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics for the epoch
        """
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics for the epoch
        """
        pass
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop with optional resume support.

        Args:
            resume_from: Path to checkpoint to resume from (optional)
                        If provided, training continues from saved epoch

        Returns:
            Training history dictionary

        Design:
            - Supports resume training by loading checkpoint and continuing from saved epoch
            - Subclasses should override load_checkpoint() to restore optimizer/scheduler/scaler
        """
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch, _ = self.load_checkpoint(resume_from, load_training_state=True)
            print(f"Resuming from epoch {start_epoch}")

        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}

            # Update history
            self._update_history(train_metrics, val_metrics)

            # Print progress
            self._print_epoch_summary(train_metrics, val_metrics)

            # Save checkpoint
            self._save_checkpoint(val_metrics)

            # 清理内存（防止内存泄漏）
            # if epoch % 10 ==0:
            self._cleanup_memory()

        return self.history
    
    def _cleanup_memory(self):
        """
        Clean up memory to prevent epoch-to-epoch memory leaks.

        Called after every epoch to release cached gradients, intermediate tensors,
        and trigger garbage collection.
        """
        import gc

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete

        # Force garbage collection (aggressive)
        gc.collect(generation=2)  # Collect all generations including old objects

        # Additional: Clear Python's internal memory pools
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux/Unix only
        except:
            pass  # Ignore on Windows
    
    def _update_history(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Update training history."""
        for key, value in train_metrics.items():
            history_key = f'train_{key}'
            if history_key not in self.history:
                self.history[history_key] = []
            self.history[history_key].append(value)
        
        for key, value in val_metrics.items():
            history_key = f'val_{key}'
            if history_key not in self.history:
                self.history[history_key] = []
            self.history[history_key].append(value)
    
    def _print_epoch_summary(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print summary for current epoch."""
        msg = f"Epoch [{self.current_epoch+1}/{self.epochs}] "
        msg += " | ".join([f"train_{k}: {v:.4f}" for k, v in train_metrics.items()])
        
        if val_metrics:
            msg += " | " + " | ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
        
        print(msg)
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        """
        Save model checkpoint if validation metric improves.

        This is a template method that subclasses should override to provide
        complete checkpoint data (optimizer, scheduler, scaler).

        Default behavior: Save basic checkpoint (model + training state).
        Subclasses (Trainer): Override to add optimizer/scheduler/scaler states.
        """
        if not val_metrics:
            return

        current_metric = val_metrics.get('loss', float('inf'))

        if current_metric < self.best_metric:
            self.best_metric = current_metric
            checkpoint_path = self.save_dir / 'best_model.pth'

            # Basic checkpoint data (subclasses should override to extend this)
            checkpoint_data = self._prepare_checkpoint_data()

            torch.save(checkpoint_data, checkpoint_path)

            print(f"Saved best model to {checkpoint_path} "
                  f"(epoch={self.current_epoch}, loss={current_metric:.4f})")

    def _prepare_checkpoint_data(self) -> Dict[str, Any]:
        """
        Prepare checkpoint data to save.

        This method should be overridden by subclasses to include
        optimizer, scheduler, scaler states.

        Returns:
            Dictionary with checkpoint data
        """
        return {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history
        }
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_path: Optional[Path] = None):
        """
        Save complete training checkpoint (model + optimizer + scheduler + scaler + training state).

        This is a template method that subclasses should call with complete checkpoint data.
        Subclasses are responsible for providing optimizer, scheduler, scaler states.

        Args:
            checkpoint_data: Dictionary containing:
                - 'epoch': Current epoch number
                - 'model_state_dict': Model state dict
                - 'optimizer_state_dict': Optimizer state dict (optional)
                - 'scheduler_state_dict': Scheduler state dict (optional)
                - 'scaler_state_dict': AMP scaler state dict (optional)
                - 'best_metric': Best validation metric
                - 'train_loss_history': Training loss history (optional)
                - 'val_loss_history': Validation loss history (optional)
                - 'history': Full training history (optional)
            checkpoint_path: Path to save checkpoint (default: save_dir / 'checkpoint.pth')

        Design:
            - BaseTrainer provides the save mechanism
            - Subclasses (Trainer) provide the complete checkpoint data
            - Supports resume training from any epoch with full state
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / 'checkpoint.pth'

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle DDP wrapped models (save unwrapped state_dict)
        model_state = checkpoint_data.get('model_state_dict')
        if model_state is None:
            # Fallback: extract from model directly
            if hasattr(self.model, 'module'):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            checkpoint_data['model_state_dict'] = model_state

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_training_state: bool = True,
        strict: bool = True
    ) -> Tuple[int, float]:
        """
        Load checkpoint and restore training state.

        This is a template method that loads basic state. Subclasses should override
        to load additional state (optimizer, scheduler, scaler).

        Args:
            checkpoint_path: Path to checkpoint file
            load_training_state: If True, restore training state (epoch, best_metric, history)
            strict: If True, strictly enforce state_dict keys match

        Returns:
            Tuple of (start_epoch, best_metric)
            - start_epoch: Next epoch to continue training from
            - best_metric: Best validation metric from checkpoint

        Design:
            - BaseTrainer loads model state and basic training state
            - Subclasses (Trainer) should override to load optimizer/scheduler/scaler
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state (handle DDP prefix)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state', None))
        if model_state is None:
            raise KeyError("No model_state_dict found in checkpoint")

        try:
            self.model.load_state_dict(model_state, strict=strict)
        except RuntimeError as e:
            # Try stripping DDP prefix
            try:
                stripped_state = self._strip_ddp_prefix(model_state)
                self.model.load_state_dict(stripped_state, strict=strict)
                print("[INFO] Loaded checkpoint with DDP prefix stripped")
            except Exception:
                raise RuntimeError(f"Failed to load model state: {e}")

        # Load training state
        start_epoch = 0
        best_metric = float('inf')

        if load_training_state:
            start_epoch = checkpoint.get('epoch', -1) + 1  # Continue from next epoch
            best_metric = checkpoint.get('best_metric', float('inf'))
            self.best_metric = best_metric
            self.current_epoch = start_epoch

            # Restore history
            if 'history' in checkpoint:
                self.history = checkpoint['history']

            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', -1)}, "
                  f"best_metric={best_metric:.4f}")
        else:
            print(f"Loaded checkpoint (inference mode)")

        return start_epoch, best_metric

    @staticmethod
    def _strip_ddp_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip 'module.' prefix from DDP wrapped model state_dict.

        Args:
            state_dict: Model state dict (potentially with 'module.' prefix)

        Returns:
            Cleaned state dict without 'module.' prefix
        """
        new_state = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            new_state[new_key] = v
        return new_state
