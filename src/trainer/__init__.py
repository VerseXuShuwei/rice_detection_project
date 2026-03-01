"""
Trainer Module for Asymmetric MIL.

Recent Updates:
    - [2026-01-15] Major Refactor: God Class elimination
        - Added TrainerBuilder (component construction)
        - Added TrainingState (state management)
        - Simplified AsymmetricMILTrainer (orchestration only)
    - [2026-01-05] Refactor: Created modular trainer structure

Modules:
    - asymmetric_mil_trainer: Main trainer class (Scout-Snipe orchestration)
    - trainer_builder: Component factory (Builder pattern)
    - training_state: Training state manager (State Machine)
    - engines: Training and validation engines
    - callbacks: Training event callbacks

Usage:
    >>> from src.trainer import AsymmetricMILTrainer
    >>> trainer = AsymmetricMILTrainer(config)
    >>> trainer.fit()

Advanced Usage (custom components):
    >>> from src.trainer import TrainerBuilder, TrainingState
    >>> builder = TrainerBuilder(config)
    >>> components = builder.build_all()
    >>> state = TrainingState(config)
"""

# NOTE: Refactored version (v2.0) - Original v1 backed up
from src.trainer.asymmetric_mil_trainer import AsymmetricMILTrainer

# Export new modules (for advanced usage)
from src.trainer.trainer_builder import TrainerBuilder
from src.trainer.training_state import TrainingState

__all__ = [
    'AsymmetricMILTrainer',
    'TrainerBuilder',  # NEW: Component factory
    'TrainingState'    # NEW: State manager
]
