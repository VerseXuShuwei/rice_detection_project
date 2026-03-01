"""
Training and validation engines for Asymmetric MIL.

Recent Updates:
    - [2026-01-05] Refactor: Created engines submodule

Modules:
    - asymmetric_mil: Scout-Snipe training and validation engines

Usage:
    >>> from src.trainer.engines import train_one_epoch, validate
"""

from src.trainer.engines.asymmetric_mil import train_one_epoch, validate

__all__ = ['train_one_epoch', 'validate']
