"""
Training Entry Point for Asymmetric MIL.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from train_topk_asymmetric.py

Usage:
    train:
    python scripts/train.py --config configs/algorithm/train_topk_asymmetric.yaml
    resume:
    python scripts/train.py --config X.yaml --resume checkpoint.pth
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
import numpy as np
import torch

from src.utils.config_io import load_config
from src.trainer.asymmetric_mil_trainer import AsymmetricMILTrainer


def set_reproducibility(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value (from config: training_strategy.seed)
        deterministic: Enable CUDNN deterministic mode (reduces performance but ensures reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode: ensures reproducibility at cost of performance
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def main(args):
    """Main training function."""
    print("=" * 80)
    print("Asymmetric MIL Training - Scout-Snipe Pipeline")
    print("=" * 80)

    # 1. Load Configuration
    print(f"\n[CONFIG] Loading config from: {args.config}")
    config = load_config(args.config)

    # 2. Set Reproducibility
    training_strategy = config.get('training_strategy', {})
    seed = training_strategy.get('seed', 42)
    deterministic = training_strategy.get('deterministic', True)
    print(f"[SEED] Setting random seed: {seed}, deterministic: {deterministic}")
    set_reproducibility(seed, deterministic)

    # 3. Initialize Trainer (all components built internally)
    print("\n[TRAINER] Initializing AsymmetricMILTrainer...")
    trainer = AsymmetricMILTrainer(config, resume_checkpoint=args.resume)

    # 4. Start Training
    print("\n[TRAINER] Starting training loop...")
    trainer.fit()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Asymmetric MIL Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/algorithm/train_topk_asymmetric.yaml',
        help='Path to config file (default: configs/algorithm/train_topk_asymmetric.yaml)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint for resume training (optional)'
    )

    args = parser.parse_args()
    main(args)
