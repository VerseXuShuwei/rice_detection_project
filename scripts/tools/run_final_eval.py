"""
Standalone Final Evaluation Runner.

Run final evaluation on an existing experiment without re-training.
Uses full val_dataset and full val_negative_pool (no sampling).
Outputs both filtered and unfiltered confusion matrices.

Usage:
    # Auto-find latest checkpoint
    python scripts/tools/run_final_eval.py \\
        --log-dir outputs/logs/asymmetric_mil_training_20260224_202948

    # Specify checkpoint explicitly
    python scripts/tools/run_final_eval.py \\
        --log-dir outputs/logs/asymmetric_mil_training_20260224_202948 \\
        --checkpoint outputs/checkpoints/asymmetric_mil_training_20260224_202948/epoch_045.pth
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.utils.config_io import load_config
from src.utils.device import get_device
from src.models.builder import get_model
from src.data.datasets import AsymmetricMILDataset
from src.data.negative_pool import NegativeTilePool
from src.evaluation.final_evaluator import FinalEvaluator


class _MockLogger:
    """Minimal logger stub for FinalEvaluator — only path attributes needed."""

    def __init__(self, log_dir: Path):
        self.run_dir = log_dir
        self.evaluation_dir = log_dir / "evaluation"
        self.heatmaps_dir = log_dir / "heatmaps"
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)

    def get_evaluation_dir(self) -> Path:
        return self.evaluation_dir

    def get_heatmaps_dir(self) -> Path:
        return self.heatmaps_dir


def _find_latest_checkpoint(log_dir: Path) -> Path:
    """Find the latest epoch_*.pth in the corresponding checkpoints directory."""
    exp_name = log_dir.name  # e.g. asymmetric_mil_training_20260224_202948
    ckpt_dir = log_dir.parent.parent / "checkpoints" / exp_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Please specify --checkpoint explicitly."
        )
    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No epoch_*.pth files found in: {ckpt_dir}\n"
            f"Please specify --checkpoint explicitly."
        )
    latest = ckpt_files[-1]
    print(f"[EVAL] Auto-selected checkpoint (latest): {latest}")
    return latest


def main(args):
    log_dir = Path(args.log_dir)
    assert log_dir.exists(), f"Log dir not found: {log_dir}"

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    else:
        checkpoint_path = _find_latest_checkpoint(log_dir)

    # Load config from the experiment log dir (same config used during training)
    config_path = log_dir / "config.yaml"
    assert config_path.exists(), f"config.yaml not found in log dir: {config_path}"

    print(f"[EVAL] Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"[EVAL] Device: {device}")

    # Build model
    print(f"[EVAL] Building model...")
    model_name = config.get('model', {}).get('name', 'mil_efficientnetv2-s')
    model = get_model(model_name, config)
    model = model.to(device)

    # Load checkpoint
    print(f"[EVAL] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[EVAL] Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    # Build val dataset
    print(f"[EVAL] Building validation dataset...")
    dataset_cfg = config.get('dataset', {})
    val_dataset = AsymmetricMILDataset(
        config=config,
        split='val'
    )
    print(f"[EVAL] Val dataset: {len(val_dataset)} bags, {val_dataset.num_classes} classes")

    # Build val negative pool
    print(f"[EVAL] Building validation negative pool...")
    val_negative_pool = NegativeTilePool(
        config=config,
        split='val'
    )
    print(f"[EVAL] Val negative pool: {len(val_negative_pool)} tiles")

    # Build mock logger pointing to the existing log dir
    logger = _MockLogger(log_dir)

    # Build evaluator
    print(f"[EVAL] Initializing FinalEvaluator...")
    evaluator = FinalEvaluator(
        model=model,
        val_dataset=val_dataset,
        val_negative_pool=val_negative_pool,
        config=config,
        device=device,
        logger=logger
    )

    # Run evaluation
    print(f"\n[EVAL] Starting evaluation...")
    metrics = evaluator.evaluate_all()
    evaluator.save_evaluation_report(metrics)

    print(f"\n[EVAL] Done. Results saved to: {log_dir / 'evaluation'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run final evaluation on an existing experiment')
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Path to experiment log directory (e.g. outputs/logs/asymmetric_mil_training_20260224_202948)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint .pth file (default: auto-find latest epoch_*.pth)'
    )
    args = parser.parse_args()
    main(args)
