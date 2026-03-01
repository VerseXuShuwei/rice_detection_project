"""
Plot Training Curves from training_log.json.

Standalone script that regenerates all training curve visualizations
from a saved training_log.json file. Useful when:
  - Training was interrupted before logger.finish()
  - Need to regenerate curves with updated plotting logic
  - Comparing across runs

Recent Updates:
  - [2026-02-06] Initial: Standalone curve regeneration script

Usage:
  python scripts/tools/plot_training_curves.py --log outputs/logs/run_xxx/training_log.json
  python scripts/tools/plot_training_curves.py --log outputs/logs/run_xxx/training_log.json --config outputs/logs/run_xxx/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.local_logger import LocalLogger


def main():
    parser = argparse.ArgumentParser(description='Regenerate training curves from training_log.json')
    parser.add_argument('--log', required=True, type=str,
                        help='Path to training_log.json')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (optional, auto-detected from same directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: training_curves/ next to log file)')
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] File not found: {log_path}")
        sys.exit(1)

    # Load training log
    with open(log_path, encoding='utf-8') as f:
        history = json.load(f)
    print(f"[INFO] Loaded {len(history)} epoch records from {log_path}")

    # Load config (for warmup_epochs)
    config = {}
    config_path = Path(args.config) if args.config else log_path.parent / 'config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        print(f"[INFO] Config loaded from {config_path}")
    else:
        print(f"[WARN] No config found, using defaults (warmup_epochs=6)")
        config = {'asymmetric_mil': {'warmup_epochs': 6}}

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = log_path.parent / 'training_curves'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal logger instance for plotting
    logger = LocalLogger.__new__(LocalLogger)
    logger.config = config
    logger.metrics_history = history
    logger.training_curves_dir = output_dir

    # Generate all plots
    logger.plot_training_curves(save=True)

    # Print summary of generated files
    generated = list(output_dir.glob('*.png'))
    print(f"\n[DONE] Generated {len(generated)} plots in {output_dir}:")
    for f in sorted(generated):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
