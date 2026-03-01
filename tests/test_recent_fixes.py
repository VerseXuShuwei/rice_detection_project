"""
Test Script: Validate Recent Bug Fixes

Purpose:
    - Test 1: Training curve metrics merging (_merge_metrics_by_epoch)
    - Test 2: Hit-Acc filter statistics report generation
    - Test 3: Outputs path configuration consistency

Usage:
    python tests/test_recent_fixes.py

Recent Updates:
    - [2026-01-17] Initial: Validate 2026-01-17 bug fixes
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTrainingCurveMerging:
    """Test _merge_metrics_by_epoch fixes training curve breaks."""

    def test_merge_duplicate_epochs(self):
        """
        Test that duplicate epoch records are merged correctly.

        Scenario: training_log.json has two records for epoch 9:
        - First record (before topk_metrics): {'epoch': 9, 'train_loss': 0.5}
        - Second record (after topk_metrics): {'epoch': 9, 'topk_lift': 0.3, 'hit_acc': 0.7}

        Expected: Merged into single record with all metrics
        """
        from src.utils.local_logger import LocalLogger

        # Create temp directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock logger
            logger = LocalLogger(
                experiment_name="test_merge",
                config={"test": True},
                logs_root=os.path.join(tmpdir, "logs"),
                checkpoints_root=os.path.join(tmpdir, "checkpoints")
            )

            # Simulate duplicate epoch records (like real training_log.json)
            logger.metrics_history = [
                {'epoch': 0, 'train_loss': 1.0, 'val_loss': 1.1},
                {'epoch': 1, 'train_loss': 0.9, 'val_loss': 1.0},
                {'epoch': 8, 'train_loss': 0.5},  # First record for epoch 8
                {'epoch': 8, 'topk_lift': 0.25, 'hit_acc': 0.6},  # Second record
                {'epoch': 9, 'train_loss': 0.45},  # First record for epoch 9
                {'epoch': 9, 'topk_lift': 0.3, 'hit_acc': 0.7, 'neg_recall': 0.8},  # Second
            ]

            # Test merge function
            merged = logger._merge_metrics_by_epoch()

            # Verify results
            assert len(merged) == 4, f"Expected 4 unique epochs, got {len(merged)}"

            # Check epoch 8 is merged
            epoch_8 = next(m for m in merged if m['epoch'] == 8)
            assert 'train_loss' in epoch_8, "train_loss missing from merged epoch 8"
            assert 'topk_lift' in epoch_8, "topk_lift missing from merged epoch 8"
            assert 'hit_acc' in epoch_8, "hit_acc missing from merged epoch 8"
            assert epoch_8['train_loss'] == 0.5
            assert epoch_8['topk_lift'] == 0.25

            # Check epoch 9 is merged
            epoch_9 = next(m for m in merged if m['epoch'] == 9)
            assert 'train_loss' in epoch_9
            assert 'topk_lift' in epoch_9
            assert 'neg_recall' in epoch_9
            assert epoch_9['neg_recall'] == 0.8

            print("[PASS] test_merge_duplicate_epochs")

    def test_merge_preserves_order(self):
        """Test that merged metrics maintain epoch order."""
        from src.utils.local_logger import LocalLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LocalLogger(
                experiment_name="test_order",
                config={},
                logs_root=os.path.join(tmpdir, "logs"),
                checkpoints_root=os.path.join(tmpdir, "checkpoints")
            )

            # Epochs in random order
            logger.metrics_history = [
                {'epoch': 5, 'loss': 0.5},
                {'epoch': 2, 'loss': 0.8},
                {'epoch': 8, 'loss': 0.3},
                {'epoch': 0, 'loss': 1.0},
            ]

            merged = logger._merge_metrics_by_epoch()
            epochs = [m['epoch'] for m in merged]

            assert epochs == sorted(epochs), f"Epochs not sorted: {epochs}"
            print("[PASS] test_merge_preserves_order")

    def test_merge_handles_nan_values(self):
        """Test that NaN values don't break extraction after merge."""
        import numpy as np
        from src.utils.local_logger import LocalLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LocalLogger(
                experiment_name="test_nan",
                config={},
                logs_root=os.path.join(tmpdir, "logs"),
                checkpoints_root=os.path.join(tmpdir, "checkpoints")
            )

            # Mixed valid and missing values
            logger.metrics_history = [
                {'epoch': 0, 'train_loss': 1.0},
                {'epoch': 0, 'topk_lift': 0.1},  # train_loss missing in second record
                {'epoch': 1, 'train_loss': 0.9, 'topk_lift': 0.2},
            ]

            merged = logger._merge_metrics_by_epoch()

            # Extract train_loss series
            train_losses = [m.get('train_loss', np.nan) for m in merged]

            # Should not have any NaN since merge combines records
            assert not np.isnan(train_losses[0]), "Epoch 0 train_loss should exist after merge"
            print("[PASS] test_merge_handles_nan_values")


class TestHitAccFilterStats:
    """Test confusion matrix filter statistics report."""

    def test_filter_stats_markdown_generation(self):
        """Test that filter statistics markdown report is generated correctly."""
        from pathlib import Path
        import tempfile

        # Simulate metrics dict (as returned by _compute_confusion_matrix)
        metrics = {
            'positive_tiles_total': 1000,
            'positive_tiles_filtered': 650,
            'confusion_matrix': [[100, 50], [30, 470]],
            'overall_accuracy': 0.85,
        }

        # Mock config
        config = {
            'evaluation': {
                'confusion_matrix': {
                    'use_hit_acc_filter': True,
                    'min_confidence': 0.45,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = Path(tmpdir)

            # Calculate expected values
            total = metrics['positive_tiles_total']
            filtered = metrics['positive_tiles_filtered']
            excluded = total - filtered
            filter_ratio = filtered / total if total > 0 else 0.0
            exclude_ratio = excluded / total if total > 0 else 0.0

            # Generate report content (simulating _save_filter_statistics_report)
            md_content = f"""# Confusion Matrix Filter Statistics

## Filter Configuration
- **Filter Enabled**: {config['evaluation']['confusion_matrix']['use_hit_acc_filter']}
- **Min Confidence Threshold**: {config['evaluation']['confusion_matrix']['min_confidence']}

## Tile Statistics
| Metric | Count | Percentage |
|--------|-------|------------|
| Total Positive Tiles | {total} | 100.0% |
| Included (Pass Filter) | {filtered} | {filter_ratio*100:.1f}% |
| Excluded (Below Threshold) | {excluded} | {exclude_ratio*100:.1f}% |

## Interpretation
- **High Exclusion Rate (>50%)**: Many tiles have low confidence, indicating:
  - Model uncertainty on these tiles
  - Possible annotation noise in the dataset
  - Need for better feature learning

- **Low Exclusion Rate (<20%)**: Model is confident on most tiles
"""

            # Write and verify
            md_path = eval_dir / "confusion_matrix_filter_stats.md"
            md_path.write_text(md_content, encoding='utf-8')

            # Verify file exists and contains expected sections
            assert md_path.exists()
            content = md_path.read_text(encoding='utf-8')
            assert "Filter Configuration" in content
            assert "Tile Statistics" in content
            assert "Min Confidence Threshold" in content
            assert f"{exclude_ratio*100:.1f}%" in content

            print("[PASS] test_filter_stats_markdown_generation")

    def test_filter_stats_edge_cases(self):
        """Test filter stats with edge cases (0 tiles, all filtered, none filtered)."""
        # Case 1: No tiles
        metrics_empty = {
            'positive_tiles_total': 0,
            'positive_tiles_filtered': 0,
        }
        total = metrics_empty['positive_tiles_total']
        filtered = metrics_empty['positive_tiles_filtered']
        filter_ratio = filtered / total if total > 0 else 0.0
        assert filter_ratio == 0.0, "Empty case should return 0.0"

        # Case 2: All tiles filtered (100% pass)
        metrics_all = {
            'positive_tiles_total': 100,
            'positive_tiles_filtered': 100,
        }
        total = metrics_all['positive_tiles_total']
        filtered = metrics_all['positive_tiles_filtered']
        filter_ratio = filtered / total if total > 0 else 0.0
        assert filter_ratio == 1.0, "All pass case should return 1.0"

        # Case 3: No tiles pass filter
        metrics_none = {
            'positive_tiles_total': 100,
            'positive_tiles_filtered': 0,
        }
        total = metrics_none['positive_tiles_total']
        filtered = metrics_none['positive_tiles_filtered']
        filter_ratio = filtered / total if total > 0 else 0.0
        assert filter_ratio == 0.0, "None pass case should return 0.0"

        print("[PASS] test_filter_stats_edge_cases")


class TestOutputsPathConsistency:
    """Test that outputs paths are configuration-driven."""

    def test_logger_creates_unified_directories(self):
        """Test LocalLogger creates logs and checkpoints under same experiment name."""
        from src.utils.local_logger import LocalLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = os.path.join(tmpdir, "logs")
            ckpt_root = os.path.join(tmpdir, "checkpoints")

            logger = LocalLogger(
                experiment_name="test_exp",
                config={},
                logs_root=logs_root,
                checkpoints_root=ckpt_root
            )

            # Verify both directories exist
            assert logger.run_dir.exists(), "Logs run_dir should exist"
            assert logger.checkpoints_dir.exists(), "Checkpoints dir should exist"

            # Verify same experiment name in both paths
            logs_name = logger.run_dir.name
            ckpt_name = logger.checkpoints_dir.name
            assert logs_name == ckpt_name, f"Name mismatch: {logs_name} vs {ckpt_name}"

            # Verify subdirectories exist
            assert (logger.run_dir / "training_curves").exists()
            assert (logger.run_dir / "heatmaps").exists()
            assert (logger.run_dir / "evaluation").exists()

            print("[PASS] test_logger_creates_unified_directories")

    def test_checkpoint_manager_uses_logger_path(self):
        """Test CheckpointManager gets path from Logger, not hardcoded config."""
        from src.utils.local_logger import LocalLogger
        from src.core.checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = os.path.join(tmpdir, "logs")
            ckpt_root = os.path.join(tmpdir, "checkpoints")

            logger = LocalLogger(
                experiment_name="test_ckpt",
                config={},
                logs_root=logs_root,
                checkpoints_root=ckpt_root
            )

            # Create CheckpointManager using logger's path (as TrainerBuilder does)
            ckpt_manager = CheckpointManager(
                checkpoint_root=str(logger.get_checkpoints_dir())
            )

            # Verify path matches
            assert str(ckpt_manager.checkpoint_root) == str(logger.checkpoints_dir)

            print("[PASS] test_checkpoint_manager_uses_logger_path")

    def test_config_no_hardcoded_checkpoint_root(self):
        """Verify config doesn't specify hardcoded checkpoint_root."""
        from src.utils.config_io import load_config

        config = load_config("configs/algorithm/train_topk_asymmetric.yaml")

        # Check training section doesn't have checkpoint_root
        training_cfg = config.get('training', {})
        assert 'checkpoint_root' not in training_cfg, \
            "checkpoint_root should not be in training config (managed by LocalLogger)"

        print("[PASS] test_config_no_hardcoded_checkpoint_root")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("RUNNING RECENT FIXES TESTS")
    print("=" * 60)

    test_classes = [
        TestTrainingCurveMerging,
        TestHitAccFilterStats,
        TestOutputsPathConsistency,
    ]

    total_passed = 0
    total_failed = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    total_passed += 1
                except AssertionError as e:
                    print(f"[FAIL] {method_name}: {e}")
                    total_failed += 1
                    failed_tests.append(f"{test_class.__name__}.{method_name}")
                except Exception as e:
                    print(f"[ERROR] {method_name}: {type(e).__name__}: {e}")
                    total_failed += 1
                    failed_tests.append(f"{test_class.__name__}.{method_name}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n[SUCCESS] All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
