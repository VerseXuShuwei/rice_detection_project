"""
Unit Tests for Critical Training Nodes

Recent Updates:
  - [2026-01-17] Feature: Initial test suite for TrainingState and Trainer methods

Key Test Areas:
  1. TrainingState API consistency
  2. Callback invocation signatures
  3. Checkpoint save/load integrity
  4. Phase transition logic

Usage:
  python tests/test_training_nodes.py
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Import modules under test
from src.trainer.training_state import TrainingState
from src.trainer.callbacks import (
    CheckpointCallback,
    DriftMonitorCallback,
    HeatmapCallback,
    Callback
)


class TestTrainingState(unittest.TestCase):
    """Test TrainingState class methods and signatures."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_config = {
            'asymmetric_mil': {
                'warmup_epochs': 8,
                'warmup_k': 4,
                'stable_k': 2,
                'hard_negative_mining': {
                    'enable': True,
                    'warmup_hard_ratio': 0.25,
                    'stable_hard_ratio': 0.15
                }
            }
        }
        self.training_state = TrainingState(self.default_config)

    def test_init_defaults(self):
        """Test TrainingState initialization."""
        self.assertTrue(self.training_state.is_warmup)
        self.assertEqual(self.training_state.current_k, 4)  # warmup_k
        self.assertEqual(self.training_state.stable_epoch_count, 0)
        self.assertEqual(self.training_state.best_score, 0.0)

    def test_get_epoch_state_signature(self):
        """Test get_epoch_state returns expected keys."""
        state = self.training_state.get_epoch_state(epoch=1)

        required_keys = ['is_warmup', 'K', 'hard_mining_enable', 'hard_ratio', 'mosaic_ratio']
        for key in required_keys:
            self.assertIn(key, state, f"Missing key: {key}")

    def test_update_best_metrics_signature(self):
        """Test update_best_metrics accepts exactly 2 positional args (hit_acc, topk_lift)."""
        # Should NOT raise TypeError
        result = self.training_state.update_best_metrics(
            hit_acc=0.8,
            topk_lift=0.3
        )
        self.assertIsInstance(result, bool)

    def test_update_best_metrics_rejects_extra_args(self):
        """Test update_best_metrics rejects extra arguments."""
        with self.assertRaises(TypeError):
            # This was the bug: passing current_score as first arg
            self.training_state.update_best_metrics(0.5, 0.8, 0.3)

    def test_update_best_metrics_logic(self):
        """Test best metrics update logic."""
        # First call should always return True (new best)
        self.assertTrue(self.training_state.update_best_metrics(0.5, 0.3))

        # Score = 0.6 * 0.5 + 0.4 * 0.3 = 0.42
        self.assertAlmostEqual(self.training_state.best_score, 0.42, places=6)

        # Worse score should return False
        self.assertFalse(self.training_state.update_best_metrics(0.3, 0.2))

        # Better score should return True
        self.assertTrue(self.training_state.update_best_metrics(0.9, 0.5))

    def test_transition_to_stable_signature(self):
        """Test transition_to_stable accepts initial_pos_acc."""
        # Should NOT raise TypeError
        self.training_state.transition_to_stable(initial_pos_acc=0.85)

        self.assertFalse(self.training_state.is_warmup)
        self.assertEqual(self.training_state.current_k, 2)  # stable_k
        self.assertEqual(self.training_state.stable_start_pos_acc, 0.85)

    def test_stable_start_pos_acc_attribute_exists(self):
        """Test stable_start_pos_acc attribute exists (not initial_pos_acc)."""
        # stable_start_pos_acc should exist
        self.assertTrue(hasattr(self.training_state, 'stable_start_pos_acc'))
        # initial_pos_acc should NOT exist (common naming confusion)
        self.assertFalse(hasattr(self.training_state, 'initial_pos_acc'))

    def test_check_concept_drift_signature(self):
        """Test check_concept_drift signature."""
        self.training_state.transition_to_stable(initial_pos_acc=0.85)

        # Should accept current_pos_acc and optional threshold
        result = self.training_state.check_concept_drift(current_pos_acc=0.80)
        self.assertIsInstance(result, bool)

        result = self.training_state.check_concept_drift(current_pos_acc=0.70, threshold=0.10)
        self.assertIsInstance(result, bool)

    def test_state_dict_roundtrip(self):
        """Test state_dict and load_state_dict consistency."""
        # Modify state
        self.training_state.transition_to_stable(initial_pos_acc=0.85)
        self.training_state.update_best_metrics(0.9, 0.4)
        self.training_state.increment_stable_epoch()

        # Save state
        saved_state = self.training_state.state_dict()

        # Create new instance and load
        new_state = TrainingState(self.default_config)
        new_state.load_state_dict(saved_state)

        # Verify loaded state matches
        self.assertEqual(new_state.is_warmup, self.training_state.is_warmup)
        self.assertEqual(new_state.current_k, self.training_state.current_k)
        self.assertEqual(new_state.best_score, self.training_state.best_score)
        self.assertEqual(new_state.stable_epoch_count, self.training_state.stable_epoch_count)

    def test_increment_stable_epoch(self):
        """Test stable epoch counter."""
        # Should not increment in warmup
        self.training_state.increment_stable_epoch()
        self.assertEqual(self.training_state.stable_epoch_count, 0)

        # Should increment after transition
        self.training_state.transition_to_stable(0.8)
        self.training_state.increment_stable_epoch()
        self.assertEqual(self.training_state.stable_epoch_count, 1)


class TestCallbackSignatures(unittest.TestCase):
    """Test Callback class method signatures."""

    def setUp(self):
        """Create mock trainer."""
        self.mock_trainer = Mock()
        self.mock_trainer.is_warmup = False
        self.mock_trainer.train_cfg = {'save_every': 5, 'heatmap_every': 5}
        self.mock_trainer.topk_metrics = {'hit_acc': 0.8, 'topk_lift': 0.3}
        self.mock_trainer.state = Mock()
        self.mock_trainer.state.is_warmup = False
        self.mock_trainer.state.update_best_metrics = Mock(return_value=True)
        self.mock_trainer.state.best_score = 0.5
        self.mock_trainer._save_regular_checkpoint = Mock()
        self.mock_trainer._save_best_checkpoint_if_improved = Mock()
        self.mock_trainer._monitor_concept_drift = Mock()
        self.mock_trainer._generate_heatmaps = Mock()

    def test_checkpoint_callback_on_epoch_end(self):
        """Test CheckpointCallback.on_epoch_end signature."""
        callback = CheckpointCallback()
        metrics = {'val_acc': 0.85, 'val_loss': 0.3}

        # Should NOT raise TypeError
        callback.on_epoch_end(self.mock_trainer, epoch=5, metrics=metrics)

        # Verify calls
        self.mock_trainer._save_regular_checkpoint.assert_called_once_with(5, metrics)
        self.mock_trainer._save_best_checkpoint_if_improved.assert_called_once()

    def test_drift_monitor_callback_on_epoch_end(self):
        """Test DriftMonitorCallback.on_epoch_end signature."""
        callback = DriftMonitorCallback()
        metrics = {'val_acc': 0.85}

        callback.on_epoch_end(self.mock_trainer, epoch=10, metrics=metrics)

        self.mock_trainer._monitor_concept_drift.assert_called_once_with(0.85)

    def test_heatmap_callback_on_epoch_end(self):
        """Test HeatmapCallback.on_epoch_end signature."""
        callback = HeatmapCallback(frequency=10)
        self.mock_trainer.visualizer = Mock()
        self.mock_trainer.visualizer.generate_monitoring_heatmaps = Mock()
        self.mock_trainer.train_dataset = Mock()
        self.mock_trainer.model = Mock()
        metrics = {'val_acc': 0.85}

        # Should trigger at epoch 10 (matches frequency)
        callback.on_epoch_end(self.mock_trainer, epoch=10, metrics=metrics)

        # Should call visualizer (not _generate_heatmaps)
        self.mock_trainer.visualizer.generate_monitoring_heatmaps.assert_called_once()


class TestTrainerMethodSignatures(unittest.TestCase):
    """Test AsymmetricMILTrainer method signatures via inspection."""

    def test_save_best_checkpoint_if_improved_signature(self):
        """Test _save_best_checkpoint_if_improved accepts (epoch, topk_metrics, metrics)."""
        import inspect
        from src.trainer.asymmetric_mil_trainer import AsymmetricMILTrainer

        sig = inspect.signature(AsymmetricMILTrainer._save_best_checkpoint_if_improved)
        params = list(sig.parameters.keys())

        # Should be: self, epoch, topk_metrics, metrics
        self.assertEqual(params, ['self', 'epoch', 'topk_metrics', 'metrics'])

    def test_save_regular_checkpoint_signature(self):
        """Test _save_regular_checkpoint signature."""
        import inspect
        from src.trainer.asymmetric_mil_trainer import AsymmetricMILTrainer

        sig = inspect.signature(AsymmetricMILTrainer._save_regular_checkpoint)
        params = list(sig.parameters.keys())

        self.assertIn('self', params)
        self.assertIn('epoch', params)
        self.assertIn('metrics', params)

    def test_evaluate_topk_and_check_transition_signature(self):
        """Test _evaluate_topk_and_check_transition signature."""
        import inspect
        from src.trainer.asymmetric_mil_trainer import AsymmetricMILTrainer

        sig = inspect.signature(AsymmetricMILTrainer._evaluate_topk_and_check_transition)
        params = list(sig.parameters.keys())

        self.assertIn('self', params)
        self.assertIn('epoch', params)


class TestPhaseTransitionLogic(unittest.TestCase):
    """Test warmup → stable phase transition."""

    def setUp(self):
        config = {
            'asymmetric_mil': {
                'warmup_epochs': 8,
                'warmup_k': 4,
                'stable_k': 2,
                'hard_negative_mining': {
                    'enable': True,
                    'warmup_hard_ratio': 0.25,
                    'stable_hard_ratio': 0.15
                }
            }
        }
        self.training_state = TrainingState(config)

    def test_should_transition_p0_criteria_all_pass(self):
        """Test P0 criteria for phase transition - all pass."""
        p0_pass = {
            'negative_recall': 90.0,
            'topk_lift': 25.0,
            'top1_confidence': 60.0
        }
        self.assertTrue(self.training_state.should_transition_to_stable(p0_pass))

    def test_should_transition_p0_criteria_neg_recall_fail(self):
        """Test P0 criteria - negative recall fail."""
        p0_fail = {
            'negative_recall': 80.0,  # < 85
            'topk_lift': 25.0,
            'top1_confidence': 60.0
        }
        self.assertFalse(self.training_state.should_transition_to_stable(p0_fail))

    def test_should_transition_p0_criteria_topk_lift_fail(self):
        """Test P0 criteria - topk lift fail."""
        p0_fail = {
            'negative_recall': 90.0,
            'topk_lift': 15.0,  # < 20
            'top1_confidence': 60.0
        }
        self.assertFalse(self.training_state.should_transition_to_stable(p0_fail))

    def test_should_transition_p0_criteria_top1_conf_fail(self):
        """Test P0 criteria - top1 confidence fail."""
        p0_fail = {
            'negative_recall': 90.0,
            'topk_lift': 25.0,
            'top1_confidence': 40.0  # < 50
        }
        self.assertFalse(self.training_state.should_transition_to_stable(p0_fail))

    def test_k_value_changes_after_transition(self):
        """Test K value updates correctly after transition."""
        self.assertEqual(self.training_state.current_k, 4)  # warmup_k

        self.training_state.transition_to_stable(0.85)

        self.assertEqual(self.training_state.current_k, 2)  # stable_k

    def test_hard_ratio_changes_after_transition(self):
        """Test hard_ratio updates correctly after transition."""
        self.assertEqual(self.training_state.current_hard_ratio, 0.25)  # warmup_hard_ratio

        self.training_state.transition_to_stable(0.85)

        self.assertEqual(self.training_state.current_hard_ratio, 0.15)  # stable_hard_ratio


class TestTrainerStateAttributeUsage(unittest.TestCase):
    """Test that all state attributes used in trainer actually exist in TrainingState."""

    def test_all_trainer_state_attributes_exist(self):
        """Verify all self.state.X attributes used in trainer exist in TrainingState."""
        from src.trainer.training_state import TrainingState

        # Attributes referenced in asymmetric_mil_trainer.py via self.state.X
        required_attributes = [
            'is_warmup',
            'current_epoch',
            'current_k',
            'stable_epoch_count',
            'best_score',
            'stable_start_pos_acc',  # NOT initial_pos_acc!
        ]

        # Methods called on self.state
        required_methods = [
            'get_epoch_state',
            'increment_stable_epoch',
            'update_best_metrics',
            'transition_to_stable',
            'load_state_dict',
            'state_dict',
            'check_concept_drift',
        ]

        config = {
            'asymmetric_mil': {
                'warmup_epochs': 8,
                'warmup_k': 4,
                'stable_k': 2,
                'hard_negative_mining': {'enable': True, 'warmup_hard_ratio': 0.25, 'stable_hard_ratio': 0.15}
            }
        }
        state = TrainingState(config)

        # Check attributes
        for attr in required_attributes:
            self.assertTrue(
                hasattr(state, attr),
                f"TrainingState missing attribute '{attr}' used in trainer"
            )

        # Check methods
        for method in required_methods:
            self.assertTrue(
                hasattr(state, method) and callable(getattr(state, method)),
                f"TrainingState missing method '{method}' used in trainer"
            )

    def test_no_typo_attributes(self):
        """Ensure commonly confused attribute names don't exist."""
        from src.trainer.training_state import TrainingState

        config = {
            'asymmetric_mil': {
                'warmup_epochs': 8,
                'warmup_k': 4,
                'stable_k': 2,
                'hard_negative_mining': {'enable': True}
            }
        }
        state = TrainingState(config)

        # These are WRONG names that should NOT exist
        wrong_names = [
            'initial_pos_acc',      # Should be stable_start_pos_acc
            'pos_acc',              # Should be stable_start_pos_acc
            'baseline_acc',         # Should be stable_start_pos_acc
            'epoch_count',          # Should be stable_epoch_count
        ]

        for wrong_name in wrong_names:
            self.assertFalse(
                hasattr(state, wrong_name),
                f"TrainingState has confusing attribute '{wrong_name}' - likely a naming bug"
            )


class TestFinalEvaluatorSignatures(unittest.TestCase):
    """Test FinalEvaluator method signatures."""

    def test_evaluate_all_signature(self):
        """Test evaluate_all() takes no positional args (V1-compatible)."""
        import inspect
        from src.evaluation.final_evaluator import FinalEvaluator

        sig = inspect.signature(FinalEvaluator.evaluate_all)
        params = list(sig.parameters.keys())

        # Should only have 'self'
        self.assertEqual(params, ['self'])

    def test_save_evaluation_report_signature(self):
        """Test save_evaluation_report accepts metrics dict."""
        import inspect
        from src.evaluation.final_evaluator import FinalEvaluator

        sig = inspect.signature(FinalEvaluator.save_evaluation_report)
        params = list(sig.parameters.keys())

        self.assertIn('self', params)
        self.assertIn('metrics', params)


class TestCheckpointManagerSignatures(unittest.TestCase):
    """Test CheckpointManager method signatures."""

    def test_save_checkpoint_supports_training_state(self):
        """Test save_checkpoint accepts training_state parameter."""
        import inspect
        from src.core.checkpoint_manager import CheckpointManager

        sig = inspect.signature(CheckpointManager.save_checkpoint)
        params = list(sig.parameters.keys())

        # Must support training_state for resume functionality
        self.assertIn('training_state', params)

    def test_save_checkpoint_full_signature(self):
        """Test save_checkpoint has all required parameters."""
        import inspect
        from src.core.checkpoint_manager import CheckpointManager

        sig = inspect.signature(CheckpointManager.save_checkpoint)
        params = list(sig.parameters.keys())

        # Required by trainer calls
        required = ['model', 'optimizer', 'epoch', 'metrics', 'config',
                    'scheduler', 'scaler', 'training_state', 'filename']
        for p in required:
            self.assertIn(p, params, f"Missing parameter: {p}")


class TestAllExternalMethodSignatures(unittest.TestCase):
    """Comprehensive test for all external method signatures used in trainer."""

    def test_warmup_evaluator_signatures(self):
        """Test WarmupEvaluator method signatures match trainer calls."""
        import inspect
        from src.evaluation.warmup_evaluator import WarmupEvaluator

        # evaluate_warmup_criteria(epoch, train_loss)
        sig = inspect.signature(WarmupEvaluator.evaluate_warmup_criteria)
        params = list(sig.parameters.keys())
        self.assertIn('epoch', params)
        self.assertIn('train_loss', params)

        # evaluate_topk_quality(k)
        sig = inspect.signature(WarmupEvaluator.evaluate_topk_quality)
        params = list(sig.parameters.keys())
        self.assertIn('k', params)

    def test_visualizer_signatures(self):
        """Test MILVisualizer method signatures match callback calls."""
        import inspect
        from src.evaluation.heatmap_visualizer import MILVisualizer

        # generate_monitoring_heatmaps(model, dataset, epoch, phase, samples_per_class)
        sig = inspect.signature(MILVisualizer.generate_monitoring_heatmaps)
        params = list(sig.parameters.keys())
        self.assertIn('model', params)
        self.assertIn('dataset', params)
        self.assertIn('epoch', params)
        self.assertIn('phase', params)

    def test_negative_pool_signatures(self):
        """Test NegativeTilePool method signatures match engine calls."""
        import inspect
        from src.data.negative_pool import NegativeTilePool

        # start_epoch() - no args
        sig = inspect.signature(NegativeTilePool.start_epoch)
        params = [p for p in sig.parameters.keys() if p != 'self']
        self.assertEqual(len(params), 0)

        # end_epoch(current_epoch)
        sig = inspect.signature(NegativeTilePool.end_epoch)
        params = list(sig.parameters.keys())
        self.assertIn('current_epoch', params)

        # sample(n, hard_ratio)
        sig = inspect.signature(NegativeTilePool.sample)
        params = list(sig.parameters.keys())
        self.assertIn('n', params)
        self.assertIn('hard_ratio', params)

        # register_batch_mapping(tile_indices, batch_indices)
        sig = inspect.signature(NegativeTilePool.register_batch_mapping)
        params = list(sig.parameters.keys())
        self.assertIn('tile_indices', params)
        self.assertIn('batch_indices', params)

        # record_prediction(batch_neg_indices, predictions)
        sig = inspect.signature(NegativeTilePool.record_prediction)
        params = list(sig.parameters.keys())
        self.assertIn('batch_neg_indices', params)
        self.assertIn('predictions', params)

    def test_local_logger_signatures(self):
        """Test LocalLogger method signatures match trainer calls."""
        import inspect
        from src.utils.local_logger import LocalLogger

        # log(metrics, step)
        sig = inspect.signature(LocalLogger.log)
        params = list(sig.parameters.keys())
        self.assertIn('metrics', params)
        self.assertIn('step', params)

        # finish() - no args
        sig = inspect.signature(LocalLogger.finish)
        params = [p for p in sig.parameters.keys() if p != 'self']
        self.assertEqual(len(params), 0)


class TestLoggerIntegration(unittest.TestCase):
    """Test LocalLogger integration."""

    def test_logger_finish_method_exists(self):
        """Test logger.finish() method exists."""
        from src.utils.local_logger import LocalLogger
        self.assertTrue(hasattr(LocalLogger, 'finish'))

    def test_logger_plot_training_curves_exists(self):
        """Test logger.plot_training_curves() method exists."""
        from src.utils.local_logger import LocalLogger
        self.assertTrue(hasattr(LocalLogger, 'plot_training_curves'))


if __name__ == '__main__':
    # Run with verbosity
    unittest.main(verbosity=2)
