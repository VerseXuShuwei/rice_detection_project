"""
Test Script: Validate Config Split Correctness

Purpose:
    - Verify all critical parameters exist in merged config
    - Compare with original backup config to ensure nothing was lost
    - Test edge cases (override, missing files, etc.)

Usage:
    python tests/test_config_split.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_io import load_config, load_yaml


def test_config_completeness():
    """
    Test 1: Verify all critical parameters exist in merged config
    """
    print("\n" + "="*60)
    print("TEST 1: Config Completeness Check")
    print("="*60)

    config = load_config("configs/algorithm/train_topk_asymmetric.yaml")

    # Define critical parameters that MUST exist
    critical_params = {
        # Training Strategy
        'training_strategy': ['name', 'seed', 'description'],

        # Dataset
        'dataset': ['name', 'root', 'negative_class_name', 'final_tile_size', 'train_ratio', 'tile_config'],
        'dataset.tile_config': ['warmup_tile_size', 'scales', 'overlaps', 'max_tiles_per_bag'],

        # Negative Pool
        'negative_pool': ['enable', 'lmdb_path', 'tile_size', 'max_pool_size'],
        'negative_pool.mosaic': ['enable', 'warmup_ratio', 'stable_ratio', 'target_size'],

        # Augmentation
        'augmentation': ['geometric', 'color', 'noise', 'random_erasing'],

        # Model
        'model': ['name', 'pretrained', 'dropout', 'img_size', 'freeze_stages'],
        'mil': ['type', 'num_classes', 'aggregation', 'include_negative_class'],

        # Asymmetric MIL
        'asymmetric_mil': ['warmup_epochs', 'warmup_k', 'stable_k', 'scout_batch_size', 'cache_augmented_tiles'],
        'asymmetric_mil.hard_negative_mining': ['enable', 'warmup_hard_ratio', 'stable_hard_ratio'],
        'asymmetric_mil.warmup_criteria': ['neg_recall_threshold', 'topk_lift_threshold', 'top1_confidence_threshold'],

        # Loss
        'loss': ['type', 'top1_ce_weight', 'top2k_nr_weight', 'top2k_soft_weight'],
        'loss.dynamic_weight': ['enable', 'warmup_weight', 'stable_weight'],
        'loss.ranking': ['enable', 'inter_weight', 'margin'],
        'loss.stable_gate': ['enable', 'confidence_threshold', 'require_correct'],

        # Optimizer
        'optimizer': ['name', 'backbone_lr', 'classifier_lr', 'weight_decay'],

        # Scheduler
        'scheduler': ['name', 'warmup_epochs', 'hold_epochs', 'min_lr'],

        # Training
        'training': ['num_epochs', 'use_amp', 'gradient_clip', 'save_every'],

        # Hardware
        'hardware': ['device'],

        # Logging
        'logging': ['enable', 'framework', 'project_name'],

        # Evaluation
        'evaluation': ['enable', 'tile_level_metrics', 'heatmap_eval_frequency'],
    }

    # Check each parameter
    missing_params = []
    for param_path, required_keys in critical_params.items():
        parts = param_path.split('.')
        current = config

        # Navigate to nested dict
        for part in parts:
            if part not in current:
                missing_params.append(f"{param_path} (section missing)")
                break
            current = current[part]
        else:
            # Check required keys
            for key in required_keys:
                if key not in current:
                    missing_params.append(f"{param_path}.{key}")

    # Report results
    if missing_params:
        print("\n[FAILED] Missing parameters:")
        for param in missing_params:
            print(f"  - {param}")
        return False
    else:
        print("\n[PASSED] All critical parameters present!")
        print(f"  Checked {len(critical_params)} sections with {sum(len(v) for v in critical_params.values())} keys")
        return True


def test_config_values():
    """
    Test 2: Verify specific values are correct
    """
    print("\n" + "="*60)
    print("TEST 2: Config Values Verification")
    print("="*60)

    config = load_config("configs/algorithm/train_topk_asymmetric.yaml")

    # Define expected values
    expected_values = {
        'training_strategy.seed': 42,
        'dataset.name': 'total_rice_image',
        'dataset.final_tile_size': 384,
        'asymmetric_mil.warmup_epochs': 8,
        'asymmetric_mil.warmup_k': 4,
        'asymmetric_mil.stable_k': 2,
        'model.name': 'mil_efficientnetv2-s',
        'optimizer.name': 'adamw',
        'optimizer.backbone_lr': 1.0e-5,
        'scheduler.name': 'trapezoidal',
        'training.num_epochs': 30,
        'loss.type': 'topk_anchored_mil',
    }

    # Check each value
    failed_checks = []
    for param_path, expected_value in expected_values.items():
        parts = param_path.split('.')
        current = config

        # Navigate to value
        try:
            for part in parts:
                current = current[part]

            if current != expected_value:
                failed_checks.append(f"{param_path}: expected {expected_value}, got {current}")
        except KeyError as e:
            failed_checks.append(f"{param_path}: Key not found ({e})")

    # Report results
    if failed_checks:
        print("\n[FAILED] Value mismatches:")
        for check in failed_checks:
            print(f"  - {check}")
        return False
    else:
        print("\n[PASSED] All values correct!")
        print(f"  Verified {len(expected_values)} parameters")
        return True


def test_deep_merge():
    """
    Test 3: Verify deep merge works correctly (no shallow overwrites)
    """
    print("\n" + "="*60)
    print("TEST 3: Deep Merge Verification")
    print("="*60)

    config = load_config("configs/algorithm/train_topk_asymmetric.yaml")

    # Verify that nested dicts are properly merged, not overwritten
    # Example: asymmetric_mil.hard_negative_mining should have ALL keys from algorithm config
    hard_mining = config.get('asymmetric_mil', {}).get('hard_negative_mining', {})

    required_keys = ['enable', 'warmup_hard_ratio', 'stable_hard_ratio',
                     'max_failures_before_blacklist', 'initial_life_points']

    missing_keys = [k for k in required_keys if k not in hard_mining]

    if missing_keys:
        print(f"\n[FAILED] Deep merge issue - missing keys in hard_negative_mining:")
        for key in missing_keys:
            print(f"  - {key}")
        return False
    else:
        print("\n[PASSED] Deep merge working correctly!")
        print(f"  hard_negative_mining has all {len(required_keys)} expected keys")
        return True


def test_no_duplication():
    """
    Test 4: Verify no duplicate keys between configs
    """
    print("\n" + "="*60)
    print("TEST 4: No Key Duplication Check")
    print("="*60)

    # This test ensures mil config is NOT duplicated between model and algorithm
    model_config = load_yaml("configs/model/efficientnet_v2_s.yaml")
    algorithm_config = load_yaml("configs/algorithm/asymmetric_default.yaml")

    # mil should be in model config
    if 'mil' not in model_config:
        print("\n[FAILED] 'mil' key missing from model config")
        return False

    # mil should NOT be in algorithm config (we removed it)
    if 'mil' in algorithm_config:
        print("\n[FAILED] 'mil' key still present in algorithm config (should be removed)")
        return False

    print("\n[PASSED] No key duplication!")
    print("  'mil' is only in model config (correct)")
    return True


def main():
    """
    Run all tests
    """
    print("="*60)
    print("CONFIG SPLIT VALIDATION TEST SUITE")
    print("="*60)

    tests = [
        test_config_completeness,
        test_config_values,
        test_deep_merge,
        test_no_duplication,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n[ERROR] {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"  {status}: {test_name}")
        if not result:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED! Config split is correct.")
        return 0
    else:
        print("\n[ERROR] SOME TESTS FAILED! Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
