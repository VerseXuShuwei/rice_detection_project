"""
Config Validation Tool - 检测配置是否被正确读取

问题背景:
    config.get('key', default) 模式会静默失败，导致：
    1. 配置不生效但不报错
    2. 隐蔽的硬编码默认值
    3. 无法确认配置是否真正被读取

用法:
    python scripts/tools/validate_config.py --config configs/algorithm/train_topk_asymmetric.yaml
"""

import argparse
from typing import Dict, Any, List, Tuple
from src.utils.config_io import load_config


# 定义所有 builder 期望读取的配置路径
# 格式: (config_path, expected_type, description)
EXPECTED_CONFIG_KEYS = [
    # trainer_builder.py
    ('training', dict, 'Training configuration'),
    ('asymmetric_mil', dict, 'MIL algorithm configuration'),
    ('dataset', dict, 'Dataset configuration'),
    ('model', dict, 'Model configuration'),
    ('model.name', str, 'Model name'),
    ('loss', dict, 'Loss function configuration'),
    ('loss.type', str, 'Loss function type'),
    ('feature_critic', dict, 'Feature critic configuration'),
    ('classes', dict, 'Class names configuration'),
    ('evaluation', dict, 'Evaluation configuration'),

    # losses/builder.py (for topk_anchored_mil)
    ('loss.top1_ce_weight', (int, float), 'Top-1 CE weight'),
    ('loss.top2k_nr_weight', (int, float), 'Top-2~K NR weight'),
    ('loss.top2k_soft_weight', (int, float), 'Top-2~K soft weight'),
    ('loss.noise_drop_threshold', (int, float), 'Noise drop threshold'),
    ('loss.stable_nr_weight', (int, float), 'Stable NR weight'),
    ('loss.alpha', (int, float), 'Alpha parameter'),
    ('loss.epsilon', (int, float), 'Epsilon parameter'),
    ('loss.dynamic_weight', dict, 'Dynamic weight config'),
    ('loss.dynamic_weight.enable', bool, 'Dynamic weight enable'),
    ('loss.dynamic_weight.warmup_weight', (int, float), 'Warmup CE weight'),
    ('loss.dynamic_weight.stable_weight', (int, float), 'Stable CE weight'),
    ('loss.ranking', dict, 'Ranking loss config'),
    ('loss.ranking.enable', bool, 'Ranking enable'),
    ('loss.stable_gate', dict, 'Stable gate config'),
    ('loss.focal_loss', dict, 'Focal loss config'),
    ('loss.anti_collapse', dict, 'Anti-collapse config'),

    # utils/builder.py
    ('optimizer', dict, 'Optimizer configuration'),
    ('optimizer.name', str, 'Optimizer name'),
    ('optimizer.backbone_lr', (int, float), 'Backbone learning rate'),
    ('optimizer.classifier_lr', (int, float), 'Classifier learning rate'),
    ('optimizer.weight_decay', (int, float), 'Weight decay'),
    ('scheduler', dict, 'Scheduler configuration'),
    ('scheduler.name', str, 'Scheduler name'),
    ('training.num_epochs', int, 'Number of epochs'),

    # asymmetric_mil config
    ('asymmetric_mil.warmup_epochs', int, 'MIL warmup epochs'),
    ('asymmetric_mil.warmup_k', int, 'Warmup K value'),
    ('asymmetric_mil.stable_k', int, 'Stable K value'),
    ('asymmetric_mil.warmup_bags_per_batch', int, 'Warmup bags per batch'),
    ('asymmetric_mil.stable_bags_per_batch', int, 'Stable bags per batch'),
    ('asymmetric_mil.warmup_neg_tiles', int, 'Warmup negative tiles'),
    ('asymmetric_mil.stable_neg_tiles', int, 'Stable negative tiles'),
    ('asymmetric_mil.hard_negative_mining', dict, 'Hard mining config'),
    ('asymmetric_mil.hard_negative_mining.enable', bool, 'Hard mining enable'),
]


def get_nested_value(config: Dict, path: str) -> Tuple[bool, Any]:
    """
    Get nested value from config using dot notation.

    Returns:
        (found, value) - found=True if key exists, value is the actual value
    """
    keys = path.split('.')
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False, None

    return True, current


def validate_config(config: Dict) -> List[Tuple[str, str, str]]:
    """
    Validate config against expected keys.

    Returns:
        List of (path, status, message) tuples
    """
    results = []

    for path, expected_type, description in EXPECTED_CONFIG_KEYS:
        found, value = get_nested_value(config, path)

        if not found:
            results.append((path, 'MISSING', f'{description} - KEY NOT FOUND'))
        elif not isinstance(value, expected_type):
            results.append((path, 'TYPE_ERROR',
                f'{description} - Expected {expected_type}, got {type(value).__name__}: {value}'))
        else:
            results.append((path, 'OK', f'{description} = {value}'))

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate config file')
    parser.add_argument('--config', type=str,
                        default='configs/algorithm/train_topk_asymmetric.yaml',
                        help='Path to config file')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all keys including OK ones')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Config Validation Report")
    print(f"{'='*60}")
    print(f"Config file: {args.config}\n")

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1

    # Validate
    results = validate_config(config)

    # Count
    missing = [r for r in results if r[1] == 'MISSING']
    type_errors = [r for r in results if r[1] == 'TYPE_ERROR']
    ok = [r for r in results if r[1] == 'OK']

    # Print issues
    if missing:
        print(f"\n{'='*60}")
        print(f"MISSING KEYS ({len(missing)})")
        print(f"{'='*60}")
        for path, status, msg in missing:
            print(f"  [MISSING] {path}")
            print(f"            {msg}")

    if type_errors:
        print(f"\n{'='*60}")
        print(f"TYPE ERRORS ({len(type_errors)})")
        print(f"{'='*60}")
        for path, status, msg in type_errors:
            print(f"  [TYPE_ERROR] {path}")
            print(f"               {msg}")

    if args.show_all:
        print(f"\n{'='*60}")
        print(f"OK KEYS ({len(ok)})")
        print(f"{'='*60}")
        for path, status, msg in ok:
            print(f"  [OK] {path} = {msg.split(' = ')[-1]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total keys checked: {len(results)}")
    print(f"  OK:          {len(ok)}")
    print(f"  MISSING:     {len(missing)}")
    print(f"  TYPE_ERROR:  {len(type_errors)}")

    if missing or type_errors:
        print(f"\n  [WARNING] CONFIG HAS ISSUES - Values may be using hardcoded defaults!")
        return 1
    else:
        print(f"\n  [PASS] Config validation passed")
        return 0


if __name__ == '__main__':
    exit(main())
