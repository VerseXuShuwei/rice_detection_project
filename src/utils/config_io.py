"""
Configuration I/O Utilities

Purpose:
    - Load and merge modular YAML configurations
    - Support Hydra-style defaults mechanism
    - Deep dictionary merging (not shallow update)
    - Seed management for reproducibility

Recent Updates:
    - [2026-01-04] REFACTOR: Complete rewrite for modular config support
        - Added load_config() with defaults resolution
        - Added deep_merge_dict() for proper config composition
        - Added seed_everything() for reproducibility

Usage:
    from src.utils.config_io import load_config, seed_everything

    # Load main config (automatically loads and merges all defaults)
    config = load_config("configs/algorithm/train_topk_asymmetric.yaml")

    # Set global seed
    seed_everything(config['training_strategy']['seed'])
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load a single YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary containing YAML content

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config if config is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {filepath}: {e}")


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries (recursive).

    Rules:
        - If both values are dicts, merge recursively
        - Otherwise, update value takes precedence
        - New keys from update are added to base

    Args:
        base: Base dictionary
        update: Dictionary to merge into base

    Returns:
        Merged dictionary (modifies base in-place)

    Example:
        >>> base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        >>> update = {'a': {'b': 10, 'e': 4}, 'f': 5}
        >>> deep_merge_dict(base, update)
        {'a': {'b': 10, 'c': 2, 'e': 4}, 'd': 3, 'f': 5}
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Both are dicts, merge recursively
            deep_merge_dict(base[key], value)
        else:
            # Overwrite with new value
            base[key] = value

    return base


def resolve_defaults(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """
    Resolve 'defaults' field and load/merge referenced configs.

    The 'defaults' field specifies which modular configs to load:
        defaults:
            dataset: rice_data              # → configs/dataset/rice_data.yaml
            model: efficientnet_v2_s        # → configs/model/efficientnet_v2_s.yaml
            trainer: default_schedule       # → configs/trainer/default_schedule.yaml
            algorithm: asymmetric_default   # → configs/algorithm/asymmetric_default.yaml

    Args:
        config: Main config dictionary (may contain 'defaults' key)
        config_dir: Root config directory (e.g., Path("configs"))

    Returns:
        Merged config dictionary
    """
    if 'defaults' not in config:
        # No defaults to resolve, return as-is
        return config

    defaults = config.pop('defaults')  # Remove 'defaults' from final config
    merged_config = {}

    # Load each default config in order
    for module_name, config_name in defaults.items():
        # Construct path: configs/<module>/<config_name>.yaml
        module_path = config_dir / module_name / f"{config_name}.yaml"

        if not module_path.exists():
            print(f"[WARNING] Default config not found: {module_path}, skipping...")
            continue

        # Load module config
        module_config = load_yaml(module_path)

        # Recursively resolve defaults in module config (if any)
        module_config = resolve_defaults(module_config, config_dir)

        # Merge into accumulated config
        merged_config = deep_merge_dict(merged_config, module_config)
        print(f"[CONFIG] Loaded and merged: {module_path.relative_to(config_dir.parent)}")

    # Finally, merge the main config (overrides everything)
    merged_config = deep_merge_dict(merged_config, config)

    return merged_config


def load_config(config_path: str, config_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration file with automatic defaults resolution.

    Workflow:
        1. Load main config file
        2. Check for 'defaults' field
        3. Load and merge all referenced configs
        4. Return final merged config

    Args:
        config_path: Path to main config file (e.g., "configs/algorithm/train_topk_asymmetric.yaml")
        config_root: Root directory for configs (default: auto-detect from config_path)

    Returns:
        Merged configuration dictionary

    Example:
        >>> config = load_config("configs/algorithm/train_topk_asymmetric.yaml")
        [CONFIG] Loaded and merged: configs/dataset/rice_data.yaml
        [CONFIG] Loaded and merged: configs/model/efficientnet_v2_s.yaml
        [CONFIG] Loaded and merged: configs/trainer/default_schedule.yaml
        [CONFIG] Loaded and merged: configs/algorithm/asymmetric_default.yaml
        [CONFIG] Main config loaded: configs/algorithm/train_topk_asymmetric.yaml
    """
    config_path = Path(config_path)

    # Auto-detect config root (assumes structure: <root>/algorithm/xxx.yaml)
    if config_root is None:
        # Go up two levels from config file: algorithm/xxx.yaml -> configs/
        config_root = config_path.parent.parent
    else:
        config_root = Path(config_root)

    # Load main config
    config = load_yaml(config_path)

    # Resolve defaults and merge
    config = resolve_defaults(config, config_root)

    print(f"[CONFIG] Main config loaded: {config_path.relative_to(config_root.parent)}")
    print(f"[CONFIG] Total keys in merged config: {len(config)}")

    return config


def seed_everything(seed: int) -> None:
    """
    Set global random seed for reproducibility.

    Sets seeds for:
        - Python random module
        - NumPy random
        - PyTorch (CPU and CUDA)
        - PyTorch CUDNN (deterministic mode)

    Args:
        seed: Random seed value

    Note:
        - Deterministic CUDNN may impact performance
        - Some operations remain non-deterministic on GPU
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Deterministic mode (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[SEED] Global seed set to {seed} (reproducibility enabled)")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

    print(f"[CONFIG] Saved config to: {save_path}")


def require_config(config: Dict[str, Any], path: str, expected_type: type = None) -> Any:
    """
    Safely get a required config value. Raises error if missing.

    This function should be used instead of config.get() when a value
    MUST exist in the config. It prevents silent failures from default values.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'loss.type', 'optimizer.backbone_lr')
        expected_type: Optional type check (e.g., str, float, dict)

    Returns:
        The config value

    Raises:
        KeyError: If the path does not exist in config
        TypeError: If the value is not of expected_type

    Example:
        >>> loss_type = require_config(config, 'loss.type', str)
        >>> lr = require_config(config, 'optimizer.backbone_lr', float)
    """
    keys = path.split('.')
    current = config

    for i, key in enumerate(keys):
        if not isinstance(current, dict):
            raise KeyError(
                f"Config path '{path}' failed at '{'.'.join(keys[:i+1])}': "
                f"expected dict, got {type(current).__name__}"
            )
        if key not in current:
            raise KeyError(
                f"Required config key '{path}' not found. "
                f"Missing key: '{key}' at level {i+1}. "
                f"Available keys: {list(current.keys())}"
            )
        current = current[key]

    if expected_type is not None and not isinstance(current, expected_type):
        raise TypeError(
            f"Config '{path}' has wrong type. "
            f"Expected {expected_type.__name__}, got {type(current).__name__}: {current}"
        )

    return current


def get_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get a config value with optional default.

    Unlike require_config(), this allows missing values but LOGS a warning
    when using the default. Use this for truly optional parameters.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'loss.epsilon')
        default: Default value if path not found

    Returns:
        The config value or default

    Example:
        >>> epsilon = get_config(config, 'loss.epsilon', 1e-8)
    """
    keys = path.split('.')
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # Log warning when using default
            import warnings
            warnings.warn(
                f"Config key '{path}' not found, using default: {default}",
                UserWarning,
                stacklevel=2
            )
            return default

    return current


# ============================================================
# Testing & Validation
# ============================================================

if __name__ == "__main__":
    # Test config loading
    print("="*60)
    print("Testing Config Loader")
    print("="*60)

    try:
        config = load_config("configs/algorithm/train_topk_asymmetric.yaml")

        print("\n" + "="*60)
        print("Merged Config Summary:")
        print("="*60)
        print(f"Training Strategy: {config.get('training_strategy', {}).get('name', 'N/A')}")
        print(f"Dataset: {config.get('dataset', {}).get('name', 'N/A')}")
        print(f"Model: {config.get('model', {}).get('name', 'N/A')}")
        print(f"Optimizer: {config.get('optimizer', {}).get('name', 'N/A')}")
        print(f"Num Epochs: {config.get('training', {}).get('num_epochs', 'N/A')}")
        print(f"Warmup Epochs (MIL): {config.get('asymmetric_mil', {}).get('warmup_epochs', 'N/A')}")
        print(f"Loss Type: {config.get('loss', {}).get('type', 'N/A')}")

        # Test seed setting
        seed_everything(config['training_strategy']['seed'])

        print("\n[SUCCESS] Config loading test passed!")

    except Exception as e:
        print(f"\n[ERROR] Config loading test failed: {e}")
        import traceback
        traceback.print_exc()
