"""
Unified Class Mapping Utilities.
统一的类别映射工具，从配置文件加载类别信息。

Recent Updates:
    - [2026-01-27] NEW: Created to eliminate hardcoded class mappings
        - Single source of truth: configs/dataset/rice_data.yaml
        - Backward compatible with legacy hardcoded values

Key Functions:
    - get_class_order(config): Get class folder order (for dataset scanning)
    - get_display_names(config): Get class_id -> display name mapping
    - get_short_names(config): Get class_id -> short name mapping
    - get_num_classes(config): Get number of disease classes (excluding background)

Usage:
    >>> from src.utils.class_mapping import get_class_order, get_display_names
    >>> config = load_config('configs/algorithm/train_topk_asymmetric.yaml')
    >>> class_order = get_class_order(config)  # ['bacterial-leaf-blight', ...]
    >>> display_names = get_display_names(config)  # {0: 'Background', 1: 'Bacterial Leaf Blight', ...}

Configuration (config['classes']):
    - class_order: List of disease folder names (1-indexed in mapping)
    - negative_class_name: Name of background class (always class 0)
    - display_names: Dict[int, str] for visualization
    - short_names: Dict[int, str] for confusion matrix
"""

from typing import Dict, List, Optional


def get_class_order(config: Dict) -> List[str]:
    """
    Get the ordered list of disease class folder names.

    Args:
        config: Configuration dict with 'classes' section

    Returns:
        List of class folder names (e.g., ['bacterial-leaf-blight', 'brown-spot', ...])

    Note:
        This order determines class_id mapping:
        - class_order[0] -> class_id = 1
        - class_order[1] -> class_id = 2
        - etc.
    """
    classes_cfg = config.get('classes', {})
    return classes_cfg.get('class_order', [])


def get_negative_class_name(config: Dict) -> str:
    """
    Get the name of the negative (background) class.

    Args:
        config: Configuration dict

    Returns:
        Negative class name (default: 'rice-healthy')
    """
    classes_cfg = config.get('classes', {})
    # Check both locations for backward compatibility
    return classes_cfg.get(
        'negative_class_name',
        config.get('dataset', {}).get('negative_class_name', 'rice-healthy')
    )


def get_display_names(config: Dict) -> Dict[int, str]:
    """
    Get class_id -> display name mapping for visualization.

    Args:
        config: Configuration dict

    Returns:
        Dict mapping class_id (0, 1, 2, ...) to display name

    Example:
        {0: 'Background', 1: 'Bacterial Leaf Blight', 2: 'Brown Spot', ...}
    """
    classes_cfg = config.get('classes', {})
    display_names = classes_cfg.get('display_names', {})

    # Convert string keys to int (YAML may load as strings)
    return {int(k): v for k, v in display_names.items()}


def get_short_names(config: Dict) -> Dict[int, str]:
    """
    Get class_id -> short name mapping for confusion matrix.

    Args:
        config: Configuration dict

    Returns:
        Dict mapping class_id to short name

    Example:
        {0: 'BG', 1: 'Bact-Leaf-Blight', 2: 'Brown-Spot', ...}
    """
    classes_cfg = config.get('classes', {})
    short_names = classes_cfg.get('short_names', {})

    # Convert string keys to int (YAML may load as strings)
    return {int(k): v for k, v in short_names.items()}


def get_num_classes(config: Dict) -> int:
    """
    Get the number of disease classes (excluding background).

    Args:
        config: Configuration dict

    Returns:
        Number of disease classes (e.g., 9 for 9 diseases)

    Note:
        Total classes including background = num_classes + 1
    """
    class_order = get_class_order(config)
    if class_order:
        return len(class_order)

    # Fallback to num_classes in config
    return config.get('num_classes', config.get('model', {}).get('num_classes', 9))


def get_class_id_to_folder(config: Dict) -> Dict[int, str]:
    """
    Get class_id -> folder name mapping.

    Args:
        config: Configuration dict

    Returns:
        Dict mapping class_id (1, 2, ...) to folder name

    Example:
        {1: 'bacterial-leaf-blight', 2: 'brown-spot', ...}
    """
    class_order = get_class_order(config)
    return {i + 1: name for i, name in enumerate(class_order)}


def get_folder_to_class_id(config: Dict) -> Dict[str, int]:
    """
    Get folder name -> class_id mapping.

    Args:
        config: Configuration dict

    Returns:
        Dict mapping folder name to class_id (1, 2, ...)

    Example:
        {'bacterial-leaf-blight': 1, 'brown-spot': 2, ...}
    """
    class_order = get_class_order(config)
    return {name: i + 1 for i, name in enumerate(class_order)}


def build_class_map_for_inference(config: Dict) -> Dict[int, str]:
    """
    Build class map for inference GUI (includes background as class 0).

    Args:
        config: Configuration dict

    Returns:
        Dict mapping class_id (0, 1, 2, ...) to display name
        Class 0 is always 'Healthy' or 'Background'

    Example:
        {0: 'Healthy', 1: 'Bacterial Leaf Blight', ...}
    """
    display_names = get_display_names(config)

    if display_names:
        # Use config display names, but rename Background to Healthy for inference
        result = display_names.copy()
        if 0 in result and result[0] == 'Background':
            result[0] = 'Healthy'
        return result

    # Fallback: build from class_order
    class_order = get_class_order(config)
    result = {0: 'Healthy'}
    for i, name in enumerate(class_order, start=1):
        # Convert folder name to display name
        display = name.replace('-', ' ').title()
        result[i] = display

    return result


__all__ = [
    'get_class_order',
    'get_negative_class_name',
    'get_display_names',
    'get_short_names',
    'get_num_classes',
    'get_class_id_to_folder',
    'get_folder_to_class_id',
    'build_class_map_for_inference',
]
