"""
Utils Module - Common utilities for the project.

Recent Updates:
    - [2026-01-27] New: class_mapping for unified class name loading from config
    - [2026-01-15] New: io_utils for unified image loading (eliminates code duplication)
    - [2026-01-13] Refactor: Extracted scheduler module from trainer

Modules:
    - config_io: YAML configuration loading with Hydra-style defaults
    - device: Device management (CUDA/MPS/CPU)
    - metrics: Evaluation metrics computation
    - resize_utils: Image resizing with aspect ratio preservation
    - cam_utils: CAM visualization (EigenCAM, GradCAM++)
    - local_logger: Local training logger (WandB-free)
    - builder: Optimizer/Scheduler factory
    - scheduler: Custom LR schedulers
    - io_utils: Image I/O utilities
    - class_mapping: Unified class mapping utilities (NEW)
    - visualization: Visualization tools
"""

# File I/O
from src.utils.io_utils import load_and_preprocess_image, check_file_exists

# Class mapping
from src.utils.class_mapping import (
    get_class_order,
    get_display_names,
    get_short_names,
    get_num_classes,
    build_class_map_for_inference,
)

__all__ = [
    'load_and_preprocess_image',
    'check_file_exists',
    'get_class_order',
    'get_display_names',
    'get_short_names',
    'get_num_classes',
    'build_class_map_for_inference',
]
