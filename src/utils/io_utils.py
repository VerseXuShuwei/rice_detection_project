"""
I/O Utilities for Image and File Operations.

Recent Updates:
    - [2026-01-15] New: Unified image loading/preprocessing to eliminate code duplication

Key Features:
    - Standardized image loading with error handling
    - Resize and preprocessing pipeline
    - Defensive file existence checks

Usage:
    >>> from src.utils.io_utils import load_and_preprocess_image
    >>> img = load_and_preprocess_image('path/to/image.jpg', target_size=384)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.resize_utils import resize_keep_aspect_ratio_crop


def load_and_preprocess_image(
    img_path: str,
    target_size: Optional[int] = None,
    color_mode: str = 'BGR'
) -> np.ndarray:
    """
    Load image and optionally resize with aspect ratio preservation.

    Unified interface to replace scattered cv2.imread + resize_keep_aspect_ratio_crop
    calls across Evaluators.

    Args:
        img_path: Path to image file
        target_size: Target size for resize (if None, return original)
                     Resize maintains aspect ratio and center crops
        color_mode: 'BGR' (OpenCV default) or 'RGB'

    Returns:
        img: Preprocessed image (H, W, C) as np.ndarray

    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image loading fails

    Example:
        >>> img = load_and_preprocess_image('image.jpg', target_size=384)
        >>> img.shape  # (384, 384, 3)
    """
    # Defensive file check
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load image
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Failed to load image: {img_path} "
                        f"(possibly corrupted or unsupported format)")

    # Optional color conversion
    if color_mode.upper() == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional resize
    if target_size is not None:
        img = resize_keep_aspect_ratio_crop(img, target_size)

    return img


def check_file_exists(file_path: str, error_msg: Optional[str] = None) -> bool:
    """
    Check if file exists with optional custom error message.

    Args:
        file_path: Path to check
        error_msg: Custom error message (if None, just return bool)

    Returns:
        True if exists

    Raises:
        FileNotFoundError: If error_msg provided and file not found
    """
    exists = Path(file_path).exists()

    if not exists and error_msg is not None:
        raise FileNotFoundError(f"{error_msg}: {file_path}")

    return exists


__all__ = [
    'load_and_preprocess_image',
    'check_file_exists'
]
