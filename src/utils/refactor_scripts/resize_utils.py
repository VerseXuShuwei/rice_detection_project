"""
Resize utilities with aspect ratio preservation.

Strategy: Short side scaling + center crop (no padding, no distortion)

Design Principles:
- Keep aspect ratio (no distortion)
- Center crop (preserve important central region)
- No black borders (avoid introducing noise)
- Compatible with both OpenCV and PIL
"""

import cv2
import numpy as np
import math
from PIL import Image
from typing import Union

try:
    import albumentations as A
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False


def resize_keep_aspect_ratio_crop(
    image: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image keeping aspect ratio, then center crop to square.
    
    Strategy:
    1. Scale short side to target_size
    2. Long side scales proportionally
    3. Center crop to target_size × target_size
    
    Args:
        image: Input image (H, W, C)
        target_size: Target square size
        interpolation: OpenCV interpolation method
    
    Returns:
        Resized and center-cropped square image (target_size, target_size, C)
    
    Example:
        >>> img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        >>> resized = resize_keep_aspect_ratio_crop(img, 300)
        >>> resized.shape
        (300, 300, 3)
        >>> # 3000x4000 -> 300x400 -> crop to 300x300 (center)
    
    Note:
        This function includes defensive checks to handle edge cases:
        - Uses math.ceil() to ensure resized dimensions >= target_size
        - Falls back to direct resize if aspect-preserving resize fails
        - Guarantees output shape is always (target_size, target_size, C)
    """
    h, w = image.shape[:2]
    
    # Boundary check: extremely small or invalid images
    if h < 1 or w < 1:
        raise ValueError(f"Invalid image size: {h}x{w}. Image must have at least 1 pixel in each dimension.")
    
    # Calculate scale factor (short side -> target_size)
    scale = target_size / min(h, w)
    
    # Use math.ceil() to ensure resized dimensions >= target_size
    # This prevents issues where int() rounds down and creates images smaller than target
    new_h = max(target_size, int(math.ceil(h * scale)))
    new_w = max(target_size, int(math.ceil(w * scale)))
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Verify resize result
    actual_h, actual_w = resized.shape[:2]
    if actual_h < target_size or actual_w < target_size:
        # Extreme case: resized image is still smaller than target
        # Fallback to direct resize (may introduce slight distortion)
        return cv2.resize(image, (target_size, target_size), interpolation=interpolation)
    
    # Center crop to target_size x target_size (indices are now guaranteed valid)
    crop_h_start = (actual_h - target_size) // 2
    crop_w_start = (actual_w - target_size) // 2
    
    cropped = resized[
        crop_h_start:crop_h_start + target_size,
        crop_w_start:crop_w_start + target_size
    ]
    
    # Final verification: ensure output has correct shape
    if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
        # Fallback: force resize if crop somehow failed
        return cv2.resize(image, (target_size, target_size), interpolation=interpolation)
    
    return cropped


def resize_keep_aspect_ratio_crop_pil(
    image: Image.Image,
    target_size: int,
    resample: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    PIL version for high-quality resize with aspect ratio preservation.
    
    Args:
        image: PIL Image
        target_size: Target square size
        resample: PIL resampling method
    
    Returns:
        Resized and center-cropped PIL Image (target_size × target_size)
    
    Example:
        >>> img = Image.new('RGB', (4000, 3000))
        >>> resized = resize_keep_aspect_ratio_crop_pil(img, 300)
        >>> resized.size
        (300, 300)
    """
    w, h = image.size
    
    # Scale short side to target_size
    scale = target_size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = image.resize((new_w, new_h), resample)
    
    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    cropped = resized.crop((left, top, right, bottom))
    
    return cropped


# Albumentations transform (only available if albumentations is installed)
if _ALBUMENTATIONS_AVAILABLE:
    class ResizeKeepAspectRatioCrop(A.DualTransform):
        """
        Albumentations transform: resize keeping aspect ratio + center crop.
        
        This transform ensures:
        - No distortion (aspect ratio preserved)
        - No black borders (center crop instead of padding)
        - Consistent output size (target_size × target_size)
        
        Usage:
            transform = A.Compose([
                ResizeKeepAspectRatioCrop(size=300),
                A.Normalize(...),
                ToTensorV2()
            ])
        
        Example:
            >>> import albumentations as A
            >>> transform = A.Compose([
            ...     ResizeKeepAspectRatioCrop(size=300),
            ...     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ... ])
            >>> img = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
            >>> transformed = transform(image=img)
            >>> transformed['image'].shape
            (300, 300, 3)
        """
        
        def __init__(
            self,
            size: int,
            interpolation: int = cv2.INTER_LINEAR,
            always_apply: bool = False,
            p: float = 1.0
        ):
            """
            Initialize transform.
            
            Args:
                size: Target square size
                interpolation: OpenCV interpolation method
                always_apply: Whether to always apply the transform
                p: Probability of applying the transform
            """
            super().__init__(always_apply, p)
            self.size = size
            self.interpolation = interpolation
        
        def apply(self, img: np.ndarray, **params) -> np.ndarray:
            """Apply transform to image."""
            return resize_keep_aspect_ratio_crop(
                img, self.size, self.interpolation
            )
        
        def get_transform_init_args_names(self):
            """Get init argument names for serialization."""
            return ("size", "interpolation")

