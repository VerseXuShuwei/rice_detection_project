"""
Feature extraction utilities using CLIP and DINOV2.

Provides unified interface for extracting features from images using
pre-trained models. Used by data_cleaning and other modules.

Usage:
    from rice_detection.utils.feature_extractor import CLIPFeatureExtractor, DINOv2FeatureExtractor
    
    # CLIP features
    clip_extractor = CLIPFeatureExtractor(model_name='ViT-B/32')
    features = clip_extractor.extract(image)
    
    # DINOv2 features
    dino_extractor = DINOv2FeatureExtractor(model_name='dinov2_vitb14')
    features = dino_extractor.extract(image)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Optional
from pathlib import Path
import logging

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

# Note: DINOV2 requires manual installation
# pip install git+https://github.com/facebookresearch/dinov2.git
DINOV2_AVAILABLE = False
try:
    # Try to import dinov2 if available
    import dinov2
    DINOV2_AVAILABLE = True
except ImportError:
    logging.warning("DINOV2 not available. Install with: pip install git+https://github.com/facebookresearch/dinov2.git")


logger = logging.getLogger(__name__)


class BaseFeatureExtractor(nn.Module):
    """Base class for feature extractors."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Feature extractor using device: {self.device}")
    
    def extract(self, images: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: Input image(s) as numpy array, tensor, or list
        
        Returns:
            Feature array of shape (N, feature_dim)
        """
        raise NotImplementedError("Subclasses must implement extract()")
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """L2 normalize features."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms


class CLIPFeatureExtractor(BaseFeatureExtractor):
    """
    CLIP-based feature extractor.
    
    Supports multiple CLIP models:
    - ViT-B/32 (default)
    - ViT-B/16
    - ViT-L/14
    - RN50, RN101, etc.
    
    Example:
        >>> extractor = CLIPFeatureExtractor('ViT-B/32')
        >>> features = extractor.extract(image)  # (1, 512)
    """
    
    AVAILABLE_MODELS = {
        'ViT-B/32': 512,
        'ViT-B/16': 512,
        'ViT-L/14': 768,
        'RN50': 1024,
        'RN101': 512,
    }
    
    def __init__(
        self,
        model_name: str = 'ViT-B/32',
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = True
    ):
        """
        Initialize CLIP feature extractor.
        
        Args:
            model_name: CLIP model name
            device: Device to run on
            normalize: Whether to L2-normalize features
        """
        super().__init__(device)
        
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP is not installed. Install with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.model_name = model_name
        self.normalize_features = normalize
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        self.feature_dim = self.AVAILABLE_MODELS.get(model_name, 512)
        logger.info(f"CLIP loaded. Feature dimension: {self.feature_dim}")
    
    @torch.no_grad()
    def extract(
        self,
        images: Union[np.ndarray, torch.Tensor, List],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract CLIP image features.
        
        Args:
            images: Input images (PIL, numpy, or tensor)
            batch_size: Batch size for processing
        
        Returns:
            Features array of shape (N, feature_dim)
        """
        # Handle different input types
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if len(images.shape) == 3:  # Single image
                images = [images]
        
        # Preprocess images
        if not isinstance(images[0], torch.Tensor):
            # Assume PIL or numpy, use CLIP preprocessing
            from PIL import Image
            if isinstance(images[0], np.ndarray):
                images = [Image.fromarray(img) for img in images]
            
            images_tensor = torch.stack([self.preprocess(img) for img in images])
        else:
            images_tensor = images if len(images.shape) == 4 else images.unsqueeze(0)
        
        # Extract features in batches
        features_list = []
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i+batch_size].to(self.device)
            features = self.model.encode_image(batch)
            features_list.append(features.cpu().numpy())
        
        features = np.vstack(features_list)
        
        # Normalize if requested
        if self.normalize_features:
            features = self.normalize(features)
        
        return features
    
    @torch.no_grad()
    def extract_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract CLIP text features.
        
        Args:
            texts: List of text strings
        
        Returns:
            Features array of shape (N, feature_dim)
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = features.cpu().numpy()
        
        if self.normalize_features:
            features = self.normalize(features)
        
        return features


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """
    DINOv2-based feature extractor.
    
    Supports DINOv2 models:
    - dinov2_vits14 (small, 384 dim)
    - dinov2_vitb14 (base, 768 dim) - default
    - dinov2_vitl14 (large, 1024 dim)
    - dinov2_vitg14 (giant, 1536 dim)
    
    Example:
        >>> extractor = DINOv2FeatureExtractor('dinov2_vitb14')
        >>> features = extractor.extract(image)  # (1, 768)
    """
    
    AVAILABLE_MODELS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = True
    ):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name: DINOv2 model name
            device: Device to run on
            normalize: Whether to L2-normalize features
        """
        super().__init__(device)
        
        if not DINOV2_AVAILABLE:
            raise ImportError(
                "DINOv2 is not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/dinov2.git"
            )
        
        self.model_name = model_name
        self.normalize_features = normalize
        
        # Load DINOv2 model
        logger.info(f"Loading DINOv2 model: {model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = self.AVAILABLE_MODELS.get(model_name, 768)
        logger.info(f"DINOv2 loaded. Feature dimension: {self.feature_dim}")
    
    @torch.no_grad()
    def extract(
        self,
        images: Union[np.ndarray, torch.Tensor, List],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract DINOv2 features.
        
        Args:
            images: Input images (numpy or tensor)
            batch_size: Batch size for processing
        
        Returns:
            Features array of shape (N, feature_dim)
        """
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:  # Single image (H, W, C)
                images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)
            elif len(images.shape) == 4:  # Batch (N, H, W, C)
                images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Unsupported image type: {type(images)}")
        
        # Ensure float and normalize to [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # Extract features in batches
        features_list = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            features = self.model(batch)
            features_list.append(features.cpu().numpy())
        
        features = np.vstack(features_list)
        
        # Normalize if requested
        if self.normalize_features:
            features = self.normalize(features)
        
        return features


class FusedFeatureExtractor:
    """
    Fused feature extractor combining CLIP and DINOv2.
    
    Extracts features from both models and combines them with weighted fusion.
    
    Example:
        >>> extractor = FusedFeatureExtractor(
        ...     clip_model='ViT-B/32',
        ...     dinov2_model='dinov2_vitb14',
        ...     clip_weight=0.5
        ... )
        >>> features = extractor.extract(images)  # (N, 512 + 768)
    """
    
    def __init__(
        self,
        clip_model: str = 'ViT-B/32',
        dinov2_model: str = 'dinov2_vitb14',
        clip_weight: float = 0.5,
        device: Optional[Union[str, torch.device]] = None,
        normalize_before_fusion: bool = True,
        normalize_after_fusion: bool = True
    ):
        """
        Initialize fused feature extractor.
        
        Args:
            clip_model: CLIP model name
            dinov2_model: DINOv2 model name
            clip_weight: Weight for CLIP features (DINOv2 weight = 1 - clip_weight)
            device: Device to run on
            normalize_before_fusion: Normalize each feature before fusion
            normalize_after_fusion: Normalize fused features
        """
        self.clip_weight = clip_weight
        self.dinov2_weight = 1.0 - clip_weight
        self.normalize_before = normalize_before_fusion
        self.normalize_after = normalize_after_fusion
        
        # Initialize extractors
        self.clip_extractor = CLIPFeatureExtractor(
            clip_model,
            device=device,
            normalize=normalize_before_fusion
        )
        self.dinov2_extractor = DINOv2FeatureExtractor(
            dinov2_model,
            device=device,
            normalize=normalize_before_fusion
        )
        
        self.feature_dim = (
            self.clip_extractor.feature_dim +
            self.dinov2_extractor.feature_dim
        )
        
        logger.info(
            f"Fused extractor: CLIP {clip_model} + DINOv2 {dinov2_model}, "
            f"weights=({clip_weight:.2f}, {self.dinov2_weight:.2f}), "
            f"total_dim={self.feature_dim}"
        )
    
    def extract(
        self,
        images: Union[np.ndarray, torch.Tensor, List],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract fused features.
        
        Args:
            images: Input images
            batch_size: Batch size for processing
        
        Returns:
            Fused features array of shape (N, clip_dim + dinov2_dim)
        """
        # Extract features from both models
        clip_features = self.clip_extractor.extract(images, batch_size)
        dinov2_features = self.dinov2_extractor.extract(images, batch_size)
        
        # Apply weights
        clip_features_weighted = clip_features * self.clip_weight
        dinov2_features_weighted = dinov2_features * self.dinov2_weight
        
        # Concatenate
        fused_features = np.concatenate(
            [clip_features_weighted, dinov2_features_weighted],
            axis=1
        )
        
        # Normalize after fusion if requested
        if self.normalize_after:
            norms = np.linalg.norm(fused_features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            fused_features = fused_features / norms
        
        return fused_features


# Factory functions for convenience
def create_clip_extractor(model_name: str = 'ViT-B/32', **kwargs) -> CLIPFeatureExtractor:
    """Create CLIP feature extractor."""
    return CLIPFeatureExtractor(model_name, **kwargs)


def create_dinov2_extractor(model_name: str = 'dinov2_vitb14', **kwargs) -> DINOv2FeatureExtractor:
    """Create DINOv2 feature extractor."""
    return DINOv2FeatureExtractor(model_name, **kwargs)


def create_fused_extractor(**kwargs) -> FusedFeatureExtractor:
    """Create fused CLIP + DINOv2 extractor."""
    return FusedFeatureExtractor(**kwargs)

