"""
Feature Debug Utilities - Component-level feature extraction and PCA visualization.

Recent Updates:
    - [2026-02-18] Initial: Created for GUI Component Diagnostics tab

Key Features:
    - Extract intermediate features from backbone, FPN, ViT, and HeatmapHead
    - PCA-based RGB visualization of high-dimensional feature maps
    - ViT attention map extraction and visualization

Usage:
    from src.inference.feature_debug import extract_component_features, features_to_pca_rgb

    features = extract_component_features(model, tile_tensor, device)
    pca_img = features_to_pca_rgb(features['fpn'])  # (H, W, 3) uint8
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, Optional, Tuple


def extract_component_features(
    model: nn.Module,
    tile_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Extract intermediate feature maps from each model component.

    Runs a single tile through the model pipeline, capturing outputs at each stage:
        backbone (Stage 4) → FPN → ViT → HeatmapHead

    Args:
        model: Trained MIL model (MILEfficientNetV2S)
        tile_tensor: (1, 3, H, W) preprocessed tile tensor
        device: torch device

    Returns:
        Dict with keys:
            'backbone': (C4, h4, w4) numpy array (Stage 4 features, 1792ch)
            'fpn': (C_fpn, h_fpn, w_fpn) numpy array (FPN output, 256ch) or None
            'vit': (C_vit, h_vit, w_vit) numpy array (ViT output, 256ch) or None
            'heatmap': (num_classes, h, w) numpy array (raw heatmap logits)
            'attn_map': (H_spatial, W_spatial) numpy array (averaged attention) or None
    """
    model.eval()
    tile_tensor = tile_tensor.to(device)
    if tile_tensor.dim() == 3:
        tile_tensor = tile_tensor.unsqueeze(0)

    result = {}

    with torch.no_grad():
        # Step 1: Backbone features
        c3, c4 = model._extract_features(tile_tensor)
        result['backbone'] = c4[0].cpu().numpy()  # (C, h, w)

        # Step 2: FPN
        if model.fpn_neck is not None and c3 is not None:
            fpn_out = model.fpn_neck(c3, c4)
            result['fpn'] = fpn_out[0].cpu().numpy()
        else:
            fpn_out = c4
            result['fpn'] = None

        # Step 3: ViT (enable attention capture temporarily)
        if model.vit_block is not None:
            model.vit_block.store_attention = True
            vit_out = model.vit_block(fpn_out)
            result['vit'] = vit_out[0].cpu().numpy()

            # Extract attention map (averaged across heads)
            if model.vit_block.last_attn_weights is not None:
                attn = model.vit_block.last_attn_weights[0]  # (num_heads, N, N)
                attn_avg = attn.mean(dim=0).cpu().numpy()  # (N, N)
                # Average attention received by each token (column mean)
                attn_per_token = attn_avg.mean(axis=0)  # (N,)
                H_s, W_s = model.vit_block.spatial_size
                attn_map = attn_per_token.reshape(H_s, W_s)
                result['attn_map'] = attn_map
            else:
                result['attn_map'] = None

            model.vit_block.store_attention = False
        else:
            vit_out = fpn_out
            result['vit'] = None
            result['attn_map'] = None

        # Step 4: HeatmapHead raw output
        if model.heatmap_head is not None:
            heatmap = model.heatmap_head.get_heatmap(vit_out)
            result['heatmap'] = heatmap[0].cpu().numpy()  # (num_classes, h, w)
        else:
            result['heatmap'] = None

    return result


def features_to_pca_rgb(
    feature_map: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convert high-dimensional feature map to RGB image via PCA(3).

    Projects (C, H, W) feature map to 3 principal components,
    then normalizes to [0, 255] uint8 for visualization.

    Args:
        feature_map: (C, H, W) numpy array
        target_size: Optional (target_h, target_w) to resize output

    Returns:
        (H, W, 3) uint8 RGB image
    """
    C, H, W = feature_map.shape

    # Flatten to (N, C) where N = H*W
    pixels = feature_map.reshape(C, H * W).T  # (N, C)

    # Center the data
    mean = pixels.mean(axis=0, keepdims=True)
    centered = pixels - mean

    # SVD-based PCA (more stable than covariance for high-dim)
    # Only compute top-3 components
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Project to top-3 components
        components = U[:, :3] * S[:3]  # (N, 3)
    except np.linalg.LinAlgError:
        # Fallback: just take first 3 channels
        components = pixels[:, :3] if C >= 3 else np.pad(
            pixels, ((0, 0), (0, 3 - C)), mode='constant'
        )

    # Normalize each component to [0, 255]
    rgb = np.zeros((H * W, 3), dtype=np.float32)
    for i in range(3):
        ch = components[:, i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            rgb[:, i] = (ch - ch_min) / (ch_max - ch_min) * 255
        else:
            rgb[:, i] = 128

    rgb_img = rgb.reshape(H, W, 3).astype(np.uint8)

    if target_size is not None:
        rgb_img = cv2.resize(rgb_img, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_LINEAR)

    return rgb_img


def attention_map_to_overlay(
    attn_map: np.ndarray,
    image_rgb: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay attention map on image.

    Args:
        attn_map: (H_attn, W_attn) attention weights
        image_rgb: (H, W, 3) original image
        alpha: Overlay transparency

    Returns:
        (H, W, 3) blended image
    """
    h, w = image_rgb.shape[:2]

    # Normalize to [0, 1]
    a_min, a_max = attn_map.min(), attn_map.max()
    if a_max - a_min > 1e-8:
        attn_norm = (attn_map - a_min) / (a_max - a_min)
    else:
        attn_norm = np.zeros_like(attn_map)

    # Resize to image dimensions
    attn_resized = cv2.resize(attn_norm.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    attn_uint8 = np.uint8(255 * attn_resized)
    attn_color = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_INFERNO)
    attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_rgb, 1 - alpha, attn_color, alpha, 0)
    return blended


__all__ = [
    'extract_component_features',
    'features_to_pca_rgb',
    'attention_map_to_overlay',
]
