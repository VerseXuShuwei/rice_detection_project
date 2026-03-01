"""
Feature-Space Critic for Asymmetric MIL

Recent Updates:
    - [2026-01-29] Feature: Auto-threshold support (reads recommended_threshold from prototypes.pth)
    - [2026-01-14] Refactor: Moved from losses/critics/ to src/critics/
    - [2026-01-14] Major: FeatureCritic now inherits nn.Module with buffer registration
    - [2026-01-13] Initial: Background suppression via prototype similarity

Key Features:
    - Pre-computed background prototypes (K-Means centers)
    - Cosine similarity-based tile filtering
    - Applied in Scout Pass only (no gradient overhead)
    - apply in **both or warm-up phase only**
    - Config-driven initialization
    - Auto-threshold: If runtime.auto_threshold=true, uses recommended_threshold from prototypes.pth

Configuration:
    feature_critic.enable: true
    feature_critic.construction.save_path: "outputs/prototypes/bg_prototypes.pth"
    feature_critic.runtime.threshold: 0.6
    feature_critic.runtime.penalty_scale: 0.5
    feature_critic.runtime.apply_phase: "both"
    feature_critic.runtime.auto_threshold: false  # If true, override threshold with prototypes.pth value
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def apply_feature_repulsion(
    outputs: torch.Tensor,
    features: torch.Tensor,
    prototypes: torch.Tensor,
    threshold: float = 0.6,
    penalty_scale: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Suppress disease class scores for tiles similar to background prototypes.

    Core Logic:
        1. Compute cosine similarity between tile features and ALL prototypes
        2. For each tile, find max similarity to ANY prototype
        3. If max_sim > threshold → tile is "background-like"
        4. Reduce disease class logits (Class 1-9) by penalty
        5. Preserve Class 0 (background class) logits

    Mathematical Formulation:
        Let:
            f_i = feature vector of tile i (normalized)
            P = {p_1, ..., p_K} = prototype set (normalized)
            sim_i = max_k cos(f_i, p_k) = max similarity to any prototype

        Then:
            logits_i[1:] *= (1 - penalty_scale * ReLU(sim_i - threshold))
            logits_i[0] unchanged  # Preserve background class

    Intuition:
        - High similarity to ANY background prototype → likely background tile
        - Reduce its disease scores to prevent Top-1 selection
        - Force model to select tiles UNLIKE background (i.e., disease tiles)

    Args:
        outputs: (B, K, num_classes) - Scout Pass logits
                 num_classes = 10 (Class 0: background, Class 1-9: diseases)
        features: (B, K, D) - Tile features from backbone (before classifier)
        prototypes: (n_clusters, D) - Pre-computed background prototypes
                    Frozen, not trainable
        threshold: Similarity threshold to trigger suppression (default 0.6)
                   Range: [0.5, 0.7] - lower = more aggressive filtering
        penalty_scale: Suppression strength (default 0.5)
                       0.5 = halve logits when sim=1.0
                       1.0 = zero out logits when sim=1.0

    Returns:
        filtered_outputs: (B, K, num_classes) - Adjusted logits
        stats: Dict with diagnostic metrics
            - background_like_ratio: Fraction of tiles above threshold
            - avg_similarity: Mean max similarity across all tiles
            - max_similarity: Highest similarity encountered
    """
    B, K, num_classes = outputs.shape
    D = features.shape[-1]

    # Reshape features for batch processing
    features_flat = features.view(B * K, D)  # (B*K, D)

    # Normalize for cosine similarity
    features_norm = F.normalize(features_flat, p=2, dim=1)  # (B*K, D)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)  # (n_clusters, D)

    # Compute similarity matrix
    # (B*K, D) @ (D, n_clusters) → (B*K, n_clusters)
    similarity_matrix = torch.mm(features_norm, prototypes_norm.T)

    # Find max similarity to ANY prototype for each tile
    max_similarity, _ = torch.max(similarity_matrix, dim=1)  # (B*K,)
    max_similarity = max_similarity.view(B, K)  # (B, K)

    # Compute penalty: ReLU(sim - threshold) to only penalize above threshold
    penalty_amount = (max_similarity - threshold).clamp(min=0)  # (B, K)
    penalty_multiplier = 1.0 - penalty_scale * penalty_amount  # (B, K)
    penalty_multiplier = penalty_multiplier.unsqueeze(-1)  # (B, K, 1) for broadcasting

    # Apply penalty to disease classes ONLY (preserve Class 0)
    filtered_outputs = outputs.clone()
    filtered_outputs[:, :, 1:] = filtered_outputs[:, :, 1:] * penalty_multiplier

    # Compute statistics
    is_background_like = (max_similarity > threshold).float()  # (B, K)
    bg_ratio = is_background_like.mean().item()
    avg_sim = max_similarity.mean().item()
    max_sim = max_similarity.max().item()

    stats = {
        'background_like_ratio': bg_ratio,
        'avg_similarity': avg_sim,
        'max_similarity': max_sim
    }

    return filtered_outputs, stats


class FeatureCritic(nn.Module):
    """
    Feature-Space Critic for Scout Pass filtering.

    Architecture:
        - Inherits nn.Module (compatible with model.to(device))
        - Prototypes stored as buffer (automatically moved with model)
        - Config-driven initialization (reads from YAML)

    Usage (Config-Driven):
        >>> # In Trainer.__init__
        >>> self.feature_critic = FeatureCritic(config)
        >>>
        >>> # In Scout Pass (engines/asymmetric_mil.py)
        >>> if self.feature_critic.loaded and not is_warmup:
        >>>     features = model.extract_features(images)  # (B, K, D)
        >>>     filtered_outputs, stats = self.feature_critic(outputs, features)

    Usage (Manual):
        >>> prototypes = torch.load('outputs/prototypes/background_prototypes.pth')['prototypes']
        >>> critic = FeatureCritic(config=None, prototypes=prototypes, threshold=0.6)
        >>> filtered_outputs, stats = critic(outputs, features)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        prototypes: Optional[torch.Tensor] = None,
        threshold: float = 0.6,
        penalty_scale: float = 0.5
    ):
        """
        Initialize Feature Critic.

        Args:
            config: Configuration dict (if provided, reads paths from config)
            prototypes: (n_clusters, D) - Pre-computed prototypes (if config is None)
            threshold: Similarity threshold (overridden by config if provided)
            penalty_scale: Suppression strength (overridden by config if provided)
        """
        super().__init__()

        # Config-driven initialization
        if config is not None:
            fc_cfg = config.get('feature_critic', {})
            runtime_cfg = fc_cfg.get('runtime', {})
            construction_cfg = fc_cfg.get('construction', {})

            # Override parameters from config
            self.threshold = runtime_cfg.get('threshold', threshold)
            self.penalty_scale = runtime_cfg.get('penalty_scale', penalty_scale)
            self.apply_phase = runtime_cfg.get('apply_phase', 'stable')
            auto_threshold = runtime_cfg.get('auto_threshold', False)

            # Load prototypes from config path
            prototype_path = construction_cfg.get('save_path', None)
            if prototype_path and os.path.exists(prototype_path):
                data = torch.load(prototype_path, map_location='cpu', weights_only=False)
                prototypes = data['prototypes']
                self.loaded = True
                print(f"[FeatureCritic] Loaded {len(prototypes)} prototypes from {prototype_path}")

                # Auto-threshold: override config threshold with recommended value from prototypes.pth
                if auto_threshold:
                    recommended = data.get('recommended_threshold', None)
                    if recommended is not None:
                        old_threshold = self.threshold
                        self.threshold = recommended
                        print(f"[FeatureCritic] Auto-threshold enabled: {old_threshold:.4f} -> {recommended:.4f}")
                    else:
                        print(f"[FeatureCritic] WARNING: auto_threshold=true but prototypes.pth has no recommended_threshold")
                        print(f"[FeatureCritic] Using config threshold: {self.threshold:.4f}")
                        print(f"[FeatureCritic] Rebuild prototypes with latest build_prototypes.py to enable auto-threshold")
            else:
                self.loaded = False
                if prototype_path:
                    print(f"[FeatureCritic] WARNING: Prototype file not found at {prototype_path}")
                    print(f"[FeatureCritic] Feature Critic disabled. Run build_prototypes.py first.")
                prototypes = None
        else:
            # Manual initialization
            self.threshold = threshold
            self.penalty_scale = penalty_scale
            self.apply_phase = 'stable'
            self.loaded = prototypes is not None

        # Register prototypes as buffer (automatically moved with model.to(device))
        if prototypes is not None:
            self.register_buffer('prototypes', prototypes)
        else:
            # Register dummy buffer to avoid errors
            self.register_buffer('prototypes', torch.empty(0))

        # Statistics accumulator
        self.stats_history = []

    def forward(
        self,
        outputs: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply feature repulsion (forward pass).

        Args:
            outputs: (B, K, num_classes)
            features: (B, K, D)

        Returns:
            filtered_outputs: (B, K, num_classes)
            stats: Dict with metrics
        """
        if not self.loaded:
            # Pass-through if critic not loaded
            return outputs, {'background_like_ratio': 0.0, 'avg_similarity': 0.0, 'max_similarity': 0.0}

        filtered_outputs, stats = apply_feature_repulsion(
            outputs, features, self.prototypes,
            self.threshold, self.penalty_scale
        )

        self.stats_history.append(stats)
        return filtered_outputs, stats

    def get_average_stats(self) -> Dict[str, float]:
        """Get average statistics over all calls since last reset."""
        if not self.stats_history:
            return {}

        avg_stats = {}
        keys = self.stats_history[0].keys()
        for key in keys:
            values = [s[key] for s in self.stats_history]
            avg_stats[f'avg_{key}'] = sum(values) / len(values)

        return avg_stats

    def reset_stats(self):
        """Clear statistics history."""
        self.stats_history = []

    def __repr__(self):
        if self.loaded:
            return (f"FeatureCritic(n_prototypes={len(self.prototypes)}, "
                    f"threshold={self.threshold:.2f}, penalty_scale={self.penalty_scale:.2f}, "
                    f"apply_phase='{self.apply_phase}')")
        else:
            return "FeatureCritic(disabled, prototypes not loaded)"
