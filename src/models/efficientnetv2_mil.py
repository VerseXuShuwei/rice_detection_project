"""
EfficientNetV2-S for WS-MIL with optional Hybrid Architecture (ViT + FPN).

Recent Updates:
    - [2026-01-29] Major: Added ViT Residual Block, FPN Neck, HeatmapHead (all configurable)
    - [2025-01-19] Major: Dynamic K support - handles variable-length bags (10-30 tiles)
    - [2025-01-19] Core: Instance Score as implicit localization (Ghost Detector)

Key Features:
    - Pure Instance Scorer: Backbone -> [Optional: FPN -> ViT] -> Head -> Logits
    - Configurable architecture via enable flags (can revert to original)
    - Instance Score IS the heatmap (no additional localization module needed)

Architecture Options:
    Original (all enable=false):
        Backbone -> GMP -> Conv1x1 -> Logits

    Hybrid (fpn+vit+heatmap enabled):
        Backbone Stage3/4 -> FPN(S3+S4→256ch, 24×24) -> ViT(256d) -> HeatmapHead -> Logits

WS-MIL Philosophy (Ghost Detector):
    - Axiom: S_bag = max(s_instance_1, ..., s_instance_N)
    - Instance Score physically means: "Confidence that this tile contains disease"
    - At inference: Instance Scores project back to heatmap directly

Configuration:
    model:
      vit_block.enable: false    # Enable ViT residual block
      fpn_neck.enable: false     # Enable FPN neck
      heatmap_head.enable: false # Enable HeatmapHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import timm

from src.core.base_model import BaseModel


class MILEfficientNetV2S(BaseModel):
    """
    Pure WS-MIL Instance Scorer with EfficientNetV2-S backbone.

    Supports optional hybrid architecture with ViT, FPN, and HeatmapHead.
    All optional components can be disabled via config to revert to original.

    Args:
        config: Configuration dict with keys:
            - num_classes: Number of output classes
            - model.freeze_stages: List of stage indices to freeze
            - model.vit_block.enable: Enable ViT residual block
            - model.fpn_neck.enable: Enable FPN neck
            - model.heatmap_head.enable: Enable HeatmapHead
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize WS-MIL Instance Scorer with optional hybrid components."""
        super().__init__(config)

        # Extract model config
        model_cfg = config.get('model', {})
        pretrained = model_cfg.get('pretrained', True)
        dropout = model_cfg.get('dropout', 0.2)
        img_size = model_cfg.get('img_size', 384)

        # Component enable flags (all default to False for backward compatibility)
        # NOTE: vit_block, fpn_neck, heatmap_head are top-level config keys
        # (not nested under 'model:' in YAML), so read from config root
        vit_cfg = config.get('vit_block', {})
        fpn_cfg = config.get('fpn_neck', {})
        heatmap_cfg = config.get('heatmap_head', {})

        self.enable_vit = vit_cfg.get('enable', False)
        self.enable_fpn = fpn_cfg.get('enable', False)
        self.enable_heatmap_head = heatmap_cfg.get('enable', False)

        # Create backbone (EfficientNetV2-S)
        # ALWAYS use standard mode (features_only=False) to preserve full feature depth
        # features_only=True truncates at 272ch, losing conv_head expansion to 1792ch
        # For FPN, we use forward hooks to capture Stage 3 intermediate features
        self.backbone = timm.create_model(
            'efficientnetv2_rw_s',
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
            global_pool='',
            features_only=False
        )

        # Freeze early stages
        freeze_stages = model_cfg.get('freeze_stages', [0, 1, 2])
        if freeze_stages:
            self._freeze_stages(freeze_stages)
            print(f"[MIL] Frozen stages: {freeze_stages}")

        # Determine feature dimensions
        self._setup_feature_dims(img_size)

        # Build optional components
        # Order matters: FPN first (reduces dim), then ViT (operates on FPN output)
        self._build_fpn_neck(fpn_cfg)
        self._build_vit_block(vit_cfg)
        self._build_head(heatmap_cfg)

        self._print_architecture_summary()

    def _setup_feature_dims(self, img_size: int):
        """Determine feature dimensions and spatial sizes from backbone.

        Uses forward hooks to capture Stage 3 (blocks[4]) for FPN,
        while always using full forward_features() for Stage 4 (1792ch).

        Also records spatial resolutions for ViT positional embedding:
            - Stage 3: 24×24 @ 384 input (FPN output resolution)
            - Stage 4: 12×12 @ 384 input
        """
        # Register Stage 3 hook for FPN (blocks[4] = 160ch, 24x24 @ 384 input)
        self._stage3_feat = None
        self._stage3_hook_handle = None
        if self.enable_fpn:
            self._stage3_hook_handle = self.backbone.blocks[4].register_forward_hook(
                self._capture_stage3
            )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            feature_map = self.backbone.forward_features(dummy_input)
            self.stage4_channels = feature_map.shape[1]  # 1792
            stage4_spatial = tuple(feature_map.shape[2:])  # (12, 12)

            if self.enable_fpn:
                assert self._stage3_feat is not None, \
                    "Stage 3 hook failed — check backbone.blocks[4] exists"
                self.stage3_channels = self._stage3_feat.shape[1]  # 160
                stage3_spatial = tuple(self._stage3_feat.shape[2:])  # (24, 24)
                print(f"[MIL] Stage 3: {self.stage3_channels}ch @ {stage3_spatial}")
                print(f"[MIL] Stage 4: {self.stage4_channels}ch @ {stage4_spatial}")
                self._stage3_feat = None  # Clear dummy
            else:
                self.stage3_channels = None
                stage3_spatial = None

        self.feature_dim = self.stage4_channels

        # ViT spatial size: FPN outputs at Stage 3 resolution, else Stage 4
        if self.enable_fpn and stage3_spatial is not None:
            self._vit_spatial_size = stage3_spatial
        else:
            self._vit_spatial_size = stage4_spatial

    def _capture_stage3(self, module, input, output):
        """Forward hook to capture Stage 3 features for FPN."""
        self._stage3_feat = output

    def remove_hooks(self):
        """Remove all registered forward hooks to prevent memory leaks.

        Call this when the model is no longer needed (e.g., before deletion)
        or when switching to a mode that doesn't require hooks (e.g., ONNX export).
        """
        if hasattr(self, '_stage3_hook_handle') and self._stage3_hook_handle is not None:
            self._stage3_hook_handle.remove()
            self._stage3_hook_handle = None

    def _build_vit_block(self, vit_cfg: Dict):
        """Build optional ViT residual block.

        ViT operates AFTER FPN (if enabled), so embed_dim = fpn_out_channels.
        Pipeline: Backbone → FPN(S3+S4→256, 24×24) → ViT(256d, 24×24) → Head

        Spatial size for positional embedding:
            - With FPN: Stage 3 resolution (24×24 @ 384 input)
            - Without FPN: Stage 4 resolution (12×12 @ 384 input)
        """
        if not self.enable_vit:
            self.vit_block = None
            return

        from src.models.components.vit_block import ViTResidualBlock

        # ViT input dim = FPN output (256) if FPN enabled, else Stage 4 (1792)
        vit_embed_dim = self.fpn_out_channels if self.enable_fpn else self.stage4_channels

        # Spatial size for positional embedding
        # FPN outputs at Stage 3 resolution; without FPN, ViT sees Stage 4 resolution
        vit_spatial = self._vit_spatial_size

        self.vit_block = ViTResidualBlock(
            embed_dim=vit_embed_dim,
            num_heads=vit_cfg.get('num_heads', 8),
            mlp_ratio=vit_cfg.get('mlp_ratio', 4.0),
            dropout=vit_cfg.get('dropout', 0.1),
            spatial_size=vit_spatial
        )

    def _build_fpn_neck(self, fpn_cfg: Dict):
        """Build optional FPN neck."""
        if not self.enable_fpn:
            self.fpn_neck = None
            self.fpn_out_channels = self.stage4_channels
            return

        from src.models.components.fpn_neck import FPNNeck

        self.fpn_out_channels = fpn_cfg.get('out_channels', 256)
        self.fpn_neck = FPNNeck(
            in_channels_s3=self.stage3_channels,
            in_channels_s4=self.stage4_channels,
            out_channels=self.fpn_out_channels
        )

    def _build_head(self, heatmap_cfg: Dict):
        """Build classification head (original or HeatmapHead)."""
        self.num_output_classes = self.num_classes + 1  # +1 for Class 0

        # Determine input channels for head
        if self.enable_fpn:
            head_in_channels = self.fpn_out_channels
        else:
            head_in_channels = self.stage4_channels

        if self.enable_heatmap_head:
            from src.models.components.heatmap_head import HeatmapHead

            self.heatmap_head = HeatmapHead(
                in_channels=head_in_channels,
                num_classes=self.num_output_classes,
                pool_mode=heatmap_cfg.get('pool_mode', 'gmp'),
                topk_pool_k=heatmap_cfg.get('topk_pool_k', 3),
            )
            self.instance_classifier = None  # Not used when heatmap_head enabled
        else:
            # Original architecture: GMP -> Conv1x1
            self.heatmap_head = None
            self.instance_classifier = nn.Conv2d(
                in_channels=head_in_channels,
                out_channels=self.num_output_classes,
                kernel_size=1,
                bias=True
            )

    def _print_architecture_summary(self):
        """Print architecture configuration summary."""
        print("=" * 60)
        print("[MIL] Model Architecture Configuration")
        print("=" * 60)

        # Backbone info
        print(f"[MIL] Backbone: EfficientNetV2-RW-S (Stage4={self.stage4_channels}ch)")

        # FPN Neck status (listed first — applied before ViT)
        if self.enable_fpn:
            print(f"[MIL] FPN Neck: ENABLED (S3={self.stage3_channels}ch + S4={self.stage4_channels}ch -> {self.fpn_out_channels}ch)")
        else:
            print("[MIL] FPN Neck: DISABLED")

        # ViT Block status (operates on FPN output if enabled)
        if self.enable_vit:
            vit_dim = self.fpn_out_channels if self.enable_fpn else self.stage4_channels
            print(f"[MIL] ViT Block: ENABLED ({vit_dim}d, {self.vit_block.num_heads}heads)")
        else:
            print("[MIL] ViT Block: DISABLED")

        # Classification Head status
        if self.enable_heatmap_head:
            print(f"[MIL] Head: HeatmapHead (Conv1x1 -> GMP)")
        else:
            print(f"[MIL] Head: Original (GMP -> Conv1x1)")

        # Pipeline summary (correct order: FPN before ViT)
        parts = ["Backbone"]
        if self.enable_fpn:
            parts.append(f"FPN({self.fpn_out_channels}ch)")
        if self.enable_vit:
            parts.append("ViT")
        if self.enable_heatmap_head:
            parts.append("HeatmapHead")
        else:
            parts.append("GMP+Conv1x1")
        print(f"[MIL] Pipeline: {' -> '.join(parts)}")
        print(f"[MIL] Output: {self.num_output_classes} classes (Class 0-{self.num_classes})")
        print("=" * 60)

    def _freeze_stages(self, stages: List[int]):
        """Freeze specified stages of the backbone."""
        for stage_idx in stages:
            for name, param in self.backbone.named_parameters():
                if f'blocks.{stage_idx}.' in name:
                    param.requires_grad = False

    def _extract_features(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Extract features from backbone.

        Always uses forward_features() for full 1792ch Stage 4.
        When FPN is enabled, Stage 3 (160ch) is captured via forward hook.

        Returns:
            (stage3_features, stage4_features) if FPN enabled
            (None, stage4_features) otherwise
        """
        self._stage3_feat = None
        c4 = self.backbone.forward_features(x)

        if self.enable_fpn:
            c3 = self._stage3_feat
            self._stage3_feat = None  # Release reference
            return c3, c4
        else:
            return None, c4

    def _apply_hybrid_components(
        self,
        c3: Optional[torch.Tensor],
        c4: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply optional FPN and ViT components.

        Pipeline order: FPN first (fuse + reduce dim), then ViT (self-attention).
        Backbone(1792ch) → FPN(S3+S4→256ch, 24×24) → ViT(256d) → Head

        Args:
            c3: Stage 3 features (None if FPN disabled)
            c4: Stage 4 features (1792ch)

        Returns:
            Processed features ready for head
        """
        # Step 1: FPN fuses Stage3+Stage4, reduces to 256ch at Stage3 resolution
        if self.fpn_neck is not None and c3 is not None:
            features = self.fpn_neck(c3, c4)
        else:
            features = c4

        # Step 2: ViT self-attention on FPN output (256ch, 24×24 = 576 tokens)
        if self.vit_block is not None:
            features = self.vit_block(features)

        return features

    def _compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute instance logits from features.

        Args:
            features: (B, C, H, W) feature maps

        Returns:
            logits: (B, num_output_classes)
        """
        if self.heatmap_head is not None:
            # HeatmapHead: Conv1x1 -> GMP
            logits = self.heatmap_head(features)
        else:
            # Original: GMP -> Conv1x1
            pooled = F.adaptive_max_pool2d(features, 1)
            logits = self.instance_classifier(pooled)
            logits = logits.flatten(1)

        return logits

    def forward(self, x: torch.Tensor, tile_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for WS-MIL training (bag-level prediction).

        Args:
            x: (B, K, C, H, W) bag of tiles
            tile_counts: Optional (B,) actual number of valid tiles per bag

        Returns:
            bag_logits: (B, num_output_classes) bag-level predictions
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,K,C,H,W), got {x.dim()}D with shape {x.shape}")

        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)

        # Extract features
        c3, c4 = self._extract_features(x)

        # Apply hybrid components
        features = self._apply_hybrid_components(c3, c4)

        # Compute instance logits
        instance_logits = self._compute_logits(features)  # (B*K, num_output_classes)

        # Reshape to (B, K, num_output_classes)
        instance_logits = instance_logits.view(B, K, self.num_output_classes)

        # Mask padded tiles (Dynamic K)
        if tile_counts is not None:
            mask = torch.arange(K, device=x.device).unsqueeze(0) < tile_counts.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(instance_logits)
            instance_logits = instance_logits.masked_fill(~mask, float('-inf'))

        # Aggregation: Per-class Max Pooling
        bag_logits, _ = torch.max(instance_logits, dim=1)

        return bag_logits

    def predict_instances(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only method: Get instance-level scores for heatmap generation.

        Args:
            x: (B, K, C, H, W) or (N, C, H, W) input

        Returns:
            instance_logits: (B, K, num_output_classes) or (N, num_output_classes)
        """
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            return_batched = True
        elif x.dim() == 4:
            B, K = 1, x.size(0)
            return_batched = False
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        # Extract and process features
        c3, c4 = self._extract_features(x)
        features = self._apply_hybrid_components(c3, c4)
        instance_logits = self._compute_logits(features)

        if return_batched:
            instance_logits = instance_logits.view(B, K, self.num_output_classes)

        return instance_logits

    def get_spatial_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial heatmap for visualization (requires heatmap_head enabled).

        Args:
            x: (N, C, H, W) input tiles

        Returns:
            heatmap: (N, num_output_classes, h, w) spatial activation maps
        """
        if self.heatmap_head is None:
            raise RuntimeError(
                "get_spatial_heatmap requires heatmap_head.enable=true in config"
            )

        c3, c4 = self._extract_features(x)
        features = self._apply_hybrid_components(c3, c4)
        heatmap = self.heatmap_head.get_heatmap(features)

        return heatmap

    def extract_features(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature vectors for Feature Critic.

        This method is specifically designed for Feature Critic compatibility.
        It ALWAYS uses pure CNN features (Stage 4 + GMP), skipping ViT/FPN
        to match the prototype construction pipeline.

        Args:
            bags: (B, K, C, H, W) bag of tiles

        Returns:
            features: (B, K, D) feature vectors (pure CNN space)

        CRITICAL (Vector Space Isomorphism):
            Feature Critic prototypes are built using pure EfficientNetV2-S (frozen CNN).
            This method MUST produce features in the same vector space regardless of
            whether ViT/FPN is enabled in the model architecture.

            Pipeline: Backbone Stage4 -> GMP -> (B, K, D)
            Skipped:  ViT, FPN (these would change the vector space)
        """
        B, K, C, H, W = bags.shape
        x = bags.view(B * K, C, H, W)

        # Extract Stage 4 features ONLY (skip FPN path)
        # _extract_features returns (c3, c4), we only need c4
        _, c4 = self._extract_features(x)

        # CRITICAL: Do NOT apply ViT here, even if enabled
        # ViT would transform features to a different vector space
        # Prototype construction uses pure CNN, so we must match that

        # Global Max Pooling (same as build_prototypes.py)
        features = F.adaptive_max_pool2d(c4, 1).flatten(1)  # (B*K, D)

        D = features.shape[1]
        features = features.view(B, K, D)

        return features

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for analysis/debugging)."""
        _, c4 = self._extract_features(x)
        return c4

    # =========================================================================
    # Full Image Inference API
    # DEPRECATED: inference_sliding_window, stitch_heatmaps, inference_full_image
    # removed in [2026-02-09]. Use src.inference.engine.UnifiedInferenceEngine instead.
    # Model retains predict_instances() and get_spatial_heatmap() as low-level API.
    # =========================================================================


__all__ = ['MILEfficientNetV2S']
