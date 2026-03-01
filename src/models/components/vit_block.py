"""
ViT Residual Block - Lightweight Vision Transformer with residual connection.

Recent Updates:
    - [2026-02-06] Enhancement: Added learnable 2D positional embedding with interpolation
    - [2026-01-29] Initial: Created for hybrid CNN+ViT architecture

Key Features:
    - Residual connection: y = x + ViT(x) preserves original features
    - Lightweight design: Single transformer layer
    - Learnable 2D positional embedding (supports resolution interpolation)
    - Maintains input dimensions (no shape change)

Mathematical Formulation:
    $$
    z = x_{seq} + PE_{2D}
    y = x + \text{scale} \cdot (\text{MHA}(\text{LN}(z)) + \text{FFN}(\text{LN}(z')))
    $$
    where PE_2D = learnable 2D positional embedding, MHA = Multi-Head Attention

Usage:
    >>> vit_block = ViTResidualBlock(embed_dim=256, num_heads=8, spatial_size=(24, 24))
    >>> x = torch.randn(4, 256, 24, 24)  # (B, C, H, W)
    >>> y = vit_block(x)  # (B, C, H, W) - same shape
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ViTResidualBlock(nn.Module):
    """
    Lightweight ViT block with residual connection and 2D positional embedding.

    Processes spatial feature maps through self-attention while preserving
    the original CNN features via residual connection.

    Args:
        embed_dim: Input/output channel dimension (must match feature dim)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio (hidden_dim = embed_dim * mlp_ratio)
        dropout: Dropout rate for attention and MLP
        spatial_size: (H, W) expected spatial resolution for positional embedding.
            If input resolution differs at runtime, PE is bilinearly interpolated.

    Input:
        x: (B, C, H, W) feature maps from FPN or backbone

    Output:
        y: (B, C, H, W) enhanced feature maps (same shape as input)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        spatial_size: Tuple[int, int] = (24, 24)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.spatial_size = spatial_size

        # Attention capture flag (off during training, on for GUI diagnostics)
        self.store_attention = False
        self.last_attn_weights: Optional[torch.Tensor] = None  # (B, num_heads, N, N)

        # Learnable 2D positional embedding: (1, H*W, C)
        # Initialized from truncated normal for stability
        num_tokens = spatial_size[0] * spatial_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Layer normalization (applied before attention)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Learnable scale for residual (initialized to small value for stability)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        """
        Get positional embedding for given spatial resolution.

        If (H, W) matches spatial_size, returns stored PE directly.
        Otherwise, bilinearly interpolates to match the input resolution.

        Args:
            H: Input height (spatial tokens)
            W: Input width (spatial tokens)

        Returns:
            pos_embed: (1, H*W, C) positional embedding
        """
        train_H, train_W = self.spatial_size
        if H == train_H and W == train_W:
            return self.pos_embed

        # Reshape to 2D grid: (1, train_H*train_W, C) -> (1, C, train_H, train_W)
        pe_2d = self.pos_embed.transpose(1, 2).view(1, self.embed_dim, train_H, train_W)

        # Bilinear interpolation to target resolution
        pe_2d = F.interpolate(pe_2d, size=(H, W), mode='bilinear', align_corners=False)

        # Reshape back: (1, C, H, W) -> (1, H*W, C)
        return pe_2d.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection and positional embedding.

        Args:
            x: (B, C, H, W) input feature maps

        Returns:
            y: (B, C, H, W) output feature maps (y = x + scale * ViT(x))
        """
        B, C, H, W = x.shape

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W

        # Add positional embedding (interpolated if resolution differs)
        x_seq = x_seq + self._get_pos_embed(H, W)

        # Store original for residual
        identity = x_seq

        # Self-attention block
        x_norm = self.norm1(x_seq)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=self.store_attention,
            average_attn_weights=False  # Return per-head weights when stored
        )
        if self.store_attention and attn_weights is not None:
            self.last_attn_weights = attn_weights.detach()  # (B, num_heads, N, N)
        x_seq = x_seq + attn_out

        # MLP block
        x_seq = x_seq + self.mlp(self.norm2(x_seq))

        # Apply scaled residual: y = original + scale * transformer_output
        # This ensures the block starts nearly as identity mapping
        x_seq = identity + self.scale * (x_seq - identity)

        # Reshape back: (B, N, C) -> (B, C, H, W)
        y = x_seq.transpose(1, 2).view(B, C, H, W)

        return y


__all__ = ['ViTResidualBlock']
