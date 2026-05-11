# =============================================================================
# EncoderBlock
# =============================================================================
"""
This file implements one encoder block — the repeated unit of the Transformer.
transformer.py stacks 6 identical instances of this block in sequence.

Each block applies two sub-modules, each preceded by a LayerNorm
and wrapped in a skip connection (Pre-Norm variant — lec7 page 39):

  X'  = X  + Attention(LayerNorm(X))    <- cross-token communication
  X'' = X' + FFN(LayerNorm(X'))         <- per-token feature refinement

Skip connections create a gradient highway through all 6 blocks — they
prevent vanishing gradients and allow the model to degrade gracefully
if a sub-module learns nothing useful.

FFN architecture:
  Linear(d_model -> d_model * mlp_ratio)   192 -> 768
  GELU
  Dropout(dropout)
  Linear(d_model * mlp_ratio -> d_model)   768 -> 192
  Dropout(dropout)

A single dropout value is applied consistently to both attention weights
and FFN activations — aligned with the original ViT paper and the YAML config.

Input  : (B, seq_len, d_model)   e.g. (64, 197, 192)
Output : (B, seq_len, d_model)   e.g. (64, 197, 192) — same shape as input

See: lec7 page 39 — Residual connections and layer normalization
See: Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020
See: Vaswani et al., "Attention Is All You Need", 2017
"""

import torch
import torch.nn as nn
from model.attention import MultiHeadSelfAttention


class EncoderBlock(nn.Module):
    """
    One Transformer encoder block — Pre-Norm with skip connections.

    Applies self-attention and FFN sequentially, each preceded by
    LayerNorm and wrapped in a residual skip connection.

    Input  : (B, seq_len, d_model)   e.g. (64, 197, 192)
    Output : (B, seq_len, d_model)   same shape as input

    Attributes:
        norm1 : LayerNorm — normalizes input before attention
        attn  : MultiHeadSelfAttention
        norm2 : LayerNorm — normalizes input before FFN
        mlp   : Sequential — two-layer FFN with GELU and Dropout
    """

    def __init__(
        self,
        d_model:   int   = 192,
        num_heads: int   = 8,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
    ):
        """
        Args:
            d_model   : token dimension throughout the block
            num_heads : number of parallel attention heads
            mlp_ratio : FFN hidden dim = d_model * mlp_ratio  (192 * 4 = 768)
            dropout   : dropout probability applied to attention weights and FFN
                        single value — consistent with YAML cfg["model"]["dropout"]
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, seq_len, d_model) — input token sequence

        Returns:
            x : (B, seq_len, d_model) — updated token sequence, same shape
        """
        x = x + self.attn(self.norm1(x))   # attention sub-layer + skip
        x = x + self.mlp(self.norm2(x))    # FFN sub-layer + skip
        return x