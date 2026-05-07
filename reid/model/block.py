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
  Dropout
  Linear(d_model * mlp_ratio -> d_model)   768 -> 192

Input  : (B, seq_len, d_model)   e.g. (64, 197, 192)
Output : (B, seq_len, d_model)   e.g. (64, 197, 192) — same shape as input

See: lec7 page 39 — Residual connections and layer normalization
See: Vaswani et al., "Attention Is All You Need", 2017
"""

import torch.nn as nn
from model.attention import MultiHeadSelfAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model=192, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(drop)
        )
    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.attn(self.norm1(x))  # skip connection around attention
        x = x + self.mlp(self.norm2(x))   # skip connection around MLP
        return x  # (B, seq_len, d_model) — same shape as input