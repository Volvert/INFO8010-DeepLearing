# =============================================================================
# Transformer
# =============================================================================
"""
This file stacks depth identical EncoderBlock instances in sequence.
Each block applies attention + skip, then FFN + skip.

The sequence shape (B, 197, 192) is preserved through all 6 blocks —
each block enriches the token representations without changing the shape.

Input  : (B, seq_len, d_model)   e.g. (64, 197, 192)
Output : (B, seq_len, d_model)   e.g. (64, 197, 192)

See: model/block.py — EncoderBlock
nn.ModuleList -> https://docs.pytorch.org/docs/2.11/generated/torch.nn.ModuleList.html
nn.LayerNorm -> https://docs.pytorch.org/docs/2.11/generated/torch.nn.LayerNorm.html
nn.GELU -> https://docs.pytorch.org/docs/2.11/generated/torch.nn.GELU.html
"""

import torch.nn as nn
from model.block import EncoderBlock

class Transformer(nn.Module):
    def __init__(self, depth=6, d_model=192, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, mlp_ratio, attn_drop, drop)
            for _ in range(depth)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)  # (B, seq_len, d_model) — same shape through all blocks
        return x