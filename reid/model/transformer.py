# =============================================================================
# Transformer
# =============================================================================
"""
Stacks depth identical EncoderBlock instances in sequence.

Each block applies attention + skip, then FFN + skip.
The sequence shape is preserved through all blocks — each block enriches
the token representations without changing the tensor dimensions.

Input  : (B, seq_len, d_model) -> (64, 197, 192)
Output : (B, seq_len, d_model) -> (64, 197, 192)

See: model/block.py — EncoderBlock
See: nn.ModuleList  — https://docs.pytorch.org/docs/2.11/generated/torch.nn.ModuleList.html
"""

import torch
import torch.nn as nn
from model.block import EncoderBlock


class Transformer(nn.Module):
    """
    Stack of depth EncoderBlock instances.

    nn.ModuleList is used instead of a plain Python list so that PyTorch
    registers all blocks as submodules — their parameters appear in
    model.parameters() and are moved to GPU with model.to(device).

    Args:
        depth : number of stacked EncoderBlocks
        d_model : token dimension throughout all blocks
        num_heads : attention heads per block
        mlp_ratio : FFN hidden dim = d_model * mlp_ratio
        dropout : dropout applied in attention and FFN of every block

    Input : (B, seq_len, d_model)
    Output : (B, seq_len, d_model)
    Attributes:
        blocks : nn.ModuleList of depth EncoderBlock instances
    """

    def __init__(
        self,
        depth: int = 6,
        d_model: int = 192,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
    super().__init__()

    self.blocks = nn.ModuleList([
        EncoderBlock(
            d_model = d_model,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
        )
        for _ in range(depth)
    ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x)
        return x