# =============================================================================
# PatchEmbedding
# =============================================================================
"""
This file implements the patch embedding module — the entry point of the ViT.

A Transformer expects a sequence of vectors as input.
An image is a 2D grid — not a sequence of token. PatchEmbedding converts it.

The image is split into N = (H*W) / P² non-overlapping patches.
Each patch (P×P pixels, 3 channels) is linearly projected to d_model dimensions.

  N = (224 × 224) / 16² = 196 patches per image
  each patch : 3×16×16 = 768 raw values -> projected to 192 (d_model)

A single Conv2d(kernel=P, stride=P) performs both operations in one GPU pass:
  - kernel=16 covers exactly one patch
  - stride=16 moves by one patch — no overlap, no gap
  - out_channels=192 is the learned linear projection

Input  : (B, 3, 224, 224)
Output : (B, 196, 192)   — sequence of 196 tokens ready for the Transformer

see: https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv2d.html
See: Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020
See: lec7 pages 51-53
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits a batch of images into non-overlapping patches and projects
    each patch to a d_model-dimensional vector.

    Input  : (B, C, H, W)              e.g. (64, 3, 224, 224)
    Output : (B, num_patches, d_model)  e.g. (64, 196, 192)

    Attributes:
        num_patches : int       — total number of patches per image (H*W / P²)
        proj        : nn.Conv2d — splits and projects in one operation
    """

    def __init__(
        self,
        img_size: int = 224,  # height and width of the input image
        patch_size: int = 16, # height and width of each patch
        in_channels: int = 3, # RGB
        d_model: int = 192,   # output dimension per token (ViT-Tiny)
    ):
        super().__init__()

        # =====================================================================
        # num_patches
        # =====================================================================
        """
        Total number of patches per image.
        224 // 16 = 14 patches per side -> 14 × 14 = 196 patches total.
        Stored as an attribute — used by vit.py to size the positional embedding.
        """
        self.num_patches: int = (img_size // patch_size) ** 2

        # =====================================================================
        # proj — Conv2d patch splitter + linear projection
        # =====================================================================
        """
        A single Conv2d with kernel=patch_size and stride=patch_size:
          in_channels  = C = 3          (RGB input)
          out_channels = d_model = 192  (learned projection to Transformer width)
          kernel_size  = patch_size     (covers exactly one patch)
          stride       = patch_size     (jumps by one patch — no overlap)

        Why kernel == stride ?
          If stride < kernel -> patches overlap -> same pixel seen multiple times
          If stride > kernel -> gaps between patches -> pixels lost
          If stride == kernel -> perfect tiling, each pixel in exactly one patch

        This is mathematically equivalent to:
          1. extract each 16×16×3 patch manually
          2. flatten to 768-d vector
          3. multiply by a learned weight matrix W ∈ R^{768 × 192}
        But Conv2d does it in one fused GPU operation.
        """
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) — batch of images

        Returns:
            tokens : (B, num_patches, d_model) — sequence of patch embeddings
        """

        # =====================================================================
        # 1. Conv2d projection
        # =====================================================================
        """
        Each 16×16 patch is projected to a 192-d vector.
        The spatial output is a 14×14 grid of 192-d vectors.
        One grid cell = one patch token.
        """
        x = self.proj(x) # (B, 3, 224, 224) -> (B, 192, 14, 14)

        # =====================================================================
        # 2. Flatten spatial dimensions
        # =====================================================================
        """
        The 14×14 spatial grid is flattened into a sequence of 196 positions.
        We flatten only the last two dimensions (H/P and W/P) — not the batch
        dimension nor the channel dimension.
        start_dim=2 means : keep dim 0 (batch) and dim 1 (channels), flatten the rest.
        """

        x = x.flatten(start_dim=2) # (B, 192, 14, 14) -> (B, 192, 196)

        # =====================================================================
        # 3. Transpose to Transformer convention
        # =====================================================================
        """
        Transformer expects (batch, sequence_length, features).
        After flatten we have (batch, features, sequence_length) — wrong order.
        Transpose dims 1 and 2 to swap features and sequence_length.
        """
        x = x.transpose(1,2) # (B, 192, 196) -> (B, 196, 192)

        return x