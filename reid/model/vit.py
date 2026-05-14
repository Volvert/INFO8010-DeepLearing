# =====================================================================
# VehicleViT
# =====================================================================
"""
This file is the main model file — the orchestrator of the entire forward pass.

It assembles all sub-modules in order and exposes a single forward() method
that maps a batch of images to a batch of L2-normalized 128-d embeddings.

Role of each sub-module called here:
  PatchEmbedding -> splits images into 196 patch tokens (batch_size, 196, 192)
  cls_token -> learnable aggregation slot prepended to the sequence
  pos_embed -> learnable positional encoding added element-wise
  Transformer -> 6 encoder blocks, each with attention + FFN
  proj_head -> Linear(192->128) + L2 normalize

Full tensor flow:
  (batch_size, 3, 224, 224) input batch
      -> (batch_size, 196, 192) patch tokens patch_embedded.py
      -> (batch_size, 197, 192) + CLS token prepended vit.py
      -> (batch_size, 197, 192) + positional embedding vit.py
      -> (batch_size, 197, 192) after Transformer transformer.py
      -> (batch_size, 192) CLS token extracted vit.py
      -> (batch_size, 128) projected + L2 norm vit.py

For vit classification tuto see: https://medium.com/@bskkim2022/paper-reimplementation-vit-vision-transformer-eed3ad20dfe7
Reduce overfitting with dropout see: https://docs.pytorch.org/docs/2.11/generated/torch.nn.Dropout.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.patch_embedded import PatchEmbedding
from model.transformer import Transformer


class VehicleViT(nn.Module):
    """
    Vision Transformer for vehicle re-identification.

    Maps each input image to a 128-dimensional L2-normalized embedding.
    Same vehicle -> close vectors. Different vehicle -> distant vectors.

    Input  : (batch_size, 3, 224, 224)
    Output : (batch_size, 128) L2-normalized embedding on the unit hypersphere

    Attributes:
        patch_embed : PatchEmbedding — splits image into patch token sequence
        cls_token : nn.Parameter (1, 1, 192) — learnable aggregation token
        pos_embed : nn.Parameter (1, 197, 192) — learned positional encoding
        transformer : Transformer — 6 encoder blocks
        norm : nn.LayerNorm — final normalization before projection
        proj_head : nn.Linear — projects 192-d CLS to 128-d embedding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 192,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        embed_dim: int = 128,
    ):
        super().__init__()

        
        # patch_embed
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )

        
        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        
        # pos_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, d_model))
        
        
        # dropout
        self.pos_drop = nn.Dropout(p=dropout)

        
        # transformer
        self.transformer = Transformer(
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        
        # norm
        self.norm = nn.LayerNorm(d_model)

        
        # proj_head
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

        
        # weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # forward
    """
    Full forward pass from raw image batch to L2-normalized embeddings.

    Step-by-step:
      1. patch_embed : (batch_size, 3, 224, 224) -> (batch_size, 196, 192)
      2. expand CLS : (1, 1, 192) -> (batch_size, 1, 192)
      3. prepend CLS : cat((batch_size,1,192), (batch_size,196,192), dim=1) -> (batch_size, 197, 192)
      4. add pos_embed: (batch_size, 197, 192) + (1, 197, 192) -> (batch_size, 197, 192)
      5. pos_drop : stochastic dropout on token sequence
      6. transformer : (batch_size, 197, 192) -> (batch_size, 197, 192)
      7. extract CLS : x[:, 0, :] -> (batch_size, 192)
      8. norm : LayerNorm -> (batch_size, 192)
      9. proj_head : Linear(192->128) -> (batch_size, 128)
     10. L2 normalize : F.normalize(x, dim=-1) -> (batch_size, 128)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] # batch size — needed to expand cls_token

        # step 1 — patch embedding
        x = self.patch_embed(x)  # (batch_size, 196, 192)

        # step 2 — expand cls_token to batch size
        cls = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, 192)
        # -1 means "keep this dimension unchanged"
        # expand does not copy memory — it creates a view

        # step 3 — prepend CLS token to patch sequence
        x = torch.cat([cls, x], dim=1) # (batch_size, 197, 192)
        # dim=1 is the sequence dimension
        # CLS occupies position 0, patches occupy positions 1-196

        # step 4 — add positional embedding
        x = x + self.pos_embed # (batch_size, 197, 192)
        # pos_embed is (1, 197, 192) — broadcast over batch dimension

        # step 5 — dropout on token sequence
        x = self.pos_drop(x) # (batch_size, 197, 192)

        # step 6 — transformer encoder
        x = self.transformer(x)  # (batch_size, 197, 192)

        # step 7 — extract CLS token (position 0)
        x = x[:, 0, :] # (batch_size, 192)
        # x[:, 0, :] means: all batches, position 0, all features

        # step 8 — layer norm
        x = self.norm(x) # (batch_size, 192)

        # step 9 — projection head
        x = self.proj_head(x) # (batch_size, 128)

        # step 10 — L2 normalize onto unit hypersphere
        x = F.normalize(x, dim=-1) # (batch_size, 128)
        # F.normalize divides each vector by its L2 norm
        # after this step: ||x[i]|| = 1 for all i
        # cosine distance = euclidean distance on the unit hypersphere

        return x