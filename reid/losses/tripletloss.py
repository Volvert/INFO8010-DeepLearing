# =============================================================================
# BatchHardTripletLoss
# =============================================================================
"""
Implements the batch-hard triplet loss used to train the ViT.

Unlike cross-entropy which learns a fixed set of classes, triplet loss learns
a distance function — a metric space where same-vehicle embeddings cluster
together and different-vehicle embeddings are pushed apart. This is required
because the 440 test identities are never seen during training.

For each anchor in the batch, batch-hard mining selects the hardest triplet:
  hardest positive : the positive farthest from the anchor    max d(a, p)
  hardest negative : the negative closest to the anchor       min d(a, n)

Loss formula (Hermans et al., 2017):
  L = max(0, d(a,p) - d(a,n) + margin)

  loss = 0  ->  d(a,n) - d(a,p) > margin  ->  triplet inactive, no gradient
  loss > 0  ->  d(a,n) - d(a,p) < margin  ->  triplet active, model learns

Forward steps:
  1. pairwise distances  : (B, B) euclidean distance matrix over L2 embeddings
  2. positive mask       : (B, B) True where labels[i] == labels[j], i != j
  3. negative mask       : (B, B) True where labels[i] != labels[j]
  4. hardest positive    : (B,)   max distance per row within positive mask
  5. hardest negative    : (B,)   min distance per row within negative mask
  6. triplet loss        : (B,)   clamp(d_pos - d_neg + margin, min=0)
  7. mean over batch     : scalar loss + active fraction

nn.TripletMarginLoss is not used — it requires pre-formed triplets and does
not support batch-hard mining. Steps 1-5 are implemented manually.

See: Hermans et al., "In Defense of the Triplet Loss", 2017
     https://arxiv.org/abs/1703.07737
See: data/batch.py — PKSampler guarantees P×K batch structure
"""

import torch
import torch.nn as nn


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with euclidean distance on L2-normalized embeddings.

    For each anchor in the batch, mines the hardest positive and hardest
    negative within the batch, then computes the margin-based triplet loss.

    Attributes:
        margin : float — minimum required gap between d(a,n) and d(a,p)
    """

    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin : triplet loss margin — from cfg["training"]["margin"]
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, embed_dim) L2-normalized
        labels:     torch.Tensor,   # (B,)           vehicle_ids
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes batch-hard triplet loss and active triplet fraction.

        Args:
            embeddings : (B, embed_dim) — L2-normalized embedding vectors
            labels     : (B,)           — vehicle identity labels

        Returns:
            loss            : scalar — mean triplet loss over the batch
            active_fraction : scalar — fraction of anchors with loss > 0
        """

        # Step 1 — pairwise euclidean distance matrix (B, B)
        # cdist computes ||emb[i] - emb[j]||_2 for all pairs (i, j)
        # diagonal = 0 (distance from a vector to itself)
        dists = torch.cdist(embeddings, embeddings)   # (B, B)

        # Step 2 — positive mask (B, B)
        # True where labels[i] == labels[j] AND i != j
        # fill_diagonal_(False) excludes self-comparisons (d(i,i) = 0)
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)   # (B, B)
        pos_mask.fill_diagonal_(False)

        # Step 3 — negative mask (B, B)
        # True where labels[i] != labels[j]
        # Using != directly — naturally False on diagonal (labels[i] != labels[i] = False)
        # This avoids the bug of ~pos_mask which sets diagonal to True after
        # fill_diagonal_(False), causing d(i,i)=0 to always be picked as hardest negative
        neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)   # (B, B)

        # Step 4 — hardest positive (B,)
        # For each anchor, find the positive with the LARGEST distance
        # torch.full_like sets non-positive cells to -inf so .max() ignores them
        hardest_pos_dists = torch.where(
            pos_mask,
            dists,
            torch.full_like(dists, float("-inf")),
        ).max(dim=1).values   # (B,)

        # Step 5 — hardest negative (B,)
        # For each anchor, find the negative with the SMALLEST distance
        # torch.full_like sets non-negative cells to +inf so .min() ignores them
        hardest_neg_dists = torch.where(
            neg_mask,
            dists,
            torch.full_like(dists, float("inf")),
        ).min(dim=1).values   # (B,)

        # Step 6 — per-anchor triplet loss (B,)
        # clamp(d_pos - d_neg + margin, min=0)
        # = 0 when d_neg - d_pos > margin (satisfied, no gradient)
        # > 0 when d_neg - d_pos < margin (violated, model learns)
        triplet_loss = torch.clamp(
            hardest_pos_dists - hardest_neg_dists + self.margin,
            min=0.0,
        )   # (B,)

        # Step 7 — aggregate
        loss            = triplet_loss.mean()
        active_fraction = (triplet_loss > 0).float().mean()

        return loss, active_fraction