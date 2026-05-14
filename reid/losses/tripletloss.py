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

Uniformity loss (Wang et al., 2020):
  L_unif = log( mean_{i != j} exp(-2 * ||z_i - z_j||^2) )

  Penalises embeddings that cluster together on the hypersphere.
  Acts as a repulsive force — pushes ALL pairs apart, preventing collapse.
  Combined loss: L_total = L_triplet + lambda_unif * L_unif

  lambda_unif = 0.1 by default — small enough to not override the triplet
  signal but strong enough to prevent collapse.

Forward steps:
  1. pairwise distances  : (B, B) euclidean distance matrix over L2 embeddings
  2. positive mask       : (B, B) True where labels[i] == labels[j], i != j
  3. negative mask       : (B, B) True where labels[i] != labels[j]
  4. hardest positive    : (B,)   max distance per row within positive mask
  5. hardest negative    : (B,)   min distance per row within negative mask
  6. triplet loss        : (B,)   clamp(d_pos - d_neg + margin, min=0)
  7. uniformity loss     : scalar log mean exp(-2 * sq_dists) over all pairs
  8. total loss          : triplet + lambda_unif * uniformity
  9. mean over batch     : scalar loss + active fraction

nn.TripletMarginLoss is not used — it requires pre-formed triplets and does
not support batch-hard mining. Steps 1-5 are implemented manually.

See: Hermans et al., "In Defense of the Triplet Loss", 2017
     https://arxiv.org/abs/1703.07737
See: Wang et al., "Understanding Contrastive Representation Learning", 2020
     https://arxiv.org/abs/2005.10242
See: data/batch.py — PKSampler guarantees P×K batch structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss + uniformity regularization.

    The uniformity loss prevents representation collapse by penalising
    embeddings that converge to the same point on the unit hypersphere.
    It acts as a repulsive force on all pairs, complementing the attractive
    force of the triplet loss on positive pairs.

    Combined loss:
        L_total = L_triplet + lambda_unif * L_unif

    Attributes:
        margin      : float — minimum required gap between d(a,n) and d(a,p)
        lambda_unif : float — weight of the uniformity regularization term
    """

    def __init__(self, margin: float = 0.3, lambda_unif: float = 0.1):
        """
        Args:
            margin      : triplet loss margin — from cfg["training"]["margin"]
            lambda_unif : uniformity loss weight — from cfg["training"]["lambda_unif"]
                          default 0.1 — small enough to not override triplet signal
        """
        super().__init__()
        self.margin      = margin
        self.lambda_unif = lambda_unif

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, embed_dim) L2-normalized
        labels:     torch.Tensor,   # (B,)           vehicle_ids
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes batch-hard triplet loss + uniformity loss and active fraction.

        Args:
            embeddings : (B, embed_dim) — L2-normalized embedding vectors
            labels     : (B,)           — vehicle identity labels

        Returns:
            loss            : scalar — triplet + uniformity loss
            active_fraction : scalar — fraction of anchors with triplet loss > 0
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

        # Step 7 — uniformity loss (scalar)
        # Measures how uniformly embeddings are spread on the hypersphere.
        # sq_dists[i,j] = ||z_i - z_j||^2 — squared euclidean distances
        # exp(-2 * sq_dists) — Gaussian kernel: close pairs → high value
        # log(mean(kernel)) — negative when pairs are spread out (good)
        #                    — near 0 when pairs cluster together (collapse)
        # Upper triangle only — avoids counting each pair twice
        sq_dists  = dists.pow(2)                           # (B, B)
        triu_mask = torch.triu(
            torch.ones(sq_dists.shape, dtype=torch.bool, device=sq_dists.device),
            diagonal=1,
        )                                                  # upper triangle, no diagonal
        sq_upper  = sq_dists[triu_mask]                    # (B*(B-1)/2,)
        unif_loss = torch.log(
            torch.exp(-2.0 * sq_upper).mean()
        )                                                  # scalar, ≤ 0

        # Step 8 — total loss
        # triplet pulls same-identity embeddings together and pushes different ones apart
        # uniformity pushes ALL embeddings apart — prevents collapse to a single point
        loss_triplet = triplet_loss.mean()
        loss_total   = loss_triplet + self.lambda_unif * unif_loss

        # Step 9 — active fraction
        active_fraction = (triplet_loss > 0).float().mean()

        return loss_total, active_fraction