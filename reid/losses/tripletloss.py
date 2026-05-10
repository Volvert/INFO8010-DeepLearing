# =============================================================================
# BatchHardTripletLoss
# =============================================================================
"""
This file implements the batch-hard triplet loss used to train the ViT.

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
  7. mean over batch     : scalar loss

nn.TripletMarginLoss is not used — it requires pre-formed triplets and does
not support batch-hard mining. Steps 1-5 are implemented manually.

See: Hermans et al., "In Defense of the Triplet Loss", 2017 https://arxiv.org/abs/1703.07737
See: data/batch.py — PKSampler
"""

import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Step 1: pairwise distances (B, B)
        dists = torch.cdist(embeddings, embeddings)

        # Step 2: positive mask (B, B)
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask.fill_diagonal_(False)  # exclude self-comparisons

        # Step 3: negative mask (B, B)
        neg_mask = ~pos_mask

        # Step 4: hardest positive (B,)
        hardest_pos_dists = torch.where(pos_mask, dists, torch.tensor(float('-inf')).to(dists.device)).max(dim=1).values

        # Step 5: hardest negative (B,)
        hardest_neg_dists = torch.where(neg_mask, dists, torch.tensor(float('inf')).to(dists.device)).min(dim=1).values

        # Step 6: triplet loss (B,)
        triplet_loss = torch.clamp(hardest_pos_dists - hardest_neg_dists + self.margin, min=0)

        # Step 7: mean over batch
        active_fraction = (triplet_loss > 0).float().mean()
        return triplet_loss.mean(), active_fraction