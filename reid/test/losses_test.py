"""
Tests for BatchHardTripletLoss.

Run: pytest test/losses_test.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from losses.tripletloss import BatchHardTripletLoss


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def loss_fn():
    return BatchHardTripletLoss(margin=0.3)


@pytest.fixture
def pk_batch():
    """P=4 identities × K=4 images — standard PK batch."""
    labels     = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    embeddings = F.normalize(torch.randn(16, 128), dim=-1)
    return embeddings, labels


# =============================================================================
# Test 1 — loss >= 0
# =============================================================================

def test_loss_non_negative(loss_fn, pk_batch):
    """
    clamp(min=0) in the formula guarantees loss >= 0.
    If this fails the formula itself is broken.
    """
    embeddings, labels = pk_batch
    loss, _ = loss_fn(embeddings, labels)
    assert loss.item() >= 0.0


# =============================================================================
# Test 2 — active_fraction in [0, 1]
# =============================================================================

def test_active_fraction_in_range(loss_fn, pk_batch):
    """active_fraction is a mean of 0/1 values — must stay in [0, 1]."""
    embeddings, labels = pk_batch
    _, active = loss_fn(embeddings, labels)
    assert 0.0 <= active.item() <= 1.0


# =============================================================================
# Test 3 — outputs are 0-d scalars
# =============================================================================

def test_outputs_are_scalars(loss_fn, pk_batch):
    """
    loss and active must be 0-dimensional tensors.
    optimizer.step() expects a scalar loss — a vector would crash backward().
    """
    embeddings, labels = pk_batch
    loss, active = loss_fn(embeddings, labels)
    assert loss.shape  == torch.Size([])
    assert active.shape == torch.Size([])


# =============================================================================
# Test 4 — regression: neg_mask diagonal bug
# =============================================================================

def test_neg_mask_excludes_diagonal(loss_fn):
    """
    Regression test for the neg_mask diagonal bug.

    Bug: neg_mask = ~pos_mask included the diagonal after fill_diagonal_(False).
    Result: d(i,i) = 0 always selected as hardest negative → active = 1.0 always.
    Fix: neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0) (diagonal naturally False).

    Setup: 4 perfectly orthogonal clusters, 2 images each.
    d(a,p) = 0, d(a,n) >> margin → loss should be 0, active should be 0.
    If the bug were present: d_neg = 0 → loss = margin > 0, active = 1.0.
    """
    embeddings = torch.zeros(8, 128)
    for i in range(8):
        embeddings[i, i % 4] = 1.0   # 4 orthogonal directions
    embeddings = F.normalize(embeddings, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss, active = loss_fn(embeddings, labels)

    assert loss.item() < 0.01, (
        f"loss={loss.item():.4f} — diagonal bug may still be present "
        f"(expected ~0 for perfectly separated embeddings)"
    )
    assert active.item() < 0.01, (
        f"active={active.item():.4f} — expected 0% active for solved embeddings"
    )


# =============================================================================
# Test 5 — larger margin → larger or equal loss
# =============================================================================

def test_larger_margin_increases_loss():
    """
    For the same embeddings, a larger margin forces a harder constraint
    → loss with margin=0.5 >= loss with margin=0.1.
    """
    embeddings = F.normalize(torch.randn(16, 128), dim=-1)
    labels     = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)

    loss_small, _ = BatchHardTripletLoss(margin=0.1)(embeddings, labels)
    loss_large, _ = BatchHardTripletLoss(margin=0.5)(embeddings, labels)

    assert loss_large.item() >= loss_small.item()