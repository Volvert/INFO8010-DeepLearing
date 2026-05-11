"""
Tests for VehicleViT.

Run: pytest test/model_test.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from model.vit import VehicleViT
from losses.tripletloss import BatchHardTripletLoss


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model():
    """VehicleViT with tiny_vit.yaml defaults."""
    return VehicleViT(
        img_size=224, patch_size=16, in_channels=3,
        d_model=192, depth=6, num_heads=8,
        mlp_ratio=4.0, dropout=0.1, embed_dim=128,
    ).eval()


@pytest.fixture
def batch():
    """Small batch of random images — no real data needed."""
    return torch.randn(4, 3, 224, 224)


# =============================================================================
# Test 1 — output shape
# =============================================================================

def test_output_shape(model, batch):
    """
    Forward pass must return (B, embed_dim) = (4, 128).
    Wrong shape → BatchHardTripletLoss crashes on torch.cdist.
    """
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4, 128), f"expected (4, 128), got {out.shape}"


def test_output_shape_batch_size_1(model):
    """Forward pass must work for batch size 1 (evaluation edge case)."""
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 128)


# =============================================================================
# Test 2 — L2 normalization
# =============================================================================

def test_embeddings_l2_normalized(model, batch):
    """
    VehicleViT applies F.normalize(x, dim=-1) as last step.
    Every embedding must have norm = 1.0 on the unit hypersphere.
    If this fails: cosine distance != euclidean distance → kNN is invalid.
    """
    with torch.no_grad():
        out = model(batch)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), (
        f"embeddings not L2-normalized — norms: {norms.tolist()}"
    )


# =============================================================================
# Test 3 — deterministic in eval mode
# =============================================================================

def test_deterministic_in_eval_mode(model, batch):
    """
    Two forward passes in model.eval() must return identical embeddings.
    Dropout is disabled in eval — if not, query/gallery embeddings
    differ each call and kNN retrieval becomes random.
    """
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    assert torch.allclose(out1, out2, atol=1e-6), (
        "eval mode is not deterministic — dropout may still be active"
    )


# =============================================================================
# Test 4 — no crash in train mode
# =============================================================================

def test_train_mode_forward(model, batch):
    """
    Forward pass in train mode (dropout active) must not crash.
    Output shape must be preserved.
    """
    model.train()
    out = model(batch)
    assert out.shape == (4, 128)


# =============================================================================
# Test 5 — trainable parameters
# =============================================================================

def test_has_trainable_parameters(model):
    """
    Model must have trainable parameters — sanity check that no layers
    were accidentally frozen or the model is not empty.
    ViT-Tiny should have ~5-6M parameters.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 1_000_000, (
        f"suspiciously few trainable parameters: {n_params:,}"
    )


# =============================================================================
# Test 6 — backward pass
# =============================================================================

def test_backward_pass(model, batch):
    """
    Most comprehensive test — validates the full computation graph.
    Runs a real forward + triplet loss + backward pass.
    If any operation in the graph is not differentiable, this will crash.
    """
    model.train()
    labels   = torch.tensor([0, 0, 1, 1])
    loss_fn  = BatchHardTripletLoss(margin=0.3)

    embeddings  = model(batch)
    loss, _     = loss_fn(embeddings, labels)
    loss.backward()

    # at least one parameter must have a non-None gradient after backward
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients computed — backward pass failed"