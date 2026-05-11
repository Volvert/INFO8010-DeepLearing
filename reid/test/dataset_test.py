"""
Tests for VehicleReIDDataset and MergedDataset.

Uses a MockDataset to avoid requiring real XML files or images on disk.
MockDataset has the same interface as VehicleReIDDataset — MergedDataset
sees no difference.

Run: pytest test/dataset_test.py -v
"""

import pytest
import torch
from torch.utils.data import Dataset
from data.dataset import MergedDataset
from data.batch   import PKSampler


# =============================================================================
# MockDataset
# =============================================================================

class MockDataset(Dataset):
    """
    Minimal stand-in for VehicleReIDDataset.
    Exposes .labels and .samples without touching the filesystem.

    Args:
        n_ids     : number of unique vehicle identities
        k_per_id  : number of images per identity
        id_offset : added to every vehicle_id (simulates MergedDataset offset)
    """

    def __init__(self, n_ids: int, k_per_id: int, id_offset: int = 0):
        self.labels:  list[int]                    = []
        self.samples: list[tuple[str, int, int]]   = []

        for vid in range(1, n_ids + 1):
            label = vid + id_offset
            for _ in range(k_per_id):
                self.labels.append(label)
                self.samples.append((f"img_{label}.jpg", label, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, vid, cid = self.samples[idx]
        return torch.zeros(3, 224, 224), vid, cid


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def real_ds():
    """10 identities × 4 images = 40 samples, IDs 1–10."""
    return MockDataset(n_ids=10, k_per_id=4, id_offset=0)


@pytest.fixture
def synthetic_ds(real_ds):
    """5 identities × 4 images = 20 samples, IDs offset above real."""
    offset = max(real_ds.labels)   # 10 → synthetic IDs = 11–15
    return MockDataset(n_ids=5, k_per_id=4, id_offset=offset)


@pytest.fixture
def merged(real_ds, synthetic_ds):
    return MergedDataset(real_ds, synthetic_ds)


# =============================================================================
# Test 1 — total length
# =============================================================================

def test_merged_len(merged, real_ds, synthetic_ds):
    """
    len(merged) must equal len(real) + len(synthetic).
    If wrong → DataLoader drops or duplicates samples silently.
    """
    assert len(merged) == len(real_ds) + len(synthetic_ds)


# =============================================================================
# Test 2 — labels list length
# =============================================================================

def test_merged_labels_length(merged):
    """
    merged.labels must have one entry per image.
    PKSampler indexes into this list — wrong length = wrong batches.
    """
    assert len(merged.labels) == len(merged)


# =============================================================================
# Test 3 — no ID collision
# =============================================================================

def test_no_id_collision(merged):
    """
    Real and synthetic IDs must be completely disjoint after offset.
    A collision means the loss treats a real and synthetic vehicle
    as the same identity → corrupted metric learning.
    """
    real_ids      = set(merged.real.labels)
    synthetic_ids = set(merged.synthetic.labels)
    collision     = real_ids & synthetic_ids

    assert len(collision) == 0, (
        f"ID collision detected: {collision}"
    )


# =============================================================================
# Test 4 — synthetic IDs above max real ID
# =============================================================================

def test_synthetic_ids_offset_correctly(merged):
    """
    min(synthetic IDs) must be strictly above max(real IDs).
    Validates the offset direction — not just that they don't collide.
    """
    max_real  = max(merged.real.labels)
    min_synth = min(merged.synthetic.labels)

    assert min_synth > max_real, (
        f"synthetic min ID {min_synth} <= real max ID {max_real} "
        f"— offset may be wrong"
    )


# =============================================================================
# Test 5 — __getitem__ routing
# =============================================================================

def test_getitem_routes_to_real(merged, real_ds):
    """
    Indices 0 … n_real-1 must come from the real dataset.
    Checks that the routing boundary is correct.
    """
    _, vid, _ = merged[0]
    _, real_vid, _ = real_ds[0]
    assert vid == real_vid


def test_getitem_routes_to_synthetic(merged, real_ds, synthetic_ds):
    """
    Index n_real must route to the first synthetic sample.
    """
    n_real          = len(real_ds)
    _, merged_vid, _ = merged[n_real]
    _, synth_vid, _  = synthetic_ds[0]
    assert merged_vid == synth_vid


# =============================================================================
# Test 6 — PKSampler compatibility
# =============================================================================

def test_pksampler_compatible(merged):
    """
    PKSampler must initialize and yield indices without error.
    This validates the full interface: merged.labels is a valid list
    of integers that PKSampler can group by identity.
    """
    P, K   = 5, 4   # 5 identities × 4 images = 20 per batch
    sampler = PKSampler(merged.labels, P=P, K=K)

    # must be iterable and yield at least one batch
    indices = list(sampler)
    assert len(indices) > 0
    assert len(indices) % (P * K) == 0, (
        f"sampler yielded {len(indices)} indices — not a multiple of P×K={P*K}"
    )