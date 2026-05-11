"""
Integration test — validates the full training pipeline end-to-end.

Uses a minimal synthetic dataset (8 identities × 4 images) to run
2 real training epochs without touching the filesystem or real data.

Output is written to runs/test/ — persists after the test so you can
inspect metrics.csv manually to verify the logger works correctly.

Run: pytest test/pipeline_test.py -v
Expected runtime: ~20 seconds on CPU.
"""

import os
import csv
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from model.vit                  import VehicleViT
from losses.tripletloss         import BatchHardTripletLoss
from engine.train               import train_one_epoch
from monitoring.logger          import Logger
from monitoring.gradient_health import GradientHealthMonitor
from monitoring.triplet_health  import TripletHealthMonitor
from data.batch                 import PKSampler


# =============================================================================
# Minimal synthetic dataset
# =============================================================================

class TinyDataset(Dataset):
    """
    8 identities × 4 images = 32 samples.
    Random tensors — no filesystem access needed.
    Exposes .labels for PKSampler compatibility.
    """
    def __init__(self, n_ids: int = 8, k: int = 4):
        self.labels  = [vid for vid in range(n_ids) for _ in range(k)]
        self.samples = self.labels[:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return torch.randn(3, 224, 224), self.samples[idx], 0


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_model(device):
    return VehicleViT(
        img_size=224, patch_size=16, in_channels=3,
        d_model=192, depth=6, num_heads=8,
        mlp_ratio=4.0, dropout=0.1, embed_dim=128,
    ).to(device)


@pytest.fixture
def tiny_loader():
    """P=4 identities × K=4 images = batch of 16."""
    dataset = TinyDataset(n_ids=8, k=4)
    sampler = PKSampler(dataset.labels, P=4, K=4)
    return DataLoader(dataset, batch_size=16, sampler=sampler, drop_last=True)


@pytest.fixture
def run_dir():
    """
    Fixed output directory — persists after the test so you can
    inspect runs/test/pipeline_test/metrics.csv manually.
    Add runs/test/ to .gitignore.
    """
    path = os.path.join("runs", "test", "pipeline_test")
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# Test 1 — one epoch without crash
# =============================================================================

def test_one_epoch_no_crash(tiny_model, tiny_loader, device):
    """
    Full forward → loss → backward → optimizer step for one epoch.
    No monitoring. If any part of the computation graph is broken, crashes.
    """
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    loss_fn   = BatchHardTripletLoss(margin=0.3)

    metrics = train_one_epoch(
        model=tiny_model, dataloader=tiny_loader,
        loss_fn=loss_fn, optimizer=optimizer, device=device,
    )

    assert "loss"            in metrics
    assert "active_triplets" in metrics
    assert "lr"              in metrics


# =============================================================================
# Test 2 — metrics are in valid ranges
# =============================================================================

def test_metrics_valid_ranges(tiny_model, tiny_loader, device):
    """loss >= 0, active in [0,1], lr > 0. NaN = something broken upstream."""
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    loss_fn   = BatchHardTripletLoss(margin=0.3)

    metrics = train_one_epoch(
        model=tiny_model, dataloader=tiny_loader,
        loss_fn=loss_fn, optimizer=optimizer, device=device,
    )

    loss   = metrics["loss"]
    active = metrics["active_triplets"]
    lr     = metrics["lr"]

    assert loss == loss,             "loss is NaN"
    assert active == active,         "active_triplets is NaN"
    assert loss   >= 0.0,            f"loss negative: {loss}"
    assert 0.0 <= active <= 1.0,     f"active_triplets out of range: {active}"
    assert lr > 0.0,                 f"lr is zero or negative: {lr}"


# =============================================================================
# Test 3 — monitors integrate without crash
# =============================================================================

def test_monitors_integrate(tiny_model, tiny_loader, device):
    """
    Monitors passed to train_one_epoch() must not break the batch loop
    and their keys must appear in the returned metrics dict.
    """
    optimizer       = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    loss_fn         = BatchHardTripletLoss(margin=0.3)
    grad_monitor    = GradientHealthMonitor(log_every_n_batches=1)
    triplet_monitor = TripletHealthMonitor(log_every_n_batches=1)

    metrics = train_one_epoch(
        model=tiny_model, dataloader=tiny_loader,
        loss_fn=loss_fn, optimizer=optimizer, device=device,
        margin=0.3,
        grad_monitor=grad_monitor,
        triplet_monitor=triplet_monitor,
    )

    assert "th_active_fraction" in metrics
    assert "th_mean_d_pos"      in metrics
    assert "th_gap"             in metrics
    assert "grad_norm_global"   in metrics
    assert "grad_exploding"     in metrics


# =============================================================================
# Test 4 — logger creates metrics.csv
# =============================================================================

def test_logger_creates_csv(tiny_model, tiny_loader, device, run_dir):
    """
    After 2 epochs, metrics.csv must exist at runs/test/pipeline_test/
    and contain 2 data rows with at least the core columns.

    Inspect manually after the test:
        runs/test/pipeline_test/metrics.csv
    """
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    loss_fn   = BatchHardTripletLoss(margin=0.3)
    logger    = Logger(
        run_name     = "pipeline_test",
        total_epochs = 2,
        runs_dir     = os.path.join("runs", "test"),
    )

    for epoch in range(2):
        logger.epoch_start()
        train_metrics = train_one_epoch(
            model=tiny_model, dataloader=tiny_loader,
            loss_fn=loss_fn, optimizer=optimizer, device=device,
        )
        logger.log_epoch(epoch, train_metrics, eval_metrics={})

    csv_path = logger.csv_path
    print(f"\n  → inspect CSV at: {csv_path}")

    assert os.path.isfile(csv_path), f"metrics.csv not found at {csv_path}"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2,       f"expected 2 rows, got {len(rows)}"
    assert "loss"  in rows[0],   "loss column missing"
    assert "epoch" in rows[0],   "epoch column missing"