# =============================================================================
# train.py — One Epoch Training Loop
# =============================================================================
"""
Single responsibility: run one full epoch of training and return all metrics.

main.py calls train_one_epoch() and receives one comprehensive dict.
It never touches the batch loop directly.

What happens inside one epoch:
    For each batch (P=16 identities × K=4 images = 64 images):
        1. zero_grad — clear previous gradients
        2. forward — images -> (B, 128) L2-normalized embeddings
        3. triplet_monitor — compute d(a,p), d(a,n), gap BEFORE backward
        4. loss — batch-hard triplet loss + active fraction
        5. backward — fill grad on all parameters
        6. grad_monitor — read grad norms AFTER backward, BEFORE step
        7. clip — clip if grad_exploding detected
        8. step — AdamW weight update
        9. accumulate — sum loss + active for epoch average

    After all batches:
        10. epoch averages — mean loss, mean active fraction
        11. triplet average — TripletHealthMonitor.epoch_average()
        12. return dict — all metrics in one flat dict

Return dict keys:

    Core (always present):
        "loss" : float — mean triplet loss over the epoch
        "active_triplets" : float — mean active triplet fraction [0, 1]
        "lr" : float — learning rate BEFORE scheduler.step()

    Triplet health (th_* prefix, present when triplet_monitor is not None):
        "th_active_fraction" : float
        "th_mean_d_pos" : float
        "th_mean_d_neg" : float
        "th_gap" : float
        "th_d_pos_std" : float
        "th_d_neg_std" : float
        "th_collapse" : bool

    Gradient health (grad_* prefix, present when grad_monitor is not None):
        "grad_norm_global" : float
        "grad_norm_patch_embed" : float
        "grad_norm_blocks.0" : float  (through blocks.5)
        ...
        "grad_norm_proj_head" : float
        "grad_vanishing" : bool
        "grad_exploding" : bool

See: monitoring/gradient_health.py — GradientHealthMonitor
See: monitoring/triplet_health.py  — TripletHealthMonitor
See: losses/tripletloss.py — BatchHardTripletLoss
See: main.py — calls this function once per epoch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses.tripletloss import BatchHardTripletLoss
from monitoring.gradient_health import GradientHealthMonitor
from monitoring.triplet_health  import TripletHealthMonitor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: BatchHardTripletLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 0.3,
    grad_monitor: GradientHealthMonitor | None = None,
    triplet_monitor: TripletHealthMonitor  | None = None,
) -> dict:

    model.train()  # activates Dropout and training-mode BatchNorm

    n_batches = len(dataloader)

    # running accumulators — summed over batches, divided at epoch end
    total_loss = 0.0
    total_active = 0.0

    # per-batch triplet health dicts — averaged at epoch end
    batch_triplet_metrics: list[dict] = []

    # last valid gradient metrics from the epoch — kept for logging
    # (only updated when grad_monitor is not None and batch is not skipped)
    last_grad_metrics: dict = {}

    for images, vehicle_ids, _ in dataloader:

        images = images.to(device)
        vehicle_ids = vehicle_ids.to(device)

       
        # Step 1 — zero_grad
        optimizer.zero_grad()

        # Step 2 — forward pass
        # images : (B, 3, 224, 224)  ->  embeddings : (B, 128) L2-normalized
        embeddings = model(images)

       
        # Step 3 — triplet health BEFORE backward
        if triplet_monitor is not None:
            th = triplet_monitor.compute(
                embeddings.detach(),
                vehicle_ids,
                margin=margin,
            )
            batch_triplet_metrics.append(th)

        # Step 4 — triplet loss       
        loss, active = loss_fn(embeddings, vehicle_ids)
       
        # Step 5 — backward
        loss.backward()
       
        # Step 6 — gradient health AFTER backward, BEFORE step
        if grad_monitor is not None:
            gm = grad_monitor.compute(model)
            if not gm.get("grad_skipped"):
                last_grad_metrics = gm

        # Step 7 — gradient clipping (conditional)
        if last_grad_metrics.get("grad_exploding"):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Step 8 — optimizer step
        optimizer.step()

        # Step 9 — accumulate
        total_loss += loss.item()
        total_active += active.item()


    # core metrics — averaged over all batches
    train_metrics = {
        "loss": total_loss / n_batches,
        "active_triplets": total_active / n_batches,
        "lr": optimizer.param_groups[0]["lr"],
    }

    # triplet health — epoch average across all batches
    if triplet_monitor is not None and batch_triplet_metrics:
        epoch_triplet = TripletHealthMonitor.epoch_average(batch_triplet_metrics)
        train_metrics.update(epoch_triplet)  # merges th_* keys into the dict

    # gradient health — last batch metrics (representative of end-of-epoch state)
    if last_grad_metrics:
        train_metrics.update(last_grad_metrics)  # merges grad_* keys

    return train_metrics