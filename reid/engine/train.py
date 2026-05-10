# =============================================================================
# train
# =============================================================================
"""
This file implements the training loop for one epoch.
main.py calls train in a loop over all epochs and handles
checkpointing, evaluation frequency and logging at the top level.

One epoch iterates over all batches from the PKSampler-structured DataLoader.
Each batch contains exactly P=16 identities × K=4 images = 64 images.

For each batch:em
  1. forward pass : ViT maps (64, 3, 224, 224) -> (64, 128) L2-normalized embeddings
  2. triplet loss : batch-hard mining + loss = max(0, d(a,p) - d(a,n) + margin)
  3. backward : loss.backward() computes gradients through all ViT parameters
  4. optimizer step : AdamW updates weights — θ = θ - lr × gradient
  5. zero_grad : clears gradients before the next batch

After all batches:
  6. scheduler step : updates the global lr once per epoch (warmup + cosine decay)
  7. monitoring : logs loss, lr, active triplet fraction

train_one_epoch() returns a dict of metrics consumed by main.py:
  {
    "loss" : float  — mean triplet loss over the epoch
    "active_triplets" : float  — fraction of non-zero triplets [0, 1]
    "lr" : float  — current learning rate after scheduler.step()
  }

See: losses/tripletloss.py — BatchHardTripletLoss
See: utils/scheduler.py — build_scheduler
See: monitoring/logger.py — metric logging
See: engine/evaluate.py — called by main.py after each epoch
See: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""

import torch
from torch.utils.data import DataLoader
from losses.tripletloss import BatchHardTripletLoss
from monitoring.logger import Logger


def train_one_epoch(
    model: torch.nn.Module, # VehicleViT in train mode
    dataloader: DataLoader, # PKSampler-structured train dataloader
    loss_fn: BatchHardTripletLoss, # batch-hard triplet loss
    optimizer: torch.optim.Optimizer,# AdamW
    device: torch.device, # GPU if available, else CPU
) -> dict:
    """
    Runs one full epoch over the training set.

    Args:
        model      : VehicleViT — called with model.train() before the loop
        dataloader : yields (image_tensor, vehicle_id, camera_id) batches
                     PKSampler guarantees P×K structure — required by triplet loss
        loss_fn    : BatchHardTripletLoss instance
        optimizer  : AdamW optimizer — stepped once per batch
        device     : torch.device — tensors moved here before forward pass

    Returns:
        dict : {
            "loss"            : mean triplet loss over all batches,
            "active_triplets" : fraction of non-zero triplets in [0, 1],
            "lr"              : current learning rate after the epoch
        }
    """
    model.train()  # set model to training mode
    total_loss = 0.0
    total_active_triplets = 0
    num_batches = 0

    for images, vehicle_ids, _ in dataloader:
        images , vehicle_ids = images.to(device), vehicle_ids.to(device)
        optimizer.zero_grad()  # clear gradients before the forward pass
        embeddings = model(images)  # forward pass: (B, 3, 224, 224) -> (B, 128)
        loss , active = loss_fn(embeddings, vehicle_ids)  # compute batch-hard triplet loss
        loss.backward()  # backward pass: compute gradients
        optimizer.step()  # update weights with AdamW
        total_loss += loss.item()  # accumulate loss for monitoring
        total_active_triplets += active.item()  # accumulate active triplet fraction
        num_batches += 1

    mean_loss = total_loss / num_batches
    active_triplet_fraction = total_active_triplets / num_batches

    return {
        "loss": mean_loss,
        "active_triplets": active_triplet_fraction,
        "lr": optimizer.param_groups[0]['lr']
        }
