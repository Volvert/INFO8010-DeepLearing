# =============================================================================
# build_scheduler
# =============================================================================
"""
This file exposes build_scheduler() — the single entry point for learning
rate scheduling. train.py calls build_scheduler() and receives a scheduler
ready to use with scheduler.step() at each epoch.

Two phases chained with SequentialLR (lec4 pages 37-39):

  Phase 1 — Linear warmup (epochs 0 -> warmup_epochs)
    lr : 0 -> base_lr
    Prevents large destructive updates when weights are still random.
    All triplets are active at epoch 0 — gradients are large and noisy.
    LinearLR(start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)

  Phase 2 — Cosine decay (epochs warmup_epochs -> total_epochs)
    lr : base_lr -> 0
    Smooth continuous decay — no brutal drops unlike step or exponential decay.
    Allows the model to settle into a good minimum without oscillating.
    CosineAnnealingLR(T_max=total_epochs - warmup_epochs, eta_min=0)

Usage in train.py:
  scheduler = build_scheduler(optimizer, warmup_epochs=5, total_epochs=50)
  for epoch in range(total_epochs):
      train_one_epoch(...)
      scheduler.step()       <- called once per epoch, after all batches

See: lec4 pages 37-39 — Scheduling, warmup and cosine decay
See: https://docs.pytorch.org/docs/2.11/optim.html#how-to-adjust-learning-rate
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)


def build_scheduler(
    optimizer:      Optimizer,
    warmup_epochs:  int = 5,
    total_epochs:   int = 50,
) -> SequentialLR:
    """
    Builds and returns a warmup + cosine decay scheduler.

    Args:
        optimizer     : AdamW optimizer from train.py
        warmup_epochs : number of linear warmup epochs (default 5)
        total_epochs  : total number of training epochs  (default 50)

    Returns:
        scheduler : SequentialLR — call scheduler.step() once per epoch
    """
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
    
    cosdecay = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=0)

    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosdecay], milestones=[warmup_epochs])

    return scheduler