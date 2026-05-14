# =============================================================================
# DataLoaders
# =============================================================================
"""
Wraps Dataset objects into PyTorch DataLoaders.

A DataLoader sits between the Dataset and the training loop.
Its job is to:
  - batch samples together
  - shuffle or sample in a controlled order (PKSampler for train)
  - parallelize image loading across multiple CPU workers
  - accelerate CPU -> GPU transfer with pin_memory

Three functions, one per split:
  get_train_dataloader  -> PKSampler, batches by identity
  get_query_dataloader  -> sequential, no shuffle, for retrieval evaluation
  get_test_dataloader   -> sequential, no shuffle, for retrieval evaluation

pin_memory is derived from device availability — not hardcoded:
  GPU available -> pin_memory=True  (page-locked memory speeds up CPU→GPU transfer)
  CPU only      -> pin_memory=False (pinning has no effect and wastes memory)

See: data/batch.py   — PKSampler
See: data/dataset.py — VehicleReIDDataset, MergedDataset
"""

import torch
from torch.utils.data import DataLoader, Dataset
from data.batch import PKSampler


# =============================================================================
# get_train_dataloader
# =============================================================================

def get_train_dataloader(
    dataset: Dataset,   # VehicleReIDDataset or MergedDataset
    P: int,
    K: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the training split.

    Uses PKSampler instead of random shuffle — each batch contains exactly
    P identities × K images. This structure is required by BatchHardTripletLoss.
    shuffle and sampler are mutually exclusive in PyTorch — PKSampler replaces shuffle.

    Args:
        dataset     : VehicleReIDDataset or MergedDataset — must expose .labels
        P           : identities per batch  (e.g. 16)
        K           : images per identity   (e.g. 4)  ->  batch_size = P × K = 64
        num_workers : parallel CPU workers for image loading

    Returns:
        DataLoader yielding (image_tensor, vehicle_id, camera_id) batches
    """
    pin_memory = torch.cuda.is_available()
    sampler    = PKSampler(dataset.labels, P=P, K=K)

    return DataLoader(
        dataset,
        batch_size = P * K,
        sampler = sampler,
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = True,
    )


# =============================================================================
# get_query_dataloader
# =============================================================================

def get_query_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the query split (image_query/).

    Sequential loading — embeddings must be extracted in a fixed, reproducible
    order so they can be correctly matched against the gallery during kNN retrieval.

    Args:
        dataset     : VehicleReIDDataset with get_test_transform() applied
        batch_size  : images per batch (from cfg["data"]["eval_batch_size"])
        num_workers : parallel CPU workers for image loading

    Returns:
        DataLoader yielding (image_tensor, vehicle_id, camera_id) batches
    """
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = False,
    )


# =============================================================================
# get_test_dataloader
# =============================================================================

def get_test_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the gallery split (image_test/).

    Sequential loading — the 31 238 gallery embeddings are extracted once
    and stored in memory for the kNN distance matrix in engine/evaluate.py.

    Args:
        dataset     : VehicleReIDDataset with get_test_transform() applied
        batch_size  : images per batch (from cfg["data"]["eval_batch_size"])
        num_workers : parallel CPU workers for image loading

    Returns:
        DataLoader yielding (image_tensor, vehicle_id, camera_id) batches
    """
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = False,
    )