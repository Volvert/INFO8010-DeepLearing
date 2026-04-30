# =============================================================================
# DataLoaders
# =============================================================================
"""
This file wraps the VehicleReIDDataset into PyTorch DataLoaders.

A DataLoader sits between the Dataset and the training loop.
Its job is to:
  - batch samples together
  - shuffle or sample in a controlled order (PK sampler for train)
  - parallelize image loading across multiple CPU workers
  - accelerate CPU to GPU transfer with pin_memory

This file exposes three functions, one per official dataset split:
  get_train_dataloader -> uses PK sampler, batches by identity
  get_query_dataloader -> sequential, no shuffle, for retrieval evaluation
  get_test_dataloader  -> sequential, no shuffle, for retrieval evaluation

See PyTorch DataLoader docs: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders
"""

from torch.utils.data import DataLoader
from data.dataset import VehicleReIDDataset
from data.batch import PKSampler


# =============================================================================
# get_train_dataloader
# =============================================================================
"""
Uses PKSampler instead of random shuffle — shuffle and sampler are mutually
exclusive in PyTorch DataLoader, you cannot use both at the same time.
Each batch is guaranteed to contain exactly P identities with K images each.
This structure is required by the batch-hard triplet loss in losses/triplet.py.
batch_size is derived from P * K — not passed directly.

pin_memory=True keeps tensors in pinned (page-locked) memory on CPU.
This accelerates the CPU -> GPU transfer during training.
"""

def get_train_dataloader(
    dataset: VehicleReIDDataset,
    P: int,
    K: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for the training split (image_train/).

    Args:
        dataset     : VehicleReIDDataset with get_train_transform() applied
        P           : number of identities per batch (e.g. 16)
        K           : number of images per identity  (e.g. 4)
                      batch_size = P * K             (e.g. 64)
        num_workers : parallel CPU workers for image loading
        pin_memory  : accelerates CPU -> GPU transfer (default True)

    Returns:
        DataLoader : yields (image_tensor, vehicle_id, camera_id) batches
    """
    sampler = PKSampler(dataset.labels, P=P, K=K)

    return DataLoader(
        dataset,
        batch_size=P * K,
        sampler=sampler,        # controls order — replaces shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # drop incomplete last batch
                                # ensures every batch has exactly P*K images
    )


# =============================================================================
# get_query_dataloader
# =============================================================================
"""
Sequential loading, no shuffle, no sampler.
Embeddings must be extracted in a fixed and reproducible order
so they can be correctly matched against the test gallery during kNN retrieval.
pin_memory=True accelerates CPU -> GPU transfer during embedding extraction.
"""

def get_query_dataloader(
    dataset: VehicleReIDDataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for the query split (image_query/).

    Args:
        dataset     : VehicleReIDDataset with get_test_transform() applied
        batch_size  : number of images per batch
        num_workers : parallel CPU workers for image loading
        pin_memory  : accelerates CPU -> GPU transfer (default True)

    Returns:
        DataLoader : yields (image_tensor, vehicle_id, camera_id) batches
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # fixed order — required for kNN matching
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,        # keep all query images — none can be dropped
    )


# =============================================================================
# get_test_dataloader
# =============================================================================
"""
Sequential loading, no shuffle, no sampler.
The 31 238 gallery embeddings are extracted once and stored in memory
for the kNN distance matrix computation in engine/evaluate.py.
pin_memory=True accelerates CPU -> GPU transfer during embedding extraction.
"""

def get_test_dataloader(
    dataset: VehicleReIDDataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for the test/gallery split (image_test/).

    Args:
        dataset     : VehicleReIDDataset with get_test_transform() applied
        batch_size  : number of images per batch
        num_workers : parallel CPU workers for image loading
        pin_memory  : accelerates CPU -> GPU transfer (default True)

    Returns:
        DataLoader : yields (image_tensor, vehicle_id, camera_id) batches
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # fixed order — required for kNN matching
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,        # keep all gallery images — none can be dropped
    )