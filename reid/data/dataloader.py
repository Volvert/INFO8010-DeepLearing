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
Uses PKSampler instead of random shuffle.
Each batch is guaranteed to contain exactly P identities with K images each.
This structure is required by the batch-hard triplet loss in losses/triplet.py.
batch_size is not passed directly — it is derived from P * K.
"""

def get_train_dataloader(
    dataset: VehicleReIDDataset,
    P: int,
    K: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the training split (image_train/).

    Args:
        dataset     : VehicleReIDDataset with get_train_transform() applied
        P           : number of identities per batch (e.g. 16)
        K           : number of images per identity  (e.g. 4)
                      batch_size = P * K             (e.g. 64)
        num_workers : parallel CPU workers for image loading

    Returns:
        DataLoader : yields (image_tensor, vehicle_id, camera_id) batches
    """
    pass


# =============================================================================
# get_query_dataloader
# =============================================================================
"""
Sequential loading, no shuffle.
Embeddings must be extracted in a fixed order so they can be matched
against the test gallery during kNN retrieval.
"""

def get_query_dataloader(
    dataset: VehicleReIDDataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the query split (image_query/).

    Args:
        dataset     : VehicleReIDDataset with get_test_transform() applied
        batch_size  : number of images per batch
        num_workers : parallel CPU workers for image loading

    Returns:
        DataLoader : yields (image_tensor, vehicle_id, camera_id) batches
    """
    pass


# =============================================================================
# get_test_dataloader
# =============================================================================
"""
Sequential loading, no shuffle.
The 31 238 gallery embeddings are extracted once and stored in memory
for the kNN distance matrix computation in engine/evaluate.py.
"""

def get_test_dataloader(
    dataset: VehicleReIDDataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the test/gallery split (image_test/).

    Args:
        dataset     : VehicleReIDDataset with get_test_transform() applied
        batch_size  : number of images per batch
        num_workers : parallel CPU workers for image loading

    Returns:
        DataLoader : yields (image_tensor, vehicle_id, camera_id) batches
    """
    pass