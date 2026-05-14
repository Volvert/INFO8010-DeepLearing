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
  get_train_dataloader -> PKSampler, batches by identity
  get_query_dataloader -> sequential, no shuffle, for retrieval evaluation
  get_test_dataloader -> sequential, no shuffle, for retrieval evaluation

pin_memory is derived from device availability — not hardcoded:
  GPU available -> pin_memory=True  (page-locked memory speeds up CPU→GPU transfer)
  CPU only -> pin_memory=False (pinning has no effect and wastes memory)

See: data/batch.py — PKSampler
See: data/dataset.py — VehicleReIDDataset, MergedDataset
See: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

import torch
from torch.utils.data import DataLoader, Dataset
from data.batch import PKSampler

def get_train_dataloader(
    dataset: Dataset,   # VehicleReIDDataset or MergedDataset
    P: int,
    K: int,
    num_workers: int = 4,
) -> DataLoader:

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

def get_query_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = False,
    )

def get_test_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:

    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = False,
    )