# =============================================================================
# PKSampler
# =============================================================================
"""
This file implements the P*K Sampler used by the training DataLoader.

In standard PyTorch training, the DataLoader picks images randomly (shuffle=True).
This is fine for classification but breaks the batch-hard triplet loss, which
requires multiple images of the same identity in each batch to mine hard triplets.

The PKSampler replaces random shuffle with a controlled sampling strategy:
  - P identities are randomly selected at each batch
  - K images are randomly selected per identity
  - batch_size = P * K (e.g. 16 * 4 = 64)

This guarantees that every batch contains:
  - K-1 positives per anchor   (same vehicle, different cameras)
  - (P-1)*K negatives per anchor (different vehicles)

Convention P*K comes from "In Defense of the Triplet Loss" (Hermans et al., 2017)
which introduced batch-hard mining for person Re-ID. Vehicle Re-ID reuses the same
terminology since it is the same problem applied to vehicles.

The sampler only manipulates indices — it never loads images.
Images are loaded by VehicleReIDDataset.__getitem__() in dataset.py.

See PyTorch Sampler docs: https://docs.pytorch.org/docs/2.11/data.html#torch.utils.data.Sampler
"""

import random
from collections import defaultdict
from torch.utils.data import Sampler


# =============================================================================
# PKSampler
# =============================================================================
"""
Inherits from torch.utils.data.Sampler
PyTorch requires two methods:
  __iter__ -> yields batches of indices in PK order
  __len__  -> returns total number of indices per epoch
"""

class PKSampler(Sampler):
    """
    Samples indices such that each batch contains exactly
    P identities with K images each.

    Attributes:
        labels     : list of vehicle_id parallel to dataset.samples
                     used to group indices by identity
        P          : number of identities per batch
        K          : number of images per identity per batch
        num_batches: number of batches per epoch
    """

    def __init__(self, labels: list, P: int, K: int):
        """
        Args:
            labels : list of vehicle_id for each image in the dataset
                     comes from dataset.labels (built by _parse_xml)
            P      : number of identities per batch (e.g. 16)
            K      : number of images per identity  (e.g. 4)
        """
        self.labels = labels
        self.P = P
        self.K = K

        # group indices by identity
        # index_per_identity[vehicle_id] = [idx1, idx2, idx3, ...]
        # built once in __init__, used at every __iter__ call
        self.index_per_identity = defaultdict(list)

        # list of unique identities — sampled P at a time in __iter__
        self.unique_identities = []

        # number of batches per epoch
        # derived from number of unique identities and P
        self.num_batches = 0

        self._build_index()

    # =========================================================================
    # _build_index
    # =========================================================================
    """
    Groups dataset indices by vehicle_id.
    Called once in __init__.

    After this method:
      self.index_per_identity[vid] = [idx, idx, ...]  for each identity vid
      self.unique_identities       = [vid1, vid2, ...]
      self.num_batches             = len(unique_identities) // P
    """

    def _build_index(self) -> None:
        """
        Builds self.index_per_identity, self.unique_identities
        and self.num_batches from self.labels.
        """
        for idx, vid in enumerate(self.labels):
            self.index_per_identity[vid].append(idx)

        self.unique_identities = list(self.index_per_identity.keys())
        self.num_batches = len(self.unique_identities) // self.P

    # =========================================================================
    # __iter__
    # =========================================================================
    """
    Called by the DataLoader at the start of each epoch.
    Yields indices one by one in PK order.

    Algorithm:
      1. shuffle the list of unique identities
      2. take P identities at a time
      3. for each identity, sample K indices (with replacement if needed)
      4. yield the P*K indices as a flat sequence
    """

    def __iter__(self):
        """
        Yields all indices for one epoch in PK order.
        Total indices yielded = num_batches * P * K.
        """
        copy_of_identities = self.unique_identities.copy()
        random.shuffle(copy_of_identities)
        for i in range(0, self.num_batches * self.P, self.P):
            selected_vids = copy_of_identities[i : i + self.P]
            for vid in selected_vids:
                list_of_idx = self.index_per_identity[vid]

                if len(list_of_idx) >= self.K:
                    chosen = random.sample(list_of_idx, self.K)
                else:
                    chosen = random.choices(list_of_idx, k=self.K)

                yield from chosen


    # =========================================================================
    # __len__
    # =========================================================================
    """
    Called by the DataLoader to know the total number of indices per epoch.
    Must be consistent with __iter__ — same total count.
    """

    def __len__(self) -> int:
        """Returns the total number of indices yielded per epoch."""
        return self.num_batches * self.P * self.K