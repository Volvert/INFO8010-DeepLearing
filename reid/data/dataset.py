# =============================================================================
# dataset.py
# =============================================================================
"""
Defines two dataset classes used throughout the project:

  VehicleReIDDataset — loads one split (train / query / test) from XML + images
  MergedDataset      — combines real + synthetic train sets for PKSampler

--- VehicleReIDDataset ---

Reads the XML annotation file and image folder, exposes the standard
PyTorch Dataset interface (.__len__, .__getitem__).

Compatible with both XML formats in the project:
  Real      (gb2312) : <Item imageName="000001.jpg" vehicleID="0269" cameraID="c026" />
  Synthetic (utf-8)  : <Item imageName="00001_c006_1.jpg" vehicleID="0001" cameraID="c006"
                             colorID="10" typeID="10" orientation="266.1" ... />

Extra attributes in the synthetic XML (colorID, typeID, orientation, etc.)
are silently ignored — xml.etree only reads what you ask for.

The id_offset parameter shifts all vehicle IDs at parse time:
  real      = VehicleReIDDataset(..., id_offset=0)            IDs : 1 – 440
  synthetic = VehicleReIDDataset(..., id_offset=max(real.labels))  IDs : 441 – 1802

This eliminates ID collisions before the datasets are combined.

--- MergedDataset ---

Concatenates one real and one synthetic VehicleReIDDataset into a single
Dataset that exposes a flat .labels list — required by PKSampler.

--- make_train_eval_split ---

Proper 3-way split with NO overlap between train and eval:
  - 40 held-out identities → never seen during training
  - train_ds  : all images of the 400 remaining identities
  - query_ds  : 1 image per held-out identity
  - gallery_ds: all remaining images of held-out identities

This is the standard academic Re-ID evaluation protocol when
the official test ground truth is unavailable (AIC21 keeps it secret).

See: data/batch.py      — PKSampler reads dataset.labels
See: data/dataloader.py — get_train_dataloader passes dataset to PKSampler
"""

import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


# =============================================================================
# VehicleReIDDataset
# =============================================================================

class VehicleReIDDataset(Dataset):
    """
    Loads vehicle images and their labels from an AIC21-format dataset.
    Returns (image_tensor, vehicle_id, camera_id) for each sample.

    Attributes:
        root      : str  — path to image directory
        transform : callable | None — torchvision transform pipeline
        samples   : list of (img_path, vehicle_id, camera_id)
        labels    : list of vehicle_id — parallel to samples
                    read by PKSampler to build P×K identity batches
    """

    def __init__(
        self,
        root:      str,
        label_xml: str,
        transform = None,
        id_offset: int = 0,
    ):
        self.root      = root
        self.transform = transform
        self.id_offset = id_offset
        self.samples: list[tuple[str, int, int]] = []
        self.labels:  list[int]                  = []
        self._parse_xml(label_xml)

    def _parse_xml(self, label_xml: str) -> None:
        tree = ET.parse(label_xml)
        root = tree.getroot()
        for item in root.iter("Item"):
            name       = item.get("imageName")
            vehicle_id = int(item.get("vehicleID", -1)) + self.id_offset
            camera_id  = int(item.get("cameraID")[1:])
            img_path   = os.path.join(self.root, name)
            self.samples.append((img_path, vehicle_id, camera_id))
            self.labels.append(vehicle_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, vehicle_id, camera_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform is not None:
            image = self.transform(image)
        return image, vehicle_id, camera_id

    def get_num_identities(self) -> int:
        return len(set(self.labels))

    def __repr__(self) -> str:
        return (
            f"VehicleReIDDataset("
            f"images={len(self.samples)}, "
            f"identities={self.get_num_identities()}, "
            f"offset={self.id_offset})"
        )


# =============================================================================
# MergedDataset
# =============================================================================

class MergedDataset(Dataset):
    """
    Concatenates a real and a synthetic VehicleReIDDataset into one Dataset.
    Exposes a flat .labels list required by PKSampler.
    """

    def __init__(self, real: VehicleReIDDataset, synthetic: VehicleReIDDataset):
        self.real      = real
        self.synthetic = synthetic
        self._n_real   = len(real)
        self.labels    = real.labels + synthetic.labels

    def __len__(self) -> int:
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, idx: int):
        if idx < self._n_real:
            return self.real[idx]
        return self.synthetic[idx - self._n_real]

    def __repr__(self) -> str:
        return (
            f"MergedDataset("
            f"real={len(self.real)}, "
            f"synthetic={len(self.synthetic)}, "
            f"total={len(self)}, "
            f"identities={len(set(self.labels))})"
        )


# =============================================================================
# _SubDataset
# =============================================================================

class _SubDataset(Dataset):
    """
    Lightweight dataset built from a pre-filtered list of samples.
    Used by make_train_eval_split() for train, query and gallery splits.
    """

    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.labels    = [s[1] for s in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, vehicle_id, camera_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform is not None:
            image = self.transform(image)
        return image, vehicle_id, camera_id

    def get_num_identities(self) -> int:
        return len(set(self.labels))

    def __repr__(self) -> str:
        return (
            f"_SubDataset("
            f"images={len(self.samples)}, "
            f"identities={self.get_num_identities()})"
        )


# =============================================================================
# make_train_eval_split
# =============================================================================

def make_train_eval_split(
    dataset:         "VehicleReIDDataset",
    n_eval_ids:      int = 40,
    seed:            int = 42,
    train_transform       = None,
    eval_transform        = None,
) -> tuple["_SubDataset", "_SubDataset", "_SubDataset"]:
    """
    Splits a VehicleReIDDataset into 3 non-overlapping splits.

    The 40 eval identities are NEVER seen during training — no data leakage.

    Split sizes (AIC21 real, 440 identities, 52717 images):
        train_ds   : ~400 identities, ~47800 images
        query_ds   : 40 images (1 per eval identity)
        gallery_ds : ~40 identities, ~4760 images

        train_ds   : _SubDataset with train_transform (augmentations)
        query_ds   : 1 image per held-out identity, eval_transform
        gallery_ds : remaining images of held-out identities, eval_transform

    Args:
        dataset         : VehicleReIDDataset — full training split
        n_eval_ids      : identities held out for evaluation (default 40)
        seed            : random seed for reproducibility
        train_transform : augmentation pipeline for training images
        eval_transform  : no-augmentation pipeline for query/gallery

    Returns:
        train_ds, query_ds, gallery_ds
    """
    rng = random.Random(seed)

    # group sample indices by vehicle_id
    id_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, vid, _) in enumerate(dataset.samples):
        id_to_indices[vid].append(idx)

    # pick n_eval_ids held-out identities — sorted then shuffled for reproducibility
    all_ids = sorted(id_to_indices.keys())
    rng.shuffle(all_ids)
    eval_ids = set(all_ids[:n_eval_ids])

    train_samples:   list = []
    query_samples:   list = []
    gallery_samples: list = []

    for vid, indices in id_to_indices.items():
        local_rng = random.Random(seed + vid)
        local_rng.shuffle(indices)

        if vid in eval_ids:
            # held-out: 1 image → query, rest → gallery
            query_samples.append(dataset.samples[indices[0]])
            gallery_samples.extend(dataset.samples[i] for i in indices[1:])
        else:
            # training identity: all images → train
            train_samples.extend(dataset.samples[i] for i in indices)

    train_ds   = _SubDataset(train_samples,   train_transform)
    query_ds   = _SubDataset(query_samples,   eval_transform)
    gallery_ds = _SubDataset(gallery_samples, eval_transform)

    return train_ds, query_ds, gallery_ds