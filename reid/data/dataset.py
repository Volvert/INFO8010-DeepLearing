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

torch.utils.data.ConcatDataset would work for __getitem__ routing but does
NOT expose .labels, which PKSampler needs to group images by identity.
MergedDataset fills that gap with minimal code.

Typical usage in main.py:

    real      = VehicleReIDDataset(
                    root      = "dataset/AIC21_Track2_ReID/image_train",
                    label_xml = "dataset/AIC21_Track2_ReID/train_label.xml",
                    transform = get_train_transform(),
                    id_offset = 0,
                )

    synthetic = VehicleReIDDataset(
                    root      = "dataset/AIC21_Track2_ReID_Simulation/sys_image_train",
                    label_xml = "dataset/AIC21_Track2_ReID_Simulation/train_label.xml",
                    transform = get_train_transform(),
                    id_offset = max(real.labels),   # 440 → synthetic becomes 441-1802
                )

    train = MergedDataset(real, synthetic)
    # train.labels  : 244 867 entries, IDs 1-1802, no collision
    # len(train)    : 244 867

See: data/batch.py      — PKSampler reads dataset.labels
See: data/dataloader.py — get_train_dataloader passes dataset to PKSampler
"""

import os
import xml.etree.ElementTree as ET
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
        """
        Args:
            root      : path to the image folder
                        e.g. "dataset/AIC21_Track2_ReID/image_train"
            label_xml : path to the XML annotation file
                        e.g. "dataset/AIC21_Track2_ReID/train_label.xml"
            transform : transform pipeline — get_train_transform() for train,
                        get_test_transform() for query and gallery
            id_offset : integer added to every vehicle_id at parse time.
                        default 0 for real data (IDs unchanged).
                        pass max(real.labels) for synthetic data to avoid
                        ID collisions when both datasets are merged.
        """
        self.root      = root
        self.transform = transform
        self.id_offset = id_offset
        self.samples: list[tuple[str, int, int]] = []
        self.labels:  list[int]                  = []

        self._parse_xml(label_xml)

    # =========================================================================
    # _parse_xml
    # =========================================================================

    def _parse_xml(self, label_xml: str) -> None:
        """
        Reads the XML annotation file and populates self.samples and self.labels.

        Called once in __init__. After this method returns:
            len(self.samples) == len(self.labels) == number of images in the split.

        The id_offset is added to vehicle_id here — at the source — so that
        all downstream code (PKSampler, BatchHardTripletLoss, MergedDataset)
        sees correct, non-colliding IDs without any extra logic.

        Args:
            label_xml : path to XML file (real or synthetic format)
        """
        tree = ET.parse(label_xml)
        root = tree.getroot()

        for item in root.iter("Item"):

            name = item.get("imageName")                    # "000001.jpg" or "00001_c006_1.jpg"

            vehicle_id = int(item.get("vehicleID", -1))    # "0269" → 269
            vehicle_id += self.id_offset                   # 269 + 0 = 269  (real)
                                                           # 1   + 440 = 441 (synthetic)

            camera_id = int(item.get("cameraID")[1:])      # "c036" → 36

            img_path = os.path.join(self.root, name)

            self.samples.append((img_path, vehicle_id, camera_id))
            self.labels.append(vehicle_id)

    # =========================================================================
    # __len__
    # =========================================================================

    def __len__(self) -> int:
        """Total number of images in this split."""
        return len(self.samples)

    # =========================================================================
    # __getitem__
    # =========================================================================

    def __getitem__(self, idx: int):
        """
        Loads one image and returns it with its labels.

        Falls back to the next sample if the image file is corrupted or missing,
        so that a single bad file never crashes a training run.

        Returns:
            image      : torch.Tensor (3, H, W) after transform, else PIL Image
            vehicle_id : int — identity label (offset already applied)
            camera_id  : int — camera index
        """
        img_path, vehicle_id, camera_id = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform is not None:
            image = self.transform(image)

        return image, vehicle_id, camera_id

    # =========================================================================
    # get_num_identities / __repr__
    # =========================================================================

    def get_num_identities(self) -> int:
        """Number of unique vehicle identities in this split."""
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

    Both datasets must already have non-colliding vehicle IDs — apply id_offset
    when instantiating the synthetic VehicleReIDDataset, not here.

    The only job of this class is to:
        1. expose a flat .labels list → required by PKSampler
        2. route __getitem__ to the correct sub-dataset

    Attributes:
        real      : VehicleReIDDataset — real training split
        synthetic : VehicleReIDDataset — synthetic training split
        labels    : list[int] — concatenation of both .labels lists
        _n_real   : int — length of real dataset (boundary for __getitem__ routing)
    """

    def __init__(
        self,
        real:      VehicleReIDDataset,
        synthetic: VehicleReIDDataset,
    ):
        """
        Args:
            real      : VehicleReIDDataset loaded with id_offset=0
            synthetic : VehicleReIDDataset loaded with id_offset=max(real.labels)
        """
        self.real      = real
        self.synthetic = synthetic
        self._n_real   = len(real)

        # flat label list — what PKSampler reads
        # real labels stay as-is, synthetic labels already have the offset
        self.labels = real.labels + synthetic.labels

    # =========================================================================
    # __len__
    # =========================================================================

    def __len__(self) -> int:
        """Total images across both datasets."""
        return len(self.real) + len(self.synthetic)

    # =========================================================================
    # __getitem__
    # =========================================================================

    def __getitem__(self, idx: int):
        """
        Routes idx to the correct sub-dataset.

        Indices 0 … n_real-1          → real dataset
        Indices n_real … n_real+n_syn-1 → synthetic dataset

        Returns:
            (image, vehicle_id, camera_id) — same tuple as VehicleReIDDataset
        """
        if idx < self._n_real:
            return self.real[idx]
        else:
            return self.synthetic[idx - self._n_real]

    # =========================================================================
    # __repr__
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"MergedDataset("
            f"real={len(self.real)}, "
            f"synthetic={len(self.synthetic)}, "
            f"total={len(self)}, "
            f"identities={len(set(self.labels))})"
        )