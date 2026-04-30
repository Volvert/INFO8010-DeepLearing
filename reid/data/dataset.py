# =============================================================================
# VehicleReIDDataset
# =============================================================================
"""
This file is the entry point for all data in the project.
Its only job is to read the XML annotation file and the image folder,
and expose a standard PyTorch Dataset interface so that the DataLoader
can feed batches to the model during training and evaluation.

PyTorch expects every dataset to implement three methods:
  __init__    -> called once when you create the dataset object
  __len__     -> called when PyTorch needs to know how many samples exist
  __getitem__ -> called thousands of times during training, once per sample

See PyTorch Dataset docs: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

See PIL Image docs: https://pillow.readthedocs.io/en/stable/
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class VehicleReIDDataset(Dataset):
    """
    Loads vehicle images and their labels from the AIC21 Track2 dataset.
    Returns (image_tensor, vehicle_id, camera_id) for each sample.

    Attributes:
        samples : list of (img_path, vehicle_id, camera_id)
                  one entry per image — the core data structure of this file
        labels  : list of vehicle_id parallel to samples
                  self.labels[i] is always the vehicle_id of self.samples[i]
                  used by PKSampler in batch.py to group images by identity
    """

    def __init__(self, root: str, label_xml: str, transform=None):
        """
        Args:
            root      : path to image folder
                        e.g. "dataset/AIC21_Track2_ReID/image_train"
            label_xml : path to XML annotation file
                        e.g. "dataset/AIC21_Track2_ReID/train_label.xml"
            transform : callable transform pipeline from data_transforms.py
                        get_train_transform() for training
                        get_test_transform()  for query and test
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.labels  = []

        self._parse_xml(label_xml)

    # =========================================================================
    # _parse_xml
    # =========================================================================
    """
    Only place where the XML file is read.
    Extracts filename, vehicle_id and camera_id for each image.
    Builds self.samples and self.labels in one pass.

    camera_id is stored because at evaluation time:
      same vehicle + different camera = true positive  (counted)
      same vehicle + same camera      = ignored        (not counted)
    """

    def _parse_xml(self, label_xml: str) -> None:
        """
        Parses the XML annotation file and populates self.samples and
        self.labels. Called once in __init__.

        After this method returns:
          len(self.samples) == len(self.labels) == total number of images
        """
        tree = ET.parse(label_xml)
        root = tree.getroot()

        for item in root.iter('Item'):
            name = item.get('imageName')           # "000001.jpg"
            vehicle_id = int(item.get('vehicleID', -1))  # -1 si absent (query/test)
            camera_id = int(item.get('cameraID')[1:])   # "c036" → 36

            img_path = os.path.join(self.root, name)

            self.samples.append((img_path, vehicle_id, camera_id))
            self.labels.append(vehicle_id)

    # =========================================================================
    # __len__
    # =========================================================================
    """
    Called by PyTorch DataLoader to know the total number of samples.
    Used to determine when one full epoch is complete.
    """

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.samples)

    # =========================================================================
    # __getitem__
    # =========================================================================
    """
    Called by PyTorch DataLoader thousands of times during training.
    Loads one image from disk, applies transforms, returns it with labels.

    try/except prevents a single corrupted image from crashing training.
    Falls back to the next sample if the image cannot be opened.
    """

    def __getitem__(self, idx: int):
        """
        Loads one image from disk and returns it with its labels.

        Returns:
            image : torch.Tensor of shape (3, H, W) or a simple image PIL Image
            vehicle_id   : int — which vehicle is this
            camera_id    : int — which camera captured this image
        """
        img_path, vehicle_id, camera_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"Warning: failed to load image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self.samples))
          
        # transform is always set in practice (get_train_transform / get_test_transform)
        # None only when instantiating the dataset manually for debugging
        if self.transform is not None:
            image = self.transform(image)

        return image, vehicle_id, camera_id

    # =========================================================================
    # __repr__
    # =========================================================================
    """
    Called when you do print(dataset).
    Useful for sanity checks after loading — verify image count and
    identity count match the expected dataset size.
    """

    def __repr__(self) -> str:
        return (
            f"VehicleReIDDataset\n"
            f"  images     : {len(self.samples)}\n"
            f"  identities : {self.get_num_identities()}\n"
            f"  root       : {self.root}"
        )

    # =========================================================================
    # get_num_identities
    # =========================================================================
    """
    Utility method — returns the number of unique vehicle identities.
    Used in __repr__ and to sanity-check the dataset after loading.
    """

    def get_num_identities(self) -> int:
        """Returns the number of unique vehicle identities in the dataset."""
        
        return len(set(self.labels))