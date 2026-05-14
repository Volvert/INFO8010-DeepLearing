# =============================================================================
# Data Transforms
# =============================================================================
"""
This file defines both transform pipelines used in the project.
It exposes two functions:
  get_train_transform() -> full augmentation + normalization for training
  get_test_transform()  -> minimal deterministic pipeline for query and test

Augmentation is applied ONLINE — transforms are applied on the fly at each
epoch inside __getitem__. No new images are created on disk. The model sees
a statistically different version of each image at every epoch.

Two purposes:
  1. Regularization — each epoch the model sees a slightly different version
                      of each image, which reduces overfitting on 52k images
  2. Invariance     — each transform teaches the model to ignore a specific
                      variation that exists in real multi-camera footage

Training transform order enforced by Compose:
  1. RandomResizedCrop    -> random crop simulating imperfect detection boxes
  2. RandomHorizontalFlip -> lateral symmetry, p=0.5
  3. ColorJitter          -> brightness, contrast, saturation variance across cameras
  4. GaussianBlur         -> low-quality or motion-blurred cameras
  5. ToTensor             -> PIL to float32 tensor [0, 1]
  6. Normalize            -> ImageNet mean/std, stabilizes gradient descent
  7. RandomErasing        -> occlusion simulation, must come after ToTensor

Test transform order:
  1. Resize -> fixed 224x224, no randomness
  2. ToTensor -> PIL to float32 tensor [0, 1]
  3. Normalize -> same ImageNet stats as training, must be identical

get_test_transform() is applied to image_query/ and image_test/ splits.
No randomness — deterministic embeddings are required for kNN retrieval.

See torchvision v2 transforms docs: https://docs.pytorch.org/vision/0.21/transforms.html
"""

from torchvision.transforms import v2 as transforms

# =============================================================================
# ImageNet normalization constants
# =============================================================================
"""
Shared between both pipelines — defined once to avoid duplication.
The test pipeline must use the exact same stats as the training pipeline.
"""
# ImageNet mean and std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """
    Returns the full augmentation + normalization pipeline for training.
    Applied to image_train/ only.

    Args:
        img_size : target image size after crop (default 224)

    Returns:
        transforms.Compose : callable pipeline, pass it to VehicleReIDDataset
    """

    # =========================================================================
    # 1. RandomResizedCrop
    # =========================================================================
    """
    Randomly crops a region of the image and resizes it to img_size.
    scale=(0.6, 1.0) ensures at least 60% of the vehicle remains visible.
    Simulates imperfect detection crops in real camera footage.
    """

    # =========================================================================
    # 2. RandomHorizontalFlip
    # =========================================================================
    """
    Flips the image left-right with probability p=0.5.
    Valid because vehicles have natural lateral symmetry.
    NO vertical flip — an upside-down vehicle is not a real viewpoint.
    """

    # =========================================================================
    # 3. ColorJitter
    # =========================================================================
    """
    Randomly varies brightness, contrast, saturation and hue.
    Simulates different lighting conditions across cameras
    (daylight, artificial light, shadows, overexposure).
    Hue kept minimal (0.05) — a red car must stay red.
    """

    # =========================================================================
    # 4. GaussianBlur
    # =========================================================================
    """
    Applies a light Gaussian blur to the image.
    Simulates low-quality cameras or fast-moving vehicles.
    Kept subtle (kernel=3, sigma=(0.1, 0.5)) — strong blur would destroy
    fine details (wheels, logos) that the model needs for Re-ID.
    """

    # =========================================================================
    # 5. ToTensor
    # =========================================================================
    """
    Converts PIL Image (H x W x C, uint8, [0, 255])
    to PyTorch Tensor  (C x H x W, float32, [0.0, 1.0]).
    Must come before Normalize and RandomErasing — both require tensors.
    """

    # =========================================================================
    # 6. Normalize
    # =========================================================================
    """
    Centers and scales each channel using ImageNet statistics:
      pixel_normalized = (pixel - mean) / std
    Ensures equal variance across all input channels which stabilizes
    gradient descent (lec4 — data normalization).
    ImageNet stats are a valid approximation for any natural photo dataset.
    """

    # =========================================================================
    # 7. RandomErasing
    # =========================================================================
    """
    Randomly masks a rectangular region of the image with p=0.5.
    scale=(0.02, 0.2) erases between 2% and 20% of the image area.
    Simulates real-world occlusions: poles, other vehicles, frame edges.
    Forces the model to build a global vehicle representation rather than
    relying on a single discriminative region.
    Must come AFTER ToTensor — operates on tensors, not PIL images.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.5),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])


def get_test_transform(img_size: int = 224) -> transforms.Compose:
    """
    Returns the minimal deterministic pipeline for query and test splits.
    Applied to image_query/ and image_test/ — no augmentation.

    Args:
        img_size : target image size (default 224, same as training)

    Returns:
        transforms.Compose : callable pipeline, pass it to VehicleReIDDataset
    """

    # =========================================================================
    # 1. Resize
    # =========================================================================
    """
    Resizes the image to a fixed img_size x img_size.
    No random crop — the full vehicle is always visible.
    """

    # =========================================================================
    # 2. ToTensor
    # =========================================================================
    """
    Converts PIL Image (H x W x C, uint8, [0, 255])
    to PyTorch Tensor  (C x H x W, float32, [0.0, 1.0]).
    """

    # =========================================================================
    # 3. Normalize
    # =========================================================================
    """
    Same ImageNet mean/std as get_train_transform().
    Must be identical — the model was trained with these stats.
    """

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])