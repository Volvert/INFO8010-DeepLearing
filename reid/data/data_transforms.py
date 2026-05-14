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
  2. Invariance — each transform teaches the model to ignore a specific
                      variation that exists in real multi-camera footage

Training transform order enforced by Compose:
  1. RandomResizedCrop  -> random crop simulating imperfect detection boxes
  2. RandomHorizontalFlip -> lateral symmetry, p=0.5
  3. ColorJitter -> brightness, contrast, saturation variance across cameras
  4. GaussianBlur -> low-quality or motion-blurred cameras
  5. ToTensor -> PIL to float32 tensor [0, 1]
  6. Normalize -> ImageNet mean/std, stabilizes gradient descent
  7. RandomErasing -> occlusion simulation, must come after ToTensor

Test transform order:
  1. Resize -> fixed 224x224, no randomness
  2. ToTensor -> PIL to float32 tensor [0, 1]
  3. Normalize -> same ImageNet stats as training, must be identical

get_test_transform() is applied to image_query/ and image_test/ splits.
No randomness — deterministic embeddings are required for kNN retrieval.

See torchvision v2 transforms docs: https://docs.pytorch.org/vision/0.21/transforms.html
"""

from torchvision.transforms import v2 as transforms

# ImageNet mean and std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:

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

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])