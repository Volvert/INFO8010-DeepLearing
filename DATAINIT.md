# Data Initialization

## Overview

This module handles everything between raw files on disk and a batch of tensors
ready for the model. Four files with strictly separated responsibilities.

| File | Role |
|---|---|
| `data/data_transforms.py` | Two image preprocessing pipelines — randomized for train, deterministic for query/test |
| `data/dataset.py` | Reads the XML annotation file, loads images on the fly, exposes the PyTorch Dataset interface |
| `data/batch.py` | Controlled P×K batch construction required by the triplet loss |
| `data/dataloader.py` | Wraps dataset + sampler into PyTorch DataLoaders — one per split |

```mermaid
flowchart LR
    XML("train_label.xml") -- "parsed once" --> D("dataset.py")
    T("data_transforms.py") -- "get_train_transform()\nget_test_transform()" --> D
    D -- "make_train_eval_split()" --> S("400 train / 40 eval")
    S -- "dataset.labels" --> B("batch.py")
    S -- "dataset" --> DL("dataloader.py")
    B -- "PKSampler" --> DL

    style XML fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style T fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style D fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style S fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    style B fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style DL fill:#E6F1FB,stroke:#378ADD,color:#0C447C
```

---

## Data Augmentation — `data/data_transforms.py`

Augmentation is applied **online** — at each call to `__getitem__`, a fresh
random transform is applied to the image. No new files are created on disk.
The model sees a statistically different version of each image at every epoch.

Two purposes: **regularization** (prevents memorizing exact pixel values) and
**invariance** (teaches the model to ignore variations that exist across cameras —
lighting, blur, partial occlusion).

### Training pipeline

| Step | Transform | Purpose |
|---|---|---|
| 1 | `RandomResizedCrop scale=(0.6, 1.0)` | Imperfect detection crop simulation |
| 2 | `RandomHorizontalFlip p=0.5` | Lateral symmetry — no vertical flip |
| 3 | `ColorJitter brightness=0.3 hue=0.05` | Cross-camera lighting variance |
| 4 | `GaussianBlur sigma=(0.1, 0.5)` | Low-quality or motion-blurred cameras |
| 5 | `ToTensor` | PIL HWC uint8 → PyTorch CHW float32 |
| 6 | `Normalize` ImageNet stats | Equal variance across channels |
| 7 | `RandomErasing p=0.5` | Occlusion simulation — forces global representation |

`ToTensor` is the mandatory boundary — `Normalize` and `RandomErasing` require
tensors and cannot appear before it.

### Test pipeline

`Resize → ToTensor → Normalize` — fully deterministic. Query and gallery
embeddings must be identical across runs for kNN retrieval to be reproducible.
The normalization constants must be strictly identical to training:

$$\text{pixel}_{\text{norm}} = \frac{\text{pixel} - \mu}{\sigma}, \quad \mu = [0.485,\ 0.456,\ 0.406], \quad \sigma = [0.229,\ 0.224,\ 0.225]$$

---

## Dataset — `data/dataset.py`

Implements the standard PyTorch `Dataset` interface. Reads `train_label.xml`
once at construction and builds two parallel lists:

```
self.samples[i] = (img_path, vehicle_id, camera_id)
self.labels[i]  = vehicle_id
```

`self.labels[i]` is always the `vehicle_id` of `self.samples[i]`.
`PKSampler` reads only `self.labels` — it never touches image files.

`vehicleID` defaults to `-1` when absent — `query_label.xml` and `test_label.xml`
do not carry `vehicleID` because it is the ground truth kept secret by the
competition organisers. Providing it as input would be data leakage.

### Local evaluation split — `make_train_eval_split()`

Since the official test ground truth is unavailable locally, a proper 3-way
split is constructed from `train_label.xml` via `make_train_eval_split()`:

```
full real train (440 identities, 52 717 images)
    │
    ├── train_ds   (400 identities, ~47 800 images) ← train_transform
    ├── query_ds   (40 identities,  40 images)       ← eval_transform
    └── gallery_ds (40 identities,  ~4 800 images)   ← eval_transform
```

The 40 eval identities are **never seen during training** — strict separation
eliminates data leakage and produces meaningful mAP values.

**Why 40 out of 440 (10%)?**

In standard classification, 80/20 splits are common because the same classes
appear in both train and test. Re-ID is fundamentally different — test identities
must be **entirely unseen** during training, since the model must generalize to
new vehicles, not memorize known ones.

This changes the split logic entirely:

- **Too few eval identities (e.g. 10)** — only 10 queries, statistically
  unreliable mAP. A single lucky or unlucky query shifts the metric significantly.

- **Too many eval identities (e.g. 80-100)** — 18-23% of identities withheld
  from training. With only 52k images and a model training from scratch without
  pretrained weights, reducing training diversity significantly hurts generalisation.

- **40 identities (9%)** — follows the standard Re-ID evaluation protocol used
  in the literature. Provides 40 queries and ~4 800 gallery images for statistically
  meaningful metrics, while preserving 400 identities (91%) for training —
  the right balance for a from-scratch model on a medium-sized dataset.

---

## Batch Construction — `data/batch.py`

### Why PKSampler

Standard random shuffle cannot guarantee that multiple images of the same vehicle
appear together in a batch. The batch-hard triplet loss requires it — it mines
the hardest positive and hardest negative **within the batch**.

PKSampler guarantees every batch contains exactly:

| | Value | Formula |
|---|---|---|
| Identities per batch | 20 | P |
| Images per identity | 8 | K |
| Batch size | 160 | P × K |
| Positives per anchor | 7 | K − 1 |
| Negatives per anchor | 152 | (P − 1) × K |

### Algorithm

At construction — `_build_index` groups all dataset indices by identity:

$$\text{index per identity[vid]} = [idx_1, idx_2, idx_3, ...]$$

At each epoch — `__iter__` shuffles the 400 training identities, slices P at a time,
samples K indices per identity, yields all P×K indices as a flat sequence.

$$\text{num}_\text{batches} = \lfloor 400/20 \rfloor = 20 \quad \Rightarrow \quad 20 \times 160 = 3200 \text{ images per epoch}$$

Not all 47 800 images are seen every epoch — coverage builds across epochs as
identities are reshuffled.

---

## DataLoaders — `data/dataloader.py`

Three functions, one per split. The differences are deliberate:

| | Train | Query | Test |
|---|---|---|---|
| Sampler | PKSampler | none | none |
| `shuffle` | forbidden | `False` | `False` |
| `drop_last` | `True` | `False` | `False` |
| `batch_size` | 160 | 128 | 128 |

`drop_last=True` for train — an incomplete batch has fewer than P×K images and
breaks the triplet loss structure.

`shuffle=False` for query and test — embeddings are stored sequentially and
matched by position against ground-truth labels. Any reordering corrupts kNN.

`pin_memory=True` on all three — keeps tensors in page-locked CPU memory,
which the GPU can access directly via DMA without OS scheduling overhead.
