# Model

## Overview

The model is a Vision Transformer (ViT-Tiny) that maps each input image to a
128-dimensional L2-normalized embedding vector. Same vehicle images produce
close vectors; different vehicle images produce distant vectors. Retrieval is
then a nearest-neighbor search in this embedding space.

$$f : \text{image} \rightarrow \mathbb{R}^{128}, \quad \|f(x)\| = 1$$

| File | Role |
|---|---|
| `model/patch_embedded.py` | Splits each image into 196 patch tokens of dimension 192 |
| `model/vit.py` | CLS token, positional embedding, projection head — orchestrates the full forward pass |
| `model/transformer.py` | Stack of 6 encoder blocks |
| `model/block.py` | One encoder block: LayerNorm + Attention + skip + LayerNorm + FFN + skip |
| `model/attention.py` | Multi-head self-attention mechanism |
| `model/init_model.py` | Instantiates and initializes the final model |

```mermaid
flowchart TD
    A["Input batch\n B × 3 × 224 × 224"] --> B
    B["Patch Embedding\n patch_embedded.py\n B × 196 × 192"] --> C
    C["CLS token prepend\n vit.py\n B × 197 × 192"] --> D
    D["Positional embedding\n vit.py\n B × 197 × 192"] --> E
    E["Transformer x6\n transformer.py + block.py + attention.py\n B × 197 × 192"] --> F
    F["Extract CLS token\n vit.py\n B × 192"] --> G
    G["Projection head\n Linear + L2 normalize\n B × 128"]

    style A fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style B fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style C fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style D fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style E fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style F fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    style G fill:#FCEBEB,stroke:#E24B4A,color:#501313
```

---

## Patch Embedding — `model/patch_embedded.py`

### Theory

A Transformer expects a sequence of vectors as input.
An image is a 2D grid — not a sequence. Patch embedding converts it.

The image is split into $N$ non-overlapping patches of size $P \times P$:

$$N = \frac{H \times W}{P^2} = \frac{224 \times 224}{16^2} = 196 \text{ tokens}$$

Each patch covers $P \times P = 16 \times 16$ pixels across 3 RGB channels = 768 raw values.
A linear projection maps those 768 values to $d_{\text{model}} = 192$ — the working
dimension of the Transformer throughout the entire network.

A single `Conv2d(kernel=P, stride=P)` performs both operations in one GPU pass:
- `kernel=16` covers exactly one patch
- `stride=16` moves by exactly one patch — no overlap, no gap
- `out_channels=192` is the linear projection learned during training

### Tensor flow

```
Input batch        :  (B,   3, 224, 224)
Conv2d k=16 s=16   :  (B, 192,  14,  14)   ← 196 patch positions on a 14×14 grid
flatten(start=2)   :  (B, 192, 196)         ← spatial grid → flat sequence
transpose(1, 2)    :  (B, 196, 192)         ← (batch, seq_len, d_model)
```

### Parameters

| Parameter | Value | Derived from |
|---|---|---|
| `img_size` | 224 | ImageNet convention |
| `patch_size` | 16 | attention is O(N²) — patch=8 would 4× memory |
| `in_channels` | 3 | RGB |
| `d_model` | 192 | ViT-Tiny standard width |
| `num_patches` | 196 | $(224/16)^2 = 14^2 = 196$ |
| Conv2d weights | $3 \times 16 \times 16 \times 192 = 147\,456$ | learned projection |

#### Diagram

```mermaid
flowchart TD
    A["Input batch\n B × 3 × 224 × 224"] --> B
    B["Conv2d\n kernel=16  stride=16  out=192\n B × 192 × 14 × 14"] --> C
    C["flatten start_dim=2\n B × 192 × 196"] --> D
    D["transpose dim 1 and 2\n B × 196 × 192\n 196 tokens  ·  d_model=192"] --> E
    E["Transformer input\n sequence of 196 patch tokens"]

    style A fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style B fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style C fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style D fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style E fill:#FFF4E5,stroke:#E8A020,color:#7A4500
```
---

## CLS Token + Positional Embedding — `model/vit.py`

### CLS token

The CLS token is a learnable vector (`nn.Parameter`, shape `1×192`) prepended
to the patch sequence before the Transformer. It has no spatial meaning — it is
a dedicated slot that aggregates information from all other tokens through
attention across all 6 Transformer layers.

```
Before Transformer :  [CLS,  p1,  p2, ..., p196]   →  197 × 192
After  Transformer :  [CLS', p1', p2', ..., p196']  →  197 × 192
                        ↑
                   only this token is extracted
```

At forward time, the stored CLS token `(1, 1, 192)` is expanded to `(B, 1, 192)`
and prepended along the sequence dimension — producing 197 tokens.

### Positional embedding

Self-attention is permutation-invariant — it only depends on dot products between
token vectors, not their order. Without positional information the Transformer
treats every patch position identically.

The positional embedding is an `nn.Parameter` of shape `1×197×192` added
element-wise to the full sequence including the CLS token:

$$x_i \leftarrow x_i + e_i^{pos} \quad \forall i \in \{0, 1, \ldots, 196\}$$

Stored with batch dimension 1 — PyTorch broadcasting applies it to every image.

### Initialization

Both parameters are initialized with `trunc_normal(std=0.02)` — small values
keep activations stable and prevent exploding signals through residual connections.

### Key operations

| Step | Operation | Input shape | Output shape |
|---|---|---|---|
| Expand CLS | expand to batch size | `1 × 1 × 192` | `B × 1 × 192` |
| Prepend CLS | concatenate on sequence dim | `B × 196 × 192` | `B × 197 × 192` |
| Add pos embed | element-wise addition | `B × 197 × 192` | `B × 197 × 192` |

| Parameter | Shape | Init |
|---|---|---|
| `cls_token` | `1 × 1 × 192` | `trunc_normal std=0.02` |
| `pos_embed` | `1 × 197 × 192` | `trunc_normal std=0.02` |

### Dropout

`Dropout(p=0.1)` is applied to the token sequence immediately after the positional
embedding addition. It randomly zeros 10% of activations during training, preventing
the model from over-relying on specific token positions. Disabled automatically at
`model.eval()`. Also applied inside each encoder block — in the FFN and on attention weights.

---

## Transformer Encoder — `model/transformer.py`

*To be documented — stack of 6 encoder blocks, each processing the full 197-token sequence.*

---

## Encoder Block — `model/block.py`

*To be documented — LayerNorm + Multi-Head Attention + skip connection + LayerNorm + FFN + skip connection.*

---

## Multi-Head Self-Attention — `model/attention.py`

*To be documented — Q K V projections, scaled dot-product attention, 8 heads × 24 dimensions.*

---

## Projection Head — `model/vit.py`

After the Transformer, the CLS token at position 0 is extracted `(B, 192)`.
It passes through a final LayerNorm, a linear projection, and L2 normalization:

```
CLS token  :  (B, 192)
LayerNorm  :  (B, 192)   ← stabilizes input to projection
Linear     :  (B, 128)   ← 192 → 128, learned projection
L2 norm    :  (B, 128)   ← projects onto unit hypersphere
```

**Why 128 dimensions ?**
Compact enough for fast nearest-neighbor retrieval at inference — the distance matrix
`1103 × 31238` is computed in one matrix multiplication. Expressive enough to
separate 440 vehicle identities.

**Why L2 normalization ?**
After normalization all vectors lie on the unit hypersphere: $\|f(x)\| = 1$.
Cosine distance becomes equivalent to Euclidean distance, which is what the
triplet loss minimizes. The embedding space has well-defined geometry.
