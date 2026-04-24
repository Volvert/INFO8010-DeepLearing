<div align="center">

# Vehicle Re-Identification - ViT from Scratch

*INFO8010 · Deep Learning · ULiège 2025-2026*

**Antoine Deckers** (s170999) · **Florent Volvert** (s203710)

</div>

---

## Goal

Match the same vehicle across a city-scale camera network - **Track 2 of the 2021 NVIDIA AI City Challenge**. Given a query image, retrieve all images of the same vehicle in a gallery of images taken by *other* cameras.

Since new vehicles appear at test time, we do not classify - we learn an embedding function

<div align="center">

*f* : image $\rightarrow$ ℝ¹²⁸

</div>

such that same vehicle $\rightarrow$ close vectors, different vehicles $\rightarrow$ distant vectors. Retrieval is then a nearest-neighbor search.

## Dataset

**AI City Challenge 2021 - Track 2.** 85 058 cropped images, 880 identities across non-overlapping cameras. Real + synthetic ([VehicleX](https://github.com/yorkeyao/VehicleX)).

---

## Pipeline

1. **Data augmentation** — `RandomResizedCrop` · `HorizontalFlip` · `ColorJitter` · light `GaussianBlur` · `RandomErasing` *(simulate occlusion)*
   - injects invariances and mitigates overfitting on 52k images

2. **Normalization** — ImageNet stat mean/std
   - equal variance across input features stabilizes gradient descent

3. **Patch embedding** — image split into 16×16 patches → linear projection
   - transformers operate on token sequences, not raw pixels

4. **CLS token + learned positional embedding + GAP**
   - attention is permutation-invariant, so position must be injected back

5. **Transformer encoder** (×*L* blocks) — each block:
   - LayerNorm → **multi-head self-attention** *(scaled dot-product, √d<sub>k</sub>)* → **residual + skip connection** 
   - LayerNorm → **FFN with GELU** → **residual + skip connection**
   - attention captures long-range relations; residuals + LayerNorm enable deep training

6. **Projection head** — CLS vector → 128-d embedding → L2-normalized
   - compact representation on the unit hypersphere (cosine ≡ euclidean)

7. **Loss** — batch-hard triplet loss with PK sampling (*P* ids × *K* images)
   - metric-learning objective suited to the open-set setting

8. **Optimization** — AdamW + linear warmup + cosine decay, weight decay, dropout
   - modern transformer default; warmup stabilizes early training

9. **Evaluation** — **Rank-1** and **mAP** on the retrieval task
   - standard retrieval metrics; mAP measures full ranking quality

## Architecture default

| | |
|---|---|
| Variant | ViT-Tiny | Why |
| Depth *L* | 6 |
| Heads | 8 |
| d<sub>model</sub> | 192 |
| Embedding dim | 128 |
| Patch size | 16 |
| Input | 224 × 224 |

## Monitoring

| Category | What to log | Why |
|---|---|---|
| **Loss** | `train_loss`, `val_loss` per epoch | detect overfitting (train down while val up) |
| **Retrieval** | `Rank-1`, `mAP` on val split | actual task metric — save best checkpoint on mAP |
| **Triplet health** | fraction of *active* triplets *(loss > 0)*, mean `d(a,p)` vs `d(a,n)` | if 0% active $\rightarrow$ nothing learns; gap should grow |
| **Gradients** | global grad norm + per-layer norm | catches vanishing / exploding gradients |
| **Optimizer** | current `lr` (warmup + cosine curve) | sanity-check schedule |
| **Weights** | mean / std of each block's params | detect dead neurons or drift |
| **Attention** | entropy of attention maps (optional) | high entropy = attention not focusing |
| **System** | GPU mem, throughput (img/s), epoch time | catch memory leaks, plan ablations |

## Target

Beat the **36.0% val mAP** cross-entropy baseline of the 2021 challenge winners.

---

<sub>References: [AI City Challenge](https://www.aicitychallenge.org/2021-challenge-tracks/) · [VehicleX](https://github.com/yorkeyao/VehicleX) · [2021 winners (DMT)](https://github.com/michuanhaohao/AICITY2021_Track2_DMT) · [DINOv3](https://arxiv.org/abs/2508.10104)</sub>
