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

## Dataset — AIC21 Track 2 ReID

The project uses the **AI City Challenge 2021 Track 2** benchmark for city-scale vehicle re-identification.
The dataset contains **85 058 cropped vehicle images** across **880 unique identities**, captured by **46 non-overlapping cameras** in real traffic conditions.

---

#### Dataset splits

| Split | Folder | Images | Identities | Role |
|---|---|---|---|---|
| Training | `image_train/` | 52 717 | 440 | Model training (+ VehicleX synthetic) |
| Gallery | `image_test/` | 31 238 | 440 | Retrieval target at evaluation |
| Query | `image_query/` | 1 103 | 440 | Images to match against the gallery |

> Training and test identities are **disjoint** — the model never sees test vehicles during training.
> This forces genuine metric learning rather than memorization.

---

### Annotation files
 
| File | Content | Used by |
|---|---|---|
| `train_label.xml` | `vehicleID` + `cameraID` per training image. The only file with both labels — `vehicleID` drives PK batch construction, `cameraID` filters same-camera pairs at evaluation | `dataset.py → _parse_xml()` |
| `test_label.xml` | `cameraID` per gallery image. No `vehicleID` by design — it is the ground truth the model must predict; providing it would be data leakage | Evaluation protocol |
| `query_label.xml` | `cameraID` per query image. Same rationale as `test_label.xml` | Evaluation protocol |
| `train_track.txt` | Each line lists all images of the same vehicle on the same camera in temporal order. Enables Query Expansion: averaging track embeddings produces a more robust query vector than a single image (technique used by 2021 winners, optional for MVP) | Query expansion (Optional) |
| `test_track.txt` | Same structure as `train_track.txt` for the gallery split | Query expansion (Optional) |
| `name_train.txt` | Plain list of the 52 717 filenames in `image_train/` | Sanity check |
| `name_test.txt` | Plain list of the 31 238 filenames in `image_test/` | Sanity check |
| `name_query.txt` | Plain list of the 1 103 filenames in `image_query/` | Sanity check |
 
> A retrieved gallery image is counted as a true positive only when it shows the same vehicle on a **different** camera.
> Same vehicle, same camera = ignored. This enforces genuine cross-camera retrieval.

**Why `cameraID` matters at evaluation:**
a match is only counted as a true positive when the retrieved image shows the **same vehicle on a different camera**.
Same vehicle + same camera = ignored. This enforces cross-camera retrieval.

#### XML format (`train_label.xml`)
```xml
<TrainingImages Version="1.0">
  <Items number="52717">
    <Item imageName="000001.jpg" vehicleID="001" cameraID="c036"/>
    <Item imageName="000002.jpg" vehicleID="001" cameraID="c043"/>
    ...
  </Items>
</TrainingImages>
```

---

#### Evaluation output format

For each query image, the model produces one `.txt` file listing gallery image IDs ranked by **ascending embedding distance**:

```
# 000001.txt  — results for image_query/000001.jpg
5021          ← closest gallery image  (image_test/005021.jpg)
12701
19500
13169
...           ← top-50 matches
```

The evaluation script reads these files and computes **Rank-1 accuracy** and **mAP** against the ground-truth labels.
An example of the expected format is provided in `tools/dist_example/`.

---

#### Two-dataset merge strategy

```mermaid
flowchart TD
    A["train_label.xml\n image_train/\n VehicleX + AIVehicle3D"] -->|"_parse_xml()"| B["TrainDataset\n samples + labels"]
    B -->|"PKSampler\n P=16 x K=4"| C["Batch 64\n 16 ids x 4 imgs"]
    C -->|"Triplet Loss"| D["Trained ViT"]
 
    E["query_label.xml\n image_query/"] -->|"_parse_xml()"| F["QueryDataset"]
    G["test_label.xml\n image_test/"] -->|"_parse_xml()"| H["GalleryDataset"]
 
    D -->|"encode"| F
    D -->|"encode"| H
 
    F --> I["1 103 x 128 \n query embeddings"]
    H --> J["31 238 x 128 \n gallery embeddings"]
 
    I --> K["kNN\n1103 x 31238"]
    J --> K
 
    K -->|"top-50"| L["000001.txt ..."]
    L --> M["Rank-1 · mAP"]
 
    style A fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style B fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style C fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style D fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style E fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    style F fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    style G fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    style H fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    style I fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style J fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style K fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style L fill:#FCEBEB,stroke:#E24B4A,color:#501313
    style M fill:#FCEBEB,stroke:#E24B4A,color:#501313
```

> **Key rule:** the synthetic dataset is used **only for training**.
> The gallery (`image_test/`) and queries (`image_query/`) are always real images exclusively.

---

#### Visualization tool

`tools/visualize.py` is a Tkinter GUI that renders the retrieval results visually.
It reads the `.txt` files from a results directory and displays the query image alongside its top-50 gallery matches.

> ⚠️ The script is written in **Python 2** — port to Python 3 before use (`Tkinter` → `tkinter`, `print` statements, etc.).

---

## Pipeline

### 1. Artificial data generation — `data/vehiclex/`
   - 1.1. **VehicleX synthetic images** — 3D-rendered cropped vehicles with controlled viewpoints, lighting and backgrounds added to the training set
   - 1.2. **Label alignment** — synthetic images are assigned real vehicle identities and camera IDs compatible with `train_label.xml`

### 2. Data augmentation — `data/data_transforms.py`
   - 2.1. **RandomResizedCrop** `scale=(0.6, 1.0)` — simulates imperfect detection crops
   - 2.2. **HorizontalFlip** `p=0.5` — lateral symmetry of vehicles; no vertical flip (invalid viewpoint)
   - 2.3. **ColorJitter** `brightness · contrast · saturation` — cross-camera lighting variance; hue kept minimal to preserve vehicle color identity
   - 2.4. **GaussianBlur** `σ ∈ [0.1, 0.5]` — low-quality or motion-blurred cameras
   - 2.5. **RandomErasing** `p=0.5, scale=(0.02, 0.2)` — occlusion simulation (poles, other vehicles); forces global representation

### 3. Normalization — `data/data_transforms.py`
   - 3.1. **ToTensor** — PIL HWC uint8 → PyTorch CHW float32 [0, 1]
   - 3.2. **Normalize** `μ=[0.485, 0.456, 0.406]  σ=[0.229, 0.224, 0.225]` — ImageNet stats; equal variance across channels stabilizes gradient descent

### 4. Batch construction — `data/batch.py` + `data/dataloader.py`
   - 4.1. **PK sampling** `P=16 identities × K=4 images = batch of 64` — PKSampler builds each batch with exactly P identities and K images each; guarantees positives and negatives are always present for the triplet loss
   - 4.2. **DataLoader** — delivers batches to the model; `pin_memory=True` accelerates CPU → GPU transfer; `num_workers=4` parallelizes image loading; `drop_last=True` ensures every batch has exactly P×K images

### 5. Patch embedding — `model/patch_embedded.py`
   - 5.1. **Conv2d** `kernel=16, stride=16, out=192` — splits 3×224×224 into 196 patches, projects each to 192-d in one operation
   - 5.2. **Flatten + Transpose** — reshapes 192×14×14 → sequence of **196 × 192** tokens

### 6. CLS token + positional embedding — `model/vit.py`
   - 6.1. **CLS token** `nn.Parameter(1×192)` — learnable vector prepended → sequence becomes **197 × 192**; aggregates image-level representation through attention
   - 6.2. **Learned positional embedding** `nn.Parameter(197×192)` — added element-wise; self-attention is permutation-invariant, position must be injected back
   - 6.3. **GAP (alternative)** — mean over 196 patch tokens; simpler aggregation, no CLS needed

### 7. Transformer encoder ×6 — `model/block.py` + `model/attention.py`

Each block applies two sub-modules, each wrapped in a **skip connection (+)**:

   - 7.1. **LayerNorm** — normalizes each token independently: `u' = γ⊙(u−μ)/σ + β`; μ, σ computed over 192 features of a single token (not the batch)
   - 7.2. **Multi-head self-attention** `8 heads × 24-d` — `Attention(Q,K,V) = softmax(QKᵀ / √24) · V`; each head learns a different relation (shape, color, position…)
   - 7.3. **Skip connection ⊕** — `X = X + Attention(LayerNorm(X))`; gradient highway, prevents vanishing
   - 7.4. **LayerNorm** — second normalization before FFN
   - 7.5. **FFN** — `Linear(192→768) → GELU → Dropout(0.1) → Linear(768→192)`; hidden dim = 4 × d_model; GELU smooth activation
   - 7.6. **Skip connection ⊕** — `X = X + FFN(LayerNorm(X))`

### 8. Projection head — `model/vit.py`
   - 8.1. **Extract CLS token** — take position 0 of the output sequence (1 × 192)
   - 8.2. **Linear** `192 → 128` — compact embedding for fast nearest-neighbor retrieval
   - 8.3. **L2 normalize** `‖f(x)‖ = 1` — projects onto unit hypersphere; cosine distance ≡ euclidean distance

### 9. Loss — `losses/triplet.py`
   - 9.1. **Batch-hard mining** — for each anchor: hardest positive `max d(a,p)` + hardest negative `min d(a,n)` within the batch
   - 9.2. **Triplet loss** `max(0, d(a,p) − d(a,n) + 0.3)` — margin=0.3 enforces a separation buffer; loss=0 → no gradient (monitor active triplet fraction)

### 10. Optimization — `engine/train.py` + `utils/scheduler.py`
   - 10.1. **AdamW** `lr=1e-4, β₁=0.9, β₂=0.999` — adaptive per-parameter learning rate; W = weight decay decoupled from gradient
   - 10.2. **Linear warmup** — lr: 0 → 1e-4 over 5 epochs; stabilizes early training when weights are random
   - 10.3. **Cosine decay** — lr smoothly decreases to 0 after warmup; avoids sharp drops
   - 10.4. **Weight decay** `λ=0.01` — penalty `λ‖θ‖²` discourages large weights, reduces overfitting
   - 10.5. **Dropout** `p=0.1` — applied in FFN and on attention weights; stochastic regularization

### 11. Evaluation — `engine/evaluate.py` + `monitoring/logger.py`
   - 11.1. **Extract embeddings** `model.eval() + no_grad()` — forward pass on all query and gallery images, no augmentation
   - 11.2. **kNN search** — cosine distance matrix: `dist = 1 − query_emb @ gallery_emb.T`; rank by ascending distance
   - 11.3. **Rank-1** — fraction of queries where top-1 retrieved image shares the same identity
   - 11.4. **mAP** — mean Average Precision; area under precision-recall curve averaged over all queries; **target > 36.0% val mAP**

## Architecture default

| Parameter | Value | Why |
|---|---|---|
| Variant | ViT-Tiny | ~5M params · fits 52k training images without overfitting |
| Depth *L* | 6 | enough layers for global attention; shallow enough to train from scratch |
| Heads | 8 | 8 parallel attention subspaces; each specializes on a different visual relation |
| d<sub>model</sub> | 192 | standard Tiny width; divisible by 8 heads → 24-d per head |
| FFN hidden dim | 768 | 4 × d<sub>model</sub>; standard transformer ratio |
| Embedding dim | 128 | compact for fast nearest-neighbor retrieval at inference |
| Patch size | 16 | 196 tokens on 224² input; attention is O(N²), patch=8 would 4× memory |
| Input | 224 × 224 | ImageNet convention; compatible with normalization stats |
| Dropout | 0.1 | applied in FFN and attention weights; stochastic regularization |
| Positional emb. | Learned | nn.Parameter 197×192; better than sinusoidal for 2D image structure |
| Aggregation | CLS token | position 0 of output sequence; aggregates via attention across all layers |
| Init | trunc\_normal std=0.02 | keeps early activations small; avoids exploding signals through residuals |


```mermaid
flowchart TD
    classDef araug fill:#EEEDFE,stroke:#7F77DD,color:#3C3489
    classDef aug fill:#EEEDFE,stroke:#7F77DD,color:#3C3489
    classDef norm fill:#E1F5EE,stroke:#1D9E75,color:#085041
    classDef batch fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    classDef model fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    classDef attn fill:#B5D4F4,stroke:#185FA5,color:#042C53
    classDef ffn fill:#C0DD97,stroke:#3B6D11,color:#173404
    classDef head fill:#FAECE7,stroke:#D85A30,color:#4A1B0C
    classDef loss fill:#FCEBEB,stroke:#E24B4A,color:#501313
    classDef optim fill:#EAF3DE,stroke:#639922,color:#173404
    classDef eval fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    classDef skip fill:#F4C0D1,stroke:#D4537E,color:#4B1528

    subgraph AA["⓪ Artificial Data Generation — data/vehiclex/"]
        AA1["VehicleX synthetic images"]:::araug
        AA2["Label alignment"]:::araug
        AA1-->AA2
    end

    subgraph DA["① Data Augmentation — data/data_transforms.py"]
        A1["RandomResizedCrop\nscale 0.6–1.0"]:::aug
        A2["HorizontalFlip p=0.5"]:::aug
        A3["ColorJitter\nbright · contrast · sat · hue 0.05"]:::aug
        A4["GaussianBlur σ 0.1–0.5"]:::aug
        A5["RandomErasing p=0.5\nscale 0.02–0.2"]:::aug
        A1-->A2-->A3-->A4-->A5
    end

    subgraph NO["② Normalization — data/data_transforms.py"]
        B1["ToTensor\nPIL HWC uint8 → CHW float32"]:::norm
        B2["Normalize\nμ=[.485,.456,.406]  σ=[.229,.224,.225]"]:::norm
        B1-->B2
    end

    subgraph BA["③ Batch Construction — data/batch.py + data/dataloader.py"]
        BA1["PKSampler\nP=16 identities × K=4 images = 64"]:::batch
        BA2["DataLoader\npin_memory=True · num_workers=4 · drop_last=True"]:::batch
        BA1-->BA2
    end

    subgraph PE["④ Patch Embedding — model/patch_embedded.py"]
        C1["Conv2d k=16 s=16\n3×224×224 → 196 × 192"]:::model
        C2["Flatten + Transpose\n196 × 192 tokens"]:::model
        C1-->C2
    end

    subgraph CLS["⑤ CLS + Positional Embedding — model/vit.py"]
        D1["CLS token\nnn.Parameter 1×192"]:::model
        D2["Prepend → 197 tokens"]:::model
        D3["+ Pos. Embedding\nnn.Parameter 197×192 learned"]:::model
        D4["GAP — mean over 196 tokens\nalternative to CLS"]:::model
        D1-->D2-->D3-->D4
    end

    subgraph TR["⑥ Transformer ×6 — model/block.py + model/attention.py"]
        E1["LayerNorm\nμ σ per token"]:::eval
        E2["Multi-Head Attention\n8 heads × 24-d  softmax·QKᵀ/√24·V"]:::attn
        E3["⊕ skip connection\nX = X + Attention·LayerNorm·X"]:::skip
        E4["LayerNorm\nμ σ per token"]:::eval
        E5["FFN\nLinear 192→768 · GELU · Dropout 0.1 · Linear 768→192"]:::ffn
        E6["⊕ skip connection\nX = X + FFN·LayerNorm·X"]:::skip
        E1-->E2-->E3-->E4-->E5-->E6
    end

    subgraph PH["⑦ Projection Head — model/vit.py"]
        F1["Extract CLS token\nposition 0 · 1×192"]:::head
        F2["Linear 192→128"]:::head
        F3["L2 Normalize  ‖f·x‖=1\nunit hypersphere"]:::head
        F4["Embedding ℝ¹²⁸"]:::head
        F1-->F2-->F3-->F4
    end

    subgraph LO["⑧ Loss — losses/triplet.py"]
        G1["Batch-hard Mining\nhardest pos max·d·a,p + hardest neg min·d·a,n"]:::loss
        G2["Triplet Loss\nmax·0, d·a,p − d·a,n + 0.3"]:::loss
        G1-->G2
    end

    subgraph OP["⑨ Optimization — engine/train.py + utils/scheduler.py"]
        H1["AdamW\nlr=1e-4  β₁=0.9  β₂=0.999"]:::optim
        H2["Linear Warmup\n0 → lr over 5 epochs"]:::optim
        H3["Cosine Decay\nlr → 0 smoothly"]:::optim
        H4["Weight decay λ=0.01\nDropout p=0.1"]:::optim
        H1-->H2-->H3-->H4
    end

    subgraph EV["⑩ Evaluation — engine/evaluate.py + monitoring/logger.py"]
        I1["Extract embeddings\nmodel.eval + no_grad"]:::eval
        I2["kNN cosine distance\ndist = 1 − query @ gallery.T"]:::eval
        I3["Rank-1"]:::eval
        I4["mAP  target > 36%"]:::eval
        I1-->I2-->I3-->I4
    end

    AA-->DA-->NO-->BA-->PE-->CLS-->TR-->PH-->LO-->OP-->EV

    style AA fill:#EEEDFE22,stroke:#7F77DD,stroke-width:2px
    style DA fill:#EEEDFE22,stroke:#7F77DD,stroke-width:2px
    style NO fill:#E1F5EE22,stroke:#1D9E75,stroke-width:2px
    style BA fill:#FFF4E522,stroke:#E8A020,stroke-width:2px
    style PE fill:#E6F1FB22,stroke:#378ADD,stroke-width:2px
    style CLS fill:#FAEEDA22,stroke:#EF9F27,stroke-width:2px
    style TR fill:#E6F1FB22,stroke:#185FA5,stroke-width:2px
    style PH fill:#FAECE722,stroke:#D85A30,stroke-width:2px
    style LO fill:#FCEBEB22,stroke:#E24B4A,stroke-width:2px
    style OP fill:#EAF3DE22,stroke:#639922,stroke-width:2px
    style EV fill:#F1EFE822,stroke:#888780,stroke-width:2px
```

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
