# Training — `engine/train.py`

## Overview

`train.py` implements one training epoch. `main.py` calls `train_one_epoch()`
in a loop over 50 epochs and handles checkpointing, evaluation and logging.

---

## Architecture

```mermaid
flowchart LR
    A["main.py"] -->|"calls once per epoch"| B["train_one_epoch()"]
    B -->|"metrics dict"| A

    SCH["utils/scheduler.py\nSequentialLR"] -->|"scheduler.step()"| B
    OPT["AdamW\noptimizer"] -->|"optimizer.step()"| B
    DL["data/dataloader.py\nPKSampler batch iterator"] -->|"64 images · 64 labels"| B
    VIT["model/vit.py\nVehicleViT"] -->|"forward pass"| B
    LOSS["losses/triplet.py\nBatchHardTripletLoss"] -->|"loss · active fraction"| B

    style A fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style B fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style SCH fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style OPT fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style DL fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style VIT fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style LOSS fill:#FCEBEB,stroke:#E24B4A,color:#501313
```

---

## Batch loop — one epoch

```mermaid
flowchart TD
    START(["epoch start\nmodel.train()"]) --> LOOP

    LOOP["for images, vehicle_ids in dataloader\n27 batches · 64 images each"] --> ZG

    ZG["optimizer.zero_grad()\nclears gradients from previous batch"] --> FW

    FW["forward pass\nmodel images\n64 × 3 × 224 × 224  →  64 × 128"] --> TL

    TL["BatchHardTripletLoss\nembeddings · vehicle_ids\nmax 0 · d a p - d a n + 0.3"] --> BW

    BW["loss.backward()\ncomputes ∂loss/∂θ\nfor every ViT parameter"] --> OS

    OS["optimizer.step()\nAdamW updates weights\nθ = θ - lr × gradient"] --> ACC

    ACC["accumulators\ntotal_loss += loss.item()\ntotal_active += active.item()"] --> CHK

    CHK{"more batches ?"} -->|"yes"| LOOP
    CHK -->|"no"| SCH

    SCH["scheduler.step()\nwarmup + cosine decay\nonce per epoch"] --> RET

    RET(["return dict\nloss · active_triplets · lr"])

    style START fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style LOOP fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style ZG fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style FW fill:#E6F1FB,stroke:#378ADD,color:#0C447C
    style TL fill:#FCEBEB,stroke:#E24B4A,color:#501313
    style BW fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style OS fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style ACC fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style CHK fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style SCH fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style RET fill:#E1F5EE,stroke:#1D9E75,color:#085041
```

---

## Key steps

### `model.train()`
Activates dropout — 10% of activations randomly zeroed at each forward pass.
Disabled automatically at `model.eval()` for deterministic kNN embeddings.

### `optimizer.zero_grad()`
PyTorch accumulates gradients by default. Without reset, gradients from batch
$t$ accumulate into batch $t+1$ — weights are updated with corrupted information.
Called **before** the forward pass, once per batch.

### Forward pass
```
(64, 3, 224, 224)  →  patch_embed  →  (64, 196, 192)
                   →  CLS + pos_embed + dropout
                   →  Transformer × 6
                   →  CLS + norm + proj_head + L2
                   →  (64, 128)   L2-normalized embeddings
```

### `BatchHardTripletLoss`
Receives `embeddings (64, 128)` and `vehicle_ids (64,)`.
Mines hardest positive and hardest negative per anchor within the batch.
Returns `loss` (scalar) and `active` (fraction of non-zero triplets).
`loss.backward()` propagates gradients through the full ViT graph.

### `optimizer.step()`
AdamW reads the gradients and updates all ViT weights:
$$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{v_t}+\epsilon}m_t - \gamma\lambda\theta_t$$
Uses the lr currently set by the scheduler.

### `scheduler.step()`
Called **once per epoch**, after all batches — not inside the batch loop.
27 batches per epoch — calling inside the loop would decay the lr 27× too fast.

---

## Connections

| Module | Role |
|---|---|
| `data/dataloader.py` | PKSampler batch iterator — 27 batches × 64 images |
| `model/vit.py` | Forward pass — images to L2-normalized embeddings |
| `losses/triplet.py` | Batch-hard triplet loss — loss + active fraction |
| `utils/scheduler.py` | Built in `main.py`, stepped once per epoch |
| `engine/evaluate.py` | Called by `main.py` every N epochs |
| `monitoring/` | Logger, triplet health, gradient health |
