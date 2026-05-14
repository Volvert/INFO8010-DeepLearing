# Optimization

## Overview

The optimization strategy combines three complementary components,
each addressing a distinct aspect of training stability and generalization.

| Component | File | Value | Role |
|---|---|---|---|
| Optimizer | `engine/train.py` | AdamW `lr=5e-4, β₁=0.9, β₂=0.999, λ=0.01` | Adaptive per-parameter learning rate + decoupled weight decay |
| Scheduler | `utils/scheduler.py` | Linear warmup 50 epochs + cosine decay to epoch 2000 | Controls the global learning rate over time |
| Regularization | `model/block.py`, `model/vit.py` | Dropout `p=0.1` | Stochastic regularization inside the model |

---

## Scheduler — `utils/scheduler.py`

### Why a fixed learning rate is not enough

The learning rate $\gamma$ controls the step size in parameter space at each gradient descent update:

$$\theta_{t+1} = \theta_t - \gamma \nabla_\theta \mathcal{L}$$

A single fixed $\gamma$ cannot be optimal throughout training — the model needs different
step sizes at different stages. Too large early on causes divergence. Too small at the end
wastes training time. The scheduler adapts $\gamma$ over time. (lec4 page 37)

---

### Phase 1 — Linear warmup (epochs 0 → 50)

At epoch 0, all ViT weights are initialized with `trunc_normal(std=0.02)` — nearly random.
The triplet loss computes distances between random embeddings → 100% of triplets are active
→ gradients are large and noisy in all directions simultaneously.

Starting with `lr=5e-4` immediately causes the first updates to be too large —
random weights produce erratic gradients that can destroy the initialization before
the model has a chance to stabilize.

```
epoch  0 :  lr = 0.0
epoch 10 :  lr = 0.2 × 5e-4
epoch 20 :  lr = 0.4 × 5e-4
epoch 30 :  lr = 0.6 × 5e-4
epoch 40 :  lr = 0.8 × 5e-4
epoch 50 :  lr = 5e-4   ← target lr reached
```

**Why linear and not cosine warmup ?**
Over 50 epochs the shape of the curve has negligible impact — what matters is
the start (0) and the end (5e-4). Linear is the simplest and most interpretable.

**Why 50 epochs ?**
With `lr=5e-4` — higher than the standard `1e-4` — the model needs a longer
stabilisation phase. 50 epochs gives the embeddings time to move from fully
random to an initial coherent structure before the full learning rate kicks in.

---

### Phase 2 — Cosine decay (epochs 50 → 2000)

After warmup, the learning rate decays smoothly following a cosine curve (lec4 page 39):

$$\gamma_t = \gamma_{\min} + \frac{1}{2}(\gamma_{\max} - \gamma_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}} \cdot \pi\right)\right)$$

```
epoch   50 :  lr = 5e-4      ← maximum
epoch  500 :  lr ≈ 4e-4
epoch 1000 :  lr ≈ 2.5e-4
epoch 1500 :  lr ≈ 1e-4
epoch 2000 :  lr ≈ 0         ← minimum
```

```
lr
5e-4 |          ___________
     |        /             \
     |      /                 \
     |    /                     \
     |  /                         \
   0 |/                             \___
     |-----|--------------------------|--→ epochs
     0    50                        2000
      warmup        cosine decay
```

**Why cosine and not step decay or exponential ?**

The course (lec4 page 37) presents three decay strategies — step, exponential and 1/t.
All three have the same limitation compared to cosine:

**Step decay** $\gamma_t = \gamma_0 \times f^{\lfloor t/s \rfloor}$ — reduces lr by a factor
$f$ every $s$ epochs. The drops are brutal and discontinuous — the loss visibly jumps
at each step. The model can oscillate just before a drop and over-adapt just after.

**Exponential decay** $\gamma_t = \gamma_0 \exp(-kt)$ — decays very fast early then stays
near zero for too long. The model has little time to explore before being locked with a
negligible lr.

**Cosine decay** — the derivative is 0 at the top and bottom of the curve.
No discontinuity, no brutal drop. The decay accelerates in the middle and slows
at the end — the model can settle finely into a good minimum without oscillating.

The course (lec4 page 39) shows the warmup + cosine graph explicitly and states
that these schedules are useful to *"escape from sharp minima and avoid overfitting"* —
exactly our situation: ViT from scratch, 52k images, high overfitting risk.

---

## AdamW — `engine/train.py`

### Why not SGD

SGD applies one global lr to all parameters:

$$\theta_{t+1} = \theta_t - \gamma \nabla_\theta \mathcal{L}$$

In a ViT, parameters are highly heterogeneous — QKV projections, FFN weights,
`pos_embed`, `cls_token` all have gradients of very different magnitudes.
A single lr is too large for some, too small for others. On 50 epochs with 52k images,
SGD converges much more slowly.

### Why Adam → AdamW

Adam adapts the lr per parameter by tracking gradient history:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla\mathcal{L} \quad \text{(gradient mean)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla\mathcal{L})^2 \quad \text{(gradient variance)}$$

$$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{v_t} + \epsilon} m_t$$

Parameters with stable gradients → large steps. Parameters with noisy gradients → small steps.
`β₁=0.9, β₂=0.999` — momentum over the last ~10 and ~1000 gradients respectively.

**The Adam weight decay problem:**
In standard Adam, weight decay is applied inside the adaptive update — it is incorrectly
scaled by $1/\sqrt{v_t}$. This weakens the regularization effect unpredictably.

**AdamW decouples weight decay from the gradient:**

$$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{v_t}+\epsilon}m_t - \gamma\lambda\theta_t$$

The penalty $\lambda\|\theta\|^2$ is applied directly to the weights, independently of
the gradient history. This produces consistent, well-calibrated regularization —
empirically shown to generalize better on Transformer architectures. (lec4 — weight decay)

`λ=0.01` — light enough not to constrain vehicle feature learning, strong enough to
discourage large weights.

---

## References

| Source | Link |
|---|---|
| lec4 page 37 — Scheduling | lec4.pdf |
| lec4 page 39 — Warmup and cosine schedule | lec4.pdf |
| lec4 — Adaptive learning rate (Adam) | lec4.pdf |
| lec4 — Weight decay | lec4.pdf |
| Loshchilov & Hutter, "Decoupled Weight Decay Regularization", 2019 | https://arxiv.org/abs/1711.05101 |
