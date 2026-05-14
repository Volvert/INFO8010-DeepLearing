# =============================================================================
# GradientHealthMonitor
# =============================================================================
"""
Reads .grad tensors after loss.backward() and returns per-group L2 norms.

Must be called AFTER loss.backward() and BEFORE optimizer.step() — this is
the only window where .grad is populated and readable.

Returns a flat dict of gradient norms with keys prefixed "grad_norm_*",
plus two anomaly flags "grad_vanishing" and "grad_exploding".
All keys are merged into train_metrics by train_one_epoch() and passed
to logger.log_epoch() for CSV storage and plot.py for visualization.

Grouping strategy:
    Parameters are grouped by architectural unit using str.startswith().
    One L2 norm per group — readable and easy to plot.
    Blocks are detected dynamically from model.named_parameters() so the
    monitor works regardless of the depth set in tiny_vit.yaml.

Thresholds:
    global_norm > 10.0  → grad_exploding = True  → clipping triggered in train.py
    any group   < 1e-6  → grad_vanishing  = True  → that layer is not learning

See: engine/train.py — calls compute() after backward(), uses grad_exploding for clipping
See: monitoring/logger.py — receives the returned dict via train_metrics
"""

"""
Monitoring code done with copilote help
"""

import re
import torch.nn as nn

EXPLODE_THRESHOLD = 10.0
VANISH_THRESHOLD  = 1e-6

# Static groups — always present in VehicleViT
_STATIC_PREFIXES = [
    ("grad_norm_patch_embed", "patch_embed"),
    ("grad_norm_cls_token",   "cls_token"),
    ("grad_norm_pos_embed",   "pos_embed"),
    ("grad_norm_norm",        "norm."),        # trailing dot avoids norm1/norm2 inside blocks
    ("grad_norm_proj_head",   "proj_head"),
]

# Regex to detect block indices dynamically — e.g. "transformer.blocks.3.*"
_BLOCK_RE = re.compile(r"^transformer\.blocks\.(\d+)\.")


class GradientHealthMonitor:
    """
    Computes per-group gradient L2 norms after each backward pass.

    Attributes:
        log_every_n_batches : only compute every N batches (default 1)
    """

    def __init__(self, log_every_n_batches: int = 1):
        self.log_every_n_batches = log_every_n_batches
        self._batch_counter      = 0

    def compute(self, model: nn.Module) -> dict:
        """
        Reads .grad from all parameters and returns gradient norms.

        Args:
            model : VehicleViT with .grad populated after loss.backward()

        Returns:
            dict with keys:
                "grad_norm_global"     : float
                "grad_norm_patch_embed": float
                "grad_norm_cls_token"  : float
                "grad_norm_pos_embed"  : float
                "grad_norm_blocks.N"   : float  for each block N in the model
                "grad_norm_norm"       : float
                "grad_norm_proj_head"  : float
                "grad_vanishing"       : bool
                "grad_exploding"       : bool
            or {"grad_skipped": True} if this batch is not a logging step.
        """
        self._batch_counter += 1
        if self._batch_counter % self.log_every_n_batches != 0:
            return {"grad_skipped": True}

        # accumulate squared gradient sums
        global_sq   = 0.0
        static_sq   = {key: 0.0 for key, _ in _STATIC_PREFIXES}
        block_sq:  dict[str, float] = {}   # "grad_norm_blocks.N" → sq_sum

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            sq = param.grad.detach().pow(2).sum().item()
            global_sq += sq

            # dynamic block detection
            m = _BLOCK_RE.match(name)
            if m:
                key = f"grad_norm_blocks.{m.group(1)}"
                block_sq[key] = block_sq.get(key, 0.0) + sq
                continue

            # static groups
            for key, prefix in _STATIC_PREFIXES:
                if name.startswith(prefix):
                    static_sq[key] += sq
                    break

        # compute norms
        result: dict = {"grad_norm_global": global_sq ** 0.5}
        result.update({k: v ** 0.5 for k, v in static_sq.items()})
        result.update({k: v ** 0.5 for k, v in sorted(block_sq.items())})

        # anomaly flags
        all_group_norms = (list(static_sq.values()) +
                           list(block_sq.values()))
        result["grad_vanishing"] = any(
            0.0 < v ** 0.5 < VANISH_THRESHOLD for v in all_group_norms
        )
        result["grad_exploding"] = result["grad_norm_global"] > EXPLODE_THRESHOLD

        return result

    def report(self, metrics: dict) -> None:
        """
        Prints a gradient health summary to the console.
        Call once per epoch with the last batch's metrics dict.
        """
        if metrics.get("grad_skipped") or not metrics:
            return

        _R = "\033[0m"; _G = "\033[92m"; _Y = "\033[93m"
        _RED = "\033[91m"; _B = "\033[1m"; _D = "\033[2m"

        def _fmt(v: float) -> str:
            if v < VANISH_THRESHOLD or v > EXPLODE_THRESHOLD:
                return f"{_RED}{v:.2e}{_R}"
            return f"{_Y if v > 2.0 else _G}{v:.2e}{_R}"

        exploding = metrics.get("grad_exploding", False)
        vanishing  = metrics.get("grad_vanishing", False)
        status = (f"  {_RED}{_B}⚠ EXPLODING{_R}" if exploding else
                  f"  {_RED}{_B}⚠ VANISHING{_R}" if vanishing else
                  f"  {_G}✓ healthy{_R}")

        print(f"\n{_B}  Gradient Health{_R}{status}")
        print(f"  {_D}{'─' * 46}{_R}")
        for key, val in metrics.items():
            if key.startswith("grad_norm_"):
                label = key.replace("grad_norm_", "")
                print(f"  {label:<16}  {_fmt(val)}")
        print(f"  {_D}{'─' * 46}{_R}\n")

    def __repr__(self) -> str:
        return (f"GradientHealthMonitor("
                f"log_every_n_batches={self.log_every_n_batches})")