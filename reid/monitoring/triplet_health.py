# =============================================================================
# TripletHealthMonitor
# =============================================================================
"""
Computes triplet health statistics from embeddings after the forward pass.

Called in train_one_epoch() BEFORE loss.backward() — embeddings are already
computed, distance matrix recomputation costs < 0.1ms on GPU.

Returns a dict with keys prefixed "th_*" merged into train_metrics by
train_one_epoch() and logged to metrics.csv by logger.py.

The 4 signals:
    th_active_fraction : fraction of anchors with loss > 0  (should decrease over training)
    th_mean_d_pos      : mean hardest positive distance      (should decrease)
    th_mean_d_neg      : mean hardest negative distance      (should increase)
    th_gap             : th_mean_d_neg - th_mean_d_pos       (should grow above margin)
    th_d_pos_std       : std of positive distances           (high = uneven clustering)
    th_d_neg_std       : std of negative distances
    th_collapse        : True if both d_pos and d_neg → 0   (embedding collapse)

See: engine/train.py  — calls compute() and epoch_average()
See: losses/tripletloss.py — same distance logic
"""

import torch
import torch.nn as nn

COLLAPSE_THRESHOLD = 0.05


class TripletHealthMonitor:
    """
    Computes per-batch triplet health statistics from embeddings and labels.

    Replicates the distance mining from BatchHardTripletLoss to extract
    richer statistics than the loss function exposes publicly.

    Attributes:
        log_every_n_batches : compute every N batches (default 1)
    """

    def __init__(self, log_every_n_batches: int = 1):
        self.log_every_n_batches = log_every_n_batches
        self._batch_counter      = 0

    def compute(
        self,
        embeddings: torch.Tensor,   # (B, 128) L2-normalized, detached
        labels:     torch.Tensor,   # (B,) vehicle_ids
        margin:     float = 0.3,
    ) -> dict:
        """
        Computes triplet health for one batch.

        Args:
            embeddings : (B, 128) detached L2-normalized embeddings
            labels     : (B,) vehicle identity labels
            margin     : triplet loss margin (from cfg["training"]["margin"])

        Returns:
            dict with th_* keys, or {"th_skipped": True} if not a logging step.
        """
        self._batch_counter += 1
        if self._batch_counter % self.log_every_n_batches != 0:
            return {"th_skipped": True}

        with torch.no_grad():
            # pairwise distances (B, B)
            dists = torch.cdist(embeddings, embeddings, p=2)

            # masks
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask.fill_diagonal_(False)
            neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)

            # hardest positive / negative per anchor
            d_pos = torch.where(
                pos_mask, dists, torch.full_like(dists, float("-inf"))
            ).max(dim=1).values

            d_neg = torch.where(
                neg_mask, dists, torch.full_like(dists, float("inf"))
            ).min(dim=1).values

            # filter valid anchors (has at least one positive and one negative)
            valid = (d_pos != float("-inf")) & (d_neg != float("inf"))
            d_pos = d_pos[valid]
            d_neg = d_neg[valid]

            # per-anchor loss
            loss_per = torch.clamp(d_pos - d_neg + margin, min=0.0)

            mean_d_pos = d_pos.mean().item()
            mean_d_neg = d_neg.mean().item()

        return {
            "th_active_fraction": (loss_per > 0).float().mean().item(),
            "th_mean_d_pos":      mean_d_pos,
            "th_mean_d_neg":      mean_d_neg,
            "th_gap":             mean_d_neg - mean_d_pos,
            "th_d_pos_std":       d_pos.std().item() if len(d_pos) > 1 else 0.0,
            "th_d_neg_std":       d_neg.std().item() if len(d_neg) > 1 else 0.0,
            "th_collapse":        mean_d_pos < COLLAPSE_THRESHOLD and
                                  mean_d_neg < COLLAPSE_THRESHOLD,
        }

    @staticmethod
    def epoch_average(batch_results: list[dict]) -> dict:
        """
        Averages th_* metrics across all batches of an epoch.

        Args:
            batch_results : list of dicts from compute() over one epoch

        Returns:
            dict with averaged th_* values.
            th_collapse is True if ANY batch triggered it.
        """
        valid = [d for d in batch_results if not d.get("th_skipped")]
        if not valid:
            return {"th_skipped": True}

        scalar_keys = [
            "th_active_fraction", "th_mean_d_pos", "th_mean_d_neg",
            "th_gap", "th_d_pos_std", "th_d_neg_std",
        ]
        result = {k: sum(d[k] for d in valid) / len(valid) for k in scalar_keys}
        result["th_collapse"] = any(d["th_collapse"] for d in valid)
        return result

    def report(self, metrics: dict, epoch: int = -1, margin: float = 0.3) -> None:
        """
        Prints a triplet health summary to the console.
        Call once per epoch with the averaged metrics dict.
        """
        if metrics.get("th_skipped") or not metrics:
            return

        _R = "\033[0m"; _G = "\033[92m"; _Y = "\033[93m"
        _RED = "\033[91m"; _B = "\033[1m"; _D = "\033[2m"

        active   = metrics["th_active_fraction"]
        d_pos    = metrics["th_mean_d_pos"]
        d_neg    = metrics["th_mean_d_neg"]
        gap      = metrics["th_gap"]
        collapse = metrics["th_collapse"]

        if collapse:
            status = f"  {_RED}{_B}⚠ COLLAPSE{_R}"
        elif active > 0.95:
            status = f"  {_Y}● early phase{_R}"
        elif active > 0.50:
            status = f"  {_Y}● mid phase{_R}"
        elif active < 0.01:
            status = f"  {_RED}⚠ stalled — 0% active{_R}"
        else:
            status = f"  {_G}● converging{_R}"

        def _fmt_gap(v):
            return (_G if v >= margin else _Y if v >= margin / 2 else _RED)(f"{v:.4f}")

        # reuse same color helpers
        def _G(s): return f"\033[92m{s}{_R}"
        def _Y(s): return f"\033[93m{s}{_R}"
        def _RED(s): return f"\033[91m{s}{_R}"

        e_str = f"epoch {epoch}" if epoch >= 0 else ""
        print(f"\n{_B}  Triplet Health  {e_str}{_R}{status}")
        print(f"  {_D}{'─' * 46}{_R}")
        print(f"  {'active':<18}  {active*100:5.1f}%")
        print(f"  {'mean d(a,p)':<18}  {d_pos:.4f}  ← should decrease")
        print(f"  {'mean d(a,n)':<18}  {d_neg:.4f}  ← should increase")
        print(f"  {'gap':<18}  {gap:.4f}  ← margin = {margin}")
        print(f"  {_D}{'─' * 46}{_R}\n")

    def __repr__(self) -> str:
        return f"TripletHealthMonitor(log_every_n_batches={self.log_every_n_batches})"