# =============================================================================
# Logger
# =============================================================================
"""
Dual-channel logger: console table + metrics.csv.

Usage in main.py:
    logger = Logger(run_name, total_epochs, runs_dir)
    for epoch in range(total_epochs):
        logger.epoch_start()
        train_metrics = train_one_epoch(...)   # includes th_* and grad_* keys
        eval_metrics  = evaluate(...)          # {"rank1", "mAP"} or {}
        logger.log_epoch(epoch, train_metrics, eval_metrics)
        if logger.is_best():
            torch.save(model.state_dict(), logger.best_ckpt_path)
    logger.summary()

Output layout:
    runs/<run_name>/
    ├── metrics.csv      one row per epoch — all metrics including th_* and grad_*
    └── best_model.pth   saved by main.py when logger.is_best() is True

See: engine/train.py    — produces train_metrics (core + th_* + grad_*)
See: engine/evaluate.py — produces eval_metrics
See: plot.py            — reads metrics.csv
"""

import os
import csv
import time
from typing import Optional

# ANSI — terminal only, never written to CSV
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

def _g(s): return f"{_GREEN}{s}{_RESET}"
def _y(s): return f"{_YELLOW}{s}{_RESET}"
def _r(s): return f"{_RED}{s}{_RESET}"
def _c(s): return f"{_CYAN}{s}{_RESET}"
def _b(s): return f"{_BOLD}{s}{_RESET}"
def _d(s): return f"{_DIM}{s}{_RESET}"

# AIC21 2021 baseline — used for color coding in console + summary
BASELINE_MAP = 0.36

# Core columns always present — th_* and grad_* appended dynamically on epoch 0
CORE_COLUMNS = [
    "epoch", "loss", "active_triplets", "lr",
    "rank1", "mAP", "epoch_time_s",
]


class Logger:
    """
    Logs training metrics to console and metrics.csv.

    metrics.csv columns:
      - Core  : epoch, loss, active_triplets, lr, rank1, mAP, epoch_time_s
      - th_*  : triplet health (added automatically when present in train_metrics)
      - grad_*: gradient norms (added automatically when present in train_metrics)

    Attributes:
        run_dir        : str   — runs/<run_name>/
        csv_path       : str   — runs/<run_name>/metrics.csv
        best_ckpt_path : str   — runs/<run_name>/best_model.pth
        best_mAP       : float — running best mAP across all epochs
    """

    def __init__(self, run_name: str, total_epochs: int, runs_dir: str = "runs"):
        self.run_name     = run_name
        self.total_epochs = total_epochs
        self.run_dir      = os.path.join(runs_dir, run_name)

        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path       = os.path.join(self.run_dir, "metrics.csv")
        self.best_ckpt_path = os.path.join(self.run_dir, "best_model.pth")

        self._best_mAP:    float           = 0.0
        self._is_best:     bool            = False
        self._t_start:     Optional[float] = None
        self._csv_columns: list[str]       = []    # built on first epoch

        self._print_banner()

    # =========================================================================
    # Public API
    # =========================================================================

    def epoch_start(self) -> None:
        """Call at the top of each epoch — starts the chronometer."""
        self._t_start = time.time()

    def log_epoch(
        self,
        epoch:         int,
        train_metrics: dict,
        eval_metrics:  dict,
    ) -> None:
        """
        Logs one epoch to console and CSV.

        Args:
            epoch         : current epoch index (0-based)
            train_metrics : dict from train_one_epoch()
                            must contain "loss", "active_triplets", "lr"
                            may also contain "th_*" and "grad_*" keys
            eval_metrics  : dict from evaluate() — {"rank1", "mAP"} or {} if skipped
        """
        epoch_time = time.time() - self._t_start if self._t_start else float("nan")

        # build flat row — core metrics first
        row: dict = {
            "epoch":           epoch,
            "loss":            train_metrics.get("loss",            float("nan")),
            "active_triplets": train_metrics.get("active_triplets", float("nan")),
            "lr":              train_metrics.get("lr",              float("nan")),
            "rank1":           eval_metrics.get("rank1",            float("nan")),
            "mAP":             eval_metrics.get("mAP",              float("nan")),
            "epoch_time_s":    epoch_time,
        }

        # append th_* and grad_* keys from train_metrics
        for key, val in train_metrics.items():
            if key.startswith("th_") or key.startswith("grad_"):
                row[key] = val

        # best mAP tracking — nan-safe (nan != nan in IEEE 754)
        mAP = row["mAP"]
        if mAP == mAP and mAP > self._best_mAP:
            self._best_mAP = mAP
            self._is_best  = True
        else:
            self._is_best  = False

        self._print_row(row)
        self._write_csv(row)

    def is_best(self) -> bool:
        """True if the last logged epoch set a new best mAP."""
        return self._is_best

    @property
    def best_mAP(self) -> float:
        """Running best mAP across all logged epochs."""
        return self._best_mAP

    def log_message(self, msg: str) -> None:
        """Prints a timestamped info message — checkpoints, warnings, etc."""
        print(f"{_d(time.strftime('%H:%M:%S'))}  {_c('ℹ')}  {msg}")

    def summary(self) -> None:
        """Prints a final summary banner after the epoch loop."""
        beat = (_g("✓ beats 36% baseline") if self._best_mAP >= BASELINE_MAP
                else _r("✗ below 36% baseline"))
        w = 62
        print()
        print(_b(f"┌{'─'*w}┐"))
        print(_b(f"│{'Training Complete':^{w}}│"))
        print(_b(f"├{'─'*w}┤"))
        print(_b(f"│  Best mAP : {self._best_mAP:.4f}  {beat:<{w-24}}│"))
        print(_b(f"│  CSV      : {self.csv_path:<{w-13}}│"))
        print(_b(f"│  Ckpt     : {self.best_ckpt_path:<{w-13}}│"))
        print(_b(f"└{'─'*w}┘"))
        print()

    def __repr__(self) -> str:
        return (f"Logger(run='{self.run_name}', "
                f"epochs={self.total_epochs}, "
                f"best_mAP={self._best_mAP:.4f})")

    # =========================================================================
    # Private
    # =========================================================================

    def _print_banner(self) -> None:
        w = 62
        print()
        print(_b(f"┌{'─'*w}┐"))
        print(_b(f"│{'VehicleViT — Training':^{w}}│"))
        print(_b(f"├{'─'*w}┤"))
        print(_b(f"│  Run    : {self.run_name:<{w-11}}│"))
        print(_b(f"│  Epochs : {self.total_epochs:<{w-11}}│"))
        print(_b(f"│  CSV    : {self.csv_path:<{w-11}}│"))
        print(_b(f"└{'─'*w}┘"))
        print()
        hdr = (f"{'Epoch':>8}  {'Loss':>8}  {'Active%':>8}  "
               f"{'LR':>10}  {'Rank-1':>8}  {'mAP':>8}  {'Time(s)':>8}")
        sep = _d("─" * len(hdr))
        print(sep); print(_d(hdr)); print(sep)

    def _print_row(self, row: dict) -> None:
        loss   = row["loss"]
        active = row["active_triplets"]
        lr     = row["lr"]
        rank1  = row["rank1"]
        mAP    = row["mAP"]
        t      = row["epoch_time_s"]

        def _loss(v):
            if v != v: return _d(f"{'nan':>8}")
            return (_r if v > 0.5 else _y if v > 0.1 else _g)(f"{v:8.4f}")

        def _active(v):
            if v != v: return _d(f"{'nan':>8}")
            p = v * 100
            return (_r if v < 0.05 else _y if v > 0.80 else _g)(f"{p:7.1f}%")

        def _metric(v):
            if v != v: return _d(f"{'—':>8}")
            return (_g if v >= BASELINE_MAP else _y)(f"{v:8.4f}")

        print(
            f"{row['epoch']+1:3d}/{self.total_epochs}  "
            f"{_loss(loss)}  {_active(active)}  "
            f"{lr:.2e if lr==lr else '       nan':>10}  "
            f"{_metric(rank1)}  {_metric(mAP)}  "
            f"{t:8.1f if t==t else 'nan':>8}  "
            f"{_c(_b('★ best')) if self._is_best else '      '}"
        )

    def _write_csv(self, row: dict) -> None:
        # first epoch — build full column list and write header
        if not self._csv_columns:
            self._csv_columns = CORE_COLUMNS + [
                k for k in row if k not in CORE_COLUMNS
            ]
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_columns).writeheader()

        # format all floats, write row
        formatted = {
            k: (f"{v:.6f}" if isinstance(v, float) and v == v
                else ("nan" if isinstance(v, float) else v))
            for k, v in row.items()
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(
                f, fieldnames=self._csv_columns, extrasaction="ignore"
            ).writerow(formatted)