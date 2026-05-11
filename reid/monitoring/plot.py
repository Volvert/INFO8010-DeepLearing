# =============================================================================
# plot.py — Training Curve Visualizer
# =============================================================================
"""
Reads metrics.csv and generates 6 plots + one dashboard.

Usage:
    # from main.py after training
    from plot import generate_all_plots
    generate_all_plots(run_dir=logger.run_dir, margin=cfg["training"]["margin"])

    # from command line
    python plot.py --run run_001
    python plot.py --run run_001 --margin 0.3

Output: runs/<run_name>/plots/
    00_dashboard.png   all 6 plots in one 2×3 figure
    01_loss.png        triplet loss + smoothed trend
    02_retrieval.png   Rank-1 + mAP vs 0.36 baseline
    03_active_pct.png  active triplet fraction arc
    04_lr.png          learning rate warmup + cosine decay
    05_triplet_gap.png d(a,p), d(a,n), gap vs margin  [optional — th_* columns]
    06_grad_norms.png  per-block gradient norms         [optional — grad_* columns]

See: monitoring/logger.py — writes metrics.csv
See: main.py              — calls generate_all_plots()
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── visual constants ────────────────────────────────────────────────────────

DPI       = 150
SIZE_1    = (8, 4)
SIZE_DB   = (16, 10)
BASELINE  = 0.36       # AIC21 2021 mAP baseline

C_BLUE   = "#0072B2"
C_RED    = "#D55E00"
C_GREEN  = "#009E73"
C_ORANGE = "#E69F00"
C_PURPLE = "#CC79A7"
C_CYAN   = "#56B4E9"
C_YELLOW = "#F0E442"

STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.fontsize": 9, "legend.framealpha": 0.8,
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _has(df: pd.DataFrame, *cols: str) -> bool:
    """True if all columns exist and have at least one non-NaN value."""
    return all(c in df.columns and not df[c].isna().all() for c in cols)

def _epochs(df: pd.DataFrame):
    return df["epoch"] + 1   # 1-indexed for display

def _block_cols(df: pd.DataFrame) -> list[str]:
    """Returns sorted list of grad_norm_blocks.* columns present in df."""
    return sorted(c for c in df.columns if c.startswith("grad_norm_blocks."))


# ─── axis-fill functions (no save, no figure creation) ───────────────────────
# Each function receives an ax and fills it. The dashboard reuses them directly,
# eliminating all duplication between individual plots and the 2×3 grid.

def _ax_loss(ax, df):
    e = _epochs(df)
    ax.plot(e, df["loss"], color=C_BLUE, lw=2, label="loss")
    if len(df) >= 5:
        ax.plot(e, df["loss"].rolling(5, center=True).mean(),
                color=C_RED, lw=1.5, ls="--", alpha=0.7, label="smoothed")
    ax.set(title="Triplet Loss", xlabel="Epoch", ylabel="Loss",
           xlim=(1, None), ylim=(0, None))
    ax.legend()

def _ax_retrieval(ax, df):
    e = _epochs(df)
    if _has(df, "rank1"):
        ax.plot(e, df["rank1"], color=C_BLUE, lw=2, marker="o", ms=3, label="Rank-1")
    if _has(df, "mAP"):
        ax.plot(e, df["mAP"],   color=C_GREEN, lw=2, marker="s", ms=3, label="mAP")
    ax.axhline(BASELINE, color=C_RED, ls="--", lw=1.5, alpha=0.8,
               label=f"baseline {BASELINE:.0%}")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set(title="Retrieval", xlabel="Epoch", ylabel="Score",
           xlim=(1, None), ylim=(0, 1.0))
    ax.legend()

def _ax_active(ax, df):
    e = _epochs(df)
    pct = df["active_triplets"] * 100
    ax.plot(e, pct, color=C_ORANGE, lw=2)
    ax.axhspan(5, 80, alpha=0.07, color=C_GREEN)
    ax.axhline(5,  color=C_RED,    ls=":", lw=1, alpha=0.7, label="5% (stalled)")
    ax.axhline(80, color=C_YELLOW, ls=":", lw=1, alpha=0.7, label="80% (too easy)")
    ax.set(title="Active Triplets", xlabel="Epoch",
           ylabel="Active (%)", xlim=(1, None), ylim=(0, 105))
    ax.legend(fontsize=8)

def _ax_lr(ax, df):
    e = _epochs(df)
    ax.semilogy(e, df["lr"], color=C_PURPLE, lw=2)
    idx = df["lr"].idxmax()
    if idx > 0:
        ax.axvline(e[idx], color=C_ORANGE, ls="--", lw=1.2, alpha=0.8,
                   label=f"warmup end")
    ax.set(title="Learning Rate", xlabel="Epoch",
           ylabel="LR (log)", xlim=(1, None))
    ax.legend(fontsize=8)

def _ax_triplet_gap(ax, df, margin: float = 0.3):
    if not _has(df, "th_mean_d_pos", "th_mean_d_neg", "th_gap"):
        ax.text(0.5, 0.5, "th_* columns\nnot logged",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Embedding Distances")
        return
    e = _epochs(df)
    ax.plot(e, df["th_mean_d_pos"], color=C_BLUE,  lw=2, label="d(a,p) ↓")
    ax.plot(e, df["th_mean_d_neg"], color=C_RED,   lw=2, label="d(a,n) ↑")
    ax.plot(e, df["th_gap"],        color=C_GREEN, lw=2, ls="-.", label="gap ↑")
    ax.fill_between(e, df["th_mean_d_pos"], df["th_mean_d_neg"],
                    alpha=0.08, color=C_GREEN)
    ax.axhline(margin, color=C_ORANGE, ls="--", lw=1.5, alpha=0.8,
               label=f"margin {margin}")
    ax.set(title="Embedding Distances", xlabel="Epoch",
           ylabel="Distance", xlim=(1, None), ylim=(0, None))
    ax.legend(fontsize=8)

def _ax_grad_norms(ax, df):
    cols = _block_cols(df)
    has_global = _has(df, "grad_norm_global")
    if not cols and not has_global:
        ax.text(0.5, 0.5, "grad_norm_* columns\nnot logged",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Gradient Norms")
        return
    e = _epochs(df)
    # colormap for blocks — evenly spaced from cool to warm
    cmap = plt.cm.coolwarm
    for i, col in enumerate(cols):
        color = cmap(i / max(len(cols) - 1, 1))
        label = col.replace("grad_norm_", "")
        ax.semilogy(e, df[col], color=color, lw=1.5, alpha=0.85, label=label)
    if has_global:
        ax.semilogy(e, df["grad_norm_global"],
                    color="black", lw=2.5, label="global", zorder=10)
    ax.axhline(1e-6, color=C_RED, ls=":",  lw=1, alpha=0.7)
    ax.axhline(10.0, color=C_RED, ls="--", lw=1, alpha=0.7)
    ax.set(title="Gradient Norms", xlabel="Epoch",
           ylabel="L2 Norm (log)", xlim=(1, None))
    ax.legend(fontsize=7, ncol=2)


# ─── public API ──────────────────────────────────────────────────────────────

def generate_all_plots(run_dir: str, margin: float = 0.3) -> None:
    """
    Generates all plots from metrics.csv and saves them to run_dir/plots/.

    Args:
        run_dir : path to the run directory (e.g. "runs/run_001")
        margin  : triplet margin from cfg["training"]["margin"] — used in triplet_gap plot
    """
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metrics.csv not found in {run_dir}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  metrics.csv is empty — nothing to plot."); return

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print(f"\n📊  Plotting {len(df)} epochs → {plots_dir}/")

    # individual plots — one figure per metric group
    _plots = [
        ("01_loss.png",        _ax_loss),
        ("02_retrieval.png",   _ax_retrieval),
        ("03_active_pct.png",  _ax_active),
        ("04_lr.png",          _ax_lr),
    ]
    for fname, fn in _plots:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=SIZE_1)
            fn(ax, df)
            fig.savefig(os.path.join(plots_dir, fname), dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"   ✓ {fname}")

    # optional plots — skipped silently if columns absent
    for fname, fn, extra in [
        ("05_triplet_gap.png", _ax_triplet_gap, {"margin": margin}),
        ("06_grad_norms.png",  _ax_grad_norms,  {}),
    ]:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=SIZE_1)
            fn(ax, df, **extra)
            fig.savefig(os.path.join(plots_dir, fname), dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"   ✓ {fname}")

    # dashboard — 2×3 grid reusing the same axis-fill functions
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=SIZE_DB)
        fig.suptitle(f"Training Dashboard — {run_name}", fontsize=16,
                     fontweight="bold", y=1.01)
        _ax_loss(axes[0, 0], df)
        _ax_retrieval(axes[0, 1], df)
        _ax_active(axes[0, 2], df)
        _ax_lr(axes[1, 0], df)
        _ax_triplet_gap(axes[1, 1], df, margin=margin)
        _ax_grad_norms(axes[1, 2], df)
        plt.tight_layout()
        path = os.path.join(plots_dir, "00_dashboard.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✓ 00_dashboard.png\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run",      required=True, help="run name, e.g. run_001")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--margin",   type=float, default=0.3)
    args = p.parse_args()
    generate_all_plots(
        run_dir=os.path.join(args.runs_dir, args.run),
        margin=args.margin,
    )