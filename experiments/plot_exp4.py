"""Generate Experiment 4 (CIFAR-10 / ResNet-18) figures from saved JSONs.

Reads the 15 JSON files in results_cifar/ (one per opt x seed) and writes
PNG figures to figures/.  Files saved at dpi=200 for report-quality.

Outputs (figures/):
    exp4_loss_curves.png             5 optimizers, log-y, smoothed mean +/- std
    exp4_loss_curves_grad_only.png   excludes Adam, linear y (SGD/PGD/RPGD/SPGD)
    exp4_test_acc.png                bar chart of mean test acc + per-seed dots
    exp4_final_loss.png              bar chart of mean final train loss + dots
    exp4_paired_spgd.png             paired diff per seed: SPGD-RPGD, SPGD-SGD

Run from project root:
    uv run python experiments/plot_exp4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPT_ORDER = ["sgd", "adam", "pgd", "rpgd", "spgd"]
OPT_LABEL = {"sgd": "SGD", "adam": "Adam", "pgd": "PGD", "rpgd": "RPGD", "spgd": "SPGD"}
OPT_COLORS = {
    "sgd":  "#4C72B0",
    "adam": "#DD8452",
    "pgd":  "#55A467",
    "rpgd": "#C44E52",
    "spgd": "#8172B3",
}

DPI = 200


def load_runs(results_dir: Path) -> pd.DataFrame:
    """Read all 15 JSONs into a DataFrame; keep losses as a column of arrays."""
    rows = []
    for p in sorted(results_dir.glob("exp4_*_e50.json")):
        d = json.load(p.open())
        rows.append(dict(
            opt=d["opt"],
            seed=int(d["seed"]),
            final_loss=float(d["final_loss"]),
            test_accuracy=float(d["test_accuracy"]),
            duration_sec=float(d["duration_sec"]),
            n_stagnation_episodes=int(d["stagnation"]["n_episodes"]),
            losses=np.asarray(d["losses"], dtype=np.float32),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no exp4_*_e50.json found in {results_dir}")
    return df


def smooth(x: np.ndarray, window: int = 200) -> np.ndarray:
    """Boxcar moving average; returns array of same length (edges via 'valid' pad)."""
    if window <= 1 or window > len(x):
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    # 'same' keeps length; pad reflects to avoid edge artifacts
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def _draw_curves(ax, df: pd.DataFrame, opts, *, smooth_win: int = 200,
                 subsample: int = 25):
    """Plot mean +/- std training-loss curve per optimizer.

    Per-step losses are smoothed with a 200-step moving average and then
    subsampled by `subsample` to keep the PNG file size sane (raw 19,550-pt
    curves x 5 opts produce visually identical plots at much higher cost).
    """
    for opt in opts:
        sub = df[df["opt"] == opt].sort_values("seed")
        if sub.empty:
            continue
        # Stack: shape (n_seeds, n_steps); all runs are 19,550 steps so this is rectangular.
        arr = np.stack([smooth(np.asarray(l), window=smooth_win) for l in sub["losses"]])
        arr = arr[:, ::subsample]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        t = np.arange(arr.shape[1]) * subsample
        ax.plot(t, mean, color=OPT_COLORS[opt], label=OPT_LABEL[opt], linewidth=1.4)
        ax.fill_between(t, mean - std, mean + std, color=OPT_COLORS[opt], alpha=0.18)


def plot_loss_curves(df: pd.DataFrame, figures_dir: Path) -> None:
    n_seeds = df["seed"].nunique()

    # Figure 1: all five optimizers on log-y -- shows Adam's dominance.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _draw_curves(ax, df, OPT_ORDER)
    ax.set_yscale("log")
    ax.set_xlabel("training step (minibatch)")
    ax.set_ylabel("training cross-entropy (log scale)")
    ax.set_title(f"Experiment 4 — CIFAR-10 / ResNet-18 training loss "
                 f"(mean ± 1 std, {n_seeds} seeds)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "exp4_loss_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")

    # Figure 2: gradient-only optimizers, linear y -- exposes SGD/PGD/RPGD/SPGD differences
    # that the log-y plot compresses into one band near the bottom.
    grad_only = [o for o in OPT_ORDER if o != "adam"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _draw_curves(ax, df, grad_only)
    ax.set_xlabel("training step (minibatch)")
    ax.set_ylabel("training cross-entropy (linear scale)")
    ax.set_title(f"Experiment 4 — Gradient-only optimizers (Adam excluded), "
                 f"mean ± 1 std, {n_seeds} seeds")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "exp4_loss_curves_grad_only.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")


def _bar_with_seed_dots(df: pd.DataFrame, value_col: str, ylabel: str,
                        title: str, out_path: Path) -> None:
    """Bar chart of mean over seeds with per-seed dots overlaid."""
    means = df.groupby("opt")[value_col].mean().reindex(OPT_ORDER)
    stds  = df.groupby("opt")[value_col].std().reindex(OPT_ORDER)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = np.arange(len(OPT_ORDER))
    colors = [OPT_COLORS[o] for o in OPT_ORDER]
    ax.bar(xs, means.values, yerr=stds.values, color=colors, alpha=0.75,
           capsize=4, edgecolor="black", linewidth=0.6)

    # Overlay per-seed dots
    for j, opt in enumerate(OPT_ORDER):
        vals = df[df["opt"] == opt][value_col].values
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full_like(vals, j, dtype=float) + jitter, vals,
                   color="black", s=22, zorder=3, alpha=0.85)

    ax.set_xticks(xs)
    ax.set_xticklabels([OPT_LABEL[o] for o in OPT_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_test_accuracy(df: pd.DataFrame, figures_dir: Path) -> None:
    _bar_with_seed_dots(
        df, "test_accuracy",
        ylabel="test accuracy",
        title="Experiment 4 — Final test accuracy (bars: mean ± std; dots: per-seed)",
        out_path=figures_dir / "exp4_test_acc.png",
    )


def plot_final_loss(df: pd.DataFrame, figures_dir: Path) -> None:
    _bar_with_seed_dots(
        df, "final_loss",
        ylabel="final training loss",
        title="Experiment 4 — Final training loss (bars: mean ± std; dots: per-seed)",
        out_path=figures_dir / "exp4_final_loss.png",
    )


def plot_paired_spgd(df: pd.DataFrame, figures_dir: Path) -> None:
    """Paired per-seed deltas: SPGD vs RPGD and SPGD vs SGD on test accuracy."""
    by_key = {(r["opt"], r["seed"]): r["test_accuracy"] for _, r in df.iterrows()}
    seeds = sorted(df["seed"].unique())

    diff_rpgd = [by_key[("spgd", s)] - by_key[("rpgd", s)] for s in seeds]
    diff_sgd  = [by_key[("spgd", s)] - by_key[("sgd",  s)] for s in seeds]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.36
    xs = np.arange(len(seeds))
    ax.bar(xs - width / 2, diff_rpgd, width, color=OPT_COLORS["rpgd"],
           edgecolor="black", linewidth=0.6, label="SPGD - RPGD (selection-rule effect)")
    ax.bar(xs + width / 2, diff_sgd,  width, color=OPT_COLORS["sgd"],
           edgecolor="black", linewidth=0.6, label="SPGD - SGD (perturbation+selection effect)")
    ax.axhline(0, color="black", linewidth=0.8)

    # Annotate mean lines
    ax.axhline(np.mean(diff_rpgd), color=OPT_COLORS["rpgd"], linestyle="--",
               linewidth=1.0, alpha=0.8)
    ax.axhline(np.mean(diff_sgd), color=OPT_COLORS["sgd"], linestyle="--",
               linewidth=1.0, alpha=0.8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("test accuracy difference (SPGD - baseline)")
    ax.set_title("Experiment 4 — Paired comparisons (same init, optimizer swapped)\n"
                 f"mean SPGD-RPGD = {np.mean(diff_rpgd):+.4f}   "
                 f"mean SPGD-SGD = {np.mean(diff_sgd):+.4f}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "exp4_paired_spgd.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results_cifar"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {results_dir} ...")
    df = load_runs(results_dir)
    print(f"  loaded {len(df)} runs across {df['opt'].nunique()} optimizers "
          f"and {df['seed'].nunique()} seeds")

    plot_loss_curves(df, figures_dir)
    plot_test_accuracy(df, figures_dir)
    plot_final_loss(df, figures_dir)
    plot_paired_spgd(df, figures_dir)

    # Print a compact summary table for the report
    print("\n=== Aggregate test accuracy / final loss by optimizer ===")
    agg = df.groupby("opt")[["test_accuracy", "final_loss", "duration_sec"]].agg(
        ["mean", "std"]
    ).reindex(OPT_ORDER)
    print(agg.round(4))


if __name__ == "__main__":
    main()
