"""Generate Experiment 6 figures from results/exp6_mc_*.csv.

Outputs (figures/):
    exp6_mc_curves.png      training MSE & held-out RMSE vs step
    exp6_mc_paired.png      per-seed paired deltas (SPGD - {SGD, RPGD, Adam})

Run from project root:
    uv run python experiments/plot_exp6.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["sgd", "rpgd", "spgd", "adam"]
METHOD_LABEL = {"sgd": "SGD", "rpgd": "RPGD", "spgd": "SPGD", "adam": "Adam"}
METHOD_COLORS = {
    "sgd":  "#55A467",
    "rpgd": "#C44E52",
    "spgd": "#8172B3",
    "adam": "#4C72B0",
}


def plot_curves(df_hist: pd.DataFrame, figures_dir: Path) -> None:
    fig, (ax_loss, ax_rmse) = plt.subplots(1, 2, figsize=(12, 4.4))

    for method in METHOD_ORDER:
        sub = df_hist[df_hist["method"] == method]
        if sub.empty:
            continue
        agg = sub.groupby("step").agg(
            loss_mean=("train_loss", "mean"),
            loss_std=("train_loss", "std"),
            rmse_mean=("test_rmse", "mean"),
            rmse_std=("test_rmse", "std"),
        ).reset_index()
        c = METHOD_COLORS[method]
        ax_loss.plot(agg["step"], agg["loss_mean"], color=c,
                     label=METHOD_LABEL[method], linewidth=1.8)
        ax_loss.fill_between(agg["step"],
                             agg["loss_mean"] - agg["loss_std"],
                             agg["loss_mean"] + agg["loss_std"],
                             color=c, alpha=0.18)
        ax_rmse.plot(agg["step"], agg["rmse_mean"], color=c,
                     label=METHOD_LABEL[method], linewidth=1.8)
        ax_rmse.fill_between(agg["step"],
                             agg["rmse_mean"] - agg["rmse_std"],
                             agg["rmse_mean"] + agg["rmse_std"],
                             color=c, alpha=0.18)
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("training MSE on observed entries")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training MSE vs step (log scale)")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3, which="both")

    ax_rmse.set_xlabel("step")
    ax_rmse.set_ylabel("held-out test RMSE")
    ax_rmse.set_title("Test RMSE vs step (lower = better)")
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(alpha=0.3)

    n_seeds = df_hist["seed"].nunique()
    fig.suptitle(
        f"Experiment 6 -- Matrix completion on MovieLens-100K "
        f"(Burer-Monteiro, mean ± 1 std across {n_seeds} seeds)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out = figures_dir / "exp6_mc_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_paired(df_fin: pd.DataFrame, figures_dir: Path) -> None:
    pivot = df_fin.pivot(index="seed", columns="method",
                         values="final_test_rmse")
    refs = [r for r in ["sgd", "rpgd", "adam"] if r in pivot.columns]
    seeds = pivot.index.tolist()

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.25
    xs = np.arange(len(seeds))
    handles_means = []
    for i, ref in enumerate(refs):
        delta = (pivot["spgd"] - pivot[ref]).values
        offset = (i - (len(refs) - 1) / 2) * width
        c = METHOD_COLORS[ref]
        ax.bar(xs + offset, delta, width=width, color=c, edgecolor="black",
               linewidth=0.6, label=f"SPGD − {METHOD_LABEL[ref]}")
        h = ax.axhline(delta.mean(), color=c, linestyle="--", linewidth=1.0,
                       alpha=0.7,
                       label=f"mean vs {METHOD_LABEL[ref]}: {delta.mean():+.4f}")
        handles_means.append(h)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Δ test RMSE  (negative = SPGD better)")
    ax.set_title("Experiment 6 -- paired per-seed deltas in held-out test RMSE")
    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "exp6_mc_paired.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_hist = pd.read_csv(results_dir / "exp6_mc_history.csv")
    df_fin = pd.read_csv(results_dir / "exp6_mc_finals.csv")
    print(f"Loaded {len(df_hist)} history rows, {len(df_fin)} final rows")

    plot_curves(df_hist, figures_dir)
    plot_paired(df_fin, figures_dir)


if __name__ == "__main__":
    main()
