"""Generate Experiment 3 figures from saved results.

Outputs (figures/):
    exp3_acc_per_dataset.png         grouped bar chart -- test accuracy per dataset
    exp3_loss_curves.png             training loss vs step (1 panel per dataset, all opts)
    exp3_loss_curves_grad_only.png   same, with Adam excluded so SGD/PGD/RPGD/SPGD
                                     differences are visible on linear y-axis

Run from project root:
    uv run python experiments/plot_exp3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OPT_ORDER = ["sgd", "adam", "pgd", "rpgd", "spgd"]
OPT_LABEL = {"sgd": "SGD", "adam": "Adam", "pgd": "PGD", "rpgd": "RPGD", "spgd": "SPGD"}
OPT_COLORS = {
    "sgd":  "#4C72B0",
    "adam": "#DD8452",
    "pgd":  "#55A467",
    "rpgd": "#C44E52",
    "spgd": "#8172B3",
}


def plot_test_accuracy(df: pd.DataFrame, figures_dir: Path) -> None:
    palette = {k: OPT_COLORS[k] for k in OPT_ORDER}
    g = sns.catplot(
        data=df, x="opt", y="test_accuracy",
        col="dataset", col_wrap=3,
        kind="bar",
        order=OPT_ORDER, hue="opt", palette=palette,
        errorbar="sd",
        height=3.4, aspect=1.1, legend=False,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("optimizer", "test accuracy")
    for ax in g.axes.flat:
        ymin = max(0.0, ax.get_ylim()[0] - 0.05)
        ax.set_ylim(bottom=ymin, top=1.02)
    g.fig.suptitle(
        "Experiment 3 — Test accuracy per dataset (mean ± 1 std across seeds)",
        y=1.03, fontsize=13,
    )
    out = figures_dir / "exp3_acc_per_dataset.png"
    g.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(g.fig)
    print(f"Wrote {out}")


def _draw_curves(ax, curves, ds, n_seeds, opts):
    for opt in opts:
        seeds_curves = [
            curves[f"{ds}_{opt}_s{s}"]
            for s in range(n_seeds)
            if f"{ds}_{opt}_s{s}" in curves
        ]
        if not seeds_curves:
            continue
        arr = np.array(seeds_curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, color=OPT_COLORS[opt], label=OPT_LABEL[opt], linewidth=1.3)
        ax.fill_between(t, mean - std, mean + std, color=OPT_COLORS[opt], alpha=0.15)


def plot_loss_curves(curves: dict, df: pd.DataFrame, figures_dir: Path) -> None:
    """Two figures: full (all 5 opts, log y) and grad-only (excl. Adam, linear y).

    The full plot shows Adam's dominance. The grad-only plot makes SPGD's gap
    over SGD/PGD/RPGD visible -- otherwise log-y compresses the 0.4-vs-0.6
    range into a single thick band at the top of the chart.
    """
    datasets = list(dict.fromkeys(df["dataset"].tolist()))
    n_seeds = df["seed"].nunique()

    # --- Figure 1: all five optimizers, log y -------------------------------
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.6 * len(datasets), 4), sharey=False
    )
    if len(datasets) == 1:
        axes = [axes]
    for j, ds in enumerate(datasets):
        ax = axes[j]
        _draw_curves(ax, curves, ds, n_seeds, OPT_ORDER)
        ax.set_yscale("log")
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("training step")
        if j == 0:
            ax.set_ylabel("training cross-entropy (log)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Experiment 3 — Training loss, all optimizers (mean ± 1 std, {n_seeds} seeds)",
        fontsize=13,
    )
    fig.tight_layout()
    out = figures_dir / "exp3_loss_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- Figure 2: gradient-only methods, linear y (Adam excluded) ----------
    grad_only = [o for o in OPT_ORDER if o != "adam"]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.6 * len(datasets), 4), sharey=False
    )
    if len(datasets) == 1:
        axes = [axes]
    for j, ds in enumerate(datasets):
        ax = axes[j]
        _draw_curves(ax, curves, ds, n_seeds, grad_only)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("training step")
        if j == 0:
            ax.set_ylabel("training cross-entropy (linear)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Experiment 3 — Training loss, gradient-only methods "
        f"(Adam excluded; mean ± 1 std, {n_seeds} seeds)",
        fontsize=13,
    )
    fig.tight_layout()
    out = figures_dir / "exp3_loss_curves_grad_only.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "exp3_openml.csv")
    with (results_dir / "exp3_loss_curves.json").open() as f:
        curves = json.load(f)

    plot_test_accuracy(df, figures_dir)
    plot_loss_curves(curves, df, figures_dir)


if __name__ == "__main__":
    main()
