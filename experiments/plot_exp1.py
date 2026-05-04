"""Generate figures for Experiment 1 from saved results.

Reads:
    results/exp1_benchmarks.csv
    results/exp1_convergence_curves.json
    results/exp1_trajectories_2d.json

Writes:
    figures/exp1_trajectories_2d.png   3 fns x 5 opts grid; 30-seed paths over contour
    figures/exp1_convergence.png       3 dims x 3 fns grid; mean+/-std running-min loss
    figures/exp1_success_rate.png      bar chart of success rate per (fn, dim, opt)

Run from project root:
    uv run python experiments/plot_exp1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from spgd_study.benchmarks import BENCHMARKS

OPT_ORDER = ["sgd", "adam", "pgd", "rpgd", "spgd"]
OPT_LABEL = {"sgd": "SGD", "adam": "Adam", "pgd": "PGD", "rpgd": "RPGD", "spgd": "SPGD"}
OPT_COLORS = {
    "sgd":  "#4C72B0",
    "adam": "#DD8452",
    "pgd":  "#55A467",
    "rpgd": "#C44E52",
    "spgd": "#8172B3",
}


def _contour_grid(fn_name: str, n: int = 200):
    """Compute the contour grid for a 2D benchmark."""
    bench = BENCHMARKS[fn_name]
    lo, hi = bench.domain
    xs = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, xs)
    pts = torch.from_numpy(np.stack([X, Y], axis=-1)).to(torch.float32)
    with torch.no_grad():
        Z = bench.fn(pts).numpy()
    return X, Y, Z, bench


# ---------------------------------------------------------------------------
# Figure 1 — 2D trajectories
# ---------------------------------------------------------------------------

def plot_trajectories_2d(trajectories: dict, n_seeds: int, figures_dir: Path) -> None:
    functions = ["rastrigin", "ackley", "rosenbrock"]
    fig, axes = plt.subplots(len(functions), len(OPT_ORDER), figsize=(20, 12))

    for i, fn_name in enumerate(functions):
        X, Y, Z, bench = _contour_grid(fn_name)
        for j, opt_name in enumerate(OPT_ORDER):
            ax = axes[i, j]
            # Use log-scale colormap for Rosenbrock (huge dynamic range).
            if fn_name == "rosenbrock":
                ax.contourf(X, Y, np.log10(Z + 1.0), levels=30, cmap="viridis", alpha=0.65)
            else:
                ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.65)

            for seed in range(n_seeds):
                key = f"{fn_name}_{opt_name}_s{seed}"
                if key not in trajectories:
                    continue
                traj = np.array(trajectories[key])
                ax.plot(traj[:, 0], traj[:, 1],
                        color=OPT_COLORS[opt_name], alpha=0.18, linewidth=0.6)

            # Mark global optimum.
            ax.scatter([bench.optimum_x], [bench.optimum_x],
                       marker="*", s=180, color="white",
                       edgecolors="black", linewidths=1.5, zorder=10)

            ax.set_xlim(bench.domain)
            ax.set_ylim(bench.domain)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(OPT_LABEL[opt_name], fontsize=12)
            if j == 0:
                ax.set_ylabel(fn_name, fontsize=12)

    fig.suptitle(
        f"Experiment 1 — Optimizer trajectories on 2D benchmarks  "
        f"({n_seeds} seeds; star = global optimum)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    out = figures_dir / "exp1_trajectories_2d.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — convergence curves
# ---------------------------------------------------------------------------

def plot_convergence_curves(curves: dict, df: pd.DataFrame, figures_dir: Path) -> None:
    functions = ["rastrigin", "ackley", "rosenbrock"]
    dims = sorted(df["dim"].unique().tolist())
    n_seeds = df["seed"].nunique()
    fig, axes = plt.subplots(len(dims), len(functions),
                             figsize=(15, 10), sharex=True)

    for i, dim in enumerate(dims):
        for j, fn_name in enumerate(functions):
            ax = axes[i, j]
            for opt_name in OPT_ORDER:
                seeds = []
                for seed in range(n_seeds):
                    key = f"{fn_name}_d{dim}_{opt_name}_s{seed}"
                    if key in curves:
                        seeds.append(curves[key])
                if not seeds:
                    continue
                arr = np.array(seeds)  # (n_seeds, n_steps), already running-min
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                t = np.arange(len(mean))
                ax.plot(t, mean, color=OPT_COLORS[opt_name],
                        label=OPT_LABEL[opt_name], linewidth=1.4)
                ax.fill_between(t, mean - std, mean + std,
                                color=OPT_COLORS[opt_name], alpha=0.15)

            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_title(f"{fn_name} | d={dim}")
            if i == len(dims) - 1:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel("running-min loss")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Experiment 1 — Convergence (running-min loss; mean ± 1 std across seeds)",
        fontsize=14,
    )
    fig.tight_layout()
    out = figures_dir / "exp1_convergence.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — success rate
# ---------------------------------------------------------------------------

def plot_success_rate(df: pd.DataFrame, figures_dir: Path) -> None:
    success = (
        df.groupby(["fn", "dim", "opt"])["success"].mean().reset_index()
    )
    g = sns.catplot(
        data=success,
        x="dim", y="success", hue="opt",
        col="fn",
        kind="bar",
        order=[2, 10, 50],
        hue_order=OPT_ORDER,
        col_order=["rastrigin", "ackley", "rosenbrock"],
        height=4, aspect=1.1,
        palette={k: OPT_COLORS[k] for k in OPT_ORDER},
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("dimension", "success rate")
    g.set(ylim=(0, 1.05))
    g._legend.set_title("optimizer")
    g.fig.suptitle(
        "Experiment 1 — Success rate (best loss within ε=0.01 of optimum)",
        y=1.03, fontsize=14,
    )
    out = figures_dir / "exp1_success_rate.png"
    g.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(g.fig)
    print(f"Wrote {out}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "exp1_benchmarks.csv")
    with (results_dir / "exp1_convergence_curves.json").open() as f:
        curves = json.load(f)
    with (results_dir / "exp1_trajectories_2d.json").open() as f:
        trajectories = json.load(f)

    n_seeds = df["seed"].nunique()
    plot_trajectories_2d(trajectories, n_seeds, figures_dir)
    plot_convergence_curves(curves, df, figures_dir)
    plot_success_rate(df, figures_dir)


if __name__ == "__main__":
    main()
