"""Generate Experiment 2 figures from saved results.

Outputs (figures/):
    exp2_loss_curves.png         mean +/- std training loss vs step (per optimizer)
    exp2_grad_norm_history.png   smoothed grad-norm trajectories with epsilon line
                                 (replaces the ~all-zero stagnation bar chart that
                                  resulted from no optimizer ever crossing eps=1e-3)
    exp2_loss_landscape.png      THE SHOWPIECE -- 1x5 grid: loss landscape in
                                 each optimizer's PCA basis with trajectory overlay.

Run from project root:
    uv run python experiments/plot_exp2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml

from spgd_study.data import load_two_moons
from spgd_study.models import two_moons_mlp
from spgd_study.nn_runner import load_flat
from spgd_study.utils import set_seed

OPT_ORDER = ["sgd", "adam", "pgd", "rpgd", "spgd"]
OPT_LABEL = {"sgd": "SGD", "adam": "Adam", "pgd": "PGD", "rpgd": "RPGD", "spgd": "SPGD"}
OPT_COLORS = {
    "sgd":  "#4C72B0",
    "adam": "#DD8452",
    "pgd":  "#55A467",
    "rpgd": "#C44E52",
    "spgd": "#8172B3",
}


def plot_loss_curves(curves: dict, df: pd.DataFrame, figures_dir: Path) -> None:
    n_seeds = df["seed"].nunique()
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt in OPT_ORDER:
        seeds_curves = [
            curves[f"{opt}_s{s}"] for s in range(n_seeds) if f"{opt}_s{s}" in curves
        ]
        if not seeds_curves:
            continue
        arr = np.array(seeds_curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, color=OPT_COLORS[opt], label=OPT_LABEL[opt], linewidth=1.4)
        ax.fill_between(t, mean - std, mean + std,
                        color=OPT_COLORS[opt], alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("training BCE loss (log scale)")
    ax.set_title(f"Two Moons training loss (mean ± 1 std across {n_seeds} seeds)")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    out = figures_dir / "exp2_loss_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_grad_norm_history(grad_curves: dict, df: pd.DataFrame,
                           figures_dir: Path, eps: float = 1e-4) -> None:
    """Plot smoothed ||grad|| vs training step for each optimizer.

    This replaces the per-optimizer stagnation bar chart, which was an
    almost-blank figure because no optimizer ever crossed eps on Two Moons.
    This plot conveys the same diagnostic visually: it shows the
    gradient-norm trajectories on log-y with eps drawn as a horizontal
    threshold, making the "no curve dips below eps" finding immediately
    obvious.  The default eps tracks the proposal's 1e-4 threshold; the
    actual value comes from cfg["stagnation_eps"].
    """
    n_seeds = df["seed"].nunique()
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt in OPT_ORDER:
        seeds_curves = [
            grad_curves[f"{opt}_s{s}"]
            for s in range(n_seeds) if f"{opt}_s{s}" in grad_curves
        ]
        if not seeds_curves:
            continue
        arr = np.array(seeds_curves)
        # mild moving-average smoothing for readability (window=10 steps)
        win = 10
        if arr.shape[1] > win:
            kernel = np.ones(win) / win
            arr = np.array([np.convolve(c, kernel, mode="same") for c in arr])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, color=OPT_COLORS[opt], label=OPT_LABEL[opt], linewidth=1.3)
        ax.fill_between(t, np.maximum(mean - std, 1e-10), mean + std,
                        color=OPT_COLORS[opt], alpha=0.15)

    ax.axhline(eps, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label=fr"$\varepsilon = {eps:g}$ (stagnation threshold)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\|\nabla f\|_2$  (log scale)")
    ax.set_title(f"Two Moons gradient-norm history (mean ± 1 std across {n_seeds} seeds)")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = figures_dir / "exp2_grad_norm_history.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_loss_landscape(weights_history: dict, figures_dir: Path, cfg: dict) -> None:
    """1x5 grid: loss landscape in a SHARED PCA basis centered on theta_0.

    Why shared: a per-optimizer PCA basis (the previous version) gives every
    panel a different 2-D plane through 1185-D weight space, with a different
    origin (each optimizer's own trajectory mean).  That makes cross-panel
    comparison misleading: the green "start" dot lands at different projected
    coordinates because the projection differs, not because the parameters
    differ.

    The shared basis fixes both:
      1. All five panels project onto the SAME 2-D plane, spanned by the
         top-2 principal directions of the union of displacements
         theta(t) - theta_0 across all optimizers.
      2. theta_0 maps to the origin (0, 0) in every panel because we center
         on theta_0 rather than the trajectory mean.

    The contours are level sets of the *training* BCE loss, evaluated on the
    Two Moons training set (the same loss the optimizers see), at every grid
    point in the shared 2-D plane.  Yellow = high loss, purple = low loss.
    """
    set_seed(0)
    X_tr, y_tr, _, _ = load_two_moons(
        n_samples=cfg["n_samples"], noise=cfg["noise"],
        test_frac=cfg["test_frac"], seed=0,
    )
    model = two_moons_mlp(hidden=cfg["hidden"])

    available = [o for o in OPT_ORDER if o in weights_history.files]
    if not available:
        print("plot_loss_landscape: no optimizer weight histories found, skipping")
        return

    # ---- shared anchor: theta_0 (paired-seed protocol guarantees this is
    # identical across optimizers; we double-check below) -------------------
    theta0 = weights_history[available[0]][0]                  # (n_params,)
    for opt in available:
        if not np.allclose(weights_history[opt][0], theta0):
            raise RuntimeError(
                f"theta_0 differs between optimizers (e.g. {available[0]} vs {opt}); "
                "the paired-seed protocol assumption is violated."
            )

    # ---- shared basis: PCA on the stacked displacements theta(t) - theta_0
    # for all five optimizers.  This basis spans the directions actually used
    # by *some* optimizer, weighted by how much they were used. -------------
    disp = np.concatenate(
        [weights_history[opt] - theta0 for opt in available], axis=0
    )
    _, _, Vt = np.linalg.svd(disp, full_matrices=False)
    u, v = Vt[0], Vt[1]                                        # (n_params,) each

    # ---- project each optimizer's trajectory into the shared basis --------
    proj = {opt: (weights_history[opt] - theta0) @ np.stack([u, v]).T
            for opt in available}

    # ---- shared sweep grid: 1.1 * max(|coord|) across all optimizers, so
    # every trajectory fits inside every panel with a small margin --------
    all_coords = np.concatenate([proj[opt] for opt in available], axis=0)
    extent_x = max(1e-3, 1.1 * float(np.abs(all_coords[:, 0]).max()))
    extent_y = max(1e-3, 1.1 * float(np.abs(all_coords[:, 1]).max()))
    a = np.linspace(-extent_x, extent_x, 60)
    b = np.linspace(-extent_y, extent_y, 60)
    A, B = np.meshgrid(a, b)
    Z = np.zeros_like(A)
    for i in range(len(a)):
        for k in range(len(b)):
            w = theta0 + a[i] * u + b[k] * v                   # shared anchor
            load_flat(model, w)
            with torch.no_grad():
                logits = model(X_tr)
                loss = F.binary_cross_entropy_with_logits(logits, y_tr)
            Z[k, i] = float(loss.item())

    # ---- Loss values along each trajectory (in this 2-D plane).  Used to
    # pick a colour-scale upper bound that's relevant to the region the
    # trajectories actually visit, rather than the corners of the grid
    # where the network sees extreme weights and BCE blows up to ~50. ------
    traj_losses = []
    for opt in available:
        for w_snap in weights_history[opt]:
            load_flat(model, w_snap)
            with torch.no_grad():
                traj_losses.append(
                    float(F.binary_cross_entropy_with_logits(model(X_tr), y_tr))
                )
    traj_losses = np.asarray(traj_losses)
    # Clip the colormap to ~3x the max BCE loss seen along any trajectory,
    # capped at the 90th percentile of the full grid Z.  This keeps the
    # high-loss-corner saturation off the colormap so the basin's structure
    # becomes legible, while still showing some out-of-basin context.
    vmin = float(Z.min())
    vmax_traj = 3.0 * float(traj_losses.max())
    vmax_grid = float(np.quantile(Z, 0.90))
    vmax = max(min(vmax_traj, vmax_grid), vmin + 1e-3)

    # Log-spaced contour levels in the trajectory-relevant range -- gives
    # finer detail near the basin minimum where the optimizers actually
    # operate.  Floor at 1e-3 to avoid log(0).
    lo = max(vmin, 1e-3)
    contour_levels = np.geomspace(lo, vmax, 11)

    fig, axes = plt.subplots(1, len(OPT_ORDER), figsize=(22, 4.7))

    for j, opt in enumerate(OPT_ORDER):
        ax = axes[j]
        if opt not in proj:
            ax.set_visible(False)
            continue

        # Clip Z to the chosen color range BEFORE drawing -- otherwise
        # contourf still allocates colour bins up to the raw Z.max() and
        # the basin gets squashed into a single shade.
        Z_clip = np.clip(Z, vmin, vmax)
        cs = ax.contourf(A, B, Z_clip,
                         levels=np.linspace(vmin, vmax, 25),
                         cmap="viridis", alpha=0.9,
                         vmin=vmin, vmax=vmax, extend="max")
        ax.contour(A, B, Z_clip, levels=contour_levels, colors="white",
                   linewidths=0.4, alpha=0.55)
        p = proj[opt]
        ax.plot(p[:, 0], p[:, 1], color="white", linewidth=1.7)
        ax.scatter(p[0, 0], p[0, 1], color="lime", s=90, marker="o",
                   edgecolors="black", linewidths=1.2, zorder=10, label="start")
        ax.scatter(p[-1, 0], p[-1, 1], color="red", s=130, marker="X",
                   edgecolors="black", linewidths=1.2, zorder=10, label="end")
        ax.set_title(OPT_LABEL[opt], fontsize=12)
        ax.set_xlabel("PC1 (shared)")
        if j == 0:
            ax.set_ylabel("PC2 (shared)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-extent_x, extent_x)
        ax.set_ylim(-extent_y, extent_y)

    # One shared colorbar for the loss-value scale.
    cbar = fig.colorbar(cs, ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.012, aspect=28)
    cbar.set_label("training BCE loss", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Two Moons — training BCE loss in a shared PCA basis "
        "(seed 0; PC1, PC2 from union of trajectories; origin = $\\theta_0$; "
        f"colour clipped to BCE $\\leq$ {vmax:.2f})",
        fontsize=13, y=1.02,
    )
    out = figures_dir / "exp2_loss_landscape.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "exp2_two_moons.csv")
    with (results_dir / "exp2_loss_curves.json").open() as f:
        curves = json.load(f)
    with (results_dir / "exp2_grad_norms.json").open() as f:
        grad_curves = json.load(f)
    weights_history = np.load(results_dir / "exp2_weights_history.npz")

    cfg = yaml.safe_load(
        (project_root / "experiments" / "configs" / "exp2.yaml").read_text()
    )

    plot_loss_curves(curves, df, figures_dir)
    plot_grad_norm_history(grad_curves, df, figures_dir,
                           eps=float(cfg["stagnation_eps"]))
    plot_loss_landscape(weights_history, figures_dir, cfg)


if __name__ == "__main__":
    main()
