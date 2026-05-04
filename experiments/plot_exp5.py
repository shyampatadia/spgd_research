"""Generate Experiment 5 ablation heatmaps from results/exp5_ablation.csv.

Outputs (figures/):
    exp5_heatmap_acc.png         mean test accuracy per (n_p, iter_p) cell
    exp5_heatmap_final_loss.png  mean final training loss per cell
    exp5_heatmap_walltime.png    mean wall-time (s) per cell -- the compute cost

Note: a stagnation-episode heatmap is intentionally NOT produced -- on
Two Moons no cell in the grid produced any stagnation episodes, so the
heatmap is uniformly zero and was dropped from the report.

Run from project root:
    uv run python experiments/plot_exp5.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DPI = 200


def _heatmap(pv: pd.DataFrame, *, cmap: str, title: str, fmt: str,
             cbar_label: str, out_path: Path, annot_size: int = 11) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    sns.heatmap(
        pv, annot=True, fmt=fmt, cmap=cmap, ax=ax,
        annot_kws={"size": annot_size},
        cbar_kws={"label": cbar_label},
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("iter_p (steps between perturbation phases)")
    ax.set_ylabel("n_p (candidates per phase)")
    ax.set_title(title)
    # Make heatmap rows read low->high top-to-bottom (default is OK; n_p
    # ascends, just confirm no inversion needed).
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv = results_dir / "exp5_ablation.csv"
    df = pd.read_csv(csv)
    n_seeds = df["seed"].nunique()
    print(f"Loaded {len(df)} rows from {csv}  ({n_seeds} seeds per cell)")

    # --- pivot tables ------------------------------------------------------
    pv_acc = df.pivot_table(index="n_p", columns="iter_p",
                            values="test_accuracy", aggfunc="mean")
    pv_loss = df.pivot_table(index="n_p", columns="iter_p",
                             values="final_loss", aggfunc="mean")
    pv_t = df.pivot_table(index="n_p", columns="iter_p",
                          values="duration_sec", aggfunc="mean")

    # --- heatmaps ----------------------------------------------------------
    _heatmap(
        pv_acc, cmap="viridis",
        title=f"Experiment 5 — Mean test accuracy ({n_seeds} seeds per cell)",
        fmt=".3f", cbar_label="test accuracy",
        out_path=figures_dir / "exp5_heatmap_acc.png",
    )
    # Note: stagnation heatmap intentionally omitted -- on Two Moons no
    # SPGD configuration in the grid produced any stagnation episodes
    # (||grad|| never crossed eps=1e-3), so the heatmap is uniformly zero
    # and uninformative.  We keep the column in the CSV for completeness.
    _heatmap(
        pv_loss, cmap="rocket_r",
        title=f"Experiment 5 — Mean final training loss ({n_seeds} seeds per cell)",
        fmt=".4f", cbar_label="final BCE loss",
        out_path=figures_dir / "exp5_heatmap_final_loss.png",
    )
    _heatmap(
        pv_t, cmap="cividis",
        title=f"Experiment 5 — Mean wall-time per run ({n_seeds} seeds per cell)",
        fmt=".2f", cbar_label="seconds",
        out_path=figures_dir / "exp5_heatmap_walltime.png",
    )

    # --- best cell summary -------------------------------------------------
    best_pos = pv_acc.stack().idxmax()
    best_val = pv_acc.stack().max()
    print(f"\nBest cell by mean test accuracy: "
          f"n_p={best_pos[0]}, iter_p={best_pos[1]} -> {best_val:.4f}")

    # Compute-vs-accuracy: cheapest cell at >= 99% of best accuracy.
    threshold = best_val * 0.99
    eligible = df.groupby(["n_p", "iter_p"]).agg(
        acc=("test_accuracy", "mean"),
        wall=("duration_sec", "mean"),
    )
    eligible = eligible[eligible["acc"] >= threshold]
    if not eligible.empty:
        cheap = eligible.sort_values("wall").iloc[0]
        cheap_idx = eligible.sort_values("wall").index[0]
        print(f"Cheapest cell within 1% of best: "
              f"n_p={cheap_idx[0]}, iter_p={cheap_idx[1]} -> "
              f"acc={cheap['acc']:.4f}, wall={cheap['wall']:.2f}s")


if __name__ == "__main__":
    main()
