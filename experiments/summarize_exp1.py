"""Print Experiment 1's summary tables from the saved CSV.

Use this to re-print the success-rate / mean-best-loss / stagnation tables
without re-running the (slow) experiment, e.g. when the run finished but a
print step crashed.

Run from project root:
    uv run python experiments/summarize_exp1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OPT_ORDER = ["sgd", "adam", "pgd", "rpgd", "spgd"]


def _ordered(pivot: pd.DataFrame) -> pd.DataFrame:
    """Reorder optimizer columns into our canonical SGD..SPGD order if present."""
    cols = [c for c in OPT_ORDER if c in pivot.columns]
    return pivot[cols]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "results" / "exp1_benchmarks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run exp1_benchmarks.py first")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}\n")

    print("=== Success rate by (fn, dim, opt) ===")
    success = _ordered(df.groupby(["fn", "dim", "opt"])["success"].mean().unstack("opt"))
    print(success.round(3).to_string())

    print("\n=== Mean best loss by (fn, dim, opt) ===")
    best = _ordered(df.groupby(["fn", "dim", "opt"])["best_loss"].mean().unstack("opt"))
    print(best.map(lambda v: f"{v:.4g}").to_string())

    print("\n=== Median best loss by (fn, dim, opt) ===  (less seed-noisy)")
    med = _ordered(df.groupby(["fn", "dim", "opt"])["best_loss"].median().unstack("opt"))
    print(med.map(lambda v: f"{v:.4g}").to_string())

    print("\n=== Mean stagnation episodes by (fn, dim, opt) ===")
    stag = _ordered(
        df.groupby(["fn", "dim", "opt"])["n_stagnation_episodes"].mean().unstack("opt")
    )
    print(stag.round(1).to_string())

    print("\n=== Mean iters_to_converge by (fn, dim, opt) ===")
    print("(equals n_steps when no run reached eps -- look at best loss instead)")
    its = _ordered(
        df.groupby(["fn", "dim", "opt"])["iters_to_converge"].mean().unstack("opt")
    )
    print(its.round(0).astype(int).to_string())

    # SPGD-vs-RPGD head-to-head on best_loss, paired by seed -- the central
    # scientific question: does the steepest-selection rule beat random pick?
    print("\n=== SPGD vs RPGD (paired by seed): mean best-loss difference ===")
    print("(negative => SPGD beats RPGD)")
    head_to_head = (
        df.pivot_table(
            index=["fn", "dim", "seed"], columns="opt", values="best_loss"
        )
        .reset_index()
    )
    if "spgd" in head_to_head.columns and "rpgd" in head_to_head.columns:
        head_to_head["spgd_minus_rpgd"] = head_to_head["spgd"] - head_to_head["rpgd"]
        diff = (
            head_to_head.groupby(["fn", "dim"])["spgd_minus_rpgd"]
            .agg(["mean", "median", "std"])
        )
        print(diff.map(lambda v: f"{v:+.4g}").to_string())


if __name__ == "__main__":
    main()
