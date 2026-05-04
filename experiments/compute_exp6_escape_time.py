"""
Compute the saddle-plateau escape-time metric promised in the proposal
(Objective 3, Methodology) from existing Exp 6 history logs.

Definition. The Burer--Monteiro factorisation of MovieLens-100K with
small N(0, 0.01^2) initialisation places the iterate inside a saddle
neighbourhood near the origin where the loss is nearly constant. We
define escape-time as the first step at which the training MSE has
fallen by a fixed fraction (default 5%) below its initial value. This
makes "escape" an objective property of the loss trajectory and is
independent of the optimiser-specific final value.

Reports:
  - per-(method, seed) escape step
  - per-method mean / std across seeds
  - paired (SPGD - SGD) and (SPGD - RPGD) deltas in steps

Run from the project root (uv run):

    uv run python experiments/compute_exp6_escape_time.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "results" / "exp6_mc_history.csv"

# Fraction of the initial training-loss drop that defines "escaped the
# saddle plateau". 0.05 = first step at which train_loss has fallen 5%
# below its starting value. Robust to log-scale noise.
ESCAPE_DROP = 0.05


def first_step_below(df_method_seed: pd.DataFrame, target: float) -> int | None:
    """Return the first step at which train_loss drops below target.

    df_method_seed must be sorted by step. Returns None if never reached.
    """
    below = df_method_seed[df_method_seed["train_loss"] < target]
    if below.empty:
        return None
    return int(below["step"].iloc[0])


def main() -> None:
    if not HIST.exists():
        raise SystemExit(f"missing {HIST} -- run experiments/exp6_matrix_completion.py first")

    df = pd.read_csv(HIST).sort_values(["method", "seed", "step"]).reset_index(drop=True)
    rows = []
    for (method, seed), grp in df.groupby(["method", "seed"], sort=True):
        grp = grp.sort_values("step")
        l0 = float(grp["train_loss"].iloc[0])
        target = l0 * (1.0 - ESCAPE_DROP)
        step = first_step_below(grp, target)
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "loss_init": l0,
                "target": target,
                "escape_step": step,
            }
        )

    per = pd.DataFrame(rows)
    print("=== Per-(method, seed) escape step (5% drop from init) ===")
    print(per.to_string(index=False))

    summary = (
        per.groupby("method")["escape_step"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    print("\n=== Per-method summary ===")
    print(summary.to_string(index=False))

    # Paired deltas (SPGD - baseline) per seed
    pivot = per.pivot(index="seed", columns="method", values="escape_step")
    print("\n=== Paired escape-step matrix ===")
    print(pivot.to_string())

    deltas = {}
    if {"spgd", "sgd"}.issubset(pivot.columns):
        d = (pivot["spgd"] - pivot["sgd"]).dropna()
        deltas["spgd_minus_sgd"] = {
            "per_seed": {int(k): int(v) for k, v in d.items()},
            "mean": float(d.mean()),
        }
    if {"spgd", "rpgd"}.issubset(pivot.columns):
        d = (pivot["spgd"] - pivot["rpgd"]).dropna()
        deltas["spgd_minus_rpgd"] = {
            "per_seed": {int(k): int(v) for k, v in d.items()},
            "mean": float(d.mean()),
        }
    print("\n=== Paired deltas (negative = SPGD escapes earlier) ===")
    print(json.dumps(deltas, indent=2))

    out_dir = ROOT / "results"
    per.to_csv(out_dir / "exp6_escape_time.csv", index=False)
    summary.to_csv(out_dir / "exp6_escape_time_summary.csv", index=False)
    with (out_dir / "exp6_escape_time_paired.json").open("w") as f:
        json.dump(
            {
                "escape_drop_fraction": ESCAPE_DROP,
                "definition": (
                    "first step at which train_loss < (1 - escape_drop_fraction) * "
                    "train_loss[step=initial]"
                ),
                "deltas": deltas,
            },
            f,
            indent=2,
        )
    print(f"\nwrote: {out_dir / 'exp6_escape_time.csv'}")
    print(f"wrote: {out_dir / 'exp6_escape_time_summary.csv'}")
    print(f"wrote: {out_dir / 'exp6_escape_time_paired.json'}")


if __name__ == "__main__":
    main()
