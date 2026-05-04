"""Experiment 5 — N_P x IterP ablation for SPGD on Two Moons.

Sweeps SPGD's two perturbation hyperparameters on the same Two Moons setup
used in Experiment 2 (full-batch, 2000 steps, 2->32->32->1 MLP, lr=5e-2,
amp=0.05).  4 x 4 grid of (n_p, iter_p) with 5 seeds per cell = 80 runs.

Outputs:
    results/exp5_ablation.csv

CLI:
    uv run python experiments/exp5_ablation.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from spgd_study.data import load_two_moons
from spgd_study.models import two_moons_mlp
from spgd_study.nn_runner import binary_accuracy, train_full_batch
from spgd_study.utils import set_seed


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (project_root / "experiments" / "configs" / "exp5.yaml").read_text()
    )

    runs = [
        (n_p, iter_p, seed)
        for n_p in cfg["n_p_grid"]
        for iter_p in cfg["iter_p_grid"]
        for seed in range(cfg["n_seeds"])
    ]
    n_cells = len(cfg["n_p_grid"]) * len(cfg["iter_p_grid"])
    print(f"Experiment 5: {len(runs)} runs "
          f"({n_cells} cells x {cfg['n_seeds']} seeds)")
    print(f"  grid: n_p     = {cfg['n_p_grid']}")
    print(f"        iter_p  = {cfg['iter_p_grid']}")
    print(f"  fixed: lr={cfg['lr']}, amp={cfg['amp']}, "
          f"n_steps={cfg['n_steps']}, hidden={cfg['hidden']}")

    rows: List[Dict] = []
    t0 = time.time()

    for n_p, iter_p, seed in tqdm(runs, desc="exp5"):
        set_seed(seed)
        X_tr, y_tr, X_te, y_te = load_two_moons(
            n_samples=cfg["n_samples"],
            noise=cfg["noise"],
            test_frac=cfg["test_frac"],
            seed=seed,
        )
        model = two_moons_mlp(hidden=cfg["hidden"])

        hyper = dict(
            lr=cfg["lr"],
            amp=cfg["amp"],
            n_p=int(n_p),
            iter_p=int(iter_p),
        )

        run_t0 = time.time()
        result = train_full_batch(
            model=model,
            opt_name="spgd",
            hyper=hyper,
            loss_fn=F.binary_cross_entropy_with_logits,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            n_steps=cfg["n_steps"],
            stagnation_eps=cfg["stagnation_eps"],
            record_weights=False,
            accuracy_fn=binary_accuracy,
        )
        run_dur = time.time() - run_t0

        rows.append(dict(
            n_p=int(n_p),
            iter_p=int(iter_p),
            seed=int(seed),
            final_loss=result.final_loss,
            test_accuracy=result.test_accuracy,
            n_stagnation_episodes=result.stagnation["n_episodes"],
            total_stagnant_steps=result.stagnation["total_stagnant_steps"],
            longest_episode=result.stagnation["longest_episode"],
            duration_sec=run_dur,
            # Compute load metric: total candidate forward passes for this run.
            n_candidate_evals=int((cfg["n_steps"] // int(iter_p)) * int(n_p)),
        ))

    duration = time.time() - t0
    print(f"\nCompleted {len(runs)} runs in {duration:.1f} s")

    # --- save outputs -------------------------------------------------------
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = results_dir / "exp5_ablation.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # --- summary tables -----------------------------------------------------
    print("\n=== Mean test accuracy per (n_p, iter_p) cell ===")
    pv_acc = df.pivot_table(index="n_p", columns="iter_p",
                            values="test_accuracy", aggfunc="mean")
    print(pv_acc.round(4).to_string())

    print("\n=== Mean stagnation episodes per (n_p, iter_p) cell ===")
    pv_stag = df.pivot_table(index="n_p", columns="iter_p",
                             values="n_stagnation_episodes", aggfunc="mean")
    print(pv_stag.round(2).to_string())

    print("\n=== Mean final loss per (n_p, iter_p) cell ===")
    pv_loss = df.pivot_table(index="n_p", columns="iter_p",
                             values="final_loss", aggfunc="mean")
    print(pv_loss.round(5).to_string())

    print("\n=== Mean wall-time (s) per (n_p, iter_p) cell ===")
    pv_t = df.pivot_table(index="n_p", columns="iter_p",
                          values="duration_sec", aggfunc="mean")
    print(pv_t.round(2).to_string())

    print("\n=== Best cell (highest mean test accuracy) ===")
    best = pv_acc.stack().idxmax()
    best_val = pv_acc.stack().max()
    print(f"  n_p={best[0]}, iter_p={best[1]}  ->  mean test acc = {best_val:.4f}")


if __name__ == "__main__":
    main()
