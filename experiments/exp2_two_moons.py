"""Experiment 2 — Two Moons MLP visual saddle analysis.

Trains a 2->32->32->1 MLP on the make_moons synthetic dataset (1000 points),
full-batch, for 2000 steps. 5 optimizers x 5 seeds = 25 runs. Seed 0 of each
optimizer additionally records its weight trajectory so plot_exp2.py can
render the PCA loss-landscape figure (the showpiece for the report).

Run from project root:
    uv run python experiments/exp2_two_moons.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from spgd_study.data import load_two_moons
from spgd_study.models import two_moons_mlp
from spgd_study.nn_runner import binary_accuracy, train_full_batch
from spgd_study.utils import set_seed


def build_hyper(opt_name: str, opt_cfg: Dict) -> Dict:
    h = {"lr": opt_cfg["lr"]}
    if opt_name in ("pgd", "rpgd", "spgd"):
        h["amp"] = opt_cfg["amp"]
        h["iter_p"] = opt_cfg["iter_p"]
    if opt_name in ("rpgd", "spgd"):
        h["n_p"] = opt_cfg["n_p"]
    return h


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (project_root / "experiments" / "configs" / "exp2.yaml").read_text()
    )

    rows: List[Dict] = []
    losses_history: Dict[str, List[float]] = {}
    grad_history: Dict[str, List[float]] = {}
    weights_history: Dict[str, np.ndarray] = {}

    runs = [
        (opt_name, seed)
        for opt_name in cfg["optimizers"]
        for seed in range(cfg["n_seeds"])
    ]
    print(f"Experiment 2: {len(runs)} runs "
          f"({len(cfg['optimizers'])} opts x {cfg['n_seeds']} seeds)")

    t0 = time.time()
    for opt_name, seed in tqdm(runs, desc="exp2"):
        # Seed before any randomness so init + perturbation samples align
        # across optimizers for the same seed (paired comparison).
        set_seed(seed)
        X_tr, y_tr, X_te, y_te = load_two_moons(
            n_samples=cfg["n_samples"],
            noise=cfg["noise"],
            test_frac=cfg["test_frac"],
            seed=seed,
        )
        model = two_moons_mlp(hidden=cfg["hidden"])

        record_weights = (seed == 0)
        hyper = build_hyper(opt_name, cfg["optimizers"][opt_name])

        result = train_full_batch(
            model=model,
            opt_name=opt_name,
            hyper=hyper,
            loss_fn=F.binary_cross_entropy_with_logits,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            n_steps=cfg["n_steps"],
            stagnation_eps=cfg["stagnation_eps"],
            record_weights=record_weights,
            weights_snapshot_every=cfg["weights_snapshot_every"],
            accuracy_fn=binary_accuracy,
        )

        rows.append(dict(
            opt=opt_name,
            seed=seed,
            final_loss=result.final_loss,
            test_accuracy=result.test_accuracy,
            n_stagnation_episodes=result.stagnation["n_episodes"],
            total_stagnant_steps=result.stagnation["total_stagnant_steps"],
            longest_episode=result.stagnation["longest_episode"],
            **{f"hp_{k}": v for k, v in hyper.items()},
        ))

        key = f"{opt_name}_s{seed}"
        losses_history[key] = result.losses
        grad_history[key] = result.grad_norm_history
        if result.weights_history is not None:
            weights_history[opt_name] = result.weights_history

    duration = time.time() - t0
    print(f"\nCompleted {len(runs)} runs in {duration:.1f} s")

    # --- save outputs ------------------------------------------------------
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "exp2_two_moons.csv", index=False)
    print(f"Wrote {results_dir / 'exp2_two_moons.csv'}")

    with (results_dir / "exp2_loss_curves.json").open("w") as f:
        json.dump(losses_history, f)
    with (results_dir / "exp2_grad_norms.json").open("w") as f:
        json.dump(grad_history, f)
    np.savez(results_dir / "exp2_weights_history.npz", **weights_history)
    print(f"Wrote {results_dir / 'exp2_weights_history.npz'}")

    # --- summary tables ----------------------------------------------------
    print("\n=== Mean test accuracy by optimizer ===")
    print(df.groupby("opt")["test_accuracy"]
            .agg(["mean", "std"]).round(4).to_string())

    print("\n=== Mean stagnation episodes by optimizer ===")
    print(df.groupby("opt")["n_stagnation_episodes"]
            .agg(["mean", "std"]).round(2).to_string())

    print("\n=== Mean final loss by optimizer ===")
    print(df.groupby("opt")["final_loss"]
            .agg(["mean", "std"]).round(5).to_string())


if __name__ == "__main__":
    main()
