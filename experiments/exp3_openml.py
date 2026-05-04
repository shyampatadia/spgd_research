"""Experiment 3 — OpenML-CC18 tabular classification.

Five tabular datasets x five optimizers x three seeds = 75 full-batch training
runs of an MLP (3 hidden layers of 64 units by default) using cross_entropy.

Datasets are downloaded from OpenML on first run and cached locally in
``.openml_cache/`` so subsequent runs are network-free. Each (dataset, seed)
pair is preprocessed once (impute -> standardize / one-hot) and reused
across all five optimizers, so any difference in outcome is attributable
to the optimizer and not to data preprocessing variance.

Outputs:
    results/exp3_openml.csv          one row per run
    results/exp3_loss_curves.json    per-step loss curves keyed by (ds, opt, seed)
    results/exp3_grad_norms.json     per-step gradient norms (same keying)

Run from project root:
    uv run python experiments/exp3_openml.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from spgd_study.data import load_openml_dataset
from spgd_study.models import MLP
from spgd_study.nn_runner import multiclass_accuracy, train_full_batch
from spgd_study.utils import set_seed

OPT_ORDER = ["sgd", "adam", "pgd", "rpgd", "spgd"]


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
        (project_root / "experiments" / "configs" / "exp3.yaml").read_text()
    )

    runs = [
        (ds_cfg, opt_name, seed)
        for ds_cfg in cfg["datasets"]
        for opt_name in cfg["optimizers"]
        for seed in range(cfg["n_seeds"])
    ]
    print(f"Experiment 3: {len(runs)} runs "
          f"({len(cfg['datasets'])} datasets x {len(cfg['optimizers'])} opts "
          f"x {cfg['n_seeds']} seeds)")

    # Pre-load each (dataset, seed) once so all five optimizers see exactly
    # the same train/test split + preprocessing fit.
    print("\nPre-loading datasets...")
    cache: Dict[str, tuple] = {}
    for ds_cfg in cfg["datasets"]:
        for seed in range(cfg["n_seeds"]):
            key = f"{ds_cfg['name']}_s{seed}"
            try:
                X_tr, y_tr, X_te, y_te, n_classes = load_openml_dataset(
                    openml_id=ds_cfg["openml_id"],
                    test_frac=cfg["test_frac"],
                    seed=seed,
                    verbose=(seed == 0),  # only chatty on the first seed per dataset
                )
                cache[key] = (X_tr, y_tr, X_te, y_te, n_classes)
                if seed == 0:
                    shape_str = str(tuple(X_tr.shape))
                    print(f"  {ds_cfg['name']:<10s}  X {shape_str:>15s}  "
                          f"n_classes {n_classes}  (seed {seed})")
            except Exception as e:
                print(f"  FAILED to load {key}: {e}")
                cache[key] = None

    rows: List[Dict] = []
    losses_history: Dict[str, List[float]] = {}
    grad_history: Dict[str, List[float]] = {}

    print()
    t0 = time.time()
    for ds_cfg, opt_name, seed in tqdm(runs, desc="exp3"):
        ds_name = ds_cfg["name"]
        cache_key = f"{ds_name}_s{seed}"
        if cache.get(cache_key) is None:
            continue
        X_tr, y_tr, X_te, y_te, n_classes = cache[cache_key]

        set_seed(seed)
        in_dim = int(X_tr.shape[1])
        model = MLP(
            in_dim=in_dim,
            hidden=cfg["hidden"],
            out_dim=n_classes,
            n_hidden=cfg["n_hidden"],
        )

        hyper = build_hyper(opt_name, cfg["optimizers"][opt_name])
        result = train_full_batch(
            model=model,
            opt_name=opt_name,
            hyper=hyper,
            loss_fn=F.cross_entropy,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            n_steps=cfg["n_steps"],
            stagnation_eps=cfg["stagnation_eps"],
            record_weights=False,
            accuracy_fn=multiclass_accuracy,
        )

        rows.append(dict(
            dataset=ds_name,
            opt=opt_name,
            seed=seed,
            n_classes=n_classes,
            n_features=in_dim,
            n_train=int(X_tr.shape[0]),
            final_loss=result.final_loss,
            test_accuracy=result.test_accuracy,
            n_stagnation_episodes=result.stagnation["n_episodes"],
            total_stagnant_steps=result.stagnation["total_stagnant_steps"],
            longest_episode=result.stagnation["longest_episode"],
            **{f"hp_{k}": v for k, v in hyper.items()},
        ))

        key = f"{ds_name}_{opt_name}_s{seed}"
        losses_history[key] = result.losses
        grad_history[key] = result.grad_norm_history

    duration = time.time() - t0
    print(f"\nCompleted {len(rows)} runs in {duration:.1f} s")

    # --- save outputs ------------------------------------------------------
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "exp3_openml.csv", index=False)
    print(f"Wrote {results_dir / 'exp3_openml.csv'}")

    with (results_dir / "exp3_loss_curves.json").open("w") as f:
        json.dump(losses_history, f)
    with (results_dir / "exp3_grad_norms.json").open("w") as f:
        json.dump(grad_history, f)

    # --- summary tables ----------------------------------------------------
    cols = [c for c in OPT_ORDER if c in df["opt"].unique()]

    print("\n=== Mean test accuracy by (dataset, opt) ===")
    pivot_acc = df.groupby(["dataset", "opt"])["test_accuracy"].mean().unstack("opt")
    print(pivot_acc[cols].round(4).to_string())

    print("\n=== Mean stagnation episodes by (dataset, opt) ===")
    pivot_stag = df.groupby(["dataset", "opt"])["n_stagnation_episodes"].mean().unstack("opt")
    print(pivot_stag[cols].round(1).to_string())

    print("\n=== Mean final training loss by (dataset, opt) ===")
    pivot_loss = df.groupby(["dataset", "opt"])["final_loss"].mean().unstack("opt")
    print(pivot_loss[cols].map(lambda v: f"{v:.4f}").to_string())

    print("\n=== SPGD vs RPGD paired by (dataset, seed): test-accuracy diff ===")
    print("(positive => SPGD beats RPGD on test acc)")
    pv = df.pivot_table(
        index=["dataset", "seed"], columns="opt", values="test_accuracy"
    ).reset_index()
    if "spgd" in pv.columns and "rpgd" in pv.columns:
        pv["spgd_minus_rpgd"] = pv["spgd"] - pv["rpgd"]
        diff = pv.groupby("dataset")["spgd_minus_rpgd"].agg(["mean", "median", "std"])
        print(diff.map(lambda v: f"{v:+.4f}").to_string())


if __name__ == "__main__":
    main()
