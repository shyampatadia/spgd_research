"""Experiment 1 — Benchmark function validation.

Iterates over the cartesian product (function x dim x optimizer x seed) defined
in experiments/configs/exp1.yaml and writes:

    results/exp1_benchmarks.csv          one row per run with summary metrics
    results/exp1_convergence_curves.json running-min loss curves (for plotting)
    results/exp1_trajectories_2d.json    2D x-trajectories (only dim==2 runs)

Plotting is in a separate script (experiments/plot_exp1.py) so figures can be
regenerated without re-running the experiment.

Run from project root:
    uv run python experiments/exp1_benchmarks.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

from spgd_study.benchmarks import BENCHMARKS
from spgd_study.runner import random_init, run_benchmark
from spgd_study.utils import set_seed


def build_hyper(fn_name: str, opt_name: str, opt_cfg: Dict) -> Dict:
    h = {"lr": opt_cfg["lr_per_function"][fn_name]}
    if opt_name in ("pgd", "rpgd", "spgd"):
        h["amp"] = opt_cfg["amp_per_function"][fn_name]
        h["iter_p"] = opt_cfg["iter_p"]
    if opt_name in ("rpgd", "spgd"):
        h["n_p"] = opt_cfg["n_p"]
    return h


def list_runs(cfg: Dict) -> List[Dict]:
    runs: List[Dict] = []
    for fn_name in cfg["functions"]:
        for dim in cfg["dims"]:
            for opt_name, opt_cfg in cfg["optimizers"].items():
                hyper = build_hyper(fn_name, opt_name, opt_cfg)
                for seed in range(cfg["n_seeds"]):
                    runs.append(
                        dict(fn=fn_name, dim=dim, opt=opt_name, seed=seed, hyper=hyper)
                    )
    return runs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "experiments" / "configs" / "exp1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    runs = list_runs(cfg)
    print(f"Experiment 1: {len(runs)} runs queued "
          f"({len(cfg['functions'])} fns x {len(cfg['dims'])} dims x "
          f"{len(cfg['optimizers'])} opts x {cfg['n_seeds']} seeds)")

    rows: List[Dict] = []
    convergence: Dict[str, List[float]] = {}
    trajectories_2d: Dict[str, List] = {}

    t0 = time.time()
    for spec in tqdm(runs, desc="exp1"):
        fn_name = spec["fn"]
        dim = spec["dim"]
        opt_name = spec["opt"]
        seed = spec["seed"]
        bench = BENCHMARKS[fn_name]

        # Re-seed the GLOBAL RNG before each run so optimizer-internal randomness
        # (perturbation samples) is reproducible across (fn, dim, opt) for the
        # same seed. random_init uses its own generator and does not consume
        # global RNG state.
        set_seed(seed)
        x0 = random_init(seed, dim, bench.domain)

        result = run_benchmark(
            fn=bench.fn,
            optimum_f=bench.optimum_f,
            opt_name=opt_name,
            hyper=spec["hyper"],
            x0=x0,
            n_steps=cfg["n_steps"],
            eps_converged=cfg["eps_converged"],
            stagnation_eps=cfg["stagnation_eps"],
            store_trajectory=(dim == 2),
        )

        rows.append(
            dict(
                fn=fn_name,
                dim=dim,
                opt=opt_name,
                seed=seed,
                lr=spec["hyper"].get("lr"),
                amp=spec["hyper"].get("amp"),
                n_p=spec["hyper"].get("n_p"),
                iter_p=spec["hyper"].get("iter_p"),
                best_loss=result.best_loss,
                final_loss=result.losses[-1],
                iters_to_converge=result.iters_to_converge,
                success=result.success,
                n_stagnation_episodes=result.stagnation["n_episodes"],
                total_stagnant_steps=result.stagnation["total_stagnant_steps"],
                longest_episode=result.stagnation["longest_episode"],
            )
        )

        key = f"{fn_name}_d{dim}_{opt_name}_s{seed}"
        convergence[key] = result.running_min_losses
        if result.trajectory_2d is not None:
            traj_key = f"{fn_name}_{opt_name}_s{seed}"
            trajectories_2d[traj_key] = result.trajectory_2d

    duration = time.time() - t0
    print(f"\nCompleted {len(runs)} runs in {duration:.1f} s "
          f"({duration / len(runs) * 1000:.1f} ms/run avg)")

    # --- save outputs ---------------------------------------------------
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = results_dir / "exp1_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    with (results_dir / "exp1_convergence_curves.json").open("w") as f:
        json.dump(convergence, f)
    with (results_dir / "exp1_trajectories_2d.json").open("w") as f:
        json.dump(trajectories_2d, f)

    # --- summary table --------------------------------------------------
    print("\n=== Success rate by (fn, dim, opt) ===")
    pivot_success = (
        df.groupby(["fn", "dim", "opt"])["success"].mean().unstack("opt")
    )
    print(pivot_success.round(3).to_string())

    print("\n=== Mean best loss by (fn, dim, opt) ===")
    pivot_best = (
        df.groupby(["fn", "dim", "opt"])["best_loss"].mean().unstack("opt")
    )
    # pandas 2.1+ removed DataFrame.applymap; use .map (element-wise on frames).
    print(pivot_best.map(lambda v: f"{v:.4g}").to_string())

    print("\n=== Mean stagnation episodes by (fn, dim, opt) ===")
    pivot_stag = (
        df.groupby(["fn", "dim", "opt"])["n_stagnation_episodes"].mean().unstack("opt")
    )
    print(pivot_stag.round(1).to_string())


if __name__ == "__main__":
    main()
