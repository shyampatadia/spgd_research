"""Experiment 6: SPGD vs SGD/Adam/RPGD on non-convex matrix completion.

Tests the SPGD paper's stated future-work hypothesis -- that SPGD has
"potential to significantly enhance neural network training" and "improve
convergence rates" (Vahedi & Ilies 2024, Conclusion) -- on a non-convex
ML benchmark with documented saddle structure: low-rank Burer-Monteiro
factorisation of the MovieLens-100K rating matrix (Ge-Lee-Ma 2016,
Sun-Qu-Wright 2018).

The SPGD and RPGD optimisers are the verified classes from
src/spgd_study/optimizers/, faithful to Algorithm 1 of the paper:
Uniform(-amp, +amp) per-coordinate candidates, <= acceptance rule, and
perturbation phase BEFORE the gradient step.  SGD and Adam are stock
torch.optim implementations.  All four methods see the same L2-regularised
loss inside the closure / forward pass -- a fair comparison.

Run from project root:
    uv run python experiments/exp6_matrix_completion.py
"""

from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from spgd_study.optimizers import RPGD, SPGD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / ".movielens_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "exp6.yaml"

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_movielens(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "ml-100k"
    if (extracted / "u.data").exists():
        return extracted
    zip_path = data_dir / "ml-100k.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens-100K from {ML100K_URL} ...")
        urllib.request.urlretrieve(ML100K_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    return extracted


def load_movielens100k(seed: int, test_frac: float, device: torch.device):
    """Load MovieLens-100K, split by seed; return GPU tensors with the train
    mean already subtracted from ratings (so we factorise residuals)."""
    extracted = _ensure_movielens(DATA_DIR)
    raw = np.loadtxt(extracted / "u.data", dtype=np.int64)
    user = raw[:, 0] - 1
    item = raw[:, 1] - 1
    rating = raw[:, 2].astype(np.float32)
    n_users = int(user.max()) + 1
    n_movies = int(item.max()) + 1

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rating))
    n_test = int(len(rating) * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    mean = float(rating[train_idx].mean())

    def to_dev(idx):
        return (
            torch.from_numpy(user[idx]).to(device),
            torch.from_numpy(item[idx]).to(device),
            torch.from_numpy(rating[idx] - mean).to(device),
        )
    ui_tr, ii_tr, r_tr = to_dev(train_idx)
    ui_te, ii_te, r_te = to_dev(test_idx)
    return ui_tr, ii_tr, r_tr, ui_te, ii_te, r_te, n_users, n_movies, mean


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MFModel(nn.Module):
    """Burer-Monteiro factorisation: \\widehat{M}_{ij} = (U V^T)_{ij}."""

    def __init__(self, n_users, n_movies, rank, init_scale, device, seed):
        super().__init__()
        torch.manual_seed(seed)
        U = torch.randn(n_users, rank, device=device) * init_scale
        V = torch.randn(n_movies, rank, device=device) * init_scale
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)

    def predict(self, ui, ii):
        return (self.U[ui] * self.V[ii]).sum(dim=1)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def loss_fn(model, ui, ii, r, l2_reg):
    """L2-regularised MSE, identical for all four methods."""
    pred = model.predict(ui, ii)
    L = ((pred - r) ** 2).mean()
    if l2_reg > 0:
        L = L + l2_reg * (model.U.pow(2).mean() + model.V.pow(2).mean())
    return L


def data_mse(model, ui, ii, r):
    """Raw data MSE (no L2 term) -- the metric we log/plot."""
    with torch.no_grad():
        pred = model.predict(ui, ii)
        return float(((pred - r) ** 2).mean().item())


def rmse(model, ui, ii, r):
    with torch.no_grad():
        pred = model.predict(ui, ii)
        return float(torch.sqrt(((pred - r) ** 2).mean()).item())


def grad_norm(model, ui, ii, r, l2_reg):
    L = loss_fn(model, ui, ii, r, l2_reg)
    grads = torch.autograd.grad(L, [model.U, model.V])
    return float(torch.sqrt(grads[0].pow(2).sum() + grads[1].pow(2).sum()).item())


# ---------------------------------------------------------------------------
# Per-method runner
# ---------------------------------------------------------------------------

def make_optimizer(method, model, cfg):
    if method == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr_sgd"])
    if method == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr_adam"],
            betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
            eps=cfg["adam_eps"],
        )
    if method == "spgd":
        return SPGD(model.parameters(),
                    lr=cfg["lr_spgd"], amp=cfg["amp"],
                    n_p=cfg["n_p"], iter_p=cfg["iter_p"])
    if method == "rpgd":
        return RPGD(model.parameters(),
                    lr=cfg["lr_spgd"], amp=cfg["amp"],
                    n_p=cfg["n_p"], iter_p=cfg["iter_p"])
    raise ValueError(method)


def run_method(method, cfg, ui_tr, ii_tr, r_tr, ui_te, ii_te, r_te,
               n_users, n_movies, device, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MFModel(n_users, n_movies, cfg["rank"], cfg["init_scale"], device, seed)
    optimizer = make_optimizer(method, model, cfg)

    l2 = cfg["l2_reg"]
    history = {"step": [], "train_loss": [], "test_rmse": [], "grad_norm": []}
    t0 = time.time()

    for step in range(1, cfg["n_steps"] + 1):
        if method in ("spgd", "rpgd"):
            def closure():
                return loss_fn(model, ui_tr, ii_tr, r_tr, l2)
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            L = loss_fn(model, ui_tr, ii_tr, r_tr, l2)
            L.backward()
            optimizer.step()

        if step % cfg["log_every"] == 0 or step == 1:
            history["step"].append(step)
            history["train_loss"].append(data_mse(model, ui_tr, ii_tr, r_tr))
            history["test_rmse"].append(rmse(model, ui_te, ii_te, r_te))
            history["grad_norm"].append(grad_norm(model, ui_tr, ii_tr, r_tr, l2))

    wall = time.time() - t0
    final = {
        "method": method,
        "seed": seed,
        "wall_sec": wall,
        "final_train_loss": history["train_loss"][-1],
        "final_test_rmse": history["test_rmse"][-1],
        "final_grad_norm": history["grad_norm"][-1],
    }
    if method in ("spgd", "rpgd"):
        final["n_perturbations"] = optimizer.n_perturbations
        final["n_accepted"] = optimizer.n_accepted
        final["n_candidate_evals"] = optimizer.n_candidate_evals
    return history, final


def main():
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("== Exp 6: SPGD vs SGD/Adam/RPGD on MovieLens-100K matrix completion ==")
    print(f"device: {device}")
    print("(SPGD/RPGD = verified optimiser classes from spgd_study.optimizers,")
    print(" paper-faithful Uniform(-amp,+amp) perturbations + <= acceptance.)")

    methods = ["sgd", "adam", "rpgd", "spgd"]
    n_seeds = cfg["n_seeds"]

    rows_history = []
    rows_final = []

    for seed in range(n_seeds):
        ui_tr, ii_tr, r_tr, ui_te, ii_te, r_te, n_users, n_movies, mean = (
            load_movielens100k(seed=seed, test_frac=cfg["test_frac"], device=device)
        )
        if seed == 0:
            print(f"train: {len(r_tr):,} obs, test: {len(r_te):,} obs, "
                  f"users: {n_users}, movies: {n_movies}, mean rating: {mean:.3f}")
            print(f"rank: {cfg['rank']}, init_scale: {cfg['init_scale']}, "
                  f"steps: {cfg['n_steps']}, l2_reg: {cfg['l2_reg']}")
            print(f"SPGD/RPGD: amp={cfg['amp']}, n_p={cfg['n_p']}, "
                  f"iter_p={cfg['iter_p']}")
            print()
        for method in methods:
            history, final = run_method(method, cfg, ui_tr, ii_tr, r_tr,
                                        ui_te, ii_te, r_te,
                                        n_users, n_movies, device, seed)
            extra = ""
            if "n_accepted" in final:
                extra = (f"  acc={final['n_accepted']}/"
                         f"{final['n_perturbations']}")
            print(f"  {method:5s}  seed={seed}  "
                  f"train_loss={final['final_train_loss']:.4f}  "
                  f"test_rmse={final['final_test_rmse']:.4f}  "
                  f"grad_norm={final['final_grad_norm']:.4f}  "
                  f"({final['wall_sec']:.1f}s){extra}")
            for k in range(len(history["step"])):
                rows_history.append({
                    "method": method,
                    "seed": seed,
                    "step": history["step"][k],
                    "train_loss": history["train_loss"][k],
                    "test_rmse": history["test_rmse"][k],
                    "grad_norm": history["grad_norm"][k],
                })
            rows_final.append(final)

    df_hist = pd.DataFrame(rows_history)
    df_fin = pd.DataFrame(rows_final)

    summary = df_fin.groupby("method").agg(
        train_loss_mean=("final_train_loss", "mean"),
        train_loss_std=("final_train_loss", "std"),
        test_rmse_mean=("final_test_rmse", "mean"),
        test_rmse_std=("final_test_rmse", "std"),
        grad_norm_mean=("final_grad_norm", "mean"),
        wall_mean=("wall_sec", "mean"),
    ).loc[methods]
    print()
    print("=== summary (mean +/- std over seeds) ===")
    print(summary.to_string())

    df_hist.to_csv(RESULTS_DIR / "exp6_mc_history.csv", index=False)
    df_fin.to_csv(RESULTS_DIR / "exp6_mc_finals.csv", index=False)
    summary.to_csv(RESULTS_DIR / "exp6_mc_summary.csv")
    with (RESULTS_DIR / "exp6_mc.json").open("w") as f:
        json.dump({"config": cfg, "finals": rows_final,
                   "summary": summary.reset_index().to_dict(orient="records")},
                  f, indent=2)

    pivot = df_fin.pivot(index="seed", columns="method", values="final_test_rmse")
    print()
    print("=== paired test_rmse deltas (negative = SPGD better) ===")
    for ref in ["sgd", "rpgd", "adam"]:
        if ref in pivot.columns and "spgd" in pivot.columns:
            d = pivot["spgd"] - pivot[ref]
            d_str = ", ".join(f"{x:+.4f}" for x in d.tolist())
            print(f"  spgd - {ref:5s}: per-seed = [{d_str}], "
                  f"mean = {d.mean():+.4f}")

    spgd_finals = [f for f in rows_final if f["method"] == "spgd"]
    rpgd_finals = [f for f in rows_final if f["method"] == "rpgd"]
    if spgd_finals:
        print()
        print("=== SPGD acceptance diagnostics ===")
        for f in spgd_finals:
            rate = (f["n_accepted"] / f["n_perturbations"]
                    if f["n_perturbations"] else 0.0)
            print(f"  spgd seed={f['seed']}  accepted={f['n_accepted']}/"
                  f"{f['n_perturbations']}  ({rate:.1%})  "
                  f"cand_evals={f['n_candidate_evals']}")
        for f in rpgd_finals:
            rate = (f["n_accepted"] / f["n_perturbations"]
                    if f["n_perturbations"] else 0.0)
            print(f"  rpgd seed={f['seed']}  accepted={f['n_accepted']}/"
                  f"{f['n_perturbations']}  ({rate:.1%})  "
                  f"cand_evals={f['n_candidate_evals']}")

    print()
    print("Wrote:")
    print(f"  {RESULTS_DIR / 'exp6_mc_history.csv'}")
    print(f"  {RESULTS_DIR / 'exp6_mc_finals.csv'}")
    print(f"  {RESULTS_DIR / 'exp6_mc_summary.csv'}")
    print(f"  {RESULTS_DIR / 'exp6_mc.json'}")


if __name__ == "__main__":
    main()
