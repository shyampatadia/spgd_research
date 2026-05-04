"""Generic single-run benchmark loop used by Experiment 1.

A "run" minimises a scalar function f: R^d -> R using one optimiser from
fixed initial point x0 for n_steps iterations, recording per-step diagnostics.

Why a dedicated runner: we need consistent metric definitions across all
five optimisers (SGD, Adam, PGD, RPGD, SPGD), including:
    - best loss SEEN (running min, robust to overshoot from a perturbation pick),
    - first iteration where running-min loss is within eps of optimum,
    - stagnation episodes detected from the gradient-norm stream.

Closure-vs-classic split: SPGD/PGD/RPGD use a forward-only closure protocol
(see spgd_study/optimizers/spgd.py); SGD/Adam follow the standard
zero_grad → forward → backward → step idiom. Both branches end the iteration
with a populated ``x.grad`` so we can read the gradient norm uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import SGD, Adam

from .diagnostics import StagnationTracker
from .optimizers import PGD, RPGD, SPGD
from .utils import grad_norm


@dataclass
class RunResult:
    """Outcome of a single optimiser run on a benchmark function."""

    losses: List[float]                    # per-iteration in-loop losses
    running_min_losses: List[float]        # cumulative best-so-far loss
    best_loss: float                       # min(running_min_losses)
    best_x: List[float]                    # x at the iteration that achieved best_loss
    final_x: List[float]                   # x after the very last gradient step
    iters_to_converge: int                 # first iter where running min within eps of optimum, else n_steps
    success: bool                          # iters_to_converge < n_steps
    stagnation: Dict                       # StagnationTracker.summary()
    grad_norm_history: List[float]         # ||grad|| at each iteration's gradient-step point
    trajectory_2d: Optional[List[Tuple[float, float]]] = None  # only stored when dim == 2


def random_init(seed: int, dim: int, domain: Tuple[float, float]) -> torch.Tensor:
    """Sample x0 uniformly in [domain[0], domain[1]]^d, seeded reproducibly.

    Uses a per-call torch.Generator so it does NOT touch the global RNG —
    that lets the global RNG stay consistent across optimisers for a given seed,
    so cross-optimiser perturbation sequences are aligned.
    """
    g = torch.Generator().manual_seed(int(seed))
    low, high = domain
    return torch.rand(int(dim), generator=g, dtype=torch.float32) * (high - low) + low


def _make_optimizer(name: str, params, hyper: Dict):
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=hyper["lr"])
    if name == "adam":
        return Adam(params, lr=hyper["lr"])
    if name == "pgd":
        return PGD(params, lr=hyper["lr"], amp=hyper["amp"], iter_p=hyper["iter_p"])
    if name == "rpgd":
        return RPGD(params, lr=hyper["lr"], amp=hyper["amp"], n_p=hyper["n_p"], iter_p=hyper["iter_p"])
    if name == "spgd":
        return SPGD(params, lr=hyper["lr"], amp=hyper["amp"], n_p=hyper["n_p"], iter_p=hyper["iter_p"])
    raise ValueError(f"unknown optimizer: {name!r}")


def run_benchmark(
    fn: Callable[[torch.Tensor], torch.Tensor],
    optimum_f: float,
    opt_name: str,
    hyper: Dict,
    x0: torch.Tensor,
    n_steps: int,
    eps_converged: float,
    stagnation_eps: float,
    store_trajectory: bool = False,
) -> RunResult:
    """Optimise ``fn`` from ``x0`` and return per-step diagnostics."""

    x = x0.clone().detach().requires_grad_(True)
    opt = _make_optimizer(opt_name, [x], hyper)
    is_perturbing = isinstance(opt, (SPGD, PGD, RPGD))
    tracker = StagnationTracker(eps=stagnation_eps)

    losses: List[float] = []
    running_min: List[float] = []
    best_loss = float("inf")
    best_x = x.detach().clone()
    iters_to_converge = -1
    trajectory: List[Tuple[float, float]] = []

    for step in range(n_steps):
        if store_trajectory:
            trajectory.append((float(x[0].item()), float(x[1].item())))

        if is_perturbing:
            def closure():
                return fn(x)
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            loss = fn(x)
            loss.backward()
            opt.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x.detach().clone()
        running_min.append(best_loss)

        # ||grad|| AFTER the step uses the gradient computed during this step
        # (PyTorch leaves it populated on x.grad after .step()).
        gn = grad_norm([x])
        tracker.update(gn)

        if iters_to_converge < 0 and (best_loss - optimum_f) < eps_converged:
            iters_to_converge = step + 1

    if iters_to_converge < 0:
        iters_to_converge = n_steps

    if store_trajectory:
        # Append the final point so the trajectory ends at x_{n_steps}.
        trajectory.append((float(x[0].item()), float(x[1].item())))

    return RunResult(
        losses=losses,
        running_min_losses=running_min,
        best_loss=best_loss,
        best_x=best_x.tolist(),
        final_x=x.detach().tolist(),
        iters_to_converge=iters_to_converge,
        success=iters_to_converge < n_steps,
        stagnation=tracker.summary(),
        grad_norm_history=tracker.grad_norm_history,
        trajectory_2d=trajectory if store_trajectory else None,
    )
