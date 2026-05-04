"""Non-convex benchmark functions used in Experiment 1.

All functions accept a tensor `x` of shape (..., d) and return a tensor of
shape (...,). They are vectorised across leading dimensions, so the same
implementation works for a single point, a batch of candidates, or a grid
sweep — useful for both optimisation runs and 2D landscape plots.

Each function exposes domain bounds and the location/value of the global
minimum via the BENCHMARKS registry, so experiment code can iterate over
them without hard-coding constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch


# -----------------------------------------------------------------------------
# Function definitions
# -----------------------------------------------------------------------------

_RASTRIGIN_A = 10.0


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    r"""Rastrigin: f(x) = A d + sum_i [x_i^2 - A cos(2 pi x_i)],  A = 10.

    Highly multimodal: a near-quadratic envelope studded with a cosine grid
    of local minima at the integer points. Global minimum at x = 0,  f = 0.
    """
    d = x.shape[-1]
    return _RASTRIGIN_A * d + (x**2 - _RASTRIGIN_A * torch.cos(2 * math.pi * x)).sum(-1)


def ackley(x: torch.Tensor) -> torch.Tensor:
    r"""Ackley: nearly flat outer region with a sharp narrow funnel at the origin.

    f(x) = -20 exp(-0.2 sqrt(mean(x_i^2)))
           - exp(mean(cos(2 pi x_i)))
           + 20 + e
    Global minimum at x = 0,  f = 0.
    """
    d = x.shape[-1]
    sum_sq = (x**2).sum(-1) / d
    sum_cos = torch.cos(2 * math.pi * x).sum(-1) / d
    return -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq)) - torch.exp(sum_cos) + 20.0 + math.e


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    r"""Rosenbrock: banana-shaped curved valley.

    f(x) = sum_{i=1}^{d-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Global minimum at x = (1, 1, ..., 1),  f = 0. The valley is easy to
    enter but hard to traverse — gradients along it are very small.
    """
    return (100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2).sum(-1)


# -----------------------------------------------------------------------------
# Registry — maps name -> metadata used by experiment runners
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    domain: Tuple[float, float]      # symmetric box [-r, r]^d for sampling x0
    optimum_x: float                 # global minimiser is this value in every coord
    optimum_f: float                 # global minimum value
    typical_amp: float               # SPGD perturbation amplitude that works on this fn


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "rastrigin":  BenchmarkSpec("rastrigin",  rastrigin,  domain=(-5.12, 5.12), optimum_x=0.0, optimum_f=0.0, typical_amp=2.5),
    "ackley":     BenchmarkSpec("ackley",     ackley,     domain=(-5.0, 5.0),   optimum_x=0.0, optimum_f=0.0, typical_amp=2.5),
    "rosenbrock": BenchmarkSpec("rosenbrock", rosenbrock, domain=(-2.048, 2.048), optimum_x=1.0, optimum_f=0.0, typical_amp=1.0),
}
