"""Perturbed Gradient Descent (PGD) — periodic single-perturbation variant.

Adds a single uniform perturbation to the parameters every ``iter_p``
gradient steps, then continues with a standard gradient step. There is
no candidate generation and no acceptance check — this is the cheapest
"perturbation-aware" optimiser in the comparison and stands in for the
"add some noise to escape" idea in Jin et al. (2017).

Note
----
The original Jin et al. PGD triggers perturbation on stagnation
(``||grad|| < eps``), not on a fixed schedule. We use a fixed schedule
here so that PGD/RPGD/SPGD share the same timing assumption and any
performance differences are attributable to selection mechanism.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch.optim.optimizer import Optimizer


class PGD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        amp: float = 0.1,
        iter_p: int = 5,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if amp <= 0:
            raise ValueError(f"amp must be positive, got {amp}")
        if iter_p < 1:
            raise ValueError(f"iter_p must be >= 1, got {iter_p}")

        defaults = dict(lr=lr, amp=amp, iter_p=iter_p)
        super().__init__(params, defaults)
        self._step_count = 0
        self.n_perturbations = 0

    @torch.no_grad()
    def _perturb_in_place(self) -> None:
        for g in self.param_groups:
            amp = g["amp"]
            for p in g["params"]:
                p.add_(amp * (2.0 * torch.rand_like(p) - 1.0))

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        if closure is None:
            raise RuntimeError("PGD requires a forward-only closure returning a loss tensor.")

        iter_p = self.param_groups[0]["iter_p"]
        if self._step_count > 0 and self._step_count % iter_p == 0:
            self._perturb_in_place()
            self.n_perturbations += 1

        self.zero_grad()
        loss = closure()
        loss.backward()

        with torch.no_grad():
            for g in self.param_groups:
                lr = g["lr"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    p.add_(p.grad, alpha=-lr)

        self._step_count += 1
        return loss
