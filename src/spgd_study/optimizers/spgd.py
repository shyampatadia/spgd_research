"""Steepest Perturbed Gradient Descent (SPGD).

Reference
---------
A. M. Vahedi & H. T. Ilies, "SPGD: Steepest Perturbed Gradient Descent
Optimization," arXiv:2411.04946, 2024.  Algorithm 1.

Algorithm
---------
Every ``iter_p`` gradient steps, sample ``n_p`` candidate perturbations
of the current parameters from Uniform(-amp, +amp) per coordinate.
Accept the candidate with the lowest loss IF it is no worse than the
current loss (<= comparison; equality acceptance is intentional, to
keep exploring on flat plateaus). Then take a standard gradient step
from the (possibly updated) point.

Closure protocol
----------------
``optimizer.step(closure)`` requires a *forward-only* closure that
returns a scalar loss tensor. The closure must NOT call ``.backward()``
or zero gradients — the optimizer handles both. Example::

    def closure():
        return loss_fn(model(x), y)

    loss = optimizer.step(closure)

This protocol differs from PyTorch's LBFGS convention; the no-backward
closure lets us evaluate ``n_p`` candidate perturbations under
``torch.no_grad()`` without paying for ``n_p`` extra backward passes.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch.optim.optimizer import Optimizer


class SPGD(Optimizer):
    """Steepest Perturbed Gradient Descent.

    Parameters
    ----------
    params : iterable of ``torch.nn.Parameter`` (or list of param groups).
    lr     : gradient step size (alpha in the paper).
    amp    : perturbation half-width; samples are drawn from
             Uniform(-amp, +amp) per coordinate.
    n_p    : number of candidate perturbations per perturbation phase.
    iter_p : number of gradient steps between perturbation phases.

    Diagnostics (read-only attributes)
    ----------------------------------
    n_perturbations    : number of perturbation phases executed so far.
    n_candidate_evals  : total number of forward evaluations spent on
                         candidate selection (≈ n_perturbations * (n_p+1)).
    n_accepted         : number of perturbation phases where some
                         candidate was accepted (i.e. moved the params).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        amp: float = 0.1,
        n_p: int = 10,
        iter_p: int = 5,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if amp <= 0:
            raise ValueError(f"amp must be positive, got {amp}")
        if n_p < 1:
            raise ValueError(f"n_p must be >= 1, got {n_p}")
        if iter_p < 1:
            raise ValueError(f"iter_p must be >= 1, got {iter_p}")

        defaults = dict(lr=lr, amp=amp, n_p=n_p, iter_p=iter_p)
        super().__init__(params, defaults)
        self._step_count = 0
        self.n_perturbations = 0
        self.n_candidate_evals = 0
        self.n_accepted = 0

    # ---- helpers --------------------------------------------------------

    @torch.no_grad()
    def _snapshot(self) -> List[torch.Tensor]:
        return [p.detach().clone() for g in self.param_groups for p in g["params"]]

    @torch.no_grad()
    def _restore(self, snap: List[torch.Tensor]) -> None:
        idx = 0
        for g in self.param_groups:
            for p in g["params"]:
                p.copy_(snap[idx])
                idx += 1

    @torch.no_grad()
    def _perturb_in_place(self) -> None:
        for g in self.param_groups:
            amp = g["amp"]
            for p in g["params"]:
                # Uniform(-amp, +amp) per coordinate
                p.add_(amp * (2.0 * torch.rand_like(p) - 1.0))

    # ---- main phases ----------------------------------------------------

    @torch.no_grad()
    def _perturbation_phase(self, closure: Callable[[], torch.Tensor]) -> None:
        saved = self._snapshot()
        current_loss = float(closure())
        self.n_candidate_evals += 1

        n_p = self.param_groups[0]["n_p"]

        best_loss = current_loss
        best_snapshot: Optional[List[torch.Tensor]] = None

        for _ in range(n_p):
            self._restore(saved)
            self._perturb_in_place()
            cand_loss = float(closure())
            self.n_candidate_evals += 1
            # <= per the paper: accept ties to keep exploring flat regions.
            if cand_loss <= best_loss:
                best_loss = cand_loss
                best_snapshot = self._snapshot()

        if best_snapshot is not None:
            self._restore(best_snapshot)
            self.n_accepted += 1
        else:
            self._restore(saved)

        self.n_perturbations += 1

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        if closure is None:
            raise RuntimeError(
                "SPGD requires a closure that returns a scalar loss tensor "
                "(forward only — do NOT call .backward())."
            )

        # Phase 1: periodic perturbation
        iter_p = self.param_groups[0]["iter_p"]
        if self._step_count > 0 and self._step_count % iter_p == 0:
            self._perturbation_phase(closure)

        # Phase 2: gradient step at the (possibly updated) parameters
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
