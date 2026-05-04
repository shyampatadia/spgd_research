"""Random-Perturbation Gradient Descent (RPGD).

This is the *control* baseline for SPGD. It generates the same N_P
candidate perturbations every Iter_P steps as SPGD, but picks the kept
candidate uniformly at random instead of by argmin. The acceptance
criterion (only accept if not worse than current) is identical.

Difference from SPGD:  selection rule only.
Difference from PGD :  same as SPGD vs PGD — N_P candidates instead of 1.

If SPGD beats RPGD ⇒ the steepest-selection rule contributes.
If SPGD ≈ RPGD     ⇒ the perturbation mechanism alone suffices.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch.optim.optimizer import Optimizer


class RPGD(Optimizer):
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
                p.add_(amp * (2.0 * torch.rand_like(p) - 1.0))

    @torch.no_grad()
    def _perturbation_phase(self, closure: Callable[[], torch.Tensor]) -> None:
        saved = self._snapshot()
        current_loss = float(closure())
        self.n_candidate_evals += 1

        n_p = self.param_groups[0]["n_p"]
        # Pick which candidate to keep up front; we still evaluate all n_p
        # to match SPGD's compute exactly.
        chosen_idx = int(torch.randint(0, n_p, (1,)).item())

        chosen_loss: Optional[float] = None
        chosen_snapshot: Optional[List[torch.Tensor]] = None

        for j in range(n_p):
            self._restore(saved)
            self._perturb_in_place()
            cand_loss = float(closure())
            self.n_candidate_evals += 1
            if j == chosen_idx:
                chosen_loss = cand_loss
                chosen_snapshot = self._snapshot()

        # Same acceptance rule as SPGD: accept iff not worse than current.
        if chosen_snapshot is not None and chosen_loss is not None and chosen_loss <= current_loss:
            self._restore(chosen_snapshot)
            self.n_accepted += 1
        else:
            self._restore(saved)

        self.n_perturbations += 1

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        if closure is None:
            raise RuntimeError("RPGD requires a forward-only closure returning a loss tensor.")

        iter_p = self.param_groups[0]["iter_p"]
        if self._step_count > 0 and self._step_count % iter_p == 0:
            self._perturbation_phase(closure)

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
