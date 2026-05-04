"""Smoke test: SPGD must escape a local minimum that traps vanilla SGD.

Setup
-----
- 2D Rastrigin function (highly multimodal; integer-grid local minima).
- Initial point [3.5, 3.5] — the gradient there points toward the (3, 3)
  basin where vanilla GD will get stuck (Rastrigin([3, 3]) = 18).
- 500 iterations.
- All four optimisers (SGD, PGD, RPGD, SPGD) start from the same seed so
  perturbation RNG sequences are identical until selection logic diverges.

Metric
------
We report the *best loss seen during training* (running min) rather than the
loss at the very last iteration. This is the right notion for non-convex
benchmarks: SPGD periodically jumps to good points via perturbation but the
following gradient step can overshoot uphill in regions of large curvature
(Rastrigin gradients near integer points have magnitude ~2*pi*A = 62.8).
"Best ever found" is also what Experiment 1 will use as its primary metric,
so the smoke test is consistent with the formal experiment.

Expected outcome
----------------
- SGD  : descends into a basin near the start; oscillates with lr=0.01
         given Rastrigin's stiff cosine derivatives; best loss ~ 18-30.
- PGD  : single noise injections may or may not escape; partial improvement.
- RPGD : N_P candidates with random pick, often modest improvement.
- SPGD : argmin over N_P candidates with amp=2.5 covers the 2D search space
         well; reliably reaches a deep basin (best loss << SGD).

Acceptance:
    SPGD best loss < 8.0  AND  SGD best loss - SPGD best loss > 5.0
"""

from __future__ import annotations

from typing import Callable, List

import torch
from torch.optim import SGD

from spgd_study.benchmarks import rastrigin
from spgd_study.optimizers import PGD, RPGD, SPGD
from spgd_study.utils import set_seed


def _run(opt_factory: Callable, x0: List[float], n_steps: int, name: str) -> float:
    """Optimise 2D Rastrigin and return the BEST loss observed during training."""
    set_seed(0)
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    opt = opt_factory([x])

    best_loss = float("inf")
    best_x = x.detach().clone()

    for _ in range(n_steps):
        if isinstance(opt, (SPGD, PGD, RPGD)):

            def closure():
                return rastrigin(x)

            loss = opt.step(closure)
        else:
            opt.zero_grad()
            loss = rastrigin(x)
            loss.backward()
            opt.step()

        v = float(loss.item())
        if v < best_loss:
            best_loss = v
            best_x = x.detach().clone()

    print(
        f"  {name:<5}  best f = {best_loss:8.4f}   at x = "
        f"[{best_x[0].item():+.4f}, {best_x[1].item():+.4f}]"
    )
    return best_loss


def main() -> None:
    x0 = [3.5, 3.5]
    n_steps = 500

    print(f"2D Rastrigin smoke test  |  x0 = {x0}  |  {n_steps} steps  |  metric = best loss seen")
    print("-" * 70)

    sgd_best = _run(lambda p: SGD(p, lr=0.01), x0, n_steps, "SGD")
    pgd_best = _run(
        lambda p: PGD(p, lr=0.01, amp=2.5, iter_p=5),
        x0, n_steps, "PGD",
    )
    rpgd_best = _run(
        lambda p: RPGD(p, lr=0.01, amp=2.5, n_p=10, iter_p=5),
        x0, n_steps, "RPGD",
    )
    spgd_best = _run(
        lambda p: SPGD(p, lr=0.01, amp=2.5, n_p=10, iter_p=5),
        x0, n_steps, "SPGD",
    )

    print("-" * 70)
    margin = sgd_best - spgd_best
    print(f"  SPGD vs SGD margin (best loss): {margin:+.4f}")

    # Acceptance criteria — looser than the underlying signal, to allow
    # for minor seed/build noise across torch versions.
    assert spgd_best < 8.0, f"SPGD best loss should be < 8 (near-origin basin), got {spgd_best:.4f}"
    assert margin > 5.0, (
        f"SPGD should beat SGD by > 5 in best-loss terms (got margin {margin:.4f})"
    )

    print()
    print("SMOKE TEST PASSED")
    print("Optimizers behave as expected -- proceed to Step 2 (Experiment 1).")


if __name__ == "__main__":
    main()
