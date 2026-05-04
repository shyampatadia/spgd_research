"""Generic full-batch training loop for the MLP experiments (Exp 2 & 3).

Why a dedicated runner: the closure protocol used by SPGD/PGD/RPGD differs
from the standard zero_grad/backward/step idiom for SGD/Adam. This runner
hides the difference and ends each step with x.grad populated for every
parameter so we can read gradient norms uniformly.

What gets recorded per step:
  - loss (scalar)
  - ||grad||_2 over all parameters
  - flat parameter snapshot (only when record_weights=True; subsampled
    every weights_snapshot_every steps to keep memory bounded)

The flat parameter history feeds the PCA-basis loss-landscape visualization
in plot_exp2.py (the showpiece figure for Experiment 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from .diagnostics import StagnationTracker
from .optimizers import PGD, RPGD, SPGD
from .utils import grad_norm


@dataclass
class NNRunResult:
    losses: List[float]
    grad_norm_history: List[float]
    stagnation: Dict
    test_accuracy: float
    final_loss: float
    weights_history: Optional[np.ndarray] = None  # (n_snapshots, n_params) or None


def _make_optimizer(name: str, params, hyper: Dict):
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=hyper["lr"], momentum=hyper.get("momentum", 0.0))
    if name == "adam":
        return Adam(params, lr=hyper["lr"])
    if name == "pgd":
        return PGD(params, lr=hyper["lr"], amp=hyper["amp"], iter_p=hyper["iter_p"])
    if name == "rpgd":
        return RPGD(params, lr=hyper["lr"], amp=hyper["amp"], n_p=hyper["n_p"], iter_p=hyper["iter_p"])
    if name == "spgd":
        return SPGD(params, lr=hyper["lr"], amp=hyper["amp"], n_p=hyper["n_p"], iter_p=hyper["iter_p"])
    raise ValueError(f"unknown optimizer {name!r}")


def flat_params(model: nn.Module) -> np.ndarray:
    """Concatenate all model parameters into a single flat numpy vector."""
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()


def load_flat(model: nn.Module, w: np.ndarray) -> None:
    """Inverse of flat_params: load a flat vector back into model parameters."""
    w_t = torch.from_numpy(w.astype(np.float32))
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(w_t[offset:offset + n].view_as(p))
        offset += n


def binary_accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    """Test accuracy for binary classification with a single-logit output."""
    with torch.no_grad():
        logits = model(X)
        preds = (logits > 0).float()
        return float((preds == y).float().mean().item())


def multiclass_accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    """Test accuracy for multiclass classification (output: (N, C) logits)."""
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=-1)
        return float((preds == y).float().mean().item())


def loader_accuracy(model, loader, device: str = "cpu") -> float:
    """Test accuracy by streaming a DataLoader (used for CIFAR-10 / Exp 4)."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds = model(X).argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    if was_training:
        model.train()
    return correct / max(1, total)


def train_minibatch(
    model: nn.Module,
    opt_name: str,
    hyper: Dict,
    loss_fn: Callable,
    train_loader,
    test_loader,
    n_epochs: int,
    stagnation_eps: float,
    device: str = "cpu",
    log_every: int = 100,
) -> NNRunResult:
    """Mini-batch training loop for image / large-tabular experiments (Exp 4).

    For SPGD/PGD/RPGD, the perturbation phase reuses the *current* minibatch
    in its closure. Candidates are scored on the same data the gradient step
    will use, so the selection criterion and the gradient signal point at
    the same loss surface for that step.

    NOTE: SPGD's iter_p counts MINIBATCH steps (not epochs). With CIFAR-10's
    ~391 batches/epoch, iter_p=200 fires roughly twice per epoch.
    """
    model = model.to(device)
    opt = _make_optimizer(opt_name, list(model.parameters()), hyper)
    is_perturbing = isinstance(opt, (SPGD, PGD, RPGD))
    tracker = StagnationTracker(eps=stagnation_eps)
    params = list(model.parameters())

    losses: List[float] = []
    step = 0

    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_perturbing:
                def closure():
                    return loss_fn(model(X), y)
                loss = opt.step(closure)
            else:
                opt.zero_grad()
                preds = model(X)
                loss = loss_fn(preds, y)
                loss.backward()
                opt.step()

            loss_val = float(loss.item())
            losses.append(loss_val)
            tracker.update(grad_norm(params))

            if step % log_every == 0:
                print(f"  epoch {epoch + 1}/{n_epochs}  step {step:>6d}  "
                      f"loss {loss_val:.4f}", flush=True)
            step += 1

    test_acc = float("nan")
    if test_loader is not None:
        test_acc = loader_accuracy(model, test_loader, device=device)

    return NNRunResult(
        losses=losses,
        grad_norm_history=tracker.grad_norm_history,
        stagnation=tracker.summary(),
        test_accuracy=test_acc,
        final_loss=losses[-1] if losses else float("nan"),
        weights_history=None,
    )


def train_full_batch(
    model: nn.Module,
    opt_name: str,
    hyper: Dict,
    loss_fn: Callable,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_steps: int,
    stagnation_eps: float,
    record_weights: bool = False,
    weights_snapshot_every: int = 10,
    accuracy_fn: Optional[Callable] = None,
) -> NNRunResult:
    """Train a model on (X_train, y_train) full-batch for n_steps."""
    opt = _make_optimizer(opt_name, list(model.parameters()), hyper)
    is_perturbing = isinstance(opt, (SPGD, PGD, RPGD))
    tracker = StagnationTracker(eps=stagnation_eps)
    params = list(model.parameters())

    losses: List[float] = []
    weights_snapshots: List[np.ndarray] = []

    if record_weights:
        weights_snapshots.append(flat_params(model))  # snapshot 0 = init

    for step in range(n_steps):
        if is_perturbing:
            def closure():
                return loss_fn(model(X_train), y_train)
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            preds = model(X_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            opt.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        tracker.update(grad_norm(params))

        if record_weights and (step + 1) % weights_snapshot_every == 0:
            weights_snapshots.append(flat_params(model))

    if record_weights and (n_steps % weights_snapshot_every) != 0:
        weights_snapshots.append(flat_params(model))  # final state

    test_acc = float("nan")
    if accuracy_fn is not None:
        test_acc = accuracy_fn(model, X_test, y_test)

    return NNRunResult(
        losses=losses,
        grad_norm_history=tracker.grad_norm_history,
        stagnation=tracker.summary(),
        test_accuracy=test_acc,
        final_loss=losses[-1],
        weights_history=(np.array(weights_snapshots) if record_weights else None),
    )
