"""Experiment 6 -- SPGD as an L_inf adversarial attack on CIFAR-10.

Compares three attack strategies against a freshly trained CIFAR-10
ResNet-18 inside the L_inf ball of radius eps:

    pgd-attack    Madry-style projected sign-gradient ascent.
    rpgd-attack   sample n_p uniform candidates per phase, accept one at random.
    spgd-attack   sample n_p uniform candidates per phase, accept the steepest
                  loss-INCREASING one (the SPGD selection rule, applied to the
                  attacker's maximisation problem).

Outputs:
    results/exp6_attack.csv           per (attack, seed) row of metrics
    results/exp6_attack_summary.csv   aggregate (mean +/- std) over seeds
    results/exp6_attack.json          raw per-seed dict for record

Run from project root:
    uv run python experiments/exp6_spgd_attack.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from spgd_study.models import cifar10_resnet18
from spgd_study.utils import set_seed

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class NormalizedModel(nn.Module):
    """Wraps a model that expects normalized input so that forward() takes raw
    [0,1] pixel tensors. Lets us define eps in pixel space (e.g. 8/255) without
    having to scale per-channel for the L_inf attack budget.
    """

    def __init__(self, model: nn.Module, mean, std) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def cifar10_pixelspace_loaders(data_dir: Path, batch_size: int):
    """Loaders that emit *unnormalized* [0,1] tensors. Augmentation on train."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True,
                                transform=train_tf)
    test_ds = datasets.CIFAR10(str(data_dir), train=False, download=True,
                               transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    return train_ds, test_ds, train_loader


def quick_train(model: nn.Module, train_loader: DataLoader,
                device: torch.device, *, epochs: int, lr: float,
                momentum: float, weight_decay: float) -> None:
    """Train the wrapped model with SGD+momentum + cosine LR schedule."""
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        print(f"  epoch {ep+1}/{epochs}  loss={running/n:.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()


@torch.no_grad()
def clean_accuracy(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> float:
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return correct / n


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
               eps: float, alpha: float, steps: int) -> torch.Tensor:
    """Madry-style L_inf PGD attack: projected sign-gradient ascent."""
    delta = torch.empty_like(x).uniform_(-eps, eps).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta.clamp_(-eps, eps)
        delta = ((x + delta).clamp(0, 1) - x).detach()
    return delta


def perturbed_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
                     eps: float, alpha: float, steps: int,
                     n_p: int, iter_p: int, amp: float,
                     mode: str) -> torch.Tensor:
    """RPGD- or SPGD-flavored attack.

    Between perturbation phases the iterate follows the same sign-gradient
    ascent step as pgd_attack. Every iter_p steps we sample n_p candidate
    deltas (each delta + Uniform(-amp,+amp), projected back into the L_inf
    ball and the [0,1] image range) and select one:
        spgd: per-image, accept the candidate with the LARGEST loss; only
              keep it if it strictly improves on the current iterate's loss.
              (steepest-loss-INCREASE selection rule)
        rpgd: per-image, accept a uniformly random one of the n_p candidates
              unconditionally.
    """
    assert mode in ("rpgd", "spgd")
    delta = torch.empty_like(x).uniform_(-eps, eps).detach()
    B = x.size(0)
    img_shape = x.shape[1:]

    for t in range(steps):
        # ---- gradient ascent step (same as PGD-attack) --------------------
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta.clamp_(-eps, eps)
        delta = ((x + delta).clamp(0, 1) - x).detach()

        # ---- perturbation phase -------------------------------------------
        if (t + 1) % iter_p == 0:
            with torch.no_grad():
                cands = []
                for _ in range(n_p):
                    c = delta + torch.empty_like(delta).uniform_(-amp, amp)
                    c = c.clamp_(-eps, eps)
                    c = (x + c).clamp(0, 1) - x
                    cands.append(c)
                cands = torch.stack(cands, dim=0)  # (n_p, B, C, H, W)
                cand_losses = torch.stack([
                    F.cross_entropy(model(x + cands[k]), y, reduction="none")
                    for k in range(n_p)
                ], dim=0)                          # (n_p, B)
                base_loss = F.cross_entropy(model(x + delta), y,
                                            reduction="none")

                if mode == "spgd":
                    chosen_idx = cand_losses.argmax(dim=0)        # (B,)
                else:
                    chosen_idx = torch.randint(0, n_p, (B,),
                                               device=delta.device)

                idx_view = chosen_idx.view(1, B, 1, 1, 1) \
                                     .expand(1, B, *img_shape)
                chosen = cands.gather(0, idx_view).squeeze(0)
                chosen_loss = cand_losses.gather(
                    0, chosen_idx.unsqueeze(0)).squeeze(0)

                if mode == "spgd":
                    accept = (chosen_loss > base_loss).view(B, 1, 1, 1)
                    delta = torch.where(accept, chosen, delta)
                else:
                    delta = chosen
                delta = delta.detach()

    return delta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_attack(model, loader, device, attack_name, kwargs):
    correct_clean, correct_adv, total = 0, 0, 0
    losses, conf_clean, conf_adv = [], [], []
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits_clean = model(x)
            pred_clean = logits_clean.argmax(dim=1)
            conf_clean.append(F.softmax(logits_clean, dim=1)
                              .gather(1, y.unsqueeze(1)).cpu().numpy())
        if attack_name == "pgd":
            delta = pgd_attack(model, x, y, **kwargs)
        else:
            delta = perturbed_attack(model, x, y, mode=attack_name, **kwargs)
        with torch.no_grad():
            logits_adv = model(x + delta)
            pred_adv = logits_adv.argmax(dim=1)
            losses.append(F.cross_entropy(logits_adv, y, reduction="none")
                          .cpu().numpy())
            conf_adv.append(F.softmax(logits_adv, dim=1)
                            .gather(1, y.unsqueeze(1)).cpu().numpy())
        correct_clean += (pred_clean == y).sum().item()
        correct_adv += (pred_adv == y).sum().item()
        total += x.size(0)
    return {
        "attack": attack_name,
        "n": total,
        "clean_acc": correct_clean / total,
        "adv_acc": correct_adv / total,
        "attack_success": 1.0 - correct_adv / total,
        "mean_adv_loss": float(np.concatenate(losses).mean()),
        "mean_conf_clean": float(np.concatenate(conf_clean).mean()),
        "mean_conf_adv": float(np.concatenate(conf_adv).mean()),
        "wall_sec": time.time() - t0,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (project_root / "experiments" / "configs" / "exp6.yaml").read_text()
    )
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / ".cifar10_cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== Exp 6: SPGD-attack vs PGD-attack vs RPGD-attack on CIFAR-10 ==")
    print(f"device: {device}")

    # ---- load data (pixel-space, no normalization) -----------------------
    set_seed(cfg["seed"])
    _, test_ds, train_loader = cifar10_pixelspace_loaders(
        data_dir, batch_size=cfg["batch_size"]
    )
    n_attack = cfg["n_attack_samples"]
    test_subset = Subset(test_ds, list(range(n_attack)))
    test_loader = DataLoader(test_subset, batch_size=cfg["attack_batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"train: {len(train_loader.dataset):,} images, "
          f"attack subset: {n_attack:,} images")

    # ---- build / train base classifier -----------------------------------
    print(f"\n[1/3] Training ResNet-18 base classifier "
          f"({cfg['train_epochs']} epochs)...")
    base_model = cifar10_resnet18(disable_bn_layers=(), num_classes=10)
    model = NormalizedModel(base_model, CIFAR10_MEAN, CIFAR10_STD).to(device)
    quick_train(
        model, train_loader, device,
        epochs=cfg["train_epochs"], lr=cfg["train_lr"],
        momentum=cfg["train_momentum"], weight_decay=cfg["train_weight_decay"],
    )
    clean = clean_accuracy(model, test_loader, device)
    print(f"clean test accuracy on attack subset: {clean:.4f}")

    # ---- run attacks ------------------------------------------------------
    print(f"\n[2/3] Running attacks ({cfg['n_seeds']} seeds each)...")
    eps, alpha = float(cfg["eps"]), float(cfg["alpha"])
    amp = cfg["amp_frac"] * eps
    common_pgd = dict(eps=eps, alpha=alpha, steps=cfg["attack_steps"])
    common_pert = dict(**common_pgd, n_p=cfg["n_p"],
                       iter_p=cfg["iter_p"], amp=amp)

    rows = []
    for attack_name, kwargs in [
        ("pgd",  common_pgd),
        ("rpgd", common_pert),
        ("spgd", common_pert),
    ]:
        for seed in range(cfg["n_seeds"]):
            set_seed(seed)
            res = run_attack(model, test_loader, device, attack_name, kwargs)
            res["seed"] = seed
            rows.append(res)
            print(f"  {attack_name:4s}  seed={seed}  "
                  f"adv_acc={res['adv_acc']:.4f}  "
                  f"adv_loss={res['mean_adv_loss']:.4f}  "
                  f"conf_adv={res['mean_conf_adv']:.4f}  "
                  f"({res['wall_sec']:.1f}s)")

    # ---- save & summarise ------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = results_dir / "exp6_attack.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    with (results_dir / "exp6_attack.json").open("w") as f:
        json.dump({
            "clean_accuracy": clean,
            "config": cfg,
            "rows": rows,
        }, f, indent=2)

    print("\n[3/3] Aggregate (mean +/- std over seeds):")
    agg = df.groupby("attack").agg(
        adv_acc_mean=("adv_acc", "mean"),
        adv_acc_std=("adv_acc", "std"),
        attack_success=("attack_success", "mean"),
        adv_loss_mean=("mean_adv_loss", "mean"),
        adv_loss_std=("mean_adv_loss", "std"),
        wall=("wall_sec", "mean"),
    )
    # display in attack order pgd, rpgd, spgd
    agg = agg.reindex(["pgd", "rpgd", "spgd"])
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    summary_path = results_dir / "exp6_attack_summary.csv"
    agg.to_csv(summary_path)
    print(f"Wrote {summary_path}")

    # ---- paired SPGD-vs-PGD (per-seed) -----------------------------------
    print("\nPaired per-seed deltas (SPGD-attack - PGD-attack):")
    pivot_acc = df.pivot(index="seed", columns="attack", values="adv_acc")
    pivot_loss = df.pivot(index="seed", columns="attack", values="mean_adv_loss")
    delta_acc = pivot_acc["spgd"] - pivot_acc["pgd"]
    delta_loss = pivot_loss["spgd"] - pivot_loss["pgd"]
    for s in pivot_acc.index:
        print(f"  seed {s}: d_robust_acc={delta_acc[s]:+.4f}  "
              f"d_loss={delta_loss[s]:+.4f}")
    print(f"  mean    : d_robust_acc={delta_acc.mean():+.4f}  "
          f"d_loss={delta_loss.mean():+.4f}")
    print("(SPGD-attack stronger when d_robust_acc < 0 and d_loss > 0.)")


if __name__ == "__main__":
    main()
