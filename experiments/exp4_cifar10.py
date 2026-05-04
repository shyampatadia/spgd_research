"""Experiment 4 — CIFAR-10 / ResNet-18 anchor experiment.

Each invocation runs ONE (optimizer, seed) configuration. Designed to be
launched as a Slurm array job on Turing (one job per array index, mapping
to one (opt, seed) pair) so the 15 runs can execute in parallel.

It can also be run locally for sanity validation -- pass --epochs 1 to
exercise the full pipeline (data load -> training loop -> save) in a
few minutes on RTX 4050.

CLI:
    --opt {sgd,adam,pgd,rpgd,spgd}    REQUIRED
    --seed N                          REQUIRED
    --epochs N                        override config; default uses config
    --device {cpu,cuda}               default: cuda if available, else cpu
    --config PATH                     default: experiments/configs/exp4.yaml
    --out-dir PATH                    default: results/
    --data-dir PATH                   default: <project>/.cifar10_cache

Examples:
    # Local 1-epoch sanity check on RTX 4050 (~3-5 min):
    uv run python experiments/exp4_cifar10.py --opt sgd  --seed 0 --epochs 1
    uv run python experiments/exp4_cifar10.py --opt spgd --seed 0 --epochs 1

    # Full 50-epoch cluster run (called from the SBATCH array script):
    python experiments/exp4_cifar10.py --opt $OPT --seed $SEED --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from spgd_study.data import load_cifar10
from spgd_study.models import cifar10_resnet18
from spgd_study.nn_runner import train_minibatch
from spgd_study.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opt", required=True,
                   choices=["sgd", "adam", "pgd", "rpgd", "spgd"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None, help="override config n_epochs")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detect if omitted)")
    p.add_argument("--config", default=None, help="path to exp4.yaml")
    p.add_argument("--out-dir", default=None, help="results output directory")
    p.add_argument("--data-dir", default=None, help="CIFAR-10 download/cache directory")
    return p.parse_args()


def build_hyper(opt_name: str, opt_cfg: dict) -> dict:
    h = {"lr": opt_cfg["lr"]}
    if "momentum" in opt_cfg:
        h["momentum"] = opt_cfg["momentum"]
    if opt_name in ("pgd", "rpgd", "spgd"):
        h["amp"] = opt_cfg["amp"]
        h["iter_p"] = opt_cfg["iter_p"]
    if opt_name in ("rpgd", "spgd"):
        h["n_p"] = opt_cfg["n_p"]
    return h


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = (Path(args.config) if args.config
                else project_root / "experiments" / "configs" / "exp4.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    n_epochs = args.epochs if args.epochs is not None else cfg["n_epochs"]
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"== Exp 4: opt={args.opt}  seed={args.seed}  epochs={n_epochs}  device={device} ==")

    set_seed(args.seed)

    train_loader, test_loader = load_cifar10(
        data_dir=args.data_dir,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        test_batch_size=cfg["test_batch_size"],
    )
    print(f"loaded CIFAR-10: {len(train_loader.dataset):,} train, "
          f"{len(test_loader.dataset):,} test, batch_size={cfg['batch_size']}")

    model = cifar10_resnet18(
        disable_bn_layers=cfg["disable_bn_layers"],
        num_classes=cfg["num_classes"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: ResNet-18 ({n_params:,} params; "
          f"BN disabled in {cfg['disable_bn_layers']})")

    hyper = build_hyper(args.opt, cfg["optimizers"][args.opt])
    print(f"hyper: {hyper}")
    print()

    t0 = time.time()
    result = train_minibatch(
        model=model,
        opt_name=args.opt,
        hyper=hyper,
        loss_fn=F.cross_entropy,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
        stagnation_eps=cfg["stagnation_eps"],
        device=device,
        log_every=cfg.get("log_every", 200),
    )
    duration = time.time() - t0

    print()
    print(f"== run complete: {duration:.1f}s ==")
    print(f"   final train loss : {result.final_loss:.4f}")
    print(f"   test accuracy    : {result.test_accuracy:.4f}")
    print(f"   stagnation episodes : {result.stagnation['n_episodes']}  "
          f"longest {result.stagnation['longest_episode']}")

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"exp4_{args.opt}_s{args.seed}_e{n_epochs}.json"

    with out_file.open("w") as f:
        json.dump(
            dict(
                opt=args.opt,
                seed=args.seed,
                n_epochs=n_epochs,
                duration_sec=duration,
                final_loss=result.final_loss,
                test_accuracy=result.test_accuracy,
                stagnation=result.stagnation,
                losses=result.losses,
                hyper=hyper,
                n_params=n_params,
                device=device,
                disable_bn_layers=cfg["disable_bn_layers"],
            ),
            f,
        )
    print(f"   wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
