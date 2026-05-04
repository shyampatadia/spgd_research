"""Common utilities: seeding and small I/O helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch (CPU + CUDA) RNGs to a deterministic state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str | os.PathLike, payload: Dict[str, Any]) -> None:
    """Write a dict to JSON, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of the concatenated gradient. Returns 0.0 if no grads are populated."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total**0.5
