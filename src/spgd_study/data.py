"""Datasets used in the empirical study.

- Two Moons (Experiments 2 and 5): synthetic 2D binary classification.
- OpenML-CC18 tabular (Experiment 3): heterogeneous tabular classification.
- CIFAR-10 (Experiment 4): loaded via torchvision in the cluster runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_moons
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def load_two_moons(
    n_samples: int = 1000,
    noise: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two Moons dataset.

    Returns (X_train, y_train, X_test, y_test) as float32 tensors. Labels are
    {0, 1} stored as float so binary_cross_entropy_with_logits accepts them
    directly without further casting.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32),
    )


def _set_openml_cache() -> None:
    """Use a project-local OpenML cache so we don't pollute ~/.openml."""
    import openml
    cache = Path(__file__).resolve().parents[2] / ".openml_cache"
    cache.mkdir(parents=True, exist_ok=True)
    openml.config.cache_directory = str(cache)


def _openml_get_dataset(openml_id: int):
    """Compat wrapper for openml's evolving API."""
    import openml
    try:
        return openml.datasets.get_dataset(int(openml_id), download_data=True)
    except TypeError:
        # openml >= 0.14 made download_data a no-op / removed it.
        return openml.datasets.get_dataset(int(openml_id))


def _openml_get_data(ds):
    """Compat wrapper for OpenMLDataset.get_data() across versions."""
    target = ds.default_target_attribute
    try:
        return ds.get_data(dataset_format="dataframe", target=target)
    except TypeError:
        # Newer openml: dataset_format kwarg dropped, dataframe is default.
        return ds.get_data(target=target)


def load_openml_dataset(
    openml_id: int,
    test_frac: float = 0.2,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load and preprocess an OpenML dataset for MLP classification.

    Preprocessing pipeline (fit on train only, applied to test):
        - missing values: mean (numeric) / mode (categorical) imputation
        - categorical:    one-hot encoded, handle_unknown="ignore" on test
        - numeric:        zero-mean / unit-variance standardisation
        - labels:         integer-encoded for cross_entropy

    Returns
    -------
    X_train : torch.float32  (N_tr, p_processed)
    y_train : torch.long     (N_tr,)
    X_test  : torch.float32  (N_te, p_processed)
    y_test  : torch.long     (N_te,)
    n_classes : int
    """
    _set_openml_cache()

    if verbose:
        print(f"[openml] fetching dataset id={openml_id} ...", flush=True)
    ds = _openml_get_dataset(openml_id)
    if verbose:
        name = getattr(ds, "name", "?")
        print(f"[openml]   metadata OK: name={name!r}, "
              f"target={ds.default_target_attribute!r}", flush=True)

    if verbose:
        print(f"[openml] downloading + parsing data ...", flush=True)
    X, y, categorical_indicator, _ = _openml_get_data(ds)
    if verbose:
        print(f"[openml]   raw shape: X={getattr(X, 'shape', '?')} "
              f"y_len={len(y) if y is not None else 0}", flush=True)

    if y is None:
        raise RuntimeError(f"OpenML dataset {openml_id}: target column came back None")

    le = LabelEncoder()
    y_int = le.fit_transform(y)
    n_classes = int(len(le.classes_))

    cat_cols = [c for c, is_cat in zip(X.columns, categorical_indicator) if is_cat]
    num_cols = [c for c, is_cat in zip(X.columns, categorical_indicator) if not is_cat]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("sc",  StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    if not transformers:
        raise RuntimeError(f"OpenML dataset {openml_id}: no usable columns")

    pre = ColumnTransformer(transformers, remainder="drop")

    X_tr_df, X_te_df, y_tr_int, y_te_int = train_test_split(
        X, y_int, test_size=test_frac, random_state=seed, stratify=y_int,
    )
    X_tr = pre.fit_transform(X_tr_df)
    X_te = pre.transform(X_te_df)

    return (
        torch.tensor(np.asarray(X_tr), dtype=torch.float32),
        torch.tensor(y_tr_int, dtype=torch.long),
        torch.tensor(np.asarray(X_te), dtype=torch.float32),
        torch.tensor(y_te_int, dtype=torch.long),
        n_classes,
    )


def load_cifar10(
    data_dir: "str | Path | None" = None,
    batch_size: int = 128,
    num_workers: int = 2,
    test_batch_size: int = 256,
):
    """CIFAR-10 train/test DataLoaders with standard CIFAR augmentation on train.

    Returns ``(train_loader, test_loader)``. Dataset is downloaded to
    ``data_dir`` on first run (about 170 MB) and cached. Default ``data_dir``
    is ``<project>/.cifar10_cache``.

    Augmentation matches the de-facto-standard CIFAR-10 baseline: random crop
    with 4-pixel padding + random horizontal flip on train; standardisation
    only on test.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / ".cifar10_cache"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 train-set statistics
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
