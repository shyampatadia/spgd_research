"""Diagnostic script to surface what's happening with OpenML on this machine.

Run from project root:
    uv run python tests/debug_openml.py

It walks the openml call sequence one step at a time, printing intermediate
results and full tracebacks on any failure. Use it to diagnose silent exits,
deprecation breakage, missing transitive deps, SSL/cert errors, or unexpected
return shapes.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


def main() -> int:
    print("=" * 70)
    print("OPENML DIAGNOSTIC")
    print("=" * 70)
    print(f"Python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform : {sys.platform}")

    # -- 1. import openml -------------------------------------------------
    print("\n[1/5] import openml")
    try:
        import openml  # type: ignore
        print(f"      ok -- version {getattr(openml, '__version__', '?')}")
    except Exception:
        print("      FAILED:")
        traceback.print_exc()
        return 1

    # -- 2. set cache -----------------------------------------------------
    print("\n[2/5] set cache directory")
    try:
        cache = Path(__file__).resolve().parents[1] / ".openml_cache"
        cache.mkdir(parents=True, exist_ok=True)
        openml.config.cache_directory = str(cache)
        print(f"      ok -- {cache}")
        print(f"      cache contents: {sorted(p.name for p in cache.iterdir())[:5]} ...")
    except Exception:
        print("      FAILED:")
        traceback.print_exc()
        return 1

    # -- 3. get_dataset metadata -----------------------------------------
    print("\n[3/5] openml.datasets.get_dataset(31)  # credit-g")
    ds = None
    try:
        try:
            ds = openml.datasets.get_dataset(31, download_data=True)
            print("      ok (with download_data=True)")
        except TypeError as e:
            print(f"      download_data kwarg rejected ({e}); retrying without it")
            ds = openml.datasets.get_dataset(31)
            print("      ok (without download_data)")
        print(f"      name             = {getattr(ds, 'name', '?')!r}")
        print(f"      target attribute = {getattr(ds, 'default_target_attribute', '?')!r}")
        print(f"      version          = {getattr(ds, 'version', '?')}")
    except Exception:
        print("      FAILED:")
        traceback.print_exc()
        return 1

    # -- 4. get_data ------------------------------------------------------
    print("\n[4/5] ds.get_data(target=...)")
    try:
        try:
            X, y, cat_ind, attr_names = ds.get_data(
                dataset_format="dataframe", target=ds.default_target_attribute
            )
            print("      ok (with dataset_format='dataframe')")
        except TypeError as e:
            print(f"      dataset_format kwarg rejected ({e}); retrying without it")
            X, y, cat_ind, attr_names = ds.get_data(target=ds.default_target_attribute)
            print("      ok (without dataset_format)")

        print(f"      X type           : {type(X).__name__}")
        print(f"      X shape          : {getattr(X, 'shape', '?')}")
        if y is None:
            print("      y                : None  <-- THIS IS THE BUG")
            print("      target column did not split off correctly")
            return 1
        print(f"      y type           : {type(y).__name__}")
        print(f"      y len            : {len(y)}")
        print(f"      y unique[:10]    : {sorted(set(y))[:10]}")
        print(f"      cat_indicator len: {len(cat_ind)}")
        n_cat = sum(1 for c in cat_ind if c)
        n_num = len(cat_ind) - n_cat
        print(f"      categorical cols : {n_cat}")
        print(f"      numeric cols     : {n_num}")
    except Exception:
        print("      FAILED:")
        traceback.print_exc()
        return 1

    # -- 5. full path through our wrapper --------------------------------
    print("\n[5/5] spgd_study.data.load_openml_dataset(31, seed=0)")
    try:
        from spgd_study.data import load_openml_dataset
        X_tr, y_tr, X_te, y_te, k = load_openml_dataset(31, seed=0, verbose=True)
        print(f"      ok")
        print(f"      X_train          : {X_tr.shape}  dtype {X_tr.dtype}")
        print(f"      y_train          : {y_tr.shape}  dtype {y_tr.dtype}  unique {sorted(set(y_tr.tolist()))}")
        print(f"      X_test           : {X_te.shape}")
        print(f"      n_classes        : {k}")
    except Exception:
        print("      FAILED:")
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("ALL CHECKS PASSED -- the OpenML pipeline is healthy on this machine.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
