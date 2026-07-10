"""Shared fixtures: synthetic imbalanced dataset mirroring the real shape traits."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def synth() -> dict:
    """Small synthetic dataset: 600 rows, rare positives, known leak column.

    Columns:
      F1 informative, F2 informative-weak, F3 noise, F4 constant,
      F5 duplicate of F1, F6 quasi-constant, F7 leak (== target),
      F8 categorical-ish ints, F3912-like leak recoding, target.
    """
    rng = np.random.default_rng(7)
    n = 600
    y = (rng.random(n) < 0.05).astype(np.int32)  # ~5% positives
    f1 = rng.normal(0, 1, n) + 2.5 * y
    f2 = rng.normal(0, 1, n) + 0.8 * y
    f3 = rng.normal(0, 1, n)
    f4 = np.zeros(n)
    f5 = f1.copy()
    f6 = np.where(rng.random(n) < 0.995, 1.0, 2.0)
    f7 = y.astype(float)                      # exact target copy
    f8 = rng.integers(0, 4, n).astype(float)
    leak_recode = np.where(y == 1, 10.0, -10.0)  # affine recoding of target
    miss_mask = rng.random(n) < 0.2
    f2m = f2.copy()
    f2m[miss_mask] = np.nan
    df = pl.DataFrame({
        "F1": f1, "F2": f2m, "F3": f3, "F4": f4, "F5": f5,
        "F6": f6, "F7": f7, "F8": f8, "F3912": leak_recode,
        "F3924": y,
    })
    return {"df": df, "y": y, "n": n}
