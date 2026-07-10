"""Per-feature profiling of the converted dataset.

Produces one row per column with dtype, cardinality, missingness, zero rate,
constant/quasi-constant status, distribution stats, top values, integer-
valuedness and exact-duplicate group membership. Written to
``artifacts/reports/data_profile.parquet`` (+ ``.csv``).

Profiling is a data-quality AUDIT over the full file. Modelling decisions
that *select* features happen strictly inside CV folds elsewhere; this audit
is used to *exclude* dangerous or dead columns and to document the dataset.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("data.profile")

TOP_VALUES_MAX_UNIQUE = 25


def _column_hash(s: pl.Series) -> str:
    """Stable content hash of a column (missing-aware, -0.0 canonicalised)."""
    if s.dtype.is_numeric():
        arr = s.cast(pl.Float64).to_numpy()
        arr = np.where(np.isnan(arr), np.float64(np.pi) * 1e17, arr)  # missing sentinel
        arr = arr + 0.0  # -0.0 -> 0.0
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    joined = "\x1f".join("\x00" if v is None else str(v) for v in s.to_list())
    return hashlib.sha256(joined.encode("utf-8", "surrogatepass")).hexdigest()


def profile_dataset(df: pl.DataFrame) -> pl.DataFrame:
    n_rows = df.height
    qc_share = float(settings.load_config("data")["audit"]["quasi_constant_share"])
    records: list[dict[str, Any]] = []
    hash_to_first: dict[str, str] = {}

    for name in df.columns:
        s = df[name]
        dtype = str(s.dtype)
        n_missing = int(s.null_count())
        non_null = s.drop_nulls()
        n_unique = int(non_null.n_unique())
        rec: dict[str, Any] = {
            "feature": name,
            "dtype": dtype,
            "n_missing": n_missing,
            "missing_rate": round(n_missing / n_rows, 6),
            "n_unique": n_unique,
            "is_constant": n_unique <= 1,
        }

        # quasi-constant: dominant non-null value share
        if n_unique >= 1 and len(non_null) > 0:
            top_share = float(non_null.value_counts().sort("count", descending=True)["count"][0]) / len(non_null)
        else:
            top_share = 1.0
        rec["top_value_share"] = round(top_share, 6)
        rec["is_quasi_constant"] = bool(rec["is_constant"] or top_share >= qc_share or n_missing / n_rows >= qc_share)

        if s.dtype.is_numeric() and len(non_null) > 0:
            arr = non_null.cast(pl.Float64).to_numpy()
            q = np.quantile(arr, [0.01, 0.25, 0.5, 0.75, 0.99])
            rec.update(
                zero_rate=round(float((arr == 0).mean()), 6),
                min=float(arr.min()), max=float(arr.max()),
                mean=float(arr.mean()), std=float(arr.std()),
                q01=float(q[0]), q25=float(q[1]), q50=float(q[2]),
                q75=float(q[3]), q99=float(q[4]),
                is_integer_valued=bool(np.all(np.isfinite(arr)) and np.all(arr == np.round(arr))),
            )
        else:
            rec.update(zero_rate=None, min=None, max=None, mean=None, std=None,
                       q01=None, q25=None, q50=None, q75=None, q99=None,
                       is_integer_valued=None)

        if n_unique <= TOP_VALUES_MAX_UNIQUE and n_unique > 0:
            vc = non_null.value_counts().sort("count", descending=True).head(5)
            rec["top_values"] = "; ".join(f"{r[0]!r}:{r[1]}" for r in vc.iter_rows())
        else:
            rec["top_values"] = None

        chash = _column_hash(s)
        rec["content_hash"] = chash
        rec["duplicate_of"] = hash_to_first.get(chash)
        if chash not in hash_to_first:
            hash_to_first[chash] = name
        records.append(rec)

    prof = pl.DataFrame(records)
    log.info(
        "profiled %d columns: %d constant, %d quasi-constant, %d exact-duplicate columns",
        prof.height,
        int(prof["is_constant"].sum()),
        int(prof["is_quasi_constant"].sum()),
        int(prof["duplicate_of"].is_not_null().sum()),
    )
    return prof


def summarize_profile(prof: pl.DataFrame, df: pl.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in df.columns if c != settings.TARGET_COLUMN]
    numeric = prof.filter(pl.col("feature") != settings.TARGET_COLUMN)
    total_cells = df.height * len(feature_cols)
    n_missing_cells = int(numeric["n_missing"].sum())
    return {
        "n_rows": df.height,
        "n_cols_total": df.width,
        "n_feature_cols": len(feature_cols),
        "n_missing_cells": n_missing_cells,
        "missing_cell_rate": round(n_missing_cells / total_cells, 6),
        "n_cols_with_missing": int((numeric["n_missing"] > 0).sum()),
        "n_constant": int(numeric["is_constant"].sum()),
        "n_quasi_constant": int(numeric["is_quasi_constant"].sum()),
        "n_exact_duplicate_cols": int(numeric["duplicate_of"].is_not_null().sum()),
        "dtype_counts": {
            k: int(v)
            for k, v in numeric["dtype"].value_counts().sort("count", descending=True).iter_rows()
        },
    }
