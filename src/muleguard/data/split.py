"""Immutable split construction.

- One stratified locked test split (fixed seed), saved to parquet with the
  exact row indices. Never touched again until final evaluation.
- Repeated stratified 5-fold assignments on the development set, saved to
  parquet (one column per repeat).
- Group hygiene: exact duplicate FEATURE-rows (identical feature vectors)
  are detected first; duplicate groups are kept entirely on one side of the
  dev/test boundary and inside a single CV fold, preventing twin leakage.

Everything is reproducible from GLOBAL_SEED; rebuilding must yield
byte-identical assignments (tested in tests/model).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.utils import save_json

log = get_logger("data.split")


def feature_row_groups(df: pl.DataFrame, quarantined: list[str]) -> np.ndarray:
    """Group id per row; rows with identical non-quarantined feature vectors share an id."""
    feat_cols = [c for c in df.columns if c not in set(quarantined) | {settings.TARGET_COLUMN}]
    hashes: dict[bytes, int] = {}
    group_ids = np.empty(df.height, dtype=np.int64)
    # hash rows chunk-wise to bound memory
    chunk = 2048
    for start in range(0, df.height, chunk):
        sub = df.slice(start, chunk).select(feat_cols)
        num_cols = [c for c in feat_cols if sub.schema[c].is_numeric()]
        other_cols = [c for c in feat_cols if not sub.schema[c].is_numeric()]
        arr = sub.select([pl.col(c).cast(pl.Float64) for c in num_cols]).to_numpy()
        arr = np.where(np.isnan(arr), np.float64(np.pi) * 1e17, arr) + 0.0
        other = (
            sub.select(other_cols).with_columns(pl.all().cast(pl.Utf8).fill_null("\x00"))
            .to_numpy() if other_cols else None
        )
        for i in range(arr.shape[0]):
            h = hashlib.blake2b(np.ascontiguousarray(arr[i]).tobytes(), digest_size=16)
            if other is not None:
                h.update("\x1f".join(other[i]).encode("utf-8", "surrogatepass"))
            key = h.digest()
            group_ids[start + i] = hashes.setdefault(key, len(hashes))
    n_groups = len(hashes)
    log.info("row-duplicate scan: %d rows -> %d unique feature vectors", df.height, n_groups)
    return group_ids


def _grouped_stratified_holdout(
    y: np.ndarray, groups: np.ndarray, test_fraction: float, seed: int
) -> np.ndarray:
    """Group-aware stratified holdout. Returns boolean mask (True = test).

    Groups (not rows) are assigned: positive groups (any positive member) and
    negative groups are shuffled separately and filled until the target row
    budget is met. Keeps prevalence close to natural and never splits a group.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    uniq, first_idx = np.unique(groups, return_index=True)
    group_pos = np.zeros(len(uniq), dtype=bool)
    group_size = np.zeros(len(uniq), dtype=np.int64)
    remap = {g: i for i, g in enumerate(uniq)}
    for g, yy in zip(groups, y):
        gi = remap[g]
        group_size[gi] += 1
        if yy == 1:
            group_pos[gi] = True

    test_mask = np.zeros(n, dtype=bool)
    for is_pos in (True, False):
        gsel = np.where(group_pos == is_pos)[0]
        order = rng.permutation(gsel)
        target_rows = test_fraction * group_size[gsel].sum()
        acc = 0
        chosen: list[int] = []
        for gi in order:
            if acc >= target_rows:
                break
            chosen.append(gi)
            acc += group_size[gi]
        chosen_set = set(uniq[gi] for gi in chosen)
        test_mask |= np.isin(groups, list(chosen_set))
    return test_mask


def make_locked_test_split(df: pl.DataFrame, groups: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = settings.load_config("data")["splits"]
    y = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    test_mask = _grouped_stratified_holdout(
        y, groups, float(cfg["locked_test_fraction"]), settings.GLOBAL_SEED
    )
    n_pos_test = int(y[test_mask].sum())
    if n_pos_test < int(cfg["min_test_positives"]):
        raise RuntimeError(
            f"locked test would hold only {n_pos_test} positives "
            f"(< {cfg['min_test_positives']}) - aborting split creation"
        )
    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": settings.GLOBAL_SEED,
        "test_fraction_config": cfg["locked_test_fraction"],
        "n_rows_total": int(len(y)),
        "n_test_rows": int(test_mask.sum()),
        "n_test_positives": n_pos_test,
        "n_dev_rows": int((~test_mask).sum()),
        "n_dev_positives": int(y[~test_mask].sum()),
        "test_prevalence": round(float(y[test_mask].mean()), 6),
        "dev_prevalence": round(float(y[~test_mask].mean()), 6),
        "group_aware": True,
    }
    return test_mask, meta


def make_cv_folds(
    df: pl.DataFrame, test_mask: np.ndarray, groups: np.ndarray
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Repeated stratified group-aware K-fold on the development rows.

    Stratification is at group level (a group is positive if any member is);
    all rows of a group land in the same fold of a given repeat.
    """
    cfg = settings.load_config("data")["splits"]
    n_splits, n_repeats = int(cfg["cv_n_splits"]), int(cfg["cv_n_repeats"])
    y = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    dev_idx = np.where(~test_mask)[0]
    dev_groups = groups[dev_idx]

    uniq = np.unique(dev_groups)
    group_pos = np.zeros(len(uniq), dtype=np.int8)
    remap = {g: i for i, g in enumerate(uniq)}
    for ri, g in zip(dev_idx, dev_groups):
        if y[ri] == 1:
            group_pos[remap[g]] = 1

    fold_cols: dict[str, np.ndarray] = {}
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.GLOBAL_SEED + rep)
        group_fold = np.full(len(uniq), -1, dtype=np.int8)
        for k, (_, va) in enumerate(skf.split(np.zeros(len(uniq)), group_pos)):
            group_fold[va] = k
        row_fold = np.array([group_fold[remap[g]] for g in dev_groups], dtype=np.int8)
        fold_cols[f"repeat_{rep}"] = row_fold

    folds_df = pl.DataFrame({"row_index": dev_idx.astype(np.int64), **fold_cols})
    meta = {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seeds": [settings.GLOBAL_SEED + r for r in range(n_repeats)],
        "positives_per_fold_repeat0": [
            int(y[dev_idx[fold_cols["repeat_0"] == k]].sum()) for k in range(n_splits)
        ],
        "group_aware": True,
    }
    return folds_df, meta


def build_all_splits(df: pl.DataFrame, quarantined: list[str]) -> dict[str, Any]:
    cfg = settings.load_config("data")["splits"]
    groups = feature_row_groups(df, quarantined)
    dup_sizes = np.bincount(groups)
    n_dup_rows = int((dup_sizes[groups] > 1).sum())

    test_mask, test_meta = make_locked_test_split(df, groups)
    folds_df, cv_meta = make_cv_folds(df, test_mask, groups)

    y = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    locked = pl.DataFrame({
        "row_index": np.arange(df.height, dtype=np.int64),
        "is_locked_test": test_mask,
        "target": y,
        "group_id": groups,
    })
    locked_path = settings.REPO_ROOT / cfg["files"]["locked_test"]
    folds_path = settings.REPO_ROOT / cfg["files"]["cv_folds"]
    dup_path = settings.REPO_ROOT / cfg["files"]["duplicate_groups"]
    locked_path.parent.mkdir(parents=True, exist_ok=True)
    locked.write_parquet(locked_path)
    folds_df.write_parquet(folds_path)
    pl.DataFrame({
        "group_id": np.arange(len(dup_sizes), dtype=np.int64),
        "n_rows": dup_sizes.astype(np.int64),
    }).filter(pl.col("n_rows") > 1).write_parquet(dup_path)

    meta = {
        "locked_test": test_meta,
        "cv": cv_meta,
        "duplicate_feature_rows": {
            "n_rows_in_multi_groups": n_dup_rows,
            "n_groups_gt1": int((dup_sizes > 1).sum()),
            "largest_group": int(dup_sizes.max()),
        },
        "files": {k: str(settings.REPO_ROOT / v) for k, v in cfg["files"].items()},
    }
    save_json(meta, settings.SPLITS_DIR / "split_metadata.json")
    return meta


def load_locked_test_mask() -> np.ndarray:
    cfg = settings.load_config("data")["splits"]
    locked = pl.read_parquet(settings.REPO_ROOT / cfg["files"]["locked_test"])
    return locked.sort("row_index")["is_locked_test"].to_numpy()


def load_cv_folds() -> pl.DataFrame:
    cfg = settings.load_config("data")["splits"]
    return pl.read_parquet(settings.REPO_ROOT / cfg["files"]["cv_folds"])
