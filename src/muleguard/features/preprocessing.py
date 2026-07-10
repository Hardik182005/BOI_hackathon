"""Fold-safe preprocessing.

`FoldPreprocessor` learns everything from the TRAINING fold only:
- allowed feature list (quarantine applied before anything else)
- categorical encoding (ordinal codes learned on train; unseen -> -1)
- constant-column removal (train statistics)
- exact-duplicate-column removal (train content hashes)
- optional median imputation + standardisation (for linear models)

Tree models receive NaN untouched (native missing handling).
The fitted object is picklable and records exactly what it removed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.utils import load_json


def load_quarantine_list() -> list[str]:
    """Quarantined feature names from the audit artifact (fail loud if absent)."""
    qpath = settings.FEATURES_DIR / "quarantined_features.json"
    if not qpath.exists():
        raise FileNotFoundError(
            "quarantined_features.json missing - run muleguard.cli.audit_data first"
        )
    return [e["feature"] for e in load_json(qpath)["quarantine"]]


def candidate_feature_columns(df: pl.DataFrame, quarantined: list[str]) -> list[str]:
    banned = set(quarantined) | {settings.TARGET_COLUMN}
    return [c for c in df.columns if c not in banned]


def encode_dataframe(df: pl.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, list[str], dict[str, list]]:
    """Encode to float32 matrix. Non-numeric columns get stable ordinal codes
    (sorted unique values); mapping returned so folds can re-apply it."""
    arrays, cat_maps = [], {}
    for c in feat_cols:
        s = df[c]
        if s.dtype.is_numeric():
            arrays.append(s.cast(pl.Float32).to_numpy())
        elif s.dtype.is_temporal():
            # deterministic ordinal: days since epoch (fixed reference)
            arrays.append((s.cast(pl.Date).cast(pl.Int32)).cast(pl.Float32).to_numpy())
        else:
            vals = sorted(v for v in s.unique().to_list() if v is not None)
            mapping = {v: float(i) for i, v in enumerate(vals)}
            cat_maps[c] = vals
            arrays.append(
                np.array([np.nan if v is None else mapping.get(v, -1.0) for v in s.to_list()],
                         dtype=np.float32)
            )
    return np.column_stack(arrays).astype(np.float32), feat_cols, cat_maps


@dataclass
class FoldPreprocessor:
    """Learned on a training fold; applied to any matrix with same columns."""

    mode: Literal["tree", "linear"] = "tree"
    keep_mask: np.ndarray | None = None
    kept_features: list[str] = field(default_factory=list)
    removed_constant: list[str] = field(default_factory=list)
    removed_duplicate: dict[str, str] = field(default_factory=dict)
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None

    def fit(self, X: np.ndarray, feature_names: list[str]) -> "FoldPreprocessor":
        n_cols = X.shape[1]
        keep = np.ones(n_cols, dtype=bool)

        # constants on TRAIN (all-missing or single non-missing value)
        for j in range(n_cols):
            col = X[:, j]
            finite = col[~np.isnan(col)]
            if len(finite) == 0 or np.all(finite == finite[0]):
                keep[j] = False
                self.removed_constant.append(feature_names[j])

        # exact duplicates on TRAIN (content hash, missing-aware)
        seen: dict[bytes, int] = {}
        for j in range(n_cols):
            if not keep[j]:
                continue
            col = X[:, j].astype(np.float64)
            col = np.where(np.isnan(col), np.float64(np.pi) * 1e17, col) + 0.0
            key = hashlib.blake2b(np.ascontiguousarray(col).tobytes(), digest_size=16).digest()
            if key in seen:
                keep[j] = False
                self.removed_duplicate[feature_names[j]] = feature_names[seen[key]]
            else:
                seen[key] = j

        self.keep_mask = keep
        self.kept_features = [feature_names[j] for j in range(n_cols) if keep[j]]

        if self.mode == "linear":
            Xk = X[:, keep]
            self.medians = np.nanmedian(Xk, axis=0)
            self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
            imputed = np.where(np.isnan(Xk), self.medians, Xk)
            self.means = imputed.mean(axis=0)
            self.stds = imputed.std(axis=0)
            self.stds = np.where(self.stds < 1e-12, 1.0, self.stds)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.keep_mask is None:
            raise RuntimeError("FoldPreprocessor not fitted")
        Xk = X[:, self.keep_mask]
        if self.mode == "linear":
            Xk = np.where(np.isnan(Xk), self.medians, Xk)
            Xk = (Xk - self.means) / self.stds
        return Xk.astype(np.float32)

    def fit_transform(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)
