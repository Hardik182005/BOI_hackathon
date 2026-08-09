"""Firewall-aware repeated-CV harness (Track A of the validation plan).

This is the successor to :mod:`muleguard.models.baselines`'s data path. The
difference is where the feature matrix comes from: everything here starts from
a :class:`muleguard.features.frame.ModelFrame`, which has already passed the
Feature Availability Firewall. A caller cannot accidentally train on a
post-resolution column, because it never has the chance to assemble a matrix.

Design rules that make the OOF number trustworthy:

* Preprocessing (constant/duplicate removal, imputation, scaling) is fitted on
  the training fold only, inside the fold loop.
* Early stopping carves its validation slice out of the TRAINING fold.
* Feature selection, when used, is passed in as a fixed list produced by an
  outer procedure that itself only ever saw training folds.
* The locked test rows are asserted absent from every fold.
* Seeds are a deterministic function of (global seed, repeat, fold), so a
  rerun reproduces the same OOF vector byte for byte.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from muleguard import settings
from muleguard.data import split as split_mod
from muleguard.features.frame import ModelFrame
from muleguard.features.preprocessing import FoldPreprocessor
from muleguard.logging import get_logger

log = get_logger("models.harness")

Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class OOFResult:
    """Out-of-fold predictions for one model configuration."""

    name: str
    row_index: np.ndarray
    y: np.ndarray
    scores: np.ndarray  # shape (n_repeats, n_dev_rows)
    feature_names: list[str]
    seconds: float = 0.0
    fold_detail: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_repeats(self) -> int:
        return self.scores.shape[0]

    def mean_scores(self) -> np.ndarray:
        """Rank-averaged score across repeats (repeat-robust ranking)."""
        ranks = np.vstack([_to_rank(s) for s in self.scores])
        return ranks.mean(axis=0)

    def per_repeat_ap(self) -> list[float]:
        return [float(average_precision_score(self.y, s)) for s in self.scores]

    def summary(self) -> dict[str, Any]:
        aps = self.per_repeat_ap()
        rocs = [float(roc_auc_score(self.y, s)) for s in self.scores]
        return {
            "model": self.name,
            "n_features": len(self.feature_names),
            "n_repeats": self.n_repeats,
            "pr_auc_mean": float(np.mean(aps)),
            "pr_auc_std": float(np.std(aps)),
            "pr_auc_min": float(np.min(aps)),
            "pr_auc_per_repeat": [round(a, 5) for a in aps],
            "roc_auc_mean": float(np.mean(rocs)),
            "pr_auc_rank_avg": float(average_precision_score(self.y, self.mean_scores())),
            "seconds": round(self.seconds, 1),
        }

    def to_frame(self) -> pl.DataFrame:
        frames = []
        for r in range(self.n_repeats):
            frames.append(pl.DataFrame({
                "row_index": self.row_index,
                "repeat": np.full(len(self.row_index), r, dtype=np.int32),
                "model": [self.name] * len(self.row_index),
                "target": self.y,
                "score": self.scores[r],
            }))
        return pl.concat(frames)


def _to_rank(s: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(s, kind="stable"), kind="stable")
    return order / max(len(s) - 1, 1)


@dataclass
class DevSplit:
    """Development rows and their repeated-fold assignments."""

    row_index: np.ndarray
    fold_ids: np.ndarray  # shape (n_repeats, n_dev_rows)

    @property
    def n_repeats(self) -> int:
        return self.fold_ids.shape[0]


_DEV_CACHE: DevSplit | None = None


def dev_split(n_repeats: int | None = None) -> DevSplit:
    """Load the saved folds and assert they never touch the locked test."""
    global _DEV_CACHE
    if _DEV_CACHE is None:
        folds = split_mod.load_cv_folds().sort("row_index")
        test_mask = split_mod.load_locked_test_mask()
        rows = folds["row_index"].to_numpy()
        if test_mask[rows].any():
            raise RuntimeError("CV folds overlap the locked test - refusing to run")
        repeat_cols = [c for c in folds.columns if c.startswith("repeat_")]
        _DEV_CACHE = DevSplit(
            row_index=rows,
            fold_ids=np.vstack([folds[c].to_numpy() for c in repeat_cols]),
        )
    if n_repeats is None:
        return _DEV_CACHE
    return DevSplit(_DEV_CACHE.row_index, _DEV_CACHE.fold_ids[:n_repeats])


def fold_seed(repeat: int, fold: int) -> int:
    return settings.GLOBAL_SEED * 1000 + repeat * 10 + int(fold)


def run_oof(
    name: str,
    scorer: Scorer,
    frame: ModelFrame,
    *,
    mode: str = "tree",
    n_repeats: int | None = None,
    feature_subset: Sequence[str] | None = None,
    record_folds: bool = False,
) -> OOFResult:
    """Repeated stratified OOF for one scorer over one ModelFrame."""
    mf = frame.subset(feature_subset) if feature_subset is not None else frame
    dev = dev_split(n_repeats)
    Xdev = mf.X[dev.row_index]
    ydev = mf.y[dev.row_index]
    names = mf.feature_names

    t0 = time.time()
    all_scores = np.full((dev.n_repeats, len(dev.row_index)), np.nan, dtype=float)
    detail: list[dict[str, Any]] = []
    for rep in range(dev.n_repeats):
        ids = dev.fold_ids[rep]
        for k in np.unique(ids):
            tr, va = ids != k, ids == k
            prep = FoldPreprocessor(mode="tree" if mode == "tree" else "linear")
            Xtr = prep.fit_transform(Xdev[tr], names)
            Xva = prep.transform(Xdev[va])
            all_scores[rep, va] = scorer(Xtr, ydev[tr], Xva, fold_seed(rep, int(k)))
            if record_folds:
                detail.append({
                    "repeat": rep, "fold": int(k),
                    "n_features_in": len(names),
                    "n_features_kept": len(prep.kept_features),
                    "n_train": int(tr.sum()), "n_valid": int(va.sum()),
                    "n_train_pos": int(ydev[tr].sum()), "n_valid_pos": int(ydev[va].sum()),
                })
        if np.isnan(all_scores[rep]).any():
            raise RuntimeError(f"{name}: repeat {rep} has unscored rows")
    result = OOFResult(
        name=name, row_index=dev.row_index, y=ydev, scores=all_scores,
        feature_names=list(names), seconds=time.time() - t0, fold_detail=detail,
    )
    s = result.summary()
    log.info("%-34s feats=%-5d PR-AUC %.4f +/- %.4f (%.0fs)",
             name, s["n_features"], s["pr_auc_mean"], s["pr_auc_std"], s["seconds"])
    return result


def fit_full_dev(
    fitter: Callable[[np.ndarray, np.ndarray, int], Any],
    frame: ModelFrame,
    *,
    mode: str = "tree",
    feature_subset: Sequence[str] | None = None,
    seed: int | None = None,
) -> tuple[Any, FoldPreprocessor, list[str]]:
    """Refit on ALL development rows for the deployable artifact.

    Returns the fitted estimator, the preprocessor fitted on the same rows,
    and the feature names in matrix order. The locked test is still untouched.
    """
    mf = frame.subset(feature_subset) if feature_subset is not None else frame
    dev = dev_split()
    Xdev, ydev = mf.X[dev.row_index], mf.y[dev.row_index]
    prep = FoldPreprocessor(mode="tree" if mode == "tree" else "linear")
    Xt = prep.fit_transform(Xdev, mf.feature_names)
    est = fitter(Xt, ydev, settings.GLOBAL_SEED if seed is None else seed)
    return est, prep, list(mf.feature_names)
