"""Stability selection with two independent selectors, fold-safe.

Inside a TRAINING fold only:
- repeated stratified subsamples (fraction configurable)
- selector A: LightGBM gain importance (small, fast trees)
- selector B: L1 logistic regression on a univariate-screened, imputed,
  standardised view (screen + impute + scale all learned on the subsample)
- a feature scores a "selection" per subsample if it lands in the top-k of
  either selector; frequency = selections / subsamples

`select_top_k` returns the k most frequently selected features for that fold.
Aggregated frequencies across folds/repeats are reported for stability
analysis; the FINAL production feature list is frozen from dev-wide
frequencies at the end (documented in FEATURE_SELECTION_REPORT.md).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("features.selection")


def _lgbm_gain_ranks(X: np.ndarray, y: np.ndarray, seed: int, n_jobs: int) -> np.ndarray:
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=150, learning_rate=0.1,
        num_leaves=15, min_child_samples=20,
        colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
        scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
        random_state=seed, deterministic=True, force_row_wise=True,
        n_jobs=n_jobs, verbose=-1,
    )
    clf.fit(X, y)
    return clf.booster_.feature_importance(importance_type="gain")


def _l1_logistic_support(
    X: np.ndarray, y: np.ndarray, seed: int, screen_k: int = 500
) -> np.ndarray:
    """|coef| per feature; univariate screen -> impute -> scale -> L1 fit.

    Returns a full-width array (0 outside the screened set).
    """
    n_cols = X.shape[1]
    # univariate screen: |pairwise-complete correlation| (fast, subsample-local)
    from muleguard.data.leakage import pointbiserial_with_target

    r = np.abs(pointbiserial_with_target(X, y))
    screen = np.argsort(-r, kind="stable")[: min(screen_k, n_cols)]
    Xs = X[:, screen].astype(np.float64)
    med = np.nanmedian(Xs, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xs = np.where(np.isnan(Xs), med, Xs)
    mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (Xs - mu) / sd
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1, class_weight="balanced",
        max_iter=2000, random_state=seed,
    )
    clf.fit(Xs, y)
    out = np.zeros(n_cols)
    out[screen] = np.abs(clf.coef_[0])
    return out


def stability_frequencies(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_subsamples: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Selection frequency per feature over stratified subsamples (train fold only)."""
    cfg = settings.load_config("train")["stability_selection"]
    n_sub = n_subsamples or int(cfg["n_subsamples"])
    frac = float(cfg["subsample_fraction"])
    top_k = int(cfg["top_k_per_fit"])
    n_jobs = int(settings.load_config("train")["n_jobs"])

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    counts = np.zeros(X.shape[1], dtype=np.int32)
    for i in range(n_sub):
        p = rng.choice(pos, size=max(2, int(len(pos) * frac)), replace=False)
        n = rng.choice(neg, size=int(len(neg) * frac), replace=False)
        idx = np.concatenate([p, n])
        Xi, yi = X[idx], y[idx]
        gain = _lgbm_gain_ranks(Xi, yi, seed=seed * 1000 + i, n_jobs=n_jobs)
        l1 = _l1_logistic_support(Xi, yi, seed=seed * 1000 + i)
        sel = set(np.argsort(-gain, kind="stable")[:top_k][gain[np.argsort(-gain, kind="stable")[:top_k]] > 0])
        sel |= set(np.argsort(-l1, kind="stable")[:top_k][l1[np.argsort(-l1, kind="stable")[:top_k]] > 0])
        for j in sel:
            counts[j] += 1
    return counts / n_sub


def select_top_k(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int,
    n_subsamples: int | None = None, seed: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Top-k features by stability frequency (ties broken by stable order)."""
    freq = stability_frequencies(X, y, feature_names, n_subsamples=n_subsamples, seed=seed)
    order = np.argsort(-freq, kind="stable")[:k]
    return [feature_names[j] for j in order], freq
