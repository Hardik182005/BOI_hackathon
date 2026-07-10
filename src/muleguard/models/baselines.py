"""Leakage-free baseline models and the OOF evaluation harness.

The harness is the single place where CV happens for every model family:
for each repeat/fold it fits preprocessing on the training fold, trains,
and writes predictions for the held-out fold. Nothing sees the locked test.

Baselines: prevalence dummy, class-weighted logistic (L2), class-weighted
elastic-net logistic, LightGBM with class weighting + early stopping on a
stratified carve-out OF THE TRAINING FOLD.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from muleguard import settings
from muleguard.data import ingest, split as split_mod
from muleguard.features.preprocessing import (
    FoldPreprocessor,
    candidate_feature_columns,
    encode_dataframe,
    load_quarantine_list,
)
from muleguard.logging import get_logger

log = get_logger("models.baselines")

ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
# signature: (X_train, y_train, X_valid_for_es, y_valid_for_es, seed) -> fitted predictor
# we simplify: functions return validation-fold scores directly


def _lgbm_params(seed: int, n_pos: int, n_neg: int, n_jobs: int) -> dict[str, Any]:
    cfg = settings.load_config("train")["baselines"]["lightgbm"]
    return dict(
        objective="binary",
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        num_leaves=int(cfg["num_leaves"]),
        min_child_samples=int(cfg["min_child_samples"]),
        colsample_bytree=float(cfg["feature_fraction"]),
        subsample=float(cfg["bagging_fraction"]),
        subsample_freq=int(cfg["bagging_freq"]),
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
    )


def fit_score_dummy(Xtr, ytr, Xva, seed: int) -> np.ndarray:
    """Prevalence dummy: every account gets the training prevalence."""
    return np.full(len(Xva), ytr.mean(), dtype=float)


def fit_score_logistic(Xtr, ytr, Xva, seed: int, penalty: str = "l2") -> np.ndarray:
    cfg = settings.load_config("train")["baselines"]["logistic"]
    kwargs: dict[str, Any] = dict(
        C=float(cfg["C"]), max_iter=int(cfg["max_iter"]),
        class_weight="balanced", random_state=seed,
    )
    if penalty == "elasticnet":
        kwargs.update(penalty="elasticnet", solver="saga", l1_ratio=0.5)
    else:
        kwargs.update(penalty="l2", solver="lbfgs")
    clf = LogisticRegression(**kwargs)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1]


def fit_score_lightgbm(
    Xtr, ytr, Xva, seed: int, params_override: dict[str, Any] | None = None,
    return_model: bool = False,
):
    import lightgbm as lgb

    train_cfg = settings.load_config("train")
    es_frac = float(train_cfg["early_stop_fraction"])
    n_jobs = int(train_cfg["n_jobs"])
    # early-stopping set carved from TRAIN fold only, stratified
    Xfit, Xes, yfit, yes = train_test_split(
        Xtr, ytr, test_size=es_frac, stratify=ytr, random_state=seed
    )
    params = _lgbm_params(seed, int(yfit.sum()), int((yfit == 0).sum()), n_jobs)
    if params_override:
        params.update(params_override)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        Xfit, yfit,
        eval_set=[(Xes, yes)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    scores = clf.predict_proba(Xva)[:, 1]
    return (scores, clf) if return_model else scores


def run_oof(
    model_name: str,
    scorer: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray],
    mode: str,
    n_repeats: int | None = None,
    feature_subset: list[str] | None = None,
    extra_allowed: list[str] | None = None,
    fold_feature_recorder: list | None = None,
) -> pl.DataFrame:
    """Run a scorer through the saved repeated folds; return OOF predictions.

    - `mode`: 'tree' (NaN passthrough) or 'linear' (impute+scale in-fold)
    - `feature_subset`: restrict candidate features (post-quarantine names)
    - `extra_allowed`: names normally quarantined to ADD BACK (used only for
      the rejected-leakage ablation; callers must label results accordingly)
    """
    df = ingest.load_dataset()
    quarantined = load_quarantine_list()
    if extra_allowed:
        quarantined = [q for q in quarantined if q not in set(extra_allowed)]
        log.warning("ABLATION RUN: re-admitting quarantined features %s", extra_allowed)

    feat_cols = candidate_feature_columns(df, quarantined)
    if feature_subset is not None:
        missing = set(feature_subset) - set(feat_cols)
        if missing:
            raise ValueError(f"feature_subset contains unavailable columns: {sorted(missing)[:5]}")
        feat_cols = [c for c in feat_cols if c in set(feature_subset)]

    X_all, feat_names, _ = encode_dataframe(df, feat_cols)
    y_all = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()

    test_mask = split_mod.load_locked_test_mask()
    folds = split_mod.load_cv_folds().sort("row_index")
    dev_rows = folds["row_index"].to_numpy()
    assert not test_mask[dev_rows].any(), "CV folds overlap locked test - refusing to run"

    Xdev, ydev = X_all[dev_rows], y_all[dev_rows]
    repeat_cols = [c for c in folds.columns if c.startswith("repeat_")]
    if n_repeats is not None:
        repeat_cols = repeat_cols[:n_repeats]

    out_frames = []
    for rep_i, rep_col in enumerate(repeat_cols):
        fold_ids = folds[rep_col].to_numpy()
        oof = np.full(len(dev_rows), np.nan, dtype=float)
        for k in np.unique(fold_ids):
            tr, va = fold_ids != k, fold_ids == k
            prep = FoldPreprocessor(mode="tree" if mode == "tree" else "linear")
            Xtr = prep.fit_transform(Xdev[tr], feat_names)
            Xva = prep.transform(Xdev[va])
            seed = settings.GLOBAL_SEED * 1000 + rep_i * 10 + int(k)
            oof[va] = scorer(Xtr, ydev[tr], Xva, seed)
            if fold_feature_recorder is not None:
                fold_feature_recorder.append({
                    "repeat": rep_i, "fold": int(k),
                    "n_features_in": len(feat_names),
                    "n_features_kept": len(prep.kept_features),
                })
        if np.isnan(oof).any():
            raise RuntimeError("OOF has unscored rows - fold definitions inconsistent")
        out_frames.append(pl.DataFrame({
            "row_index": dev_rows,
            "repeat": np.full(len(dev_rows), rep_i, dtype=np.int32),
            "model": [model_name] * len(dev_rows),
            "target": ydev,
            "score": oof,
        }))
    return pl.concat(out_frames)
