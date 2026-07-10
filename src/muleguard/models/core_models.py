"""Core tuned model scorers: LightGBM, XGBoost, CatBoost.

Each scorer follows the harness signature (Xtr, ytr, Xva, seed) -> scores and
carves its early-stopping set from the training fold only. Hyperparameters
come from Optuna best-trial JSONs when present, else safe defaults.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.utils import load_json

log = get_logger("models.core")


def best_params(model: str) -> dict[str, Any]:
    path = settings.OPTUNA_DIR / f"best_{model}.json"
    if path.exists():
        return load_json(path)["params"]
    return {}


def _es_split(Xtr, ytr, seed):
    frac = float(settings.load_config("train")["early_stop_fraction"])
    return train_test_split(Xtr, ytr, test_size=frac, stratify=ytr, random_state=seed)


def fit_score_lgbm_tuned(Xtr, ytr, Xva, seed: int, params: dict | None = None,
                         return_model: bool = False):
    import lightgbm as lgb

    n_jobs = int(settings.load_config("train")["n_jobs"])
    Xf, Xes, yf, yes = _es_split(Xtr, ytr, seed)
    p = dict(
        objective="binary", n_estimators=3000, learning_rate=0.03,
        num_leaves=31, min_child_samples=25, colsample_bytree=0.6,
        subsample=0.8, subsample_freq=1, reg_alpha=0.0, reg_lambda=0.0,
        scale_pos_weight=(yf == 0).sum() / max(yf.sum(), 1),
        random_state=seed, deterministic=True, force_row_wise=True,
        n_jobs=n_jobs, verbose=-1,
    )
    p.update(params if params is not None else best_params("lightgbm"))
    clf = lgb.LGBMClassifier(**p)
    clf.fit(Xf, yf, eval_set=[(Xes, yes)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    s = clf.predict_proba(Xva)[:, 1]
    return (s, clf) if return_model else s


def fit_score_xgb_tuned(Xtr, ytr, Xva, seed: int, params: dict | None = None,
                        return_model: bool = False):
    import xgboost as xgb

    n_jobs = int(settings.load_config("train")["n_jobs"])
    Xf, Xes, yf, yes = _es_split(Xtr, ytr, seed)
    p = dict(
        objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
        max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.0, reg_lambda=1.0, tree_method="hist",
        scale_pos_weight=(yf == 0).sum() / max(yf.sum(), 1),
        random_state=seed, n_jobs=n_jobs, eval_metric="aucpr",
        early_stopping_rounds=100,
    )
    p.update(params if params is not None else best_params("xgboost"))
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xf, yf, eval_set=[(Xes, yes)], verbose=False)
    s = clf.predict_proba(Xva)[:, 1]
    return (s, clf) if return_model else s


def fit_score_catboost_tuned(Xtr, ytr, Xva, seed: int, params: dict | None = None,
                             return_model: bool = False):
    from catboost import CatBoostClassifier

    Xf, Xes, yf, yes = _es_split(Xtr, ytr, seed)
    p = dict(
        iterations=3000, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="PRAUC",
        scale_pos_weight=(yf == 0).sum() / max(yf.sum(), 1),
        random_seed=seed, od_type="Iter", od_wait=100,
        thread_count=int(settings.load_config("train")["n_jobs"]),
        verbose=False, allow_writing_files=False,
    )
    p.update(params if params is not None else best_params("catboost"))
    clf = CatBoostClassifier(**p)
    clf.fit(Xf, yf, eval_set=(Xes, yes))
    s = clf.predict_proba(Xva)[:, 1]
    return (s, clf) if return_model else s


SCORERS = {
    "lightgbm": fit_score_lgbm_tuned,
    "xgboost": fit_score_xgb_tuned,
    "catboost": fit_score_catboost_tuned,
}
