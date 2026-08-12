"""Nested repeated CV tournament (final-validation §8, §10, §11).

Run:  python -m muleguard.cli.nested_cv [--trials 25] [--repeats 3] [--inner 4]

Produces the primary unbiased estimate for this validation cycle:

    artifacts/splits/nested_cv_assignments.parquet   outer+inner fold map
    artifacts/metrics/nested_cv.json                 leaderboard + chosen params
    artifacts/predictions/nested_oof.parquet         every outer-val prediction

Search spaces follow §11: bounded Optuna, average precision as the inner
objective, early stopping where the estimator supports it, and no optimisation
of plain accuracy anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.features.frame import build_model_frame
from muleguard.logging import configure, get_logger
from muleguard.models import harness, nested

log = get_logger("cli.nested_cv")

N_JOBS = 4


def _spw(y: np.ndarray) -> float:
    return float((y == 0).sum() / max(y.sum(), 1))


# --------------------------------------------------------------------------
# model families - each returns fit_predict(Xtr, ytr, Xva, seed) -> scores
# --------------------------------------------------------------------------

def lgbm_factory(p: dict[str, Any]):
    import lightgbm as lgb

    def fp(Xtr, ytr, Xva, seed):
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=p.get("n_estimators", 400),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 15), max_depth=p.get("max_depth", 4),
            min_child_samples=p.get("min_child_samples", 20),
            min_split_gain=p.get("min_split_gain", 0.0),
            colsample_bytree=p.get("colsample_bytree", 0.6),
            subsample=p.get("subsample", 0.8), subsample_freq=1,
            reg_alpha=p.get("reg_alpha", 0.0), reg_lambda=p.get("reg_lambda", 1.0),
            scale_pos_weight=_spw(ytr), random_state=seed, deterministic=True,
            force_row_wise=True, n_jobs=N_JOBS, verbose=-1)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    return fp


def lgbm_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": t.suggest_int("num_leaves", 7, 63),
        "max_depth": t.suggest_int("max_depth", 3, 8),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 60),
        "min_split_gain": t.suggest_float("min_split_gain", 0.0, 0.5),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def xgb_factory(p: dict[str, Any]):
    import xgboost as xgb

    def fp(Xtr, ytr, Xva, seed):
        m = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05),
            min_child_weight=p.get("min_child_weight", 1.0),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.6),
            gamma=p.get("gamma", 0.0), reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 1.0), scale_pos_weight=_spw(ytr),
            tree_method="hist", n_jobs=N_JOBS, random_state=seed,
            eval_metric="aucpr", verbosity=0)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    return fp


def xgb_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": t.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": t.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def cat_factory(p: dict[str, Any]):
    from catboost import CatBoostClassifier

    def fp(Xtr, ytr, Xva, seed):
        m = CatBoostClassifier(
            iterations=p.get("iterations", 400), depth=p.get("depth", 4),
            learning_rate=p.get("learning_rate", 0.05),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            random_strength=p.get("random_strength", 1.0),
            bagging_temperature=p.get("bagging_temperature", 1.0),
            scale_pos_weight=_spw(ytr), random_seed=seed, verbose=0,
            allow_writing_files=False, thread_count=N_JOBS,
            eval_metric="PRAUC", bootstrap_type="Bayesian")
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    return fp


def cat_space(t):
    return {
        "iterations": t.suggest_int("iterations", 200, 800, step=100),
        "depth": t.suggest_int("depth", 3, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "random_strength": t.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 5.0),
    }


def hgb_factory(p: dict[str, Any]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fp(Xtr, ytr, Xva, seed):
        w = np.where(ytr == 1, _spw(ytr), 1.0)
        m = HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300), max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 1.0),
            random_state=seed, early_stopping=False)
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xva)[:, 1]
    return fp


def hgb_space(t):
    return {
        "max_iter": t.suggest_int("max_iter", 150, 600, step=50),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 20.0, log=True),
    }


def extratrees_factory(p: dict[str, Any]):
    from sklearn.ensemble import ExtraTreesClassifier

    def fp(Xtr, ytr, Xva, seed):
        m = ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 500), max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            max_features=p.get("max_features", "sqrt"),
            class_weight="balanced_subsample", random_state=seed, n_jobs=N_JOBS)
        m.fit(np.nan_to_num(Xtr, nan=0.0), ytr)
        return m.predict_proba(np.nan_to_num(Xva, nan=0.0))[:, 1]
    return fp


def extratrees_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 300, 900, step=100),
        "max_depth": t.suggest_categorical("max_depth", [None, 6, 10, 16]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.1]),
    }


def logistic_factory(p: dict[str, Any]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fp(Xtr, ytr, Xva, seed):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=p.get("C", 1.0), class_weight="balanced", max_iter=2000,
                solver="liblinear", penalty=p.get("penalty", "l2"), random_state=seed))
        m.fit(np.nan_to_num(Xtr, nan=0.0), ytr)
        return m.predict_proba(np.nan_to_num(Xva, nan=0.0))[:, 1]
    return fp


def logistic_space(t):
    return {
        "C": t.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": t.suggest_categorical("penalty", ["l1", "l2"]),
    }


def dummy_factory(p: dict[str, Any]):
    def fp(Xtr, ytr, Xva, seed):
        rng = np.random.default_rng(seed)
        return rng.random(len(Xva)) * 1e-6 + float(ytr.mean())
    return fp


FAMILIES: dict[str, tuple[Any, Any, int]] = {
    # name              factory              search space        trials
    "dummy_prevalence": (dummy_factory,      None,               0),
    "logistic_l1l2":    (logistic_factory,   logistic_space,     12),
    "extratrees":       (extratrees_factory, extratrees_space,   10),
    "histgb":           (hgb_factory,        hgb_space,          15),
    "lightgbm":         (lgbm_factory,       lgbm_space,         25),
    "catboost":         (cat_factory,        cat_space,          20),
    "xgboost":          (xgb_factory,        xgb_space,          25),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None,
                    help="override the per-family trial budget")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated family subset")
    args = ap.parse_args()
    configure()

    t_start = time.time()
    frame = build_model_frame()
    dev = harness.dev_split(args.repeats)
    y = frame.y[dev.row_index]
    log.info("nested CV: dev=%d rows (+%d) outer=5-fold x %d repeats inner=%d-fold",
             len(dev.row_index), int(y.sum()), args.repeats, args.inner)

    folds = nested.build_outer_folds(frame, n_repeats=args.repeats, n_inner=args.inner)

    # --- persist the split map before any model runs (§8) ------------------
    parts = []
    for f in folds:
        # inner_ids is aligned element-wise with train_idx, so no lookup is needed
        parts.append(pl.DataFrame({
            "repeat": np.full(len(f.train_idx), f.repeat, dtype=np.int32),
            "outer_fold": np.full(len(f.train_idx), f.fold, dtype=np.int32),
            "row_index": dev.row_index[f.train_idx].astype(np.int64),
            "role": ["train"] * len(f.train_idx),
            "inner_fold": f.inner_ids.astype(np.int32)}))
        parts.append(pl.DataFrame({
            "repeat": np.full(len(f.valid_idx), f.repeat, dtype=np.int32),
            "outer_fold": np.full(len(f.valid_idx), f.fold, dtype=np.int32),
            "row_index": dev.row_index[f.valid_idx].astype(np.int64),
            "role": ["outer_valid"] * len(f.valid_idx),
            "inner_fold": np.full(len(f.valid_idx), -1, dtype=np.int32)}))
    split_df = pl.concat(parts)
    settings.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = settings.SPLITS_DIR / "nested_cv_assignments.parquet"
    split_df.write_parquet(split_path)
    log.info("wrote %s (%d rows)", split_path, split_df.height)

    wanted = set(args.only.split(",")) if args.only else set(FAMILIES)
    results: dict[str, nested.NestedResult] = {}
    for name, (factory, space, trials) in FAMILIES.items():
        if name not in wanted:
            continue
        n_trials = args.trials if args.trials is not None else trials
        res = nested.run_nested(
            name, folds, factory, space if n_trials else None,
            n_trials=max(n_trials, 1), n_repeats=args.repeats,
            n_dev=len(dev.row_index), row_index=dev.row_index, y=y)
        results[name] = res

    # --- outputs -----------------------------------------------------------
    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.ARTIFACTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    pred_frames = []
    for name, r in results.items():
        for rep in range(r.scores.shape[0]):
            pred_frames.append(pl.DataFrame({
                "model": [name] * len(r.row_index),
                "repeat": np.full(len(r.row_index), rep, dtype=np.int32),
                "row_index": r.row_index, "target": r.y, "score": r.scores[rep]}))
    pl.concat(pred_frames).write_parquet(
        settings.ARTIFACTS_DIR / "predictions" / "nested_oof.parquet")

    board = sorted((r.summary() for r in results.values()),
                   key=lambda s: -s["pr_auc_mean"])
    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "design": {
            "outer": f"stratified 5-fold x {args.repeats} repeats "
                     f"(identical to the flat tournament folds)",
            "inner": f"stratified {args.inner}-fold within outer-train",
            "selection": "LightGBM gain ranking fitted on each inner-train "
                         "partition only; frequencies pooled within the outer fold",
            "tuning": "bounded Optuna TPE, objective = inner-CV average precision",
            "feature_sizes_offered": list(nested.FEATURE_SET_SIZES),
            "outer_validation_use": "predicted once, never fed back",
        },
        "leaderboard": board,
        "chosen_per_fold": {n: r.chosen for n, r in results.items()},
        "selection_seconds_total": round(sum(f.seconds_selection for f in folds), 1),
        "total_seconds": round(time.time() - t_start, 1),
    }
    path = settings.METRICS_DIR / "nested_cv.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nNESTED REPEATED CV - primary unbiased estimate")
    print(f"{'model':<20}{'PR-AUC':>9}{'std':>9}{'ROC':>8}{'feat':>6}{'sec':>8}")
    for s in board:
        print(f"{s['model']:<20}{s['pr_auc_mean']:>9.5f}{s['pr_auc_std']:>9.5f}"
              f"{s['roc_auc_mean']:>8.4f}{s['feature_size_mode']:>6}{s['seconds']:>8.0f}")
    print(f"\nwrote {path}")
    print(f"total {out['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
