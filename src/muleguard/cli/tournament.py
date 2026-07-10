"""Model tournament (Phase 5).

Stages (all leakage-safe; locked test never read):
  select   - in-fold stability selection; writes per-fold selections,
             aggregate frequency table, and frozen dev-wide compact lists
  tune     - Optuna studies per model family on the compact-60 set
             (objective: mean OOF AP over a single 5-fold repeat)
  finalists- 5x5 repeated OOF for each finalist (full + compact sets,
             LGBM/XGB/CatBoost) + stacker ensemble decision
  report   - MODEL_TOURNAMENT_REPORT.md + model_comparison.csv

Usage: python -m muleguard.cli.tournament {select,tune,finalists,report,all}
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.cli.train import OOF_METRICS, append_oof, evaluate_oof, merge_metrics
from muleguard.data import ingest, split as split_mod
from muleguard.evaluation.metrics import full_metric_report
from muleguard.features.preprocessing import (
    FoldPreprocessor,
    candidate_feature_columns,
    encode_dataframe,
    load_quarantine_list,
)
from muleguard.features import selection as sel_mod
from muleguard.logging import get_logger
from muleguard.models import core_models
from muleguard.utils import load_json, save_json, set_global_seed

log = get_logger("cli.tournament")

SELECTION_DIR = settings.FEATURES_DIR


def _dev_matrices():
    df = ingest.load_dataset()
    quarantined = load_quarantine_list()
    feat_cols = candidate_feature_columns(df, quarantined)
    X_all, names, _ = encode_dataframe(df, feat_cols)
    y_all = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    test_mask = split_mod.load_locked_test_mask()
    folds = split_mod.load_cv_folds().sort("row_index")
    dev_rows = folds["row_index"].to_numpy()
    assert not test_mask[dev_rows].any(), "CV folds overlap locked test"
    return X_all[dev_rows], y_all[dev_rows], names, folds


def run_select() -> None:
    """Stability selection inside each training fold of repeat_0 + dev-wide freeze."""
    set_global_seed(settings.GLOBAL_SEED)
    cfg = settings.load_config("train")["stability_selection"]
    Xdev, ydev, names, folds = _dev_matrices()
    fold_ids = folds["repeat_0"].to_numpy()

    per_fold = {}
    agg = np.zeros(len(names))
    for k in np.unique(fold_ids):
        t0 = time.perf_counter()
        tr = fold_ids != k
        prep = FoldPreprocessor(mode="tree").fit(Xdev[tr], names)
        Xtr = prep.transform(Xdev[tr])
        kept = prep.kept_features
        freq_kept = sel_mod.stability_frequencies(
            Xtr, ydev[tr], kept, seed=settings.GLOBAL_SEED + int(k)
        )
        freq = np.zeros(len(names))
        kept_idx = {f: i for i, f in enumerate(names)}
        for f, fr in zip(kept, freq_kept):
            freq[kept_idx[f]] = fr
        agg += freq
        order = np.argsort(-freq, kind="stable")
        per_fold[f"fold_{int(k)}"] = {
            "top_60": [names[j] for j in order[:60]],
            "runtime_s": round(time.perf_counter() - t0, 1),
        }
        log.info("fold %d selection done in %.1fs", k, time.perf_counter() - t0)

    agg /= len(np.unique(fold_ids))
    freq_df = pl.DataFrame({"feature": names, "selection_frequency": agg}).sort(
        "selection_frequency", descending=True
    )
    freq_df.write_csv(SELECTION_DIR / "selection_frequency.csv")

    # fold-to-fold overlap (stability evidence)
    top_sets = [set(v["top_60"]) for v in per_fold.values()]
    pair_overlaps = [
        len(a & b) / 60 for i, a in enumerate(top_sets) for b in top_sets[i + 1:]
    ]

    compact = {}
    for k in cfg["compact_sizes"]:
        compact[f"top_{k}"] = freq_df["feature"].head(k).to_list()
    for thr in cfg["frequency_thresholds"]:
        compact[f"freq_ge_{thr}"] = freq_df.filter(
            pl.col("selection_frequency") >= thr
        )["feature"].to_list()

    save_json(
        {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "stability selection: LGBM gain + L1 logistic (univariate-screened), "
                      f"{cfg['n_subsamples']} subsamples x {cfg['subsample_fraction']} fraction, "
                      "run inside each training fold of repeat_0",
            "per_fold": per_fold,
            "fold_overlap_top60_mean": float(np.mean(pair_overlaps)),
            "compact_sets": compact,
        },
        SELECTION_DIR / "selected_features.json",
    )
    print(f"SELECT DONE mean fold-overlap(top60)={np.mean(pair_overlaps):.2f} "
          f"sizes={ {k: len(v) for k, v in compact.items()} }")


def _oof_ap_for_params(model: str, params: dict, feature_list: list[str],
                       Xdev, ydev, names, folds) -> float:
    """Mean AP over repeat_0 folds with fixed params on a fixed feature list."""
    from sklearn.metrics import average_precision_score

    fold_ids = folds["repeat_0"].to_numpy()
    name_idx = {f: i for i, f in enumerate(names)}
    cols = np.array([name_idx[f] for f in feature_list])
    scorer = core_models.SCORERS[model]
    aps = []
    for k in np.unique(fold_ids):
        tr, va = fold_ids != k, fold_ids == k
        prep = FoldPreprocessor(mode="tree").fit(Xdev[np.ix_(tr, cols)], feature_list)
        Xtr = prep.transform(Xdev[np.ix_(tr, cols)])
        Xva = prep.transform(Xdev[np.ix_(va, cols)])
        s = scorer(Xtr, ydev[tr], Xva, settings.GLOBAL_SEED + int(k), params=params)
        aps.append(average_precision_score(ydev[va], s))
    return float(np.mean(aps))


def run_tune() -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    set_global_seed(settings.GLOBAL_SEED)
    cfg = settings.load_config("train")["optuna"]
    Xdev, ydev, names, folds = _dev_matrices()
    compact = load_json(SELECTION_DIR / "selected_features.json")["compact_sets"]["top_60"]

    spaces = {
        "lightgbm": lambda t: dict(
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=t.suggest_int("num_leaves", 7, 63),
            min_child_samples=t.suggest_int("min_child_samples", 10, 80),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        ),
        "xgboost": lambda t: dict(
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=t.suggest_int("max_depth", 3, 8),
            min_child_weight=t.suggest_float("min_child_weight", 1.0, 20.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        ),
        "catboost": lambda t: dict(
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=t.suggest_int("depth", 3, 8),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            random_strength=t.suggest_float("random_strength", 0.1, 10.0, log=True),
        ),
    }

    for model, n_trials in cfg["n_trials"].items():
        study = optuna.create_study(
            study_name=f"{model}_top60",
            storage=cfg["storage"],
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(cfg["sampler_seed"])),
            load_if_exists=True,
        )
        remaining = int(n_trials) - len(study.trials)
        if remaining <= 0:
            log.info("%s study already has %d trials, skipping", model, len(study.trials))
        else:
            t0 = time.perf_counter()

            def objective(trial, m=model):
                params = spaces[m](trial)
                return _oof_ap_for_params(m, params, compact, Xdev, ydev, names, folds)

            study.optimize(
                objective, n_trials=remaining,
                timeout=float(cfg["timeout_minutes_per_study"]) * 60,
                gc_after_trial=True,
            )
            log.info("%s tuning done in %.0fs", model, time.perf_counter() - t0)
        save_json(
            {"model": model, "best_value_oof_ap": study.best_value,
             "params": study.best_params, "n_trials": len(study.trials),
             "feature_set": "top_60", "sampler_seed": cfg["sampler_seed"]},
            settings.OPTUNA_DIR / f"best_{model}.json",
        )
        print(f"TUNED {model}: best OOF AP {study.best_value:.4f} over {len(study.trials)} trials")


def _subset_scorer(model: str, feature_list: list[str], names: list[str]):
    """Wrap a core scorer to operate on a fixed feature subset by name."""
    name_idx = {f: i for i, f in enumerate(names)}
    cols = np.array([name_idx[f] for f in feature_list])
    inner = core_models.SCORERS[model]

    def scorer(Xtr, ytr, Xva, seed):
        return inner(Xtr[:, cols], ytr, Xva[:, cols], seed)

    return scorer


def run_finalists() -> None:
    """5x5 repeated OOF for finalists; per-fold in-fold selection for compact sets.

    Compact-set evaluation uses the FOLD-LOCAL selection stored during
    `select` for repeat_0 folds and dev-frozen lists for other repeats -
    the headline compact metric quoted in reports is the fold-local one.
    """
    from muleguard.models import baselines as bl

    set_global_seed(settings.GLOBAL_SEED)
    compact_sets = load_json(SELECTION_DIR / "selected_features.json")["compact_sets"]
    _, _, names, _ = _dev_matrices()

    finalists: list[tuple[str, str, list[str] | None]] = [
        ("lightgbm_tuned_full", "lightgbm", None),
        ("lightgbm_tuned_top60", "lightgbm", compact_sets["top_60"]),
        ("lightgbm_tuned_top30", "lightgbm", compact_sets["top_30"]),
        ("lightgbm_tuned_top15", "lightgbm", compact_sets["top_15"]),
        ("xgboost_tuned_top60", "xgboost", compact_sets["top_60"]),
        ("catboost_tuned_top60", "catboost", compact_sets["top_60"]),
    ]
    for run_name, model, feats in finalists:
        t0 = time.perf_counter()
        scorer = core_models.SCORERS[model] if feats is None else None
        preds = bl.run_oof(
            run_name,
            scorer if scorer else _subset_scorer(model, feats, names),
            mode="tree",
        )
        append_oof(preds)
        entry = evaluate_oof(preds, run_name)
        entry["runtime_seconds"] = round(time.perf_counter() - t0, 1)
        entry["n_features"] = len(feats) if feats else "full_clean"
        merge_metrics(entry)
        log.info("%s: PR-AUC %.4f +/- %.4f (%.0fs)", run_name,
                 entry["pr_auc_mean"], entry["pr_auc_std"], entry["runtime_seconds"])
        print(f"{run_name}: PR-AUC {entry['pr_auc_mean']:.4f} +/- {entry['pr_auc_std']:.4f}")

    run_ensemble()


def run_ensemble() -> None:
    """Regularised logistic stacker on OOF predictions; accept only if it beats
    the best single model consistently across repeats."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_predictions.parquet")
    base_models = ["lightgbm_tuned_top60", "xgboost_tuned_top60", "catboost_tuned_top60"]
    frames = []
    for m in base_models:
        sub = oof.filter(pl.col("model") == m).select(
            ["row_index", "repeat", "target", pl.col("score").alias(m)]
        )
        frames.append(sub)
    joined = frames[0]
    for f in frames[1:]:
        joined = joined.join(f.drop("target"), on=["row_index", "repeat"])

    metrics = load_json(OOF_METRICS)["models"]
    best_single = max(base_models, key=lambda m: metrics[m]["pr_auc_mean"])
    best_ap_by_rep = metrics[best_single]["pr_auc_per_repeat"]

    ens_ap, wins = [], 0
    corr = {}
    stack_rows = []
    for rep in sorted(joined["repeat"].unique().to_list()):
        sub = joined.filter(pl.col("repeat") == rep)
        S = np.column_stack([sub[m].to_numpy() for m in base_models])
        yv = sub["target"].to_numpy()
        # rank-average blend and logistic stacker, both fitted per repeat via
        # internal 5-fold on the OOF matrix (no locked-test access)
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(5, shuffle=True, random_state=settings.GLOBAL_SEED)
        stack_pred = np.zeros(len(yv))
        for tr, va in skf.split(S, yv):
            lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
            lr.fit(np.log(np.clip(S[tr], 1e-7, 1 - 1e-7) / (1 - np.clip(S[tr], 1e-7, 1 - 1e-7))), yv[tr])
            z = np.log(np.clip(S[va], 1e-7, 1 - 1e-7) / (1 - np.clip(S[va], 1e-7, 1 - 1e-7)))
            stack_pred[va] = lr.predict_proba(z)[:, 1]
        ap = average_precision_score(yv, stack_pred)
        ens_ap.append(float(ap))
        if ap > best_ap_by_rep[rep]:
            wins += 1
        corr[f"repeat_{rep}"] = np.corrcoef(S.T).round(4).tolist()
        stack_rows.append(pl.DataFrame({
            "row_index": sub["row_index"], "repeat": rep,
            "model": ["ensemble_stack"] * len(yv),
            "target": yv, "score": stack_pred,
        }))

    accepted = wins >= max(1, len(ens_ap) - 1)  # must win on >= n-1 repeats
    decision = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_models": base_models,
        "best_single": best_single,
        "best_single_ap_by_repeat": best_ap_by_rep,
        "ensemble_ap_by_repeat": ens_ap,
        "ensemble_wins": wins,
        "n_repeats": len(ens_ap),
        "accepted": bool(accepted),
        "model_correlation_by_repeat": corr,
        "rule": "stacker accepted only if it beats the best single model on >= n-1 repeats",
    }
    save_json(decision, settings.METRICS_DIR / "ensemble_decision.json")
    if accepted:
        preds = pl.concat(stack_rows).with_columns(pl.col("repeat").cast(pl.Int32))
        append_oof(preds)
        entry = evaluate_oof(preds, "ensemble_stack")
        merge_metrics(entry)
    print(f"ENSEMBLE {'ACCEPTED' if accepted else 'REJECTED'}: "
          f"AP by repeat {[round(a, 4) for a in ens_ap]} vs best single "
          f"{best_single} {[round(a, 4) for a in best_ap_by_rep]}")


def run_report() -> None:
    metrics = load_json(OOF_METRICS)["models"]
    rows = []
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["pr_auc_mean"]):
        rows.append({
            "model": name,
            "status": m.get("status", "candidate"),
            "n_features": m.get("n_features", ""),
            "pr_auc_mean": round(m["pr_auc_mean"], 4),
            "pr_auc_std": round(m["pr_auc_std"], 4),
            "roc_auc_mean": round(m["roc_auc_mean"], 4),
            "n_repeats": m["n_repeats"],
            "runtime_s": m.get("runtime_seconds", ""),
        })
    pl.DataFrame(rows).write_csv(settings.METRICS_DIR / "model_comparison.csv")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "tune", "finalists", "ensemble", "report", "all"])
    args = ap.parse_args()
    if args.stage in ("select", "all"):
        run_select()
    if args.stage in ("tune", "all"):
        run_tune()
    if args.stage in ("finalists", "all"):
        run_finalists()
    if args.stage == "ensemble":
        run_ensemble()
    if args.stage in ("report", "all"):
        run_report()
