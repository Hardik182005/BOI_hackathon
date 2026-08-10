"""Trinetra Validation Shield (addendum UPDATE 3).

Two jobs, one idea: find the features that work here and will not work there.

**Development time** - `feature_stability_report()` scores every selected
feature on five axes and assigns a flag:

    STABLE       behaves consistently across folds and subgroups
    WATCH        one axis is marginal; keep, but report it
    SHIFT_PRONE  distribution/missingness/importance moves enough that the
                 feature may not transfer to a differently-drawn sample
    LEAKAGE      the firewall rejected it (should never appear in a model)

A SHIFT_PRONE flag is **not** an automatic removal. The specification is
explicit: test the model with and without those features and keep whatever
reproducibly works. The flag is an instruction to measure, not to delete.

**Upload time** - `adversarial_validation()` trains a throwaway classifier to
separate training rows (label 0) from uploaded rows (label 1). This is not the
mule detector and its output never touches a risk score. It answers one
question for the judge: *is your data drawn from a materially different
distribution than ours?*

    AUC ~= 0.50   the two samples are hard to tell apart
    AUC elevated  some shift; predictions still valid, confidence lower
    AUC very high substantial shift; report it prominently

Hard rules enforced here: the uploaded target is never read by this module,
the mule model is never refitted on uploaded rows, and no prediction is
altered because of an adversarial AUC.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from muleguard import settings
from muleguard.features import dictionary as fd
from muleguard.features.frame import ModelFrame, augmented_registry
from muleguard.features.preprocessing import FoldPreprocessor
from muleguard.logging import get_logger
from muleguard.models import harness, selection

log = get_logger("models.shield")

STABLE, WATCH, SHIFT_PRONE, LEAKAGE = "STABLE", "WATCH", "SHIFT_PRONE", "LEAKAGE"

# Fixed before measurement, as with the robustness grades.
SHIELD_THRESHOLDS = {
    "selection_frequency_stable_min": 0.40,
    "importance_cv_max": 1.25,       # std/mean of fold gain share
    "missingness_range_max": 0.15,   # max-min non-null rate across folds
    "distribution_psi_max": 0.25,    # population stability index across folds
    "psi_watch": 0.10,
}


def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 20 or len(b) < 20:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    pa = np.histogram(a, bins=edges)[0].astype(float)
    pb = np.histogram(b, bins=edges)[0].astype(float)
    pa = np.clip(pa / max(pa.sum(), 1), 1e-6, None)
    pb = np.clip(pb / max(pb.sum(), 1), 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def feature_stability_report(
    frame: ModelFrame,
    features: Sequence[str],
    *,
    n_repeats: int = 2,
    selection_result: selection.StabilityResult | None = None,
) -> dict[str, Any]:
    """Flag every candidate feature STABLE / WATCH / SHIFT_PRONE / LEAKAGE."""
    from muleguard.features import firewall

    reg = augmented_registry()
    mf = frame.subset(features)
    dev = harness.dev_split(n_repeats)
    Xdev = mf.X[dev.row_index]
    names = mf.feature_names
    idx = {n: i for i, n in enumerate(names)}

    # per-fold missingness and distribution
    miss: dict[str, list[float]] = {n: [] for n in names}
    psis: dict[str, list[float]] = {n: [] for n in names}
    for rep in range(dev.n_repeats):
        ids = dev.fold_ids[rep]
        for k in np.unique(ids):
            tr, va = ids != k, ids == k
            for n in names:
                col_tr, col_va = Xdev[tr, idx[n]], Xdev[va, idx[n]]
                miss[n].append(float(np.mean(~np.isnan(col_va))))
                psis[n].append(_psi(col_tr, col_va))

    # per-fold importance dispersion
    sel = selection_result or selection.stability_select(
        mf, top_k=len(names), n_repeats=n_repeats, pool_label="shield")
    sel_map = {f: (freq, gain, cv) for f, freq, gain, cv in
               zip(sel.feature, sel.frequency, sel.mean_gain, sel.gain_cv)}

    rows = []
    counts = {STABLE: 0, WATCH: 0, SHIFT_PRONE: 0, LEAKAGE: 0}
    for n in names:
        rec = fd.describe(n, reg)
        freq, gain, imp_cv = sel_map.get(n, (0.0, 0.0, 0.0))
        miss_range = float(np.max(miss[n]) - np.min(miss[n]))
        psi_max = float(np.max(psis[n]))

        reasons: list[str] = []
        if rec["availability_class"] in firewall.FORBIDDEN_CLASSES:
            flag = LEAKAGE
            reasons.append(f"availability class {rec['availability_class']}")
        else:
            shift = []
            if psi_max > SHIELD_THRESHOLDS["distribution_psi_max"]:
                shift.append(f"cross-fold PSI {psi_max:.2f}")
            if miss_range > SHIELD_THRESHOLDS["missingness_range_max"]:
                shift.append(f"missingness range {miss_range:.2f}")
            watch = []
            if freq < SHIELD_THRESHOLDS["selection_frequency_stable_min"]:
                watch.append(f"selection frequency {freq:.2f}")
            if SHIELD_THRESHOLDS["psi_watch"] < psi_max <= SHIELD_THRESHOLDS["distribution_psi_max"]:
                watch.append(f"cross-fold PSI {psi_max:.2f}")
            if imp_cv > SHIELD_THRESHOLDS["importance_cv_max"]:
                watch.append(f"importance CV {imp_cv:.2f}")
            if shift:
                flag, reasons = SHIFT_PRONE, shift
            elif watch:
                flag, reasons = WATCH, watch
            else:
                flag = STABLE
        counts[flag] += 1
        rows.append({
            "feature": n,
            "variable_name": rec["variable_name"],
            "feature_family": rec["feature_family"],
            "availability_class": rec["availability_class"],
            "selection_frequency": round(freq, 3),
            "mean_gain_share": round(gain, 5),
            "cross_fold_psi_max": round(psi_max, 4),
            "missingness_range": round(miss_range, 4),
            "flag": flag,
            "reasons": reasons,
        })

    rows.sort(key=lambda r: ({STABLE: 0, WATCH: 1, SHIFT_PRONE: 2, LEAKAGE: 3}[r["flag"]],
                             -r["selection_frequency"]))
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_features": len(rows),
        "counts": counts,
        "thresholds": SHIELD_THRESHOLDS,
        "policy": (
            "SHIFT_PRONE is a measurement instruction, not a deletion order. "
            "The tournament reports the model with and without these features "
            "and keeps whichever is reproducibly better."
        ),
        "features": rows,
        "shift_prone": [r["feature"] for r in rows if r["flag"] == SHIFT_PRONE],
        "watch": [r["feature"] for r in rows if r["flag"] == WATCH],
    }


def adversarial_validation(
    train_X: np.ndarray,
    upload_X: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    """Can a classifier tell training rows from uploaded rows?

    Deliberately a plain logistic model on median-imputed, standardised
    columns: the point is a calibrated *diagnostic*, not the strongest possible
    separator. A gradient-booster would find some idiosyncratic split and
    report alarming AUC on samples that are practically interchangeable.

    Never sees any target, and its output never modifies a risk score.
    """
    seed = settings.GLOBAL_SEED if seed is None else seed
    X = np.vstack([train_X, upload_X])
    y = np.concatenate([np.zeros(len(train_X), dtype=int),
                        np.ones(len(upload_X), dtype=int)])
    if len(upload_X) < 20 or len(train_X) < 20:
        return {
            "status": "INSUFFICIENT_ROWS",
            "n_train": int(len(train_X)), "n_upload": int(len(upload_X)),
            "note": "at least 20 rows on each side are needed for a meaningful AUC",
        }

    prep = FoldPreprocessor(mode="linear").fit(X, list(feature_names))
    Xs = prep.transform(X)
    kept = prep.kept_features

    oof = np.zeros(len(y))
    coefs = np.zeros(Xs.shape[1])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(Xs, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        clf.fit(Xs[tr], y[tr])
        oof[va] = clf.predict_proba(Xs[va])[:, 1]
        coefs += np.abs(clf.coef_[0]) / n_splits
    auc = float(roc_auc_score(y, oof))

    order = np.argsort(-coefs)[:15]
    reg = augmented_registry()
    top_shift = []
    for j in order:
        if coefs[j] <= 1e-9:
            continue
        name = kept[j]
        rec = fd.describe(name, reg)
        tr_col = train_X[:, list(feature_names).index(name)]
        up_col = upload_X[:, list(feature_names).index(name)]
        top_shift.append({
            "feature": name,
            "variable_name": rec["variable_name"],
            "abs_coefficient": round(float(coefs[j]), 4),
            "psi": round(_psi(tr_col, up_col), 4),
            "train_missing_rate": round(float(np.mean(np.isnan(tr_col))), 4),
            "upload_missing_rate": round(float(np.mean(np.isnan(up_col))), 4),
            "train_median": _safe_median(tr_col),
            "upload_median": _safe_median(up_col),
        })

    if auc < 0.60:
        band, verdict = "COMPATIBLE", (
            "the two samples are hard to tell apart; the model is operating "
            "inside the distribution it was fitted on")
    elif auc < 0.75:
        band, verdict = "MILD_SHIFT", (
            "some distribution shift is detectable; predictions remain valid "
            "but confidence intervals should be read as wider than reported")
    elif auc < 0.90:
        band, verdict = "MODERATE_SHIFT", (
            "substantial shift; review the top shifted features before acting "
            "on borderline scores")
    else:
        band, verdict = "SEVERE_SHIFT", (
            "the uploaded sample is easily separable from training data; treat "
            "absolute probabilities with caution and prefer the ranking")

    return {
        "status": "OK",
        "adversarial_auc": round(auc, 4),
        "shift_band": band,
        "verdict": verdict,
        "n_train": int(len(train_X)),
        "n_upload": int(len(upload_X)),
        "n_features_compared": len(kept),
        "top_shifted_features": top_shift,
        "guarantees": [
            "this classifier never reads any target column",
            "the mule model is not refitted on uploaded rows",
            "no prediction is modified because of this AUC",
        ],
        "interpretation_scale": {
            "< 0.60": "COMPATIBLE", "0.60-0.75": "MILD_SHIFT",
            "0.75-0.90": "MODERATE_SHIFT", ">= 0.90": "SEVERE_SHIFT",
        },
    }


def _safe_median(col: np.ndarray) -> float | None:
    finite = col[~np.isnan(col)]
    return round(float(np.median(finite)), 4) if len(finite) else None


def shift_prone_ablation(scorer, frame: ModelFrame, features: Sequence[str],
                         shift_prone: Sequence[str], *,
                         n_repeats: int = 3, mode: str = "tree") -> dict[str, Any]:
    """Measure the model with and without the SHIFT_PRONE features."""
    keep = [f for f in features if f not in set(shift_prone)]
    full = harness.run_oof("shield_with_shift_prone", scorer, frame, mode=mode,
                           n_repeats=n_repeats, feature_subset=list(features))
    if not shift_prone or len(keep) < 3:
        return {
            "n_shift_prone": len(shift_prone),
            "with_shift_prone_pr_auc": round(float(np.mean(full.per_repeat_ap())), 5),
            "without_shift_prone_pr_auc": None,
            "decision": "NO_ABLATION_NEEDED",
        }
    trimmed = harness.run_oof("shield_without_shift_prone", scorer, frame, mode=mode,
                              n_repeats=n_repeats, feature_subset=keep)
    a = float(np.mean(full.per_repeat_ap()))
    b = float(np.mean(trimmed.per_repeat_ap()))
    sa, sb = float(np.std(full.per_repeat_ap())), float(np.std(trimmed.per_repeat_ap()))
    decision = ("DROP_SHIFT_PRONE" if b >= a - 1e-4 and sb <= sa else "KEEP_SHIFT_PRONE")
    return {
        "n_shift_prone": len(shift_prone),
        "shift_prone_features": list(shift_prone),
        "with_shift_prone_pr_auc": round(a, 5),
        "with_shift_prone_pr_auc_std": round(sa, 5),
        "without_shift_prone_pr_auc": round(b, 5),
        "without_shift_prone_pr_auc_std": round(sb, 5),
        "decision": decision,
        "rule": (
            "drop the shift-prone features only if removing them does not lose "
            "PR-AUC and does not increase fold variance; otherwise keep them "
            "and disclose the exposure"
        ),
    }
