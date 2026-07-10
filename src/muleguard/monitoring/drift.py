"""Drift monitoring: PSI / Jensen-Shannon over features, scores, missingness.

A baseline snapshot is frozen from the dev data at build time. New batches
are compared against it; per-feature PSI, score-distribution PSI, missingness
shift and OOD rate feed the drift status endpoint and dashboard.

Thresholds follow common practice: PSI < 0.1 stable, 0.1-0.25 moderate,
> 0.25 alert - stated as convention, not a discovered result.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from muleguard.logging import get_logger

log = get_logger("monitoring.drift")

PSI_MODERATE = 0.10
PSI_ALERT = 0.25


def _hist_proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    finite = x[~np.isnan(x)]
    if len(finite) == 0:
        return np.full(len(edges) + 1, np.nan)
    idx = np.searchsorted(edges, finite, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    props = counts / counts.sum()
    return np.clip(props, 1e-6, None)


def psi(reference: np.ndarray, current: np.ndarray, edges: np.ndarray) -> float:
    p = _hist_proportions(reference, edges)
    q = _hist_proportions(current, edges)
    if np.isnan(p).any() or np.isnan(q).any():
        return float("nan")
    return float(np.sum((p - q) * np.log(p / q)))


def make_baseline(X_dev: np.ndarray, feature_names: list[str],
                  dev_scores: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    baseline: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_rows": int(X_dev.shape[0]),
        "features": {},
        "score_edges": np.quantile(dev_scores, np.linspace(0, 1, n_bins + 1)[1:-1]).tolist(),
        "score_reference": dev_scores.tolist(),
        "missing_rate_mean": float(np.isnan(X_dev).mean()),
    }
    for j, name in enumerate(feature_names):
        col = X_dev[:, j]
        finite = col[~np.isnan(col)]
        if len(finite) < 20 or np.all(finite == finite[0]):
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]))
        baseline["features"][name] = {
            "edges": edges.tolist(),
            "reference_sample": finite[
                np.random.default_rng(0).choice(len(finite), min(2000, len(finite)), replace=False)
            ].tolist(),
            "missing_rate": float(np.isnan(col).mean()),
        }
    return baseline


def drift_report(baseline: dict[str, Any], X_new: np.ndarray,
                 feature_names: list[str], new_scores: np.ndarray) -> dict[str, Any]:
    per_feature = {}
    name_idx = {f: j for j, f in enumerate(feature_names)}
    for name, ref in baseline["features"].items():
        j = name_idx.get(name)
        if j is None:
            continue
        col = X_new[:, j]
        value = psi(np.asarray(ref["reference_sample"]), col, np.asarray(ref["edges"]))
        miss_shift = float(np.isnan(col).mean() - ref["missing_rate"])
        per_feature[name] = {"psi": value, "missing_rate_shift": miss_shift}

    score_psi = psi(
        np.asarray(baseline["score_reference"]), new_scores,
        np.asarray(baseline["score_edges"]),
    )
    psis = [v["psi"] for v in per_feature.values() if not np.isnan(v["psi"])]
    n_alert = sum(1 for p in psis if p > PSI_ALERT)
    n_moderate = sum(1 for p in psis if PSI_MODERATE < p <= PSI_ALERT)
    status = "ALERT" if (n_alert > 0 or (not np.isnan(score_psi) and score_psi > PSI_ALERT)) \
        else ("MODERATE" if (n_moderate > 2 or (not np.isnan(score_psi) and score_psi > PSI_MODERATE))
              else "STABLE")
    worst = sorted(per_feature.items(), key=lambda kv: -(kv[1]["psi"] or 0))[:10]
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "score_psi": score_psi,
        "n_features_alert": n_alert,
        "n_features_moderate": n_moderate,
        "worst_features": [{"feature": k, **v} for k, v in worst],
        "thresholds": {"moderate": PSI_MODERATE, "alert": PSI_ALERT,
                       "note": "conventional PSI bands, not tuned results"},
        "n_rows_scored": int(X_new.shape[0]),
    }
