"""Honest metric suite for rare-event classification.

Primary: PR-AUC (Average Precision). Secondary: ROC-AUC, recall/precision at
alert budgets, recall at fixed FPR, FP per 1,000 legitimate accounts, Brier,
ECE, confusion matrices at thresholds, bootstrap CIs.

No plain accuracy anywhere.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def recall_precision_at_budget(y: np.ndarray, scores: np.ndarray, budget: int) -> dict[str, float]:
    """Alert the top-`budget` scored accounts; measure catch rate and purity."""
    budget = min(budget, len(scores))
    order = np.argsort(-scores, kind="stable")
    top = order[:budget]
    tp = float(y[top].sum())
    total_pos = float(y.sum())
    return {
        "budget": budget,
        "recall": tp / total_pos if total_pos else 0.0,
        "precision": tp / budget if budget else 0.0,
        "true_positives": int(tp),
    }


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_target: float) -> dict[str, float]:
    """Highest recall achievable while FPR (on negatives) stays <= target."""
    neg_scores = scores[y == 0]
    if len(neg_scores) == 0:
        return {"fpr_target": fpr_target, "recall": 0.0, "threshold": float("inf")}
    thr = float(np.quantile(neg_scores, 1.0 - fpr_target))
    alerts = scores >= thr
    tp = float((y[alerts] == 1).sum())
    fp = float((y[alerts] == 0).sum())
    return {
        "fpr_target": fpr_target,
        "threshold": thr,
        "recall": tp / max(float(y.sum()), 1.0),
        "achieved_fpr": fp / max(float((y == 0).sum()), 1.0),
        "fp_per_1000_legit": 1000.0 * fp / max(float((y == 0).sum()), 1.0),
    }


def expected_calibration_error(
    y: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Standard binned ECE plus per-bin detail for calibration curves."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = float(probs[m].mean()), float(y[m].mean()), float(m.mean())
        ece += w * abs(acc - conf)
        bins.append({"bin": b, "mean_predicted": conf, "observed_rate": acc,
                     "count": int(m.sum())})
    return {"ece": ece, "n_bins": n_bins, "bins": bins}


def confusion_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, int | float]:
    pred = scores >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fp_per_1000_legit": 1000.0 * fp / max(tn + fp, 1),
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Stratified bootstrap CI: resample positives and negatives separately so
    every replicate keeps the natural prevalence and >=1 positive."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    stats = []
    for _ in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        stats.append(metric_fn(y[idx], scores[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(metric_fn(y, scores)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def full_metric_report(
    y: np.ndarray,
    scores: np.ndarray,
    budgets: Sequence[int] = (25, 50, 100),
    fpr_targets: Sequence[float] = (0.005, 0.01),
    probs: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    split_label: str = "unspecified",
) -> dict[str, Any]:
    """One-call honest evaluation. `scores` rank accounts; `probs` (if given)
    are calibrated probabilities used for Brier/ECE."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    report: dict[str, Any] = {
        "split": split_label,
        "n": int(len(y)),
        "n_positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": bootstrap_ci(y, scores, average_precision_score, n_boot=n_boot, seed=seed),
        "roc_auc": bootstrap_ci(y, scores, roc_auc_score, n_boot=n_boot, seed=seed),
    }
    dyn_budgets = sorted(set(list(budgets) + [max(1, round(0.01 * len(y)))]))
    report["recall_at_budget"] = [recall_precision_at_budget(y, scores, b) for b in dyn_budgets]
    report["recall_at_fpr"] = [recall_at_fpr(y, scores, f) for f in fpr_targets]

    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        report["brier"] = float(brier_score_loss(y, probs))
        report["calibration"] = expected_calibration_error(y, probs)

    prec, rec, thr = precision_recall_curve(y, scores)
    keep = np.unique(np.linspace(0, len(prec) - 1, min(200, len(prec))).astype(int))
    report["pr_curve"] = {
        "precision": prec[keep].tolist(),
        "recall": rec[keep].tolist(),
    }
    return report
