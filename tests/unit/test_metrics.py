"""Metric correctness against hand-computed values."""
import numpy as np
import pytest

from muleguard.evaluation.metrics import (
    bootstrap_ci,
    confusion_at_threshold,
    expected_calibration_error,
    recall_at_fpr,
    recall_precision_at_budget,
    full_metric_report,
)


def test_recall_precision_at_budget_hand_case():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.05, 0.15, 0.25, 0.35, 0.4])
    r = recall_precision_at_budget(y, scores, budget=3)
    # top-3 scores: 0.9(y=1), 0.8(y=1), 0.4(y=1) -> 3 of 3 positives caught
    assert r["true_positives"] == 3
    assert r["recall"] == 1.0
    assert r["precision"] == 1.0


def test_budget_larger_than_n_is_clamped():
    y = np.array([1, 0])
    scores = np.array([0.9, 0.1])
    r = recall_precision_at_budget(y, scores, budget=10)
    assert r["budget"] == 2


def test_confusion_at_threshold_hand_case():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    c = confusion_at_threshold(y, s, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == 0.5 and c["recall"] == 0.5


def test_recall_at_fpr_respects_negative_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(1000)].astype(int)
    s = np.r_[rng.normal(2, 1, 20), rng.normal(0, 1, 1000)]
    r = recall_at_fpr(y, s, 0.01)
    assert r["achieved_fpr"] <= 0.011  # quantile threshold keeps FPR at target
    assert 0.0 <= r["recall"] <= 1.0


def test_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < probs).astype(int)
    e = expected_calibration_error(y, probs, n_bins=10)
    assert e["ece"] < 0.02


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(30), np.zeros(500)].astype(int)
    s = np.r_[rng.normal(1.5, 1, 30), rng.normal(0, 1, 500)]
    from sklearn.metrics import average_precision_score
    ci = bootstrap_ci(y, s, average_precision_score, n_boot=200, seed=0)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_full_report_never_reports_accuracy():
    y = np.array([0, 1] * 50)
    s = np.linspace(0, 1, 100)
    rpt = full_metric_report(y, s, n_boot=50)
    assert "accuracy" not in str(rpt.keys()).lower()
    assert rpt["pr_auc"]["point"] > 0
    assert rpt["split"] == "unspecified"


def test_probability_bounds_guard():
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    rpt = full_metric_report(y, probs, probs=probs, n_boot=10)
    assert 0.0 <= rpt["brier"] <= 1.0
