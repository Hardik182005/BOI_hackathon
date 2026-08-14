"""The resamplers and the definitive table.

Both of these produce numbers that end up in front of a judge, and both have a
failure mode that looks like success: a resampler that quietly moves a real row
into the wrong class, and a table whose status column flatters a model that
should have been rejected. The tests here are aimed at those two, not at
arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from muleguard.cli.final_accuracy_table import (
    BUDGET,
    COLUMNS,
    _at_k,
    _repeat_metrics,
    _status,
)
from muleguard.cli.smote_ablation import (
    random_oversample,
    random_undersample,
    smote,
)


def _imbalanced(n_neg: int = 400, n_pos: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n_neg, 6)), rng.normal(3, 1, (n_pos, 6))])
    y = np.concatenate([np.zeros(n_neg, int), np.ones(n_pos, int)])
    return X, y


# -- SMOTE ------------------------------------------------------------------


def test_smote_only_ever_adds_positives():
    X, y = _imbalanced()
    Xr, yr, n = smote(X, y, ratio=0.25, seed=1)
    assert n > 0
    assert int((yr == 0).sum()) == int((y == 0).sum())
    assert int((yr == 1).sum()) == int((y == 1).sum()) + n


def test_smote_never_alters_an_existing_row():
    """The original matrix must survive untouched at the top of the result."""
    X, y = _imbalanced()
    Xr, yr, _ = smote(X, y, ratio=0.25, seed=1)
    assert np.array_equal(Xr[: len(X)], X, equal_nan=True)
    assert np.array_equal(yr[: len(y)], y)


def test_smote_reaches_the_requested_ratio():
    X, y = _imbalanced()
    _, yr, _ = smote(X, y, ratio=0.25, seed=1)
    assert (yr == 1).sum() / (yr == 0).sum() == pytest.approx(0.25, abs=0.01)


def test_synthetic_rows_lie_between_two_real_positives():
    """Interpolation, not extrapolation: nothing may land outside the convex
    range of the minority class, or SMOTE has invented a new kind of account."""
    X, y = _imbalanced()
    P = X[y == 1]
    Xr, yr, n = smote(X, y, ratio=0.25, seed=3)
    new = Xr[len(X):]
    assert len(new) == n
    assert (new >= P.min(axis=0) - 1e-9).all()
    assert (new <= P.max(axis=0) + 1e-9).all()


def test_a_coordinate_missing_in_a_parent_stays_missing():
    """Averaging a present value with an absent one is not knowledge."""
    X, y = _imbalanced(n_neg=100, n_pos=6)
    X[y == 1, 0] = np.nan  # every positive is missing feature 0
    Xr, _, n = smote(X, y, ratio=0.3, seed=5)
    assert n > 0
    assert np.isnan(Xr[len(X):, 0]).all()


def test_smote_is_a_noop_below_two_positives():
    X, y = _imbalanced(n_neg=50, n_pos=1)
    Xr, yr, n = smote(X, y, ratio=0.5, seed=0)
    assert n == 0 and Xr.shape == X.shape


def test_smote_is_a_noop_when_already_above_the_ratio():
    X, y = _imbalanced(n_neg=10, n_pos=40)
    _, _, n = smote(X, y, ratio=0.25, seed=0)
    assert n == 0


def test_smote_is_deterministic_for_a_seed():
    X, y = _imbalanced()
    a, _, _ = smote(X, y, ratio=0.25, seed=11)
    b, _, _ = smote(X, y, ratio=0.25, seed=11)
    assert np.array_equal(a, b, equal_nan=True)


def test_a_constant_feature_does_not_divide_by_zero():
    X, y = _imbalanced()
    X[:, 2] = 1.0
    Xr, _, n = smote(X, y, ratio=0.25, seed=2)
    assert n > 0 and np.isfinite(Xr[:, 2]).all()


# -- the controls -----------------------------------------------------------


def test_random_oversample_duplicates_only_positives():
    X, y = _imbalanced()
    Xr, yr, n = random_oversample(X, y, ratio=0.25, seed=1)
    new = Xr[len(X):]
    real_pos = {tuple(r) for r in X[y == 1]}
    assert n > 0
    assert all(tuple(r) in real_pos for r in new)
    assert (yr[len(y):] == 1).all()


def test_random_undersample_keeps_every_positive():
    """Discarding a real mule to balance a class ratio would be indefensible."""
    X, y = _imbalanced()
    _, yr, dropped = random_undersample(X, y, ratio=0.25, seed=1)
    assert int((yr == 1).sum()) == int((y == 1).sum())
    assert dropped == int((y == 0).sum()) - int((yr == 0).sum())


def test_random_undersample_is_a_noop_when_already_balanced():
    X, y = _imbalanced(n_neg=10, n_pos=40)
    Xr, _, dropped = random_undersample(X, y, ratio=0.25, seed=0)
    assert dropped == 0 and Xr.shape == X.shape


# -- the table --------------------------------------------------------------


def test_the_table_has_exactly_the_27_spec_columns():
    assert len(COLUMNS) == 27
    assert len(set(COLUMNS)) == 27


def test_at_k_counts_hits_not_ranks():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    rec, prec = _at_k(y, s, 2)
    assert rec == 0.5 and prec == 0.5


def test_at_k_clamps_to_the_available_rows():
    y = np.array([1, 0])
    rec, prec = _at_k(y, np.array([0.9, 0.1]), 100)
    assert rec == 1.0 and prec == 0.5


def test_precision_equals_precision_at_the_budget():
    """The stated, deliberate consequence of a fixed top-K operating point."""
    rng = np.random.default_rng(0)
    y = np.zeros(2000, int)
    y[rng.choice(2000, 20, replace=False)] = 1
    s = rng.random(2000) + 0.4 * y
    m = _repeat_metrics(y, s)
    assert m["precision"] == pytest.approx(m[f"precision_at_top{BUDGET}"])
    assert m["recall"] == pytest.approx(m[f"recall_at_top{BUDGET}"])


def test_a_perfect_ranking_scores_one_where_it_can():
    y = np.zeros(500, int)
    y[:10] = 1
    s = np.linspace(1.0, 0.0, 500)
    m = _repeat_metrics(y, s)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["recall_at_top25"] == pytest.approx(1.0)


def test_mcc_stays_inside_its_range():
    rng = np.random.default_rng(1)
    for seed_shift in range(5):
        y = (rng.random(800) < 0.01).astype(int)
        s = rng.random(800) + seed_shift * 0.0
        assert -1.0 <= _repeat_metrics(y, s)["mcc"] <= 1.0


# -- the status rule, which is the part that can lie ------------------------


CHAMP, CPR, CSD = "champ", 0.769, 0.027


@pytest.mark.parametrize("safe,eligible,pr,expect", [
    (False, True, 0.95, "REJECTED"),   # leaky and better is still rejected
    (True, False, 0.91, "REJECTED"),   # accurate but unservable
    (True, True, 0.75, "CHALLENGER"),  # inside the champion's band
    (True, True, 0.80, "CHALLENGER"),  # above it
    (True, True, 0.60, "FINALIST"),
    (True, True, 0.10, "REJECTED"),
])
def test_status_rule(safe, eligible, pr, expect):
    assert _status("other", pr, safe, eligible, CHAMP, CPR, CSD) == expect


def test_the_champion_is_the_champion_regardless_of_the_band():
    assert _status(CHAMP, CPR, True, True, CHAMP, CPR, CSD) == "CHAMPION"


def test_a_leaky_model_can_never_be_a_challenger():
    """The whole firewall argument collapses if this ever returns CHALLENGER."""
    for pr in (0.94, 0.99, 1.0):
        assert _status("REJECTED_leakage", pr, False, True,
                       CHAMP, CPR, CSD) == "REJECTED"
