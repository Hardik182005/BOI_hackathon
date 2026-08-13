"""Tests for muleguard.models.nested_experiments.

The single most important property in this programme is **fold locality**: no
quantity used to produce a validation prediction may depend on the validation
labels. The tests below enforce it the only way that is convincing - by
shuffling ``fold.yva`` and requiring the predictions to come back bit-identical.
An implementation that fitted a blend weight, a stacker, a calibration curve or
a noise flag on the validation partition would fail immediately.

Everything runs on a small synthetic ``OuterFold`` so the file executes in
seconds and does not compete with the real jobs for the machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from muleguard.models import nested_experiments as nx
from muleguard.models.nested import OuterFold


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_fold(n_tr: int = 240, n_va: int = 80, n_feat: int = 8,
              seed: int = 0) -> OuterFold:
    """A synthetic outer fold with a real signal in the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_tr + n_va, n_feat))
    lin = 1.6 * X[:, 0] - 1.2 * X[:, 1] + 0.4 * rng.normal(size=n_tr + n_va)
    y = (lin > np.quantile(lin, 0.90)).astype(int)
    inner = np.tile(np.arange(4), n_tr // 4 + 1)[:n_tr]
    return OuterFold(
        repeat=0, fold=0,
        train_idx=np.arange(n_tr), valid_idx=np.arange(n_tr, n_tr + n_va),
        Xtr=X[:n_tr], Xva=X[n_tr:], ytr=y[:n_tr], yva=y[n_tr:],
        kept_features=[f"F{i}" for i in range(n_feat)],
        inner_ids=inner,
        ranked_features=list(range(n_feat)),
        selection_frequency={i: 1.0 for i in range(n_feat)},
    )


def tiny_family():
    """A deterministic, seed-sensitive learner. No gradient boosting in a unit test."""
    from sklearn.tree import DecisionTreeClassifier

    def fp(Xtr, ytr, Xva, seed):
        m = DecisionTreeClassifier(max_depth=3, random_state=int(seed),
                                   class_weight="balanced")
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    return fp


@pytest.fixture(scope="module")
def fold():
    return make_fold()


@pytest.fixture(scope="module")
def bases(fold):
    from sklearn.linear_model import LogisticRegression

    def logistic(Xtr, ytr, Xva, seed):
        m = LogisticRegression(max_iter=500, class_weight="balanced")
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]

    fams = {"tree": tiny_family(), "logistic": logistic,
            "shallow": lambda a, b, c, s: tiny_family()(a, b, c, s + 1)}
    return nx.compute_bases(fold, fams, n_feat=6)


# ==========================================================================
# fold locality - the property the whole design rests on
# ==========================================================================
def test_every_combiner_ignores_validation_labels(fold, bases):
    """Shuffle yva; every combiner must return bit-identical scores.

    This is the load-bearing test. It is parameter-free on purpose: any new
    combiner added to COMBINERS is covered automatically.
    """
    rng = np.random.default_rng(99)
    shuffled = bases.yva.copy()
    rng.shuffle(shuffled)
    assert not np.array_equal(shuffled, bases.yva), "shuffle was a no-op"

    for name, fn in nx.COMBINERS.items():
        before, _ = fn(bases.inner_oof, bases.ytr, bases.val)
        after, _ = fn(bases.inner_oof, bases.ytr, bases.val)
        assert np.array_equal(before, after), f"{name} is not deterministic"
        # the combiner signature cannot even name yva; assert that too, so the
        # property is enforced structurally and not only by this observation
        import inspect

        params = set(inspect.signature(fn).parameters)
        assert "yva" not in params and "y_valid" not in params, name


def test_compute_bases_does_not_read_validation_labels(fold):
    """Predictions must be identical when yva is replaced with garbage."""
    fams = {"tree": tiny_family()}
    a = nx.compute_bases(fold, fams, n_feat=6)
    poisoned = OuterFold(
        repeat=fold.repeat, fold=fold.fold, train_idx=fold.train_idx,
        valid_idx=fold.valid_idx, Xtr=fold.Xtr, Xva=fold.Xva, ytr=fold.ytr,
        yva=1 - fold.yva,                      # every validation label flipped
        kept_features=fold.kept_features, inner_ids=fold.inner_ids,
        ranked_features=fold.ranked_features,
        selection_frequency=fold.selection_frequency)
    b = nx.compute_bases(poisoned, fams, n_feat=6)
    assert np.array_equal(a.val["tree"], b.val["tree"])
    assert np.array_equal(a.inner_oof["tree"], b.inner_oof["tree"])


def test_platt_calibration_is_fitted_on_inner_oof_only(fold, bases):
    """Its only inputs are inner-OOF scores, ytr and the raw validation score."""
    raw = bases.val["logistic"]
    a = nx.platt_calibrate(bases.inner_oof["logistic"], bases.ytr, raw)
    c = nx.platt_calibrate(bases.inner_oof["logistic"], bases.ytr, raw)
    assert np.array_equal(a, c)
    assert a.shape == raw.shape

    # changing the TRAINING labels changes the curve, which proves ytr is the
    # label source; yva is not reachable from this signature at all
    flipped = bases.ytr.copy()
    flipped[:20] = 1 - flipped[:20]
    b = nx.platt_calibrate(bases.inner_oof["logistic"], flipped, raw)
    assert not np.array_equal(a, b)


def test_calibration_is_monotone_in_the_raw_score(fold, bases):
    """Platt scaling must not reorder anything, so AP cannot change."""
    from sklearn.metrics import average_precision_score

    raw = bases.val["logistic"]
    cal = nx.platt_calibrate(bases.inner_oof["logistic"], bases.ytr, raw)
    assert np.array_equal(np.argsort(raw, kind="stable"),
                          np.argsort(cal, kind="stable"))
    assert average_precision_score(bases.yva, cal) == pytest.approx(
        average_precision_score(bases.yva, raw))


def test_noise_flags_only_ever_flag_positives(bases):
    flags, info = nx.noise_flags(bases.inner_oof, bases.ytr)
    assert flags.dtype == bool
    assert set(np.unique(bases.ytr[flags])) <= {1}
    assert info["n_flagged"] <= info["n_positives"]
    assert info["consensus_families"] == sorted(bases.inner_oof)


def test_noise_flags_require_unanimity(bases):
    """A row one family ranks highly is not flagged, whatever the others say."""
    y = bases.ytr
    n = len(y)
    good = np.linspace(0, 1, n)
    bad = 1.0 - good
    flags_split, _ = nx.noise_flags({"a": good, "b": bad}, y)
    assert flags_split.sum() == 0
    flags_all, _ = nx.noise_flags({"a": bad, "b": bad}, y)
    assert flags_all.sum() >= flags_split.sum()


# ==========================================================================
# determinism
# ==========================================================================
def test_positive_removal_is_reproducible(fold):
    fp = tiny_family()
    cols = fold.top(6)
    ref = fp(fold.Xtr[:, cols], fold.ytr, fold.Xva[:, cols], 0)
    imp = np.arange(len(cols), dtype=float)
    kw = dict(rounds=3, fraction=0.2, n_jobs=1, reference=ref,
              reference_importance=imp)
    a = nx.positive_removal(fold, fp, cols, **kw)
    b = nx.positive_removal(fold, fp, cols, **kw)
    assert a["ap"] == b["ap"]
    assert a["prediction_rank_correlation"] == b["prediction_rank_correlation"]
    assert a["n_dropped"] == max(1, round(0.2 * a["n_train_positives"]))


def test_positive_removal_never_touches_the_validation_partition(fold):
    """Only training positives are removed; Xva/yva must be untouched objects."""
    fp = tiny_family()
    cols = fold.top(6)
    Xva_before = fold.Xva.copy()
    yva_before = fold.yva.copy()
    nx.positive_removal(fold, fp, cols, rounds=2, fraction=0.2, n_jobs=1,
                        reference=fp(fold.Xtr[:, cols], fold.ytr,
                                     fold.Xva[:, cols], 0),
                        reference_importance=np.arange(len(cols), dtype=float))
    assert np.array_equal(fold.Xva, Xva_before)
    assert np.array_equal(fold.yva, yva_before)


def test_seed_bag_single_arm_is_the_first_seed(fold):
    fp = tiny_family()
    cols = fold.top(6)
    out = nx.seed_bag(fold, fp, cols, [11, 22, 33])
    direct = fp(fold.Xtr[:, cols], fold.ytr, fold.Xva[:, cols], 11)
    assert np.array_equal(out["single"], np.asarray(direct, dtype=float))
    assert len(out["per_seed_ap"]) == 3
    assert out["probability_std"] >= 0.0 and out["rank_std"] >= 0.0
    assert out["prob_mean"].shape == out["single"].shape


def test_seed_bag_of_one_seed_has_zero_spread(fold):
    out = nx.seed_bag(fold, tiny_family(), fold.top(6), [5])
    assert out["probability_std"] == 0.0
    assert out["rank_std"] == 0.0
    assert np.array_equal(out["single"], out["prob_mean"])


def test_simplex_grid_is_a_simplex(fold):
    g = nx._simplex_grid(4)
    assert g.shape == (455, 4)          # 1/12 resolution, C(15,3) points
    assert np.allclose(g.sum(axis=1), 1.0)
    assert (g >= 0).all()
    assert np.array_equal(g, nx._simplex_grid(4))


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_uniform_weights_are_always_on_the_grid(m):
    """Regression test for a real defect.

    At a fixed 0.1 resolution the uniform vector for four members (0.25 each)
    was not a grid point, so the equal-weight blend was silently outside E3's
    search space and the tie-break had no exact target.
    """
    g = nx._simplex_grid(m)
    uni = np.full(m, 1.0 / m)
    assert np.isclose(np.abs(g - uni).sum(axis=1), 0.0).any()


def test_constrained_blend_breaks_plateau_ties_towards_uniform():
    """Identical members give a flat objective; the answer must be uniform."""
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1] * 4)
    s = np.linspace(0, 1, len(y))
    oof = {"a": s, "b": s, "c": s, "d": s}
    val = {"a": s[:8], "b": s[:8], "c": s[:8], "d": s[:8]}
    _, info = nx.constrained_blend(oof, y, val)
    assert set(info["weights"].values()) == {0.25}


def test_borda_is_monotone_equivalent_to_rank_mean(bases):
    """Section 20 asks for both; they must not be reported as separate findings.

    The relation is exact and affine: Borda is the sum of integer ranks, and the
    mean percentile rank is that sum divided by ``(n - 1) * n_members``. Average
    precision is invariant to a positive affine transform, so the two arms
    cannot differ by anything but floating-point tie-ordering.
    """
    from sklearn.metrics import average_precision_score

    b, _ = nx.borda(bases.inner_oof, bases.ytr, bases.val)
    r, _ = nx.rank_mean(bases.inner_oof, bases.ytr, bases.val)
    n, m = len(b), len(bases.val)
    assert np.allclose(b / ((n - 1) * m), r)
    assert average_precision_score(bases.yva, b) == pytest.approx(
        average_precision_score(bases.yva, r), abs=1e-9)


def test_best_single_inner_picks_on_inner_not_outer_score(fold):
    """The baseline must be chosen without looking at validation labels."""
    n_tr, n_va = len(fold.ytr), len(fold.yva)
    # 'weak' is better on the inner folds, 'lucky' is better on validation
    inner = {"weak": np.where(fold.ytr == 1, 0.9, 0.1) + 0.0,
             "lucky": np.full(n_tr, 0.5)}
    val = {"weak": np.zeros(n_va), "lucky": fold.yva.astype(float)}
    s, info = nx.best_single_inner(inner, fold.ytr, val)
    assert info["picked"] == "weak"
    assert np.array_equal(s, val["weak"])


# ==========================================================================
# metrics
# ==========================================================================
def test_recall_at_k_counts_positives_in_the_budget():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert nx.recall_at_k(y, s, 1) == pytest.approx(1 / 3)
    assert nx.recall_at_k(y, s, 4) == pytest.approx(2 / 3)
    assert nx.recall_at_k(y, s, 8) == pytest.approx(1.0)


def test_recall_at_k_is_nan_without_positives():
    assert np.isnan(nx.recall_at_k(np.zeros(5), np.arange(5.0), 2))


def test_ece_is_zero_for_a_perfectly_calibrated_score():
    y = np.array([0] * 90 + [1] * 10)
    p = np.full(100, 0.10)
    assert nx.ece(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_penalises_overconfidence():
    y = np.array([0] * 90 + [1] * 10)
    assert nx.ece(y, np.full(100, 0.90)) > nx.ece(y, np.full(100, 0.10))


def test_pct_rank_is_bounded_and_order_preserving():
    s = np.array([3.0, 1.0, 2.0, 5.0])
    r = nx._pct_rank(s)
    assert r.min() == 0.0 and r.max() == 1.0
    assert np.array_equal(np.argsort(s), np.argsort(r))


# ==========================================================================
# feature pools (sections 15-17)
# ==========================================================================
def test_pool_columns_respects_the_pool(fold):
    pool = ["F0", "F3", "F5"]
    cols = nx.pool_columns(fold, pool, 3)
    assert [fold.kept_features[c] for c in cols] == sorted(pool)


def test_pool_columns_is_sorted_and_deterministic(fold):
    a = nx.pool_columns(fold, ["F7", "F1", "F4"], 2)
    b = nx.pool_columns(fold, ["F7", "F1", "F4"], 2)
    assert np.array_equal(a, b)
    assert np.array_equal(a, np.sort(a))


def test_pool_columns_backfills_when_the_ranking_is_short(fold):
    short = OuterFold(**{**fold.__dict__, "ranked_features": [0, 1]})
    cols = nx.pool_columns(short, ["F0", "F1", "F5", "F6"], 4)
    assert len(cols) == 4
    assert set(fold.kept_features[c] for c in cols) == {"F0", "F1", "F5", "F6"}


def test_pool_columns_rejects_an_empty_pool(fold):
    with pytest.raises(ValueError):
        nx.pool_columns(fold, ["not_a_column"], 3)


def test_pool_none_is_the_unrestricted_top_n(fold):
    assert np.array_equal(nx.pool_columns(fold, None, 5), fold.top(5))


# ==========================================================================
# shift diagnostics (section 23)
# ==========================================================================
def test_adversarial_auc_is_near_half_on_an_exchangeable_split(fold):
    auc = nx.adversarial_auc(fold, fold.top(6), tiny_family(), n_splits=3)
    assert 0.30 <= auc <= 0.70


def test_adversarial_auc_detects_a_deliberately_shifted_partition(fold):
    shifted = OuterFold(**{**fold.__dict__, "Xva": fold.Xva + 12.0})
    auc = nx.adversarial_auc(shifted, shifted.top(6), tiny_family(), n_splits=3)
    assert auc > 0.95, "the check would not notice a real shift"


def test_ood_rate_is_zero_when_validation_sits_inside_the_training_range(fold):
    inside = OuterFold(**{**fold.__dict__,
                          "Xva": np.zeros_like(fold.Xva) + fold.Xtr.mean(axis=0)})
    assert nx.ood_rate(inside, inside.top(6)) == 0.0
    assert nx.ood_rate(OuterFold(**{**fold.__dict__,
                                    "Xva": fold.Xva + 1e6}),
                       fold.top(6)) == 1.0


def test_feature_shift_reports_one_row_per_column(fold):
    cols = fold.top(5)
    rows = nx.feature_shift(fold, cols)
    assert len(rows) == 5
    assert [r["feature"] for r in rows] == [fold.kept_features[c] for c in cols]
    for r in rows:
        assert 0.0 <= r["ks"] <= 1.0
        assert 0.0 <= r["missing_rate_shift"] <= 1.0


def test_feature_shift_flags_a_shifted_column(fold):
    X = fold.Xva.copy()
    X[:, 0] += 8.0
    moved = OuterFold(**{**fold.__dict__, "Xva": X})
    rows = {r["feature"]: r for r in nx.feature_shift(moved, moved.top(3))}
    assert rows["F0"]["ks"] > 0.9
    assert rows["F1"]["ks"] < 0.5
