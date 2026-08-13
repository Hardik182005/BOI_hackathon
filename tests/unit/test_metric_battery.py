"""Tests for the §24 - §27 metric battery.

The tests that matter most here are the ones in section 2. Every other property
in this module would still hold if the intervals were computed by resampling the
wrong thing - repeats instead of accounts - and the resulting numbers would look
entirely plausible. So the axis is pinned two ways: by constructing input where a
repeat-axis bootstrap must produce zero width and an account-axis bootstrap
cannot, and by reimplementing the stratified resample independently in the test
and demanding exact agreement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, fbeta_score, matthews_corrcoef,
                             roc_auc_score)

from muleguard import settings
from muleguard.models import metric_battery as mb

ART = settings.METRICS_DIR / "metric_battery.json"
OOF = settings.PREDICTIONS_DIR / "oof_v2.parquet"
TOURNEY = settings.METRICS_DIR / "tournament_v2.json"


@pytest.fixture(scope="module")
def toy():
    """1,200 accounts, 30 positives, three repeats, no ties.

    Deliberately not degenerate: the positives are spread through the ranking so
    that recall at a small budget is neither 0 nor 1 and the intervals have room
    to move.
    """
    rng = np.random.default_rng(11)
    n, n_pos = 1200, 30
    y = np.zeros(n, dtype=int)
    y[rng.permutation(n)[:n_pos]] = 1
    S = np.vstack([rng.normal(0, 1, n) + y * 2.0 for _ in range(3)])
    S += rng.normal(0, 1e-9, S.shape)          # break exact ties
    return y, S


# ==========================================================================
# 1. the ranking core must agree with the reference implementation
# ==========================================================================
def test_ap_and_roc_match_sklearn_including_heavy_ties():
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(120):
        n = int(rng.integers(30, 400))
        y = (rng.random(n) < rng.uniform(0.01, 0.4)).astype(int)
        if y.sum() in (0, n):
            continue
        if trial % 3 == 0:
            s = rng.integers(0, 4, size=n).astype(float)     # many ties
        elif trial % 3 == 1:
            s = np.full(n, 0.5)                              # all tied
        else:
            s = rng.random(n)
        ap, roc = mb.ap_roc_from_sorted(*mb._ordered(y, s))
        worst = max(worst, abs(ap - average_precision_score(y, s)),
                    abs(roc - roc_auc_score(y, s)))
    assert worst < 1e-12, f"fast AP/ROC disagrees with sklearn by {worst}"


def test_ap_and_roc_are_nan_when_one_class_is_absent():
    y = np.zeros(50, dtype=int)
    s = np.linspace(0, 1, 50)
    ap, roc = mb.ap_roc_from_sorted(*mb._ordered(y, s))
    assert np.isnan(ap) and np.isnan(roc)


def test_fused_bundle_equals_the_independent_closures(toy):
    """The fast path shares sorts; it must not therefore compute anything else."""
    y, S = toy
    budgets = [10, 25, 50, 100]
    slow = mb.bootstrap_intervals(y, S, mb.ranking_interval_statistics(budgets),
                                  n_boot=40, seed=3, scheme="stratified")
    fast = mb.bootstrap_intervals(y, S, mb.ranking_bundle(budgets),
                                  n_boot=40, seed=3, scheme="stratified")
    for name in slow:
        for field in ("point", "ci_low", "ci_high"):
            assert slow[name][field] == pytest.approx(fast[name][field], abs=1e-12), (
                f"{name}.{field} differs between the fused and unfused paths")


# ==========================================================================
# 2. the interval must be computed over ACCOUNTS, not over repeats
# ==========================================================================
def test_interval_is_over_accounts_not_over_repeats(toy):
    """Three identical repeats: zero repeat-spread, but a real account interval.

    This is the test that catches an interval computed on the wrong axis. With
    the repeats made identical, any scheme that resamples the repeat axis - or
    that reports the standard deviation across repeats dressed up as an interval
    - must produce a width of exactly zero. An interval over accounts cannot:
    30 positives in 1,200 rows leave plenty of room for average precision to
    move when the accounts themselves are resampled.
    """
    y, S = toy
    S = np.vstack([S[0], S[0], S[0]])

    point = mb.ranking_point_estimates(y, S)
    # Not exactly 0.0: np.std of three identical floats leaves a rounding crumb.
    assert point["pr_auc"]["std"] < 1e-15, "identical repeats must have zero spread"

    out = mb.bootstrap_intervals(y, S, mb.ranking_bundle([25, 50]),
                                 n_boot=300, seed=5, scheme="stratified")
    pr = out["pr_auc"]
    assert pr["resample_unit"] == "account"
    assert pr["ci_width"] > 0.02, (
        "PR-AUC interval collapsed although the accounts still vary - the "
        f"bootstrap is resampling the wrong axis (width {pr['ci_width']})")
    assert pr["ci_low"] < pr["point"] < pr["ci_high"]
    assert pr["n_boot_effective"] == 300


def test_stratified_interval_matches_an_independent_reimplementation(toy):
    """Reimplement the resample in the test and demand exact agreement.

    The reimplementation below draws positives and negatives separately, in that
    order, from ``default_rng(seed)``, and evaluates every repeat on the same
    account sample before averaging. If the module resampled repeats, resampled
    without stratifying, averaged before resampling, or consumed the generator
    in a different order, these quantiles would not match to 1e-12.
    """
    y, S = toy
    n_boot, seed = 200, 17
    out = mb.bootstrap_intervals(y, S, mb.ranking_bundle([25]),
                                 n_boot=n_boot, seed=seed, scheme="stratified")

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        q = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        vals.append(float(np.mean([average_precision_score(y[idx], S[r][idx])
                                   for r in range(S.shape[0])])))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    assert out["pr_auc"]["ci_low"] == pytest.approx(lo, abs=1e-12)
    assert out["pr_auc"]["ci_high"] == pytest.approx(hi, abs=1e-12)
    assert out["pr_auc"]["point"] == pytest.approx(
        float(np.mean([average_precision_score(y, S[r]) for r in range(S.shape[0])])),
        abs=1e-12)


def test_every_replicate_keeps_the_positive_count_under_stratification(toy):
    y, _ = toy
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = mb._draw(rng, y, pos_idx, neg_idx, "stratified")
        assert int(y[idx].sum()) == len(pos_idx)
        assert len(idx) == len(y)


def test_resampling_accounts_lets_the_positive_count_move(toy):
    y, _ = toy
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(2)
    counts = {int(y[mb._draw(rng, y, pos_idx, neg_idx, "resample_accounts")].sum())
              for _ in range(100)}
    assert len(counts) > 1, "the account scheme must let the mule count vary"


def test_account_scheme_is_the_wider_of_the_two(toy):
    """Not a tautology: it is the claim the documentation makes to a judge."""
    y, S = toy
    kw = dict(n_boot=400, seed=9)
    strat = mb.bootstrap_intervals(y, S, mb.ranking_bundle([25]),
                                   scheme="stratified", **kw)["pr_auc"]
    acct = mb.bootstrap_intervals(y, S, mb.ranking_bundle([25]),
                                  scheme="resample_accounts", **kw)["pr_auc"]
    assert acct["ci_width"] >= strat["ci_width"]


def test_a_transposed_score_matrix_is_rejected(toy):
    y, S = toy
    with pytest.raises(ValueError, match="axis 1 must be accounts"):
        mb.ranking_point_estimates(y, S.T)


def test_unknown_scheme_is_rejected(toy):
    y, S = toy
    with pytest.raises(ValueError, match="unknown scheme"):
        mb.bootstrap_intervals(y, S, mb.ranking_bundle([25]), scheme="jackknife")


# ==========================================================================
# 3. degenerate inputs are reported, never silently turned into zeros
# ==========================================================================
def test_all_negative_labels_report_undefined_rather_than_zero():
    y = np.zeros(400, dtype=int)
    S = np.vstack([np.linspace(0, 1, 400)] * 2)

    sup = mb.label_support(y)
    assert sup["ranking_metrics_defined"] is False
    assert sup["degenerate"] is True
    assert "no positives" in sup["why_not"]
    assert sup["recall_resolution_pct_points"] is None

    point = mb.ranking_point_estimates(y, S)
    assert point["defined"] is False
    assert point["pr_auc"] is None, "an all-negative split must not report 0.0"

    out = mb.bootstrap_intervals(y, S, mb.ranking_bundle([25]), n_boot=10)
    assert out["pr_auc"]["point"] is None
    assert out["pr_auc"]["ci_low"] is None
    assert out["pr_auc"]["n_boot_effective"] == 0

    doc = mb.build_battery(y=y, S=S, n_boot=10, protocol="FLAT")
    assert doc["ranking"]["defined"] is False
    assert doc["banking_workload"]["defined"] is False
    assert doc["support"]["n_positives"] == 0


def test_all_positive_labels_report_undefined():
    y = np.ones(200, dtype=int)
    S = np.linspace(0, 1, 200).reshape(1, -1)
    sup = mb.label_support(y)
    assert sup["ranking_metrics_defined"] is False
    assert "no negatives" in sup["why_not"]
    assert mb.ranking_point_estimates(y, S)["pr_auc"] is None


def test_a_single_positive_flags_a_degenerate_interval_instead_of_certainty():
    """One mule, ranked first: the stratified interval has nothing to vary.

    The correct behaviour is to report the collapse, not to present a zero-width
    interval as precision. The account-resampling scheme still moves, because a
    replicate can draw the single mule zero or several times, so the two schemes
    together tell the reader what is really known.
    """
    n = 500
    y = np.zeros(n, dtype=int)
    y[0] = 1
    s = np.linspace(1.0, 0.0, n)               # the one positive is ranked first
    S = s.reshape(1, -1)

    sup = mb.label_support(y)
    assert sup["n_positives"] == 1
    assert sup["degenerate"] is True
    assert sup["recall_resolution_pct_points"] == 100.0

    strat = mb.bootstrap_intervals(y, S, mb.ranking_bundle([10]),
                                   n_boot=200, seed=4, scheme="stratified")
    assert strat["pr_auc"]["point"] == pytest.approx(1.0)
    assert strat["pr_auc"]["ci_width"] == 0.0
    assert strat["pr_auc"]["degenerate"] is True, (
        "a zero-width interval must be flagged, not presented as certainty")

    acct = mb.bootstrap_intervals(y, S, mb.ranking_bundle([10]),
                                  n_boot=200, seed=4, scheme="resample_accounts")
    assert acct["pr_auc"]["n_boot_effective"] < 200, (
        "replicates with no mule at all must be dropped, not scored")

    doc = mb.build_battery(y=y, S=S, n_boot=50, protocol="FLAT")
    assert doc["support"]["degenerate"] is True
    assert doc["stability"]["rank_stability"]["measurable"] is False


def test_recall_ceiling_is_reported_for_budgets_below_the_positive_count():
    y = np.zeros(600, dtype=int)
    y[:64] = 1
    S = np.random.default_rng(0).random((1, 600))
    pts = mb.budget_point_estimates(y, S, [10, 100])
    assert pts[10]["recall_ceiling_at_this_budget"] == pytest.approx(10 / 64)
    assert pts[100]["recall_ceiling_at_this_budget"] == 1.0


def test_budgets_larger_than_the_split_are_dropped():
    grid = [row["budget"] for row in mb.resolve_budgets(40)]
    assert 10 in grid and 18 in grid and 25 in grid
    assert max(grid) <= 40
    assert 100 not in grid


# ==========================================================================
# 4. classification metrics at a threshold
# ==========================================================================
def test_threshold_metrics_agree_with_sklearn(toy):
    y, S = toy
    p = 1.0 / (1.0 + np.exp(-S[0]))
    thr = float(np.quantile(p, 0.96))
    m = mb.threshold_metrics(y, p, thr)
    pred = (p >= thr).astype(int)
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["f2"] == pytest.approx(fbeta_score(y, pred, beta=2))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["accuracy"] == pytest.approx((pred == y).mean())
    assert m["tp"] + m["fp"] == m["alerts"]
    assert m["number_needed_to_review_per_mule"] == pytest.approx(
        m["alerts"] / m["tp"])


def test_accuracy_is_uninformative_at_low_prevalence():
    """The reason §24 asks for balanced accuracy beside accuracy."""
    y = np.zeros(7264, dtype=int)
    y[:64] = 1
    p = np.zeros(7264)                          # alert on nothing
    m = mb.threshold_metrics(y, p, 0.5)
    assert m["alerts"] == 0
    assert m["accuracy"] > 0.99
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["recall"] == 0.0
    assert m["precision"] is None               # no alerts: undefined, not zero


def test_f2_favours_recall_more_than_f1_does():
    p = 0.2
    r = 0.9
    assert mb._fbeta(p, r, 2.0) > mb._fbeta(p, r, 1.0)
    assert mb._fbeta(0.9, 0.2, 2.0) < mb._fbeta(0.9, 0.2, 1.0)


# ==========================================================================
# 5. probability quality and calibration (§26)
# ==========================================================================
def test_brier_is_compared_against_the_base_rate_predictor():
    y = np.zeros(7264, dtype=int)
    y[:64] = 1
    base = float(y.mean())
    q = mb.probability_quality(y, np.full(len(y), base))
    assert q["brier"] == pytest.approx(q["brier_base_rate_reference"])
    assert q["brier_skill_vs_base_rate"] == pytest.approx(0.0, abs=1e-12), (
        "a base-rate predictor must score exactly zero skill")


def test_quantile_binning_spreads_mass_that_uniform_binning_piles_into_one_bin():
    y = np.zeros(7264, dtype=int)
    y[:64] = 1
    rng = np.random.default_rng(3)
    p = np.clip(rng.beta(0.4, 200, size=len(y)) + y * 0.05, 0, 1)
    uni = mb.calibration_error(y, p, strategy="uniform")
    qua = mb.calibration_error(y, p, strategy="quantile")
    assert max(b["count"] for b in uni["bins"]) > 0.9 * len(y), (
        "at this prevalence equal-width binning should be dominated by one bin")
    assert max(b["count"] for b in qua["bins"]) < 0.3 * len(y)
    assert len(qua["bins"]) >= len(uni["bins"])


def test_calibration_comparison_reports_three_variants_and_a_selection(toy):
    y, S = toy
    raw = 1.0 / (1.0 + np.exp(-S.mean(axis=0)))
    cmp_ = mb.calibration_comparison(y, raw, seed=42)
    assert set(cmp_["variants"]) == {"uncalibrated", "platt", "isotonic"}
    for v in cmp_["variants"].values():
        assert v["brier"] is not None and v["log_loss"] is not None
        assert v["ece_quantile"] is not None
    assert cmp_["selection"]["winner"] in ("platt", "isotonic")
    assert "no locked-test label" in cmp_["fitted_on"]


# ==========================================================================
# 6. threshold policy (§27)
# ==========================================================================
def test_threshold_candidates_are_advisory_and_never_applied(toy):
    y, S = toy
    p = 1.0 / (1.0 + np.exp(-S.mean(axis=0)))
    rows = mb.threshold_candidates(y, p)
    assert rows, "no candidate thresholds were produced"
    rules = {r["rule"] for r in rows}
    assert "max_f1" in rules and "max_f2" in rules
    assert any(r.startswith("fixed_alert_budget") for r in rules)
    assert any(r.startswith("fixed_fpr") for r in rules)
    for r in rows:
        assert r["applied"] is False
        assert r["status"] == mb.ADVISORY_STATUS


def test_max_f1_and_max_f2_rows_each_maximise_their_own_objective(toy):
    y, S = toy
    p = 1.0 / (1.0 + np.exp(-S.mean(axis=0)))
    rows = {r["rule"]: r for r in mb.threshold_candidates(y, p)}
    assert rows["max_f1"]["f1"] >= rows["max_f2"]["f1"] - 1e-12
    assert rows["max_f2"]["f2"] >= rows["max_f1"]["f2"] - 1e-12


def test_frozen_tiers_are_located_read_only(toy):
    y, S = toy
    p = 1.0 / (1.0 + np.exp(-S.mean(axis=0)))
    frozen = {"critical_risk": 0.9, "urgent_risk": 0.5, "standard_risk": 0.1,
              "policy_version": "1.0"}
    rows = mb.locate_thresholds(y, p, frozen)
    assert [r["tier"] for r in rows] == ["CRITICAL_REVIEW", "URGENT_REVIEW",
                                         "STANDARD_REVIEW"]
    for r in rows:
        assert r["frozen"] is True and r["modified_by_this_run"] is False
        assert r["policy_version"] == "1.0"
    alerts = [r["alerts"] for r in rows]
    assert alerts == sorted(alerts), "a lower threshold must alert on at least as many"


def test_holdout_protocol_withholds_the_threshold_search(toy):
    y, S = toy
    p = 1.0 / (1.0 + np.exp(-S.mean(axis=0)))
    doc = mb.build_battery(y=y, S=S, calibrated=p, n_boot=20,
                           frozen_thresholds={"critical_risk": 0.9,
                                              "policy_version": "1.0"},
                           protocol="HOLDOUT_REFERENCE",
                           allow_threshold_search=False)
    assert doc["threshold_candidates"] is None
    assert "forbids" in doc["threshold_candidates_withheld"]
    assert doc["at_frozen_thresholds"], "evaluating a frozen constant is still allowed"


# ==========================================================================
# 7. aggregation reconciliation and stability
# ==========================================================================
def test_reconciliation_names_the_per_repeat_mean_as_the_headline(toy):
    y, S = toy
    rec = mb.aggregation_reconciliation(y, S)
    assert rec["headline_choice"] == "mean_of_per_repeat"
    assert rec["mean_of_per_repeat"] == pytest.approx(
        float(np.mean([average_precision_score(y, s) for s in S])))
    assert rec["ap_of_mean_score"] >= rec["mean_of_per_repeat"], (
        "averaging independent fits should not rank worse than the average fit")
    assert rec["max_gap_vs_headline"] >= 0.0


def test_fold_level_spread_is_wider_than_the_repeat_level_spread(toy):
    y, S = toy
    rng = np.random.default_rng(6)
    fold_ids = np.vstack([rng.integers(0, 5, size=S.shape[1]) for _ in range(3)])
    fl = mb.fold_level_ap(y, S, fold_ids)
    assert fl["n_folds"] == 15
    rep_std = mb.ranking_point_estimates(y, S)["pr_auc"]["std"]
    assert fl["pr_auc_std"] > rep_std, (
        "a fold holds a fifth of the positives, so its estimate must be noisier")


def test_fold_ids_with_the_wrong_shape_are_rejected(toy):
    y, S = toy
    with pytest.raises(ValueError, match="does not match"):
        mb.fold_level_ap(y, S, np.zeros((2, S.shape[1]), dtype=int))


def test_rank_stability_needs_more_than_one_repeat(toy):
    y, S = toy
    assert mb.rank_stability(S[:1])["measurable"] is False
    out = mb.rank_stability(S, budgets=[25])
    assert out["measurable"] is True
    assert 0.0 <= out["top_budget_jaccard"]["top_25_jaccard"] <= 1.0


def test_reported_mean_can_never_beat_every_repeat():
    """Guards the arithmetic that hid a bug in an earlier capacity report."""
    sp = mb._spread([0.1, 0.2, 0.3])
    assert sp["min"] <= sp["mean"] <= sp["max"]
    assert sp["n_repeats"] == 3


# ==========================================================================
# 8. the assembled document
# ==========================================================================
def test_build_battery_covers_the_sections_the_spec_lists(toy):
    y, S = toy
    p = 1.0 / (1.0 + np.exp(-S.mean(axis=0)))
    doc = mb.build_battery(
        y=y, S=S, calibrated=p, n_boot=60, protocol="FLAT",
        frozen_thresholds={"critical_risk": 0.95, "urgent_risk": 0.6,
                           "standard_risk": 0.2, "policy_version": "1.0"})
    for key in ("ranking", "analyst_budgets", "banking_workload",
                "probability_quality", "calibration_comparison",
                "at_frozen_thresholds", "threshold_candidates", "intervals",
                "interval_method", "stability", "aggregation_reconciliation"):
        assert doc[key] is not None, f"missing section {key}"
    for scheme in mb.SCHEMES:
        block = doc["intervals"][scheme]
        assert "pr_auc" in block and "roc_auc" in block
        assert any(k.startswith("f1_at_") for k in block), "§25 requires an F1 CI"
        assert any(k.startswith("mcc_at_") for k in block), "§25 requires an MCC CI"
        assert any(k.startswith("recall_at_top_") for k in block)
        assert any(k.startswith("precision_at_top_") for k in block)
    assert doc["intervals"]["stratified"]["pr_auc"]["computed_on"] == \
        "per_repeat_score_matrix"
    assert doc["intervals"]["stratified"]["f1_at_critical"]["computed_on"] == \
        "calibrated_probability_vector"
    assert doc["frozen_policy"]["modified_by_this_run"] is False
    assert "OOD_REVIEW" in doc["frozen_policy"]["tiers_not_score_based"]


def test_battery_without_a_calibrated_vector_refuses_to_substitute_raw_scores(toy):
    y, S = toy
    doc = mb.build_battery(y=y, S=S, calibrated=None, n_boot=20, protocol="FLAT")
    assert doc["probability_quality"] is None
    assert doc["at_frozen_thresholds"] is None
    assert "NOT substituted" in doc["calibration_note"]


def test_pr_auc_interval_brackets_the_reported_point_estimate(toy):
    y, S = toy
    doc = mb.build_battery(y=y, S=S, n_boot=200, seed=8, protocol="FLAT")
    point = doc["ranking"]["pr_auc"]["mean"]
    for scheme in mb.SCHEMES:
        iv = doc["intervals"][scheme]["pr_auc"]
        assert iv["point"] == pytest.approx(point, abs=1e-12), (
            f"the {scheme} interval is attached to a different point estimate "
            "than the one reported beside it")
        assert iv["ci_low"] <= point <= iv["ci_high"]


# ==========================================================================
# 9. CLI guards
# ==========================================================================
def test_a_holdout_source_forces_the_reference_protocol():
    from muleguard.cli import metric_battery as cli
    for name in ("holdout_predictions.parquet", "locked_test_predictions.parquet",
                 "final_locked_test_predictions.parquet"):
        protocol, allow = cli._protocol_for(Path(name), "FLAT")
        assert protocol == "HOLDOUT_REFERENCE"
        assert allow is False, "no threshold may be searched against the holdout"
    assert cli._protocol_for(Path("oof_v2.parquet"), None) == ("FLAT", True)


def test_the_headline_run_is_the_shipped_family_not_the_alphabetical_first(monkeypatch):
    """Within a protocol tier the shipped family decides, not string order.

    `sorted()` alone nominated NESTED:catboost over NESTED:xgboost because "c"
    precedes "x", which made the document's headline figure describe a model the
    product does not serve.
    """
    from muleguard.cli import metric_battery as mb

    monkeypatch.setattr(mb, "_champion_family", lambda: "xgboost")
    runs = {"FLAT:xgboost_top_120": {}, "NESTED:catboost": {},
            "NESTED:xgboost": {}, "HOLDOUT_REFERENCE:retired": {}}
    assert mb._primary_key(runs) == "NESTED:xgboost"


def test_the_headline_still_prefers_nested_over_flat_for_the_shipped_family(monkeypatch):
    """Preferring the shipped family must not promote a flat run over a nested one."""
    from muleguard.cli import metric_battery as mb

    monkeypatch.setattr(mb, "_champion_family", lambda: "xgboost")
    assert mb._primary_key({"FLAT:xgboost_top_120": {},
                            "NESTED:catboost": {}}) == "NESTED:catboost"


def test_a_missing_champion_record_falls_back_to_order(monkeypatch):
    """No champion on disk is not a reason to return nothing."""
    from muleguard.cli import metric_battery as mb

    monkeypatch.setattr(mb, "_champion_family", lambda: None)
    assert mb._primary_key({"NESTED:catboost": {}, "NESTED:xgboost": {}}) == "NESTED:catboost"


def test_an_inferred_nested_label_follows_the_repeats_not_the_filename():
    """The finished run overwrote the preliminary one at the same path.

    Labelling by filename alone would file a completed 3-repeat nested run as
    NESTED_PRELIMINARY, which `_interpretation` marks unusable for selection -
    the primary protocol would silently disqualify itself.
    """
    from muleguard.cli import metric_battery as cli
    nested = Path("nested_oof.parquet")
    assert cli._protocol_for(nested, None)[0] == "NESTED_PRELIMINARY", \
        "with nothing known about the store, the cautious label stands"
    assert cli._protocol_for(nested, None, 1)[0] == "NESTED_PRELIMINARY"
    assert cli._protocol_for(nested, None, cli.FULL_NESTED_REPEATS)[0] == "NESTED"
    # An explicit flag still wins over anything inferred.
    assert cli._protocol_for(nested, "NESTED_PRELIMINARY", 3)[0] == \
        "NESTED_PRELIMINARY"


def test_nested_outranks_flat_when_choosing_the_headline_run():
    from muleguard.cli import metric_battery as cli
    runs = {"FLAT:xgboost_top_120": {}, "HOLDOUT_REFERENCE:x": {},
            "NESTED_PRELIMINARY:xgboost": {}}
    assert cli._primary_key(runs) == "FLAT:xgboost_top_120"
    runs["NESTED:xgboost"] = {}
    assert cli._primary_key(runs) == "NESTED:xgboost"


def test_the_holdout_interpretation_is_never_marked_selectable():
    from muleguard.cli import metric_battery as cli
    assert cli._interpretation("HOLDOUT_REFERENCE")["usable_for_selection"] is False
    assert cli._interpretation("NESTED_PRELIMINARY")["usable_for_selection"] is False
    assert cli._interpretation("NESTED")["usable_for_selection"] is True


# ==========================================================================
# 10. artifact-backed checks (skipped when the artifacts are absent)
# ==========================================================================
def test_flat_champion_pr_auc_reproduces_the_tournament_figure():
    if not (OOF.exists() and TOURNEY.exists()):
        pytest.skip("stored OOF predictions or tournament artifact not present")
    import polars as pl

    tourney = json.loads(TOURNEY.read_text())
    model = "xgboost_top_120"
    if model not in tourney.get("models", {}):
        pytest.skip(f"{model} not in the tournament artifact")
    sub = pl.read_parquet(OOF).filter(pl.col("model") == model)
    reps = sorted(sub["repeat"].unique().to_list())
    S, y = [], None
    for r in reps:
        s = sub.filter(pl.col("repeat") == r).sort("row_index")
        S.append(s["score"].to_numpy())
        y = s["target"].to_numpy()
    got = mb.ranking_point_estimates(np.asarray(y), np.vstack(S))
    assert got["pr_auc"]["mean"] == pytest.approx(
        tourney["models"][model]["pr_auc_mean"], abs=1e-9), (
        "the battery no longer reproduces the stored tournament PR-AUC")
    assert got["pr_auc"]["std"] == pytest.approx(
        tourney["models"][model]["pr_auc_std"], abs=1e-9)


def test_written_artifact_labels_every_run_and_claims_no_retraining():
    if not ART.exists():
        pytest.skip("metric_battery.json has not been generated yet")
    doc = json.loads(ART.read_text())
    assert doc["runs"], "artifact contains no runs"
    for key, run in doc["runs"].items():
        assert run["protocol"] in ("FLAT", "NESTED", "NESTED_PRELIMINARY",
                                   "HOLDOUT_REFERENCE")
        assert run["provenance"]["retraining_performed"] is False
        assert run["provenance"]["thresholds_modified"] is False
        assert run["provenance"]["predictions_source"]["sha256_prefix"]
        if run["protocol"] == "HOLDOUT_REFERENCE":
            assert run["interpretation"]["usable_for_selection"] is False
            assert run["threshold_candidates"] is None, (
                f"{key} searched for a threshold on the held-out split")


def test_written_artifact_avoids_overclaiming_vocabulary():
    if not ART.exists():
        pytest.skip("metric_battery.json has not been generated yet")
    text = ART.read_text().lower()
    # Whole-word matching, not substring: "proven" occurs inside "provenance",
    # and a check that fires on its own metadata block teaches the next person
    # to delete the check rather than to fix the wording.
    for word in ("state of the art", "guaranteed", "proven", "flawless",
                 "production ready", "perfect model"):
        assert re.search(rf"\b{re.escape(word)}\b", text) is None, (
            f"artifact overclaims with {word!r}")
