"""Analyst Capacity Optimizer: the properties a capacity answer has to have.

Three kinds of assertion live here.

1. Structural properties that must hold for any ranking at all - recall never
   falls as the budget grows, a recommended threshold reproduces the budget it
   promises, and the two input modes cannot disagree.
2. Safety properties - the frozen policy thresholds are read-only, and no
   answer ever claims to have applied anything.
3. Agreement with artifacts computed by entirely different code, so that a
   silent change in either one is caught rather than averaged away.

The artifact-backed tests skip when the artifacts are absent, so this file is
runnable on a fresh clone; the property tests run everywhere.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

from muleguard import settings
from muleguard.models import capacity

CURVE_PATH = settings.METRICS_DIR / "capacity_curve.json"
LENS_PATH = settings.METRICS_DIR / "lens_stack_oof_v2.json"
OOF_PATH = settings.PREDICTIONS_DIR / "oof_v2.parquet"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy() -> dict:
    """A small ranking with known structure and no ties.

    30 positives in 1,200 rows. The signal is strong enough that the top of the
    ranking is mostly positive, which is what makes the monotonicity and
    threshold assertions meaningful rather than vacuous.
    """
    rng = np.random.default_rng(11)
    n_pos, n_neg = 30, 1170
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
    raw = np.r_[rng.normal(3.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
    scores = 1.0 / (1.0 + np.exp(-raw))
    repeats = {}
    for r in range(3):
        jitter = np.random.default_rng(100 + r).normal(0, 0.15, len(y))
        repeats[r] = (y, 1.0 / (1.0 + np.exp(-(raw + jitter))))
    return {"y": y, "scores": scores, "repeats": repeats}


@pytest.fixture(scope="module")
def toy_curve(toy: dict) -> dict:
    points = capacity.build_curve(
        y=toy["y"], scores=toy["scores"], repeats=toy["repeats"],
        budgets=list(range(1, 301)), n_boot=200, seed=42,
        # A deliberately non-identity but strictly increasing map, to prove the
        # curve is invariant to calibration everywhere except the threshold.
        calibrate=lambda s: np.sqrt(np.asarray(s, dtype=float)),
    )
    return capacity.make_curve_document(
        points=points,
        evaluation={"n_accounts": len(toy["y"]), "n_positives": int(toy["y"].sum()),
                    "split": "toy"},
        frozen_policy={"critical_risk": 0.9, "urgent_risk": 0.5,
                       "standard_risk": 0.1, "anomaly_escalation_pct": 99.0,
                       "policy_version": "1.0"},
        provenance={"generator": "test"},
    )


@pytest.fixture(scope="module")
def real_curve() -> dict:
    if not CURVE_PATH.exists():
        pytest.skip("capacity_curve.json not built; run muleguard.cli.capacity_curve")
    return json.loads(CURVE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. structural properties
# --------------------------------------------------------------------------
def test_recall_is_monotonically_non_decreasing_in_budget(toy_curve):
    """Reviewing more accounts can never find fewer mules."""
    recalls = [p["recall"] for p in toy_curve["points"]]
    assert recalls == sorted(recalls), "recall fell as the review budget grew"
    tps = [p["true_positives"] for p in toy_curve["points"]]
    assert all(b >= a for a, b in zip(tps, tps[1:]))


def test_false_alarms_are_monotonically_non_decreasing(toy_curve):
    """The false-alarm mode relies on this to be well defined."""
    for field in ("false_positives", "fp_per_1000_legitimate", "fp_per_1000_screened"):
        vals = [p[field] for p in toy_curve["points"]]
        assert vals == sorted(vals), f"{field} decreased with budget"


def test_recall_and_precision_are_calibration_invariant(toy):
    """Only the threshold lives on the calibrated scale; the curve does not."""
    ks = [5, 10, 25, 50, 100]
    raw = capacity.capacity_points(toy["y"], toy["scores"], ks)
    squashed = capacity.capacity_points(toy["y"], np.sqrt(toy["scores"]), ks)
    assert [p["recall"] for p in raw] == [p["recall"] for p in squashed]
    assert [p["precision"] for p in raw] == [p["precision"] for p in squashed]


def test_recommended_threshold_produces_the_promised_budget(toy_curve, toy):
    """A threshold that does not reproduce the budget is not a recommendation.

    Checked two ways: against the count the curve recorded, and by applying the
    number to the score vector from scratch.
    """
    scores = toy["scores"]
    for k in (1, 7, 25, 50, 137, 300):
        plan = capacity.answer_for_budget(toy_curve, k)
        assert plan["answered_at_budget"] == k
        thr = plan["recommended_threshold"]["model_score"]
        assert plan["recommended_threshold"]["alerts_produced_on_evaluation_split"] == k
        assert int((scores >= thr).sum()) == k, (
            f"threshold {thr!r} alerts {(scores >= thr).sum()} accounts, not {k}")


def test_threshold_is_reported_on_the_calibrated_scale(toy_curve, toy):
    """The advisory number must be comparable to the frozen policy thresholds."""
    plan = capacity.answer_for_budget(toy_curve, 50)
    rec = plan["recommended_threshold"]
    assert rec["calibrated_risk"] == pytest.approx(np.sqrt(rec["model_score"]))
    assert rec["band_under_frozen_policy"] in {
        "at or above CRITICAL_REVIEW",
        "between URGENT_REVIEW and CRITICAL_REVIEW",
        "between STANDARD_REVIEW and URGENT_REVIEW",
        "below STANDARD_REVIEW",
    }


def test_budget_larger_than_the_split_is_capped_and_says_so(toy_curve):
    plan = capacity.answer_for_budget(toy_curve, 10_000)
    assert plan["capped_to_evaluation_size"] is True
    assert plan["answered_exactly"] is False
    assert plan["answered_at_budget"] <= toy_curve["evaluation"]["n_accounts"]


def test_zero_or_negative_capacity_is_rejected(toy_curve):
    with pytest.raises(ValueError):
        capacity.answer_for_budget(toy_curve, 0)
    with pytest.raises(ValueError):
        capacity.answer_for_fp_per_1000(toy_curve, -1.0)
    with pytest.raises(ValueError):
        capacity.answer_for_fp_per_1000(toy_curve, 5.0, basis="nonsense")


# --------------------------------------------------------------------------
# 2. the two input modes must agree
# --------------------------------------------------------------------------
def test_fp_mode_agrees_with_budget_mode(toy_curve):
    """Whatever budget the false-alarm mode lands on, the budget mode must
    return exactly the same row - same recall, same precision, same threshold."""
    for limit in (0.0, 0.5, 2.0, 5.0, 20.0, 100.0):
        fp_plan = capacity.answer_for_fp_per_1000(toy_curve, limit)
        k = fp_plan["answered_at_budget"]
        k_plan = capacity.answer_for_budget(toy_curve, k)
        assert fp_plan["expected"] == k_plan["expected"]
        assert (fp_plan["recommended_threshold"]["calibrated_risk"]
                == k_plan["recommended_threshold"]["calibrated_risk"])


def test_fp_mode_returns_the_largest_budget_inside_the_tolerance(toy_curve):
    """It must be the binding constraint, not merely a budget that fits."""
    points = toy_curve["points"]
    for limit in (0.5, 2.0, 5.0):
        k = capacity.answer_for_fp_per_1000(toy_curve, limit)["answered_at_budget"]
        inside = [p["budget"] for p in points
                  if p["fp_per_1000_legitimate"] <= limit + 1e-12]
        assert k == max(inside)
        bigger = [p for p in points if p["budget"] > k]
        if bigger:
            assert min(p["fp_per_1000_legitimate"] for p in bigger) > limit


def test_the_two_fp_denominators_are_not_silently_conflated(toy_curve):
    """`per 1,000 screened` is the smaller number, and the API must not swap
    one for the other behind the user's back."""
    for p in toy_curve["points"]:
        assert p["fp_per_1000_screened"] <= p["fp_per_1000_legitimate"] + 1e-12
    strict = capacity.answer_for_fp_per_1000(toy_curve, 5.0, basis="legitimate")
    loose = capacity.answer_for_fp_per_1000(toy_curve, 5.0, basis="screened")
    assert loose["answered_at_budget"] >= strict["answered_at_budget"]
    assert strict["input"]["basis"] == "legitimate"


# --------------------------------------------------------------------------
# 3. safety: the frozen policy is read, never written
# --------------------------------------------------------------------------
def test_frozen_policy_thresholds_are_not_mutated(toy_curve):
    """Answering capacity questions must leave the policy object untouched."""
    before = copy.deepcopy(toy_curve["frozen_policy"])
    doc_before = copy.deepcopy(toy_curve)
    for k in (1, 25, 100, 300):
        capacity.answer_for_budget(toy_curve, k)
    for limit in (0.0, 1.0, 10.0):
        capacity.answer_for_fp_per_1000(toy_curve, limit)
    assert toy_curve["frozen_policy"] == before
    assert toy_curve == doc_before, "answering a question mutated the curve document"


def test_frozen_policy_file_is_untouched_by_curve_generation():
    """The on-disk policy snapshot still carries policy_version 1.0 and the
    three thresholds the bundle was frozen with."""
    if not LENS_PATH.exists():
        pytest.skip("lens_stack_oof_v2.json not present")
    frozen = json.loads(LENS_PATH.read_text(encoding="utf-8"))["policy_thresholds"]
    assert frozen["policy_version"] == "1.0"
    for key in ("critical_risk", "urgent_risk", "standard_risk"):
        assert isinstance(frozen[key], float)
    # The bands must still nest, which is the invariant build_lenses_v2 enforced.
    assert frozen["critical_risk"] > frozen["urgent_risk"] > frozen["standard_risk"]


def test_every_recommendation_is_advisory_and_unapplied(toy_curve):
    for plan in (capacity.answer_for_budget(toy_curve, 50),
                 capacity.answer_for_fp_per_1000(toy_curve, 5.0)):
        rec = plan["recommended_threshold"]
        assert rec["status"] == capacity.ADVISORY_STATUS
        assert rec["applied"] is False
        assert "human" in rec["note"].lower()


# --------------------------------------------------------------------------
# 4. uncertainty is carried, not dropped
# --------------------------------------------------------------------------
def test_intervals_bracket_the_point_estimate(toy_curve):
    for p in toy_curve["points"]:
        assert p["recall_ci_low"] <= p["recall"] <= p["recall_ci_high"] + 1e-12
        assert (p["recall_ci_low_resampled_accounts"] <= p["recall"]
                <= p["recall_ci_high_resampled_accounts"] + 1e-12)
        assert p["precision_ci_low"] <= p["precision"] <= p["precision_ci_high"] + 1e-12


def test_per_repeat_spread_is_reported_for_every_point(toy_curve):
    for p in toy_curve["points"]:
        assert len(p["recall_per_repeat"]) == 3
        assert p["recall_repeat_min"] <= p["recall_repeat_mean"] <= p["recall_repeat_max"]


def test_degenerate_stratified_interval_is_flagged_not_hidden(toy):
    """Inside the leading run of true positives the stratified scheme has
    nothing to vary. The curve has to say so rather than present a zero-width
    interval as confidence."""
    run = capacity.leading_true_positive_run(toy["y"], toy["scores"])
    assert run >= 1
    points = capacity.build_curve(y=toy["y"], scores=toy["scores"],
                                  budgets=[1, run, run + 1, 100], n_boot=200, seed=42)
    inside = next(p for p in points if p["budget"] == 1)
    assert inside["inside_leading_true_positive_run"] is True
    outside = next(p for p in points if p["budget"] == 100)
    assert outside["inside_leading_true_positive_run"] is False


def test_plan_carries_the_small_positive_count_warning(toy_curve):
    plan = capacity.answer_for_budget(toy_curve, 50)
    note = plan["uncertainty_note"]
    assert str(toy_curve["evaluation"]["n_positives"]) in note
    assert "interval" in note


# --------------------------------------------------------------------------
# 5. agreement with artifacts produced by other code
# --------------------------------------------------------------------------
def test_curve_reproduces_the_frozen_recall_at_budget_block(real_curve):
    """The lens artifact's own recall_at_budget rows were computed by
    ``muleguard.evaluation.metrics`` at bundle-freeze time. If this module
    disagreed with them, one of the two would be wrong - so it is asserted,
    not assumed."""
    if not LENS_PATH.exists():
        pytest.skip("lens_stack_oof_v2.json not present")
    lens = json.loads(LENS_PATH.read_text(encoding="utf-8"))
    by_k = {p["budget"]: p for p in real_curve["points"]}
    checked = 0
    for row in lens["oof_calibrated_report"]["recall_at_budget"]:
        k = row["budget"]
        if k not in by_k:
            continue
        assert by_k[k]["recall"] == pytest.approx(row["recall"])
        assert by_k[k]["precision"] == pytest.approx(row["precision"])
        assert by_k[k]["true_positives"] == row["true_positives"]
        checked += 1
    assert checked >= 3, "no budgets in common to check against the lens artifact"


def test_fp_mode_reproduces_the_frozen_recall_at_fpr_block(real_curve):
    """The FP/1,000 mode and the artifact's recall_at_fpr block are two
    independent routes to the same operating point; they must land together."""
    if not LENS_PATH.exists():
        pytest.skip("lens_stack_oof_v2.json not present")
    lens = json.loads(LENS_PATH.read_text(encoding="utf-8"))
    for row in lens["oof_calibrated_report"]["recall_at_fpr"]:
        plan = capacity.answer_for_fp_per_1000(real_curve, row["fp_per_1000_legit"])
        assert plan["expected"]["recall"] == pytest.approx(row["recall"], abs=1e-9)
        assert plan["expected"]["fp_per_1000_legitimate"] <= row["fp_per_1000_legit"] + 1e-9


def test_real_curve_is_built_from_development_oof_only(real_curve):
    assert real_curve["evaluation"]["locked_test_used"] is False
    assert real_curve["provenance"]["retraining_performed"] is False
    assert "oof_v2.parquet" in real_curve["provenance"]["predictions_source"]
    blob = json.dumps(real_curve).lower()
    assert "locked_test" not in blob.replace("locked_test_used", "")


def test_real_curve_champion_matches_the_promoted_model(real_curve):
    if not LENS_PATH.exists():
        pytest.skip("lens_stack_oof_v2.json not present")
    lens = json.loads(LENS_PATH.read_text(encoding="utf-8"))
    assert real_curve["provenance"]["champion_model"] == lens["winner"]
    assert real_curve["frozen_policy"] == lens["policy_thresholds"]


def test_real_thresholds_reproduce_their_budgets_on_the_stored_predictions(real_curve):
    """End to end on the real data: apply the recommended threshold to the
    repeat-averaged OOF scores and count the alerts."""
    if not OOF_PATH.exists():
        pytest.skip("oof_v2.parquet not present")
    champion = real_curve["provenance"]["champion_model"]
    oof = pl.read_parquet(OOF_PATH).filter(pl.col("model") == champion)
    avg = (oof.group_by("row_index")
           .agg(pl.col("score").mean().alias("score"),
                pl.col("target").first().alias("target"))
           .sort("row_index"))
    scores = avg["score"].to_numpy()
    y = avg["target"].to_numpy()
    for k in (25, 50, 100):
        plan = capacity.answer_for_budget(real_curve, k)
        thr = plan["recommended_threshold"]["model_score"]
        assert int((scores >= thr).sum()) == k
        assert int(y[scores >= thr].sum()) == plan["expected"]["mules_caught"]


def test_no_forbidden_verdict_vocabulary_in_the_curve_artifact(real_curve):
    blob = json.dumps(real_curve)
    for word in ("GUILTY", "CRIMINAL", "PERMANENTLY_SAFE", "CERTIFIED_CLEAN",
                 "AUTO_FREEZE"):
        assert word not in blob
