"""Section 24: the classifier must be bit-identical, and the test must say so.

This is the load-bearing suite of the whole upgrade. Everything else added by
the Cohort Radar and the Control-Attribution guardrail is defensible only if
the numbers underneath are provably the same numbers. Section 24 sets the
tolerance at 1e-12 and adds an instruction that matters more than the number:
if a difference appears, do not explain it away as small - find the cause. So
nothing here rounds before comparing.

The suite reads the frozen baseline captured before any USP code existed and
re-derives the same quantities now.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from muleguard import settings
from muleguard.usp import baseline

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built")

needs_baseline = pytest.mark.skipif(
    not baseline.BASELINE_PATH.exists(),
    reason="pre-change baseline not captured - python -m muleguard.cli.usp_baseline")


@pytest.fixture(scope="module")
def before() -> dict:
    return json.loads(baseline.BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def after() -> dict:
    return baseline.invariant_snapshot()


@pytest.fixture(scope="module")
def report(before, after) -> dict:
    return baseline.compare(before, after)


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


@needs_baseline
def test_the_bundle_on_disk_is_the_one_that_was_frozen(report):
    assert report["model_sha_equal"], report["identity_differences"]
    assert report["model_sha_before"] == report["model_sha_after"]


@needs_baseline
def test_the_selected_feature_list_did_not_move(report):
    assert report["feature_hash_equal"], report["identity_differences"]


@needs_baseline
def test_the_calibrator_and_thresholds_did_not_move(report):
    assert report["calibrator_equal"], report["identity_differences"]
    assert report["thresholds_equal"], report["identity_differences"]


@needs_baseline
def test_the_quarantine_list_did_not_move(report):
    assert report["quarantine_equal"]


@needs_baseline
def test_no_identity_field_changed_at_all(report):
    assert report["identity_differences"] == {}

# --------------------------------------------------------------------------
# section 24: probabilities, to the last bit
# --------------------------------------------------------------------------


@needs_baseline
def test_the_probe_rows_were_actually_compared(report):
    """Guards against a green run that compared nothing.

    ``compare`` reports ``compared: False`` when either snapshot lacks probes.
    Without this assertion the probability tests below would pass on a report
    that never scored a row.
    """
    probes = report["probes"]
    assert probes["compared"] is True, "probe comparison did not run"
    assert probes["n_rows_only_in_one_snapshot"] == 0
    assert probes["n_rows_compared"] >= min(baseline.N_PROBE_ROWS, 100), (
        f"only {probes['n_rows_compared']} rows compared")


@needs_baseline
def test_calibrated_probabilities_are_identical_to_1e_12(report):
    diff = report["probability_max_abs_diff"]
    assert diff is not None, "probe rows were not scored - the check did not run"
    assert diff <= baseline.PROBABILITY_TOLERANCE, (
        f"calibrated probability moved by {diff:.3e}. Section 24: do not explain "
        f"this away as small. Find the cause.")


@needs_baseline
def test_raw_uncalibrated_probabilities_are_identical_too(report):
    """The calibrated figures agreeing is not sufficient on its own.

    A monotone calibrator can absorb a small change in raw output and hide it.
    Comparing before calibration is what makes the equality a statement about
    the model rather than about the mapping bolted on after it.
    """
    diff = report["raw_probability_max_abs_diff"]
    assert diff is not None
    assert diff <= baseline.PROBABILITY_TOLERANCE, f"raw score moved by {diff:.3e}"


@needs_baseline
def test_every_probe_row_agrees_not_merely_the_maximum(before, after):
    """A maximum is a summary. This walks the rows.

    Reported as a count of disagreeing rows rather than as a mean, because a
    mean of many zeroes and one real difference reads as zero.
    """
    pb = {r["row_index"]: r for r in before["probes"]["rows"]}
    pa = {r["row_index"]: r for r in after["probes"]["rows"]}
    assert set(pb) == set(pa)
    assert pb, "no probe rows in the baseline"
    moved = [row for row in pb
             if abs(pb[row]["calibrated_risk"] - pa[row]["calibrated_risk"])
             > baseline.PROBABILITY_TOLERANCE]
    assert moved == [], f"{len(moved)} of {len(pb)} probe rows changed probability"


@needs_baseline
def test_every_model_family_in_the_ensemble_agrees_row_by_row(before, after):
    """Not just the ensemble output - each member's raw score.

    An ensemble mean can stay put while two members move in opposite
    directions. Checking the members is what rules that out.
    """
    pb = {r["row_index"]: r for r in before["probes"]["rows"]}
    pa = {r["row_index"]: r for r in after["probes"]["rows"]}
    families = set()
    moved: list[str] = []
    for row in sorted(set(pb) & set(pa)):
        b, a = pb[row]["raw_scores"], pa[row]["raw_scores"]
        assert set(b) == set(a), f"row {row}: model families changed"
        families |= set(b)
        for fam in b:
            if abs(b[fam] - a[fam]) > baseline.PROBABILITY_TOLERANCE:
                moved.append(f"row {row} / {fam}")
    assert families, "no per-family raw scores were recorded"
    assert moved == [], f"{len(moved)} family scores moved, e.g. {moved[:5]}"


@needs_baseline
def test_tiers_match_row_by_row(report):
    assert report["tier_mismatch_count"] == 0, (
        f"{report['tier_mismatch_count']} rows changed review tier: "
        f"{report['probes']['tier_mismatches']}")


@needs_baseline
def test_the_policy_decision_did_not_change_for_any_probe_row(report):
    """Section 23 in numbers: no account acquired an action it did not have."""
    probes = report["probes"]
    assert probes["policy_action_mismatch_count"] == 0, (
        probes["policy_action_mismatches"])


@needs_baseline
def test_the_supporting_signals_did_not_move_either(before, after):
    """Conformal status, OOD status, agreement, merchant safeguard.

    These are not the score, but they are shown next to it and they gate
    review routing. If the upgrade had disturbed a shared code path, this is
    where it would surface first.
    """
    pb = {r["row_index"]: r for r in before["probes"]["rows"]}
    pa = {r["row_index"]: r for r in after["probes"]["rows"]}
    bad: list[str] = []
    for row in sorted(set(pb) & set(pa)):
        b, a = pb[row], pa[row]
        for key in ("conformal_status", "ood_status", "auto_action",
                    "merchant_safeguard_applied"):
            if b.get(key) != a.get(key):
                bad.append(f"row {row}: {key} {b.get(key)!r} -> {a.get(key)!r}")
        if abs(b["model_agreement"] - a["model_agreement"]) > baseline.PROBABILITY_TOLERANCE:
            bad.append(f"row {row}: model_agreement moved")
    assert bad == [], bad[:10]


@needs_baseline
def test_the_probe_hash_is_the_same_string(before, after):
    """One digest over (row, probability, tier) for the whole probe set.

    Redundant with the per-row checks by design: a single hash is what a judge
    can compare by eye against the baseline file, and it fails loudly if any
    row was reordered, dropped or silently re-scored.
    """
    assert before["probes"]["rows_hash"] == after["probes"]["rows_hash"]


# --------------------------------------------------------------------------
# section 25/38: published accuracy
# --------------------------------------------------------------------------


@needs_baseline
def test_every_accuracy_metric_is_bit_identical(report):
    moved = {k: v for k, v in report["metric_differences"].items()
             if v["difference"] != 0.0}
    assert moved == {}, f"accuracy metrics changed: {moved}"


@needs_baseline
def test_the_headline_metrics_are_present_and_were_actually_recomputed(report):
    """A metric that is absent cannot differ. This checks it was there to check.

    ``compare`` only reports on the keys the snapshots contain, so an empty
    accuracy block would pass the equality test above while proving nothing.
    """
    diffs = report["metric_differences"]
    for key in ("pr_auc", "roc_auc", "recall", "precision", "f1", "f2", "mcc",
                "brier", "ece", "balanced_accuracy", "recall_at_top50",
                "false_positives_per_1000_legitimate"):
        assert key in diffs, f"{key} was not in the accuracy comparison"
        assert diffs[key]["before"] is not None
        assert diffs[key]["after"] is not None


@needs_baseline
def test_recomputed_pr_auc_still_agrees_with_the_published_figure(after):
    """Section 25: the shipped number must be the number the code produces.

    A tolerance rather than bit-identity, because the registry stores the
    headline rounded. Bit-identity against a rounded artifact would be a test
    that can only fail.
    """
    agreement = after["accuracy"].get("agreement_with_published")
    if not agreement or agreement.get("registry_oof_pr_auc") is None:
        pytest.skip("no published PR-AUC in the registry to compare against")
    assert agreement["matches_registry"], agreement


@needs_baseline
def test_the_frozen_artifacts_on_disk_were_not_rewritten(before, after):
    """Section 26: no retraining, no new split, no fresh locked-test run.

    Hashes rather than contents. If someone re-ran training to "retest", the
    out-of-fold predictions and the locked-test artifacts would change even
    when the metrics happened to land in the same place.
    """
    hb, ha = before["artifact_hashes"], after["artifact_hashes"]
    assert set(hb) == set(ha)
    changed = {k: (hb[k], ha[k]) for k in hb if hb[k] != ha[k]}
    assert changed == {}, f"frozen artifacts were rewritten: {sorted(changed)}"


@needs_baseline
def test_the_verdict_itself_is_pass(report):
    """Last. The individual assertions above say why; this says whether."""
    assert report["verdict"] == "PASS", report["findings"]
    assert report["findings"] == []
