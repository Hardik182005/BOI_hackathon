"""The bookkeeping layer must fail loudly rather than quietly look complete.

Three artifacts make claims about the whole project rather than about one model:
the experiment ledger (§60), the artifact reconciliation (§57/§58) and the plot
manifest (§56). Each of them is only worth anything if it is honest about what is
missing. These tests pin the honesty, not the numbers:

* an experiment nobody registered must show up, not vanish;
* a spec-named artifact whose source is absent must read PENDING, never be
  invented;
* a plot whose evidence is absent must be skipped, never drawn from a stand-in.

They run against the registries themselves with a temporary metrics directory,
so they do not depend on which runs happen to have finished.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from muleguard.cli import experiment_ledger as led
from muleguard.evaluation import plots_final


# ---------------------------------------------------------------------------
# §60 experiment ledger
# ---------------------------------------------------------------------------
def test_every_registered_source_is_unique():
    """Two entries pointing at one file would double-count that experiment."""
    paths = [s.artifact for s in led.SOURCES]
    assert len(paths) == len(set(paths)), \
        f"duplicate ledger sources: {sorted({p for p in paths if paths.count(p) > 1})}"


def test_unregistered_artifact_is_reported(tmp_path, monkeypatch):
    """A metrics file no entry claims must surface as UNREGISTERED_REVIEW."""
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "brand_new_experiment.json").write_text(
        json.dumps({"pr_auc": 0.9}), encoding="utf-8")
    monkeypatch.setattr(led.settings, "METRICS_DIR", metrics)
    monkeypatch.setattr(led.settings, "REPO_ROOT", tmp_path)

    rows, unregistered = led.build_rows()

    assert any("brand_new_experiment.json" in u for u in unregistered)
    flagged = [r for r in rows if r["status"] == "UNREGISTERED_REVIEW"]
    assert len(flagged) == 1
    assert "brand_new_experiment.json" in flagged[0]["artifact_path"]


def test_main_exits_non_zero_when_an_experiment_is_unregistered(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "forgotten.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(led.settings, "METRICS_DIR", metrics)
    monkeypatch.setattr(led.settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(led, "OUT_DIR", tmp_path / "experiments")
    monkeypatch.setattr(led, "OUT_CSV", tmp_path / "experiments" / "ledger.csv")

    assert led.main() == 1, "an unregistered experiment must fail the run"


def test_missing_artifact_becomes_pending_not_absent(tmp_path, monkeypatch):
    """A registered run that has not happened is a visible gap, not a silence."""
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    monkeypatch.setattr(led.settings, "METRICS_DIR", metrics)
    monkeypatch.setattr(led.settings, "REPO_ROOT", tmp_path)

    rows, unregistered = led.build_rows()

    assert not unregistered
    assert len(rows) == len(led.SOURCES)
    assert {r["status"] for r in rows} == {"PENDING_RUN"}


def test_pre_firewall_results_are_never_labelled_accepted():
    """Generation-1 numbers came from a model that could see quarantined columns."""
    for name in ("retired_gen1_pre_firewall_stack", "catboost_tuned_top60",
                 "RETIRED_run"):
        assert led._is_pre_firewall(name), name
    assert not led._is_pre_firewall("xgboost_top_120")


def test_quarantine_column_lists_all_thirteen():
    assert len(led.FIREWALL_13.split("|")) == 13
    for column in ("F2230", "F3912", "__UNNAMED__0"):
        assert column in led.FIREWALL_13


# ---------------------------------------------------------------------------
# §57/§58 artifact reconciliation
# ---------------------------------------------------------------------------
def test_reconciler_reports_pending_when_the_source_is_absent(tmp_path, monkeypatch):
    from muleguard.cli import reconcile_artifacts as rec

    monkeypatch.setattr(rec, "ROOT", tmp_path)
    monkeypatch.setattr(rec.settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rec, "TESTING", tmp_path / "artifacts/testing")
    monkeypatch.setattr(rec, "MANIFEST", tmp_path / "artifacts/testing/manifest.json")
    monkeypatch.setattr(rec, "INDEX_DOC", tmp_path / "docs/INDEX.md")
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)

    rc = rec.main()
    manifest = json.loads((tmp_path / "artifacts/testing/manifest.json").read_text(encoding="utf-8"))
    statuses = {a["spec"]: a["status"] for a in manifest["artifacts"]}

    # Nothing was derivable, so nothing may claim to have been derived.
    assert rec.DERIVED not in statuses.values()
    assert set(statuses.values()) <= {rec.PENDING, rec.MISSING}
    # A requirement with a named run behind it is PENDING, not MISSING.
    assert statuses["artifacts/testing/nested_cv_results.json"] == rec.PENDING
    assert rc == 1, "a requirement with no source and no pending run must fail"


def test_every_report_pointer_names_a_different_file():
    """A pointer to itself would be a loop, and a duplicate would be a fork."""
    from muleguard.cli import reconcile_artifacts as rec

    targets = [t for t, _ in rec.REPORTS.values() if t]
    assert len(targets) == len(set(targets)), "two required names share one report"
    for spec, (target, note) in rec.REPORTS.items():
        assert note, f"{spec} has no description of what it covers"


# ---------------------------------------------------------------------------
# §56 plots
# ---------------------------------------------------------------------------
def test_missing_evidence_skips_the_plot_rather_than_drawing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(plots_final, "ROOT", tmp_path)

    with pytest.raises(plots_final.Skip):
        plots_final.plot_shap_importance()
    with pytest.raises(plots_final.Skip):
        plots_final.plot_adversarial_validation()


def test_every_required_plot_has_a_producer_when_it_can_be_pending():
    """A skippable plot must say which run would produce its evidence."""
    pending_capable = {"fold AP distribution", "top SHAP importance",
                       "feature-subset ablation", "positive-removal stability",
                       "adversarial validation / shift plot"}
    for label in pending_capable:
        _, produced_by = plots_final.PLOTS[label]
        assert produced_by, f"{label} can be skipped but names no producing run"


def test_the_positive_removal_plot_reads_the_keys_the_nested_arm_actually_writes():
    """This plot spent a run silently skipped. The nested artifact existed, so the
    nested branch was taken, but it was probed for `ap_per_round` - a key the
    stage never writes - and the fallback then looked for flat-artifact names in a
    nested file. Both misses degraded to Skip, which reads as 'evidence not
    produced yet' rather than 'plot is looking in the wrong place'. Pinning the
    real schema here means the next rename fails loudly instead of quietly
    dropping a required plot from the set."""
    d = json.loads((ROOT / "artifacts/metrics/nested_positive_removal.json")
                   .read_text(encoding="utf-8"))
    assert d["per_fold"], "the arm records no per-fold rows"
    for row in d["per_fold"]:
        assert "reference_ap" in row and "stressed_ap_mean" in row
    assert "mean_paired_diff" in d["paired_degradation"]
    assert "ci95_of_mean" in d["paired_degradation"]


def test_a_nested_artifact_without_fold_rows_skips_rather_than_drawing_an_empty_axis():
    with pytest.raises(plots_final.Skip):
        plots_final._plot_positive_removal_nested({"per_fold": []})


def test_titles_carry_split_model_and_data_version():
    subtitle = plots_final._subtitle("dev OOF, nested outer folds", "xgboost_top_120")
    assert "split:" in subtitle and "model:" in subtitle and "data:" in subtitle
    assert "xgboost_top_120" in subtitle


# ---------------------------------------------------------------------------
# promotion eligibility
# ---------------------------------------------------------------------------
def test_a_model_too_slow_to_serve_cannot_be_promoted():
    """TabPFN scored highest in the tournament and costs 438 s per score.

    The veto used to live only in challenger_review, so the promotion path
    itself would have promoted it the next time the report was regenerated.
    """
    from muleguard.cli import tournament_v2 as t

    assert not t._promotable({"measured": True, "single_row_seconds": 438.1})
    assert t._promotable({"measured": True, "single_row_seconds": 0.004})
    # never timed: recorded as unknown, not assumed slow
    assert t._promotable({})
    assert t._promotable({"measured": False, "single_row_seconds": 900.0})


def test_the_promotion_artifact_names_what_it_benched():
    """The record has to show the model was benched, not beaten."""
    decision = json.loads((Path(__file__).resolve().parents[2] /
                           "artifacts/metrics/promotion_decision_v2.json"
                           ).read_text(encoding="utf-8"))
    benched = {row["model"] for row in decision.get("excluded_for_serving_cost", [])}
    assert decision["promoted"] not in benched
    if decision["raw_pr_auc_leader_including_unservable"] != decision["promoted"]:
        assert decision["raw_pr_auc_leader_including_unservable"] in benched, \
            "a model outscored the champion and the artifact does not say why it lost"


# ---------------------------------------------------------------------------
# §63/§64/§65 final verdict
# ---------------------------------------------------------------------------
def test_verdict_stays_pending_while_evidence_is_missing(tmp_path, monkeypatch):
    """An empty repository must not be able to produce a PASS."""
    from muleguard.cli import final_verdict as fv

    monkeypatch.setattr(fv, "ROOT", tmp_path)
    monkeypatch.setattr(fv.Evidence, "champion_features",
                        lambda self: (None, "no bundle in this fixture"))
    payload = fv.build(fv.Evidence())

    assert payload["verdict"] == fv.PENDING_EVIDENCE
    assert payload["verdict_is_one_of_the_three_permitted"] is False
    assert all(c["status"] == fv.NOT_MET for c in payload["pass_criteria"])


def test_a_quarantined_feature_in_the_bundle_forces_fail(tmp_path, monkeypatch):
    from muleguard.cli import final_verdict as fv

    monkeypatch.setattr(fv, "ROOT", tmp_path)
    monkeypatch.setattr(fv.Evidence, "champion_features",
                        lambda self: (["F1", "F3924"], "fixture bundle"))
    payload = fv.build(fv.Evidence())

    assert payload["verdict"] == fv.FAIL
    blocked = [b for b in payload["release_blockers"] if b["status"] == fv.BLOCKED]
    assert any("F3924" in b["detail"] for b in blocked)


def test_evidence_recorded_before_the_champion_is_stale(tmp_path, monkeypatch):
    """A suite that passed against the previous champion has not judged this one."""
    from muleguard.cli import final_verdict as fv

    monkeypatch.setattr(fv, "ROOT", tmp_path)
    (tmp_path / "artifacts/testing").mkdir(parents=True)
    (tmp_path / "artifacts/testing/backend_results.json").write_text(json.dumps(
        {"generated_utc": "2026-07-10T23:01:32+00:00", "n_checks": 15,
         "n_passed": 15, "all_passed": True}), encoding="utf-8")
    ev = fv.Evidence()
    ev.champion = "xgboost_top_120"
    ev.promoted_at = fv._parse_ts("2026-08-12T07:39:08+00:00")

    status, detail, _ = fv._suite(ev, "artifacts/testing/backend_results.json")

    assert status == fv.STALE
    assert "re-run" in detail
    # and the same artifact stops being stale once it postdates the promotion
    ev.promoted_at = fv._parse_ts("2026-07-01T00:00:00+00:00")
    assert fv._suite(ev, "artifacts/testing/backend_results.json")[0] == fv.CLEAR


def test_pass_needs_both_clear_blockers_and_met_criteria(tmp_path, monkeypatch):
    from muleguard.cli import final_verdict as fv

    monkeypatch.setattr(fv, "ROOT", tmp_path)
    clear = fv.Item("all good", lambda ev: (fv.CLEAR, "fixture", []))
    unmet = fv.Item("not yet", lambda ev: (fv.UNVERIFIED, "fixture", []))

    monkeypatch.setattr(fv, "BLOCKERS", [clear])
    monkeypatch.setattr(fv, "CRITERIA", [("ML", clear)])
    assert fv.build(fv.Evidence())["verdict"] == fv.PASS

    monkeypatch.setattr(fv, "CRITERIA", [("ML", unmet)])
    assert fv.build(fv.Evidence())["verdict"] == fv.PENDING_EVIDENCE


def test_every_section_63_blocker_line_is_checked():
    """The blocker list is the spec's, verbatim - not a subset somebody trimmed."""
    from muleguard.cli import final_verdict as fv

    spec = [
        "F3924 enters model features",
        "F3898/F3899 enter accepted model",
        "F3912/F3913/F3914/F3915 enter accepted model",
        "F2230 remains suspicious and is still included",
        "F3916-18 are used without availability evidence",
        "train-validation overlap",
        "preprocessing fitted on validation",
        "feature selection fitted on validation",
        "test labels used for tuning",
        "external validation triggers retraining",
        "fake metrics",
        "hardcoded dashboard metrics",
        "prediction row order changes",
        "model artifact cannot reproduce saved predictions",
        "UI/backend score mismatch",
        "Ollama changes scoring",
        "core requires internet",
        "core requires MCP",
        "core requires Claude in Chrome",
        "P0 defect",
        "unapproved P1 defect",
    ]
    assert [b.text for b in fv.BLOCKERS] == spec


def test_a_pending_field_always_names_the_run_that_would_fill_it():
    from muleguard.cli import final_verdict as fv

    assert fv._pending("make thing").startswith("PENDING - produced by:")


def test_the_plot_set_covers_every_figure_section_56_names():
    required = {
        "precision-recall curve", "ROC curve", "calibration curve",
        "confusion matrix", "Recall@TopK curve", "Precision@TopK curve",
        "score distribution by class", "fold AP distribution", "seed stability",
        "feature selection frequency", "top SHAP importance", "leakage ablation",
        "alert-context ablation", "feature-subset ablation",
        "positive-removal stability", "adversarial validation / shift plot",
    }
    assert set(plots_final.PLOTS) == required


# ---------------------------------------------------------------------------
# nested promotion: does the shipped champion survive the primary protocol?
# ---------------------------------------------------------------------------
def _nested_payload(leaderboard, repeats=3):
    return {
        "generated_utc": "2026-08-13T00:00:00+00:00",
        "design": {"outer": f"stratified 5-fold x {repeats} repeats",
                   "inner": "stratified 4-fold within outer-train"},
        "leaderboard": leaderboard,
    }


def _family(name, ap, std=0.01, roc=0.95, repeats=3):
    return {"model": name, "protocol": "NESTED", "n_repeats": repeats,
            "pr_auc_mean": ap, "pr_auc_std": std, "roc_auc_mean": roc,
            "feature_size_mode": 120}


def _patch_nested(monkeypatch, tmp_path, payload, deployed_family):
    from muleguard.cli import nested_promotion as np_

    path = tmp_path / "nested_cv.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(np_, "NESTED_JSON", path)
    monkeypatch.setattr(np_, "NESTED_OOF", tmp_path / "absent.parquet")
    monkeypatch.setattr(np_, "_deployed",
                        lambda: {"model": f"{deployed_family}_top_120",
                                 "family": deployed_family})
    return np_


def test_a_partial_nested_run_decides_nothing(tmp_path, monkeypatch):
    """The preliminary run had one family and would have "confirmed" it.

    A protocol that answers before it has a field to compare is not evidence,
    and the honest answer while it is still running is "not yet".
    """
    np_ = _patch_nested(monkeypatch, tmp_path,
                        _nested_payload([_family("xgboost", 0.66792, repeats=1)],
                                        repeats=1),
                        "xgboost")
    payload, rc = np_.decide()
    assert payload["verdict"] == np_.PENDING
    assert rc == 2
    assert "nested_cv" in payload["fills_when"]


def test_mixed_repeat_counts_are_treated_as_a_superseded_run(tmp_path, monkeypatch):
    """Families measured over different repeat counts are not comparable."""
    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("catboost", 0.80653), _family("histgb", 0.76735),
        _family("xgboost", 0.66792, repeats=1),
    ]), "xgboost")
    payload, rc = np_.decide()
    assert payload["verdict"] == np_.PENDING
    assert rc == 2


def test_the_dummy_never_counts_as_a_candidate(tmp_path, monkeypatch):
    """A dummy winning would mean the metric is broken, not that it is good."""
    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("dummy_prevalence", 0.0091), _family("catboost", 0.80653),
        _family("histgb", 0.76735), _family("xgboost", 0.70046),
    ]), "catboost")
    payload, _ = np_.decide()
    assert "dummy_prevalence" not in payload["nested_run"]["families_scored"]
    assert payload["nested_promoted"] != "dummy_prevalence"


def test_a_different_nested_winner_is_reported_as_a_challenge(tmp_path, monkeypatch):
    """The finding is stated and it exits non-zero - it is not acted on."""
    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("catboost", 0.80653, std=0.00845),
        _family("histgb", 0.76735, std=0.02949),
        _family("xgboost", 0.70046, std=0.02362),
    ]), "xgboost")
    payload, rc = np_.decide()
    assert payload["verdict"] == np_.CHALLENGED
    assert rc == 1
    assert payload["nested_promoted"] == "catboost"
    assert payload["nested_gap_over_deployed"] == pytest.approx(0.10607, abs=1e-5)
    # a challenge has to say what acting on it would cost, or it reads as a
    # to-do rather than a decision with the locked test attached
    assert any("locked test" in s for s in payload["what_acting_on_this_costs"])


def test_the_shipped_family_winning_is_a_confirmation(tmp_path, monkeypatch):
    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("xgboost", 0.80653, std=0.00845),
        _family("catboost", 0.76735, std=0.02949),
        _family("histgb", 0.70046, std=0.02362),
    ]), "xgboost")
    payload, rc = np_.decide()
    assert payload["verdict"] == np_.CONFIRMED
    assert rc == 0


def test_a_challenge_carries_the_paired_interval_not_two_marginal_ones(
        tmp_path, monkeypatch):
    """Same rows, same folds - so the interval belongs on the difference.

    Built so the two families are separated by a constant nudge on the positives
    only: every paired replicate must then favour the challenger, while each
    family's own marginal interval would be wide enough to overlap.
    """
    import polars as pl

    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("catboost", 0.80653, std=0.00845),
        _family("histgb", 0.76735, std=0.02949),
        _family("xgboost", 0.75393, std=0.00740),
    ]), "xgboost")

    rng = __import__("numpy").random.default_rng(0)
    n, rows = 600, []
    y = (rng.random(n) < 0.1).astype(int)
    for repeat in range(3):
        base = rng.random(n) + 0.35 * y
        for model, bonus in (("xgboost", 0.0), ("catboost", 0.25)):
            rows.append(pl.DataFrame({
                "model": [model] * n, "repeat": [repeat] * n,
                "row_index": list(range(n)), "target": y,
                "score": base + bonus * y}))
    oof = tmp_path / "nested_oof.parquet"
    pl.concat(rows).write_parquet(oof)
    monkeypatch.setattr(np_, "NESTED_OOF", oof)

    payload, rc = np_.decide()
    assert rc == 1 and payload["verdict"] == np_.CHALLENGED
    paired = payload["paired_check"]
    assert paired["deployed_family"] == "xgboost"
    assert paired["promoted_family"] == "catboost"
    assert paired["repeats_favouring_promoted"] == "3/3"
    assert paired["paired_delta_mean"] > 0
    assert paired["paired_delta_ci95"][0] > 0 and paired["excludes_zero"]


def test_a_confirmed_champion_is_not_given_a_paired_check(tmp_path, monkeypatch):
    """There is no difference to put an interval on when nothing changed."""
    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("xgboost", 0.80653), _family("catboost", 0.76735),
        _family("histgb", 0.70046),
    ]), "xgboost")
    payload, _rc = np_.decide()
    assert "paired_check" not in payload


def test_serving_cost_vetoes_the_nested_leader_too(tmp_path, monkeypatch):
    """The 5 s budget is a property of the product, not of one protocol."""
    np_ = _patch_nested(monkeypatch, tmp_path, _nested_payload([
        _family("tabpfn", 0.92), _family("catboost", 0.80653),
        _family("histgb", 0.76735), _family("xgboost", 0.70046),
    ]), "catboost")
    monkeypatch.setattr(np_, "_serving_cost_by_family",
                        lambda: {"tabpfn": {"measured": True,
                                            "single_row_seconds": 438.1}})
    payload, rc = np_.decide()
    assert payload["nested_promoted"] == "catboost"
    assert rc == 0
    benched = {r["model"] for r in payload["benched_on_serving_cost"]}
    assert benched == {"tabpfn"}


# ---------------------------------------------------------------------------
# the "final_" names must describe the model that is actually shipped
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / "artifacts/features/final_selected_features.json").exists(),
                    reason="selection artifacts not built in this checkout")
def test_the_final_selection_artifact_is_the_post_firewall_one():
    """A spec-named file is read as current whether or not it is.

    These two were first written as copies of the pre-firewall top-60 run and
    were never refreshed, so `final_selected_features.json` described a feature
    set the deployed model does not use. The reconciler now rebuilds them from
    the v2 selection; this test is what stops them drifting back.
    """
    final = json.loads((ROOT / "artifacts/features/final_selected_features.json")
                       .read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / "artifacts/features/selected_features_v2.json")
                    .read_text(encoding="utf-8"))
    gen1 = json.loads((ROOT / "artifacts/features/selected_features.json")
                      .read_text(encoding="utf-8"))

    assert final.get("generated_utc") == v2.get("generated_utc")
    assert final.get("generated_utc") != gen1.get("generated_utc")
    assert final["__provenance__"]["spec_name_of"] == \
        "artifacts/features/selected_features_v2.json"


@pytest.mark.skipif(not (ROOT / "artifacts/features/final_selected_features.json").exists(),
                    reason="selection artifacts not built in this checkout")
def test_no_quarantined_feature_reaches_the_final_selection():
    """The firewall is the reason generation 1 was retired; assert it held."""
    q = json.loads((ROOT / "artifacts/features/quarantined_features.json")
                   .read_text(encoding="utf-8"))
    quarantined = {row["feature"] for row in q["quarantine"]}
    assert len(quarantined) == 13, "the firewall list is 13 columns"
    final = json.loads((ROOT / "artifacts/features/final_selected_features.json")
                       .read_text(encoding="utf-8"))
    named = {f for v in final.values() if isinstance(v, list)
             for f in v if isinstance(f, str)}
    assert not (named & quarantined)


# ---------------------------------------------------------------------------
# the §58 top-level report
# ---------------------------------------------------------------------------
def test_the_report_never_upgrades_a_pending_verdict():
    """A narrative wrapper must not be a second place the verdict is decided."""
    from muleguard.cli import final_report as fr

    ev = fr.Evidence()
    text = fr.render(ev, {"champion": "xgboost_top_120",
                          "verdict": "PENDING_EVIDENCE",
                          "release_blockers": [], "pass_criteria": [],
                          "top_risks": []})
    assert "PENDING_EVIDENCE" in text
    assert "**Verdict: `PASS`**" not in text
    # and it has to say why that string is not one of the three §65 allows
    assert "not one of the three verdict strings" in text


def test_the_report_states_the_refusals_not_just_the_results():
    from muleguard.cli import final_report as fr

    text = fr.render(fr.Evidence(), {"verdict": "PENDING_EVIDENCE"})
    for phrase in ("zero false positives", "guarantee about any individual account",
                   "superiority over other competitors"):
        assert phrase in text, f"§61 refusal missing: {phrase}"


# ---------------------------------------------------------------------------
# §61 honesty rules, enforced on the documents rather than trusted to discipline
# ---------------------------------------------------------------------------
# release_gate c21 scans shipped *source* for the five forbidden verdict labels
# (GUILTY, CERTIFIED_CLEAN, ...). Nothing scanned the prose, which is the half a
# judge actually reads. These are absolute claims about the system, so no
# measurement can support them and no context makes them true.
OVERCLAIMS = (
    r"100\s*%\s*accurate",
    r"(?:near-)?perfect(?:ly)? detect\w*",
    r"never miss(?:es)?\b",
    r"guarantees?(?: to)? (?:detect|catch|find)",
    r"catches (?:all|every)\b",
    r"state[- ]of[- ]the[- ]art accuracy",
    r"eliminates? false positives",
    r"no human review",
)
# A refusal has to be able to name what it refuses - that is the whole point of
# §61 - so a line that negates the phrase passes. Emphasis markers are stripped
# first, because "do **not** claim" is a negation with markdown in the middle.
NEGATIONS = ("no claim", "not claim", "never claim", "do not", "does not",
             "cannot", "refuse", "without claiming", "would be")


def _prose_files():
    return sorted(ROOT.glob("docs/*.md")) + [ROOT / "README.md"]


def test_the_docs_do_not_overclaim():
    hits = []
    for path in _prose_files():
        if not path.exists():
            continue
        for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.replace("*", "").replace("_", " ").lower()
            if any(marker in line for marker in NEGATIONS):
                continue
            for pattern in OVERCLAIMS:
                if re.search(pattern, line):
                    hits.append(f"{path.name}:{n}: {raw.strip()[:90]}")
    assert not hits, "documents overclaim:\n" + "\n".join(hits)


def test_every_test_file_the_docs_cite_exists():
    """Citing a test that was never written is the quietest kind of overclaim.

    Four documents were found citing `test_dictionary.py`, `test_validation_lab.py`,
    `test_evidence_packet.py` and two route files that do not exist; two of those
    were renames, two were coverage nobody ever wrote. A reader auditing this
    project checks the test table first, and a plausible filename in a coverage
    column reads as evidence. This makes the claim mechanical.
    """
    cited = re.compile(r"`(tests/[a-zA-Z0-9_./-]+\.py)`")
    missing = []
    for path in _prose_files():
        if not path.exists():
            continue
        for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for rel in cited.findall(raw):
                if not (ROOT / rel).exists():
                    missing.append(f"{path.name}:{n}: {rel}")
    assert not missing, ("documents cite test files that do not exist - rename the "
                         "reference or say the coverage is absent:\n" + "\n".join(missing))


def test_the_overclaim_guard_would_actually_fire(tmp_path, monkeypatch):
    """A guard that has never failed is indistinguishable from a guard that cannot."""
    doc = tmp_path / "docs"
    doc.mkdir()
    (doc / "SALES.md").write_text("The model is 100% accurate and never misses.",
                                  encoding="utf-8")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="overclaim"):
        test_the_docs_do_not_overclaim()


def test_a_challenged_champion_is_not_softened_in_the_report(monkeypatch):
    """If the primary protocol disagrees, the report has to say so plainly."""
    from muleguard.cli import final_report as fr

    ev = fr.Evidence()
    real = ev.get

    def fake(rel):
        if rel.endswith("nested_promotion_decision.json"):
            return {"verdict": "CHAMPION_CHALLENGED",
                    "why": "the nested protocol promotes catboost, not the shipped xgboost",
                    "nested_gap_over_deployed": 0.10607,
                    "what_acting_on_this_costs": ["re-open the locked test"]}
        return real(rel)

    monkeypatch.setattr(ev, "get", fake)
    text = fr._protocols(ev)
    assert "promotes a different model" in text
    assert "0.10607" in text
    assert "re-open the locked test" in text


def _challenged_decision() -> dict:
    """A challenge in which the shipped family is beaten by two, not by one."""
    return {
        "verdict": "CHAMPION_CHALLENGED",
        "deployed": {"model": "xgboost_top_120", "family": "xgboost"},
        "nested_promoted": "catboost",
        "not_done_automatically": "This file records the finding.",
        "paired_check": {"paired_delta_mean": 0.05279,
                         "paired_delta_ci95": [0.02374, 0.08602],
                         "repeats_favouring_promoted": "3/3"},
        "leaderboard": [{"model": "catboost", "nested_pr_auc_mean": 0.80653},
                        {"model": "histgb", "nested_pr_auc_mean": 0.76735},
                        {"model": "xgboost", "nested_pr_auc_mean": 0.75393},
                        {"model": "lightgbm", "nested_pr_auc_mean": 0.70},
                        {"model": "extratrees", "nested_pr_auc_mean": 0.66},
                        {"model": "logistic_l1l2", "nested_pr_auc_mean": 0.60}],
    }


def test_the_shipped_rank_is_counted_rather_than_inferred_from_being_beaten():
    """Losing to the leader does not make a model the runner-up.

    The tempting sentence is "the primary protocol ranks it second", and it is
    wrong here by one place. The ordinal has to come from the leaderboard.
    """
    from muleguard.cli import final_verdict as fv

    assert "places third of 6" in fv._deployed_rank(_challenged_decision())


def test_the_tournament_table_carries_the_arbiter_verdict(monkeypatch):
    """Two protocols disagreeing in one table is not a finding until named.

    Section D used to end at the last row, leaving the reader to guess which
    protocol binds; the risk list then pointed at a challenge that appeared
    nowhere above it.
    """
    from muleguard.cli import final_verdict as fv

    ev = fv.Evidence()
    real = ev.get

    def fake(rel):
        if rel.endswith("nested_promotion_decision.json"):
            return _challenged_decision()
        return real(rel)

    monkeypatch.setattr(ev, "get", fake)
    text = fv.section_d(ev)
    assert "CHAMPION_CHALLENGED" in text
    assert "0.05279" in text and "0.02374" in text
    assert "not** taken" in text


def test_a_confirmed_champion_gets_no_challenge_paragraph(monkeypatch):
    """The arbiter is quoted either way, but a confirmation is not a warning."""
    from muleguard.cli import final_verdict as fv

    ev = fv.Evidence()
    real = ev.get
    confirmed = dict(_challenged_decision(), verdict="CHAMPION_CONFIRMED",
                     nested_promoted="xgboost")

    def fake(rel):
        if rel.endswith("nested_promotion_decision.json"):
            return confirmed
        return real(rel)

    monkeypatch.setattr(ev, "get", fake)
    text = fv.section_d(ev)
    assert "CHAMPION_CONFIRMED" in text
    assert "places third" not in text and "0.05279" not in text


def _scope_for(monkeypatch, verdict: str, decision: dict) -> str:
    from muleguard.cli import final_verdict as fv

    ev = fv.Evidence()
    real = ev.get

    def fake(rel):
        if rel.endswith("nested_promotion_decision.json"):
            return decision
        return real(rel)

    monkeypatch.setattr(ev, "get", fake)
    return fv._verdict_scope({"verdict": verdict}, ev)


def test_a_pass_says_what_it_certifies_and_what_it_does_not(monkeypatch):
    """PASS is a claim about the programme, not about the model.

    Every section 64 criterion asks whether work was done and done cleanly; none
    asks whether the shipped model is the strongest candidate. Printed bare next
    to a section D that says the champion loses under the primary protocol, the
    word invites exactly the reading the rest of the report spends pages
    refusing, so the range it covers is stated where it is issued.
    """
    from muleguard.cli import final_verdict as fv

    scope = _scope_for(monkeypatch, fv.PASS, _challenged_decision())
    assert "certifies the validation programme" in scope
    assert "not the optimality of the champion" in scope
    assert "catboost" in scope, "the scope note must name the model that beat it"


def test_the_scope_note_does_not_invent_a_challenge_that_did_not_happen(monkeypatch):
    """The paragraph tracks the arbiter; it is not boilerplate."""
    from muleguard.cli import final_verdict as fv

    confirmed = dict(_challenged_decision(), verdict="CHAMPION_CONFIRMED",
                     nested_promoted="xgboost")
    scope = _scope_for(monkeypatch, fv.PASS, confirmed)
    assert "certifies the validation programme" in scope
    assert "swap was declined" not in scope and "risk 5" not in scope
