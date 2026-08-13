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


def test_titles_carry_split_model_and_data_version():
    subtitle = plots_final._subtitle("dev OOF, nested outer folds", "xgboost_top_120")
    assert "split:" in subtitle and "model:" in subtitle and "data:" in subtitle
    assert "xgboost_top_120" in subtitle


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
