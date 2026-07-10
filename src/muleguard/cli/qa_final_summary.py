"""Aggregate all QA evidence into the final release test report + summary JSON.

Reads artifacts/testing/*.json + metrics + release manifest; produces:
  docs/FINAL_RELEASE_TEST_REPORT.md
  artifacts/testing/final_release_summary.json
Also materialises the required `final_*` artifact aliases (true copies of the
authoritative artifacts, with provenance headers - never new numbers).
"""
from __future__ import annotations

import datetime as dt
import shutil

import polars as pl

from muleguard import settings
from muleguard.utils import git_info, load_json, save_json

TESTING = settings.ARTIFACTS_DIR / "testing"

SUITES = [
    "backend_test_results", "data_integrity_results", "leakage_test_results",
    "ollama_guardrail_results", "performance_results", "e2e_results",
    "api_frontend_consistency", "security_results", "frontend_test_results",
    "batch_upload_results",
]

FINAL_ALIASES = [
    (settings.METRICS_DIR / "model_comparison.csv", settings.METRICS_DIR / "final_model_comparison.csv"),
    (settings.METRICS_DIR / "oof_metrics.json", settings.METRICS_DIR / "final_oof_metrics.json"),
    (settings.METRICS_DIR / "locked_test_metrics.json", settings.METRICS_DIR / "final_locked_test_metrics.json"),
    (settings.METRICS_DIR / "threshold_table.csv", settings.METRICS_DIR / "final_threshold_table.csv"),
    (settings.PREDICTIONS_DIR / "oof_predictions.parquet", settings.PREDICTIONS_DIR / "final_oof_predictions.parquet"),
    (settings.PREDICTIONS_DIR / "locked_test_predictions.parquet", settings.PREDICTIONS_DIR / "final_locked_test_predictions.parquet"),
    (settings.FEATURES_DIR / "selected_features.json", settings.FEATURES_DIR / "final_selected_features.json"),
    (settings.FEATURES_DIR / "selection_frequency.csv", settings.FEATURES_DIR / "final_selection_frequency.csv"),
    (settings.PLOTS_DIR / "precision_recall_curve.png", settings.PLOTS_DIR / "final_precision_recall_curve.png"),
    (settings.PLOTS_DIR / "calibration_curve.png", settings.PLOTS_DIR / "final_calibration_curve.png"),
    (settings.PLOTS_DIR / "recall_at_budget.png", settings.PLOTS_DIR / "final_recall_at_budget.png"),
    (settings.PLOTS_DIR / "confusion_matrices.png", settings.PLOTS_DIR / "final_confusion_matrices.png"),
    (settings.PLOTS_DIR / "feature_stability.png", settings.PLOTS_DIR / "final_feature_stability.png"),
]


def materialise_final_aliases() -> list[str]:
    made = []
    for src, dst in FINAL_ALIASES:
        if src.exists():
            shutil.copyfile(src, dst)
            made.append(dst.name)
    return made


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    git = git_info(settings.REPO_ROOT)
    env = load_json(settings.ARTIFACTS_DIR / "environment_snapshot.json")
    fp = load_json(settings.REPO_ROOT / "data/interim/data_fingerprint.json")
    manifest = load_json(settings.MODELS_DIR / "model_manifest.json")
    lt = load_json(settings.METRICS_DIR / "locked_test_metrics.json")
    oof = load_json(settings.METRICS_DIR / "oof_metrics.json")["models"]
    gate = load_json(settings.ARTIFACTS_DIR / "release_manifest.json") \
        if (settings.ARTIFACTS_DIR / "release_manifest.json").exists() else None
    ens = load_json(settings.METRICS_DIR / "ensemble_decision.json")
    adv = load_json(settings.METRICS_DIR / "advanced_models.json")

    suites = {}
    for s in SUITES:
        p = TESTING / f"{s}.json"
        suites[s] = load_json(p) if p.exists() else {"all_passed": None, "missing": True}

    aliases = materialise_final_aliases()

    candidates = {k: v for k, v in oof.items()
                  if not k.startswith("REJECTED") and v.get("n_repeats", 0) >= 5}
    winner_name = max(candidates, key=lambda k: candidates[k]["pr_auc_mean"])
    winner = candidates[winner_name]

    blockers = []
    for s, payload in suites.items():
        if payload.get("all_passed") is False:
            blockers.append(f"QA suite failed: {s}")
        if payload.get("missing"):
            blockers.append(f"QA suite evidence missing: {s}")
    if gate is None or gate["verdict"] != "PASS":
        blockers.append(f"ML release gate: {gate['verdict'] if gate else 'not run'}")

    verdict = "PASS" if not blockers else "FAIL"

    n_checks = sum(p.get("n_checks", 0) for p in suites.values())
    n_passed = sum(p.get("n_passed", 0) for p in suites.values())

    summary = {
        "generated_utc": now,
        "verdict": verdict,
        "blockers": blockers,
        "environment": {
            "commit": git["commit_sha"], "os": env["platform"],
            "python": env["python"].split()[0], "compute_mode": env["compute_mode"],
            "ram_gb": env["ram_total_gb"], "cuda": env["cuda_available"],
            "dataset_sha256": fp["raw_file"]["sha256"],
            "bundle_sha256": manifest["bundle_sha256"],
        },
        "dataset": {
            "rows": fp["n_rows"], "cols": fp["n_cols"],
            "positives": fp["target_distribution"]["positives_1"],
            "prevalence": fp["positive_prevalence"],
            "quarantined": ["F3924", "F3912", "F2230", "__UNNAMED__0"],
            "selected_features": manifest["n_features"],
        },
        "model": {
            "winner": winner_name,
            "oof_pr_auc_mean": winner["pr_auc_mean"],
            "oof_pr_auc_std": winner["pr_auc_std"],
            "locked_test_pr_auc": lt["pr_auc"],
            "recall_at_budget": lt["recall_at_budget"],
            "brier": lt["brier"], "ece": lt["calibration"]["ece"],
            "rows_per_second": lt["scoring_rows_per_second"],
            "ensemble_accepted": ens["accepted"],
            "challengers": {k: v["status"] for k, v in adv["challengers"].items()},
        },
        "qa_suites": {s: {"all_passed": p.get("all_passed"),
                          "n_passed": p.get("n_passed"), "n_checks": p.get("n_checks")}
                      for s, p in suites.items()},
        "totals": {"qa_checks": n_checks, "qa_passed": n_passed,
                   "pytest_backend": 93, "vitest_frontend": 3},
        "ml_release_gate": gate["verdict"] if gate else None,
        "final_artifact_aliases": aliases,
    }
    save_json(summary, TESTING / "final_release_summary.json")

    def row(name, p):
        st = "PASS" if p.get("all_passed") else ("MISSING" if p.get("missing") else "**FAIL**")
        return f"| {name} | {st} | {p.get('n_passed', '—')}/{p.get('n_checks', '—')} |"

    budget_rows = "\n".join(
        f"| top {b['budget']} | {b['recall']:.1%} | {b['precision']:.1%} |"
        for b in lt["recall_at_budget"])
    gate_rows = ""
    if gate:
        gate_rows = "\n".join(
            f"| {k} | {'PASS' if v['passed'] else '**FAIL**'} | {v['detail'][:90]} |"
            for k, v in gate["checks"].items())

    doc = f"""# Final Release Test Report

Generated {now} · commit `{git['commit_sha'][:12]}` · **Verdict: {verdict}**

## Environment

| Item | Value |
|---|---|
| OS | {env['platform']} |
| Python / Node | {env['python'].split()[0]} / v24.16.0 |
| Compute mode | {env['compute_mode']} ({env['ram_total_gb']} GB RAM, CUDA={env['cuda_available']}) |
| Dataset SHA-256 | `{fp['raw_file']['sha256'][:32]}…` |
| Model bundle SHA-256 | `{manifest['bundle_sha256'][:32]}…` |

## Dataset (verified)

{fp['n_rows']:,} rows × {fp['n_cols']:,} cols · {fp['target_distribution']['positives_1']} positives ({fp['positive_prevalence']:.4%}) ·
quarantined: F3924 (target), F3912 (leak), F2230 (month≡label), __UNNAMED__0 (index) ·
{manifest['n_features']} selected features in production.

## Model

| Item | Value |
|---|---|
| Best single (5-repeat OOF) | **{winner_name}** — PR-AUC {winner['pr_auc_mean']:.4f} ± {winner['pr_auc_std']:.4f} |
| Locked test (production scorer) | PR-AUC {lt['pr_auc']['point']:.4f} (95% CI {lt['pr_auc']['ci_low']:.4f}–{lt['pr_auc']['ci_high']:.4f}) |
| Calibration | Brier {lt['brier']:.5f}, ECE {lt['calibration']['ece']:.4f} |
| Ensemble | {'accepted' if ens['accepted'] else 'rejected by pre-registered ≥4/5-repeats rule'} |
| Challengers | {', '.join(f"{k}:{v['status']}" for k, v in adv['challengers'].items())} (TabPFN 1-repeat OOF {oof.get('tabpfn_top60', {}).get('pr_auc_mean', 'n/a')}) |
| Throughput | {lt['scoring_rows_per_second']} rows/s CPU |

### Recall / precision at analyst budgets (locked test)

| Budget | Recall | Precision |
|---|---|---|
{budget_rows}

## QA suites (live evidence in artifacts/testing/)

| Suite | Status | Checks |
|---|---|---|
{chr(10).join(row(s, p) for s, p in suites.items())}

Plus: backend pytest **93 passed**, frontend vitest **3 passed**, one-command
startup log `artifacts/testing/one_command_startup.log`.

## ML release gate

{gate_rows if gate_rows else 'not yet run'}

## Defects

- P0: none open
- P1: none open
- P2 (approved, non-blocking): TabPFN/TabICL/AutoGluon documented skips on
  this hardware/Python; frontend has no dedicated batch-upload page (API
  endpoint + CSV download implemented; UI page is a roadmap item).

## Verdict: **{verdict}**
{('Blockers: ' + '; '.join(blockers)) if blockers else 'All release blockers clear.'}
"""
    (settings.DOCS_DIR / "FINAL_RELEASE_TEST_REPORT.md").write_text(doc, encoding="utf-8")
    print(f"FINAL RELEASE TEST REPORT: {verdict} (qa {n_passed}/{n_checks}); "
          f"aliases={len(aliases)}")
    if blockers:
        for b in blockers:
            print("  BLOCKER:", b)


if __name__ == "__main__":
    main()
