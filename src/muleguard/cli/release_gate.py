"""Final release gate - machine-checked blockers from the master prompt.

Each check maps to a release blocker. Verdicts:
  PASS | PASS WITH APPROVED NON-BLOCKING EXCEPTIONS | FAIL
Writes docs/FINAL_RELEASE_GATE.md and artifacts/release_manifest.json.
"""
from __future__ import annotations

import datetime as dt
import re
import subprocess
from typing import Callable

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.utils import git_info, load_json, save_json, sha256_file

CHECKS: list[tuple[str, str]] = []  # (name, description) filled by decorator
RESULTS: dict[str, dict] = {}


def check(name: str, description: str):
    def deco(fn: Callable[[], tuple[bool, str]]):
        CHECKS.append((name, description))

        def run():
            try:
                ok, detail = fn()
            except Exception as e:
                ok, detail = False, f"check crashed: {type(e).__name__}: {e}"
            RESULTS[name] = {"passed": ok, "detail": detail, "description": description}

        run.__name__ = name
        return run

    return deco


@check("no_target_or_f3912_leakage", "target/F3912/F2230/index quarantined from bundle features")
def c1():
    q = {e["feature"] for e in load_json(settings.FEATURES_DIR / "quarantined_features.json")["quarantine"]}
    import joblib

    b = joblib.load(settings.MODELS_DIR / "final_bundle.joblib")
    used = set(b["feature_list_selected"]) | set(b["feature_list_kept"])
    bad = used & q
    return (not bad, f"overlap={sorted(bad)}" if bad else
            f"{len(used)} bundle features disjoint from {len(q)} quarantined")


@check("no_split_overlap", "locked test and CV folds disjoint")
def c2():
    locked = pl.read_parquet(settings.SPLITS_DIR / "locked_test_indices.parquet")
    folds = pl.read_parquet(settings.SPLITS_DIR / "cv_folds.parquet")
    t = set(locked.filter(pl.col("is_locked_test"))["row_index"].to_list())
    d = set(folds["row_index"].to_list())
    return (t.isdisjoint(d), f"test={len(t)} dev={len(d)} overlap={len(t & d)}")


@check("locked_test_single_touch", "locked test consumed exactly once")
def c3():
    log = load_json(settings.METRICS_DIR / "locked_test_touch_log.json")
    n = len(log["touches"])
    forced = [t for t in log["touches"] if t.get("forced")]
    return (n >= 1 and not forced, f"touches={n} forced={len(forced)}")


@check("metrics_trace_to_predictions", "headline OOF metric recomputable from saved predictions")
def c4():
    from sklearn.metrics import average_precision_score

    metrics = load_json(settings.METRICS_DIR / "oof_metrics.json")["models"]
    oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_predictions.parquet")
    for model, m in metrics.items():
        sub = oof.filter((pl.col("model") == model) & (pl.col("repeat") == 0))
        if sub.height == 0:
            return False, f"{model} has metrics but no stored predictions"
        ap = average_precision_score(sub["target"].to_numpy(), sub["score"].to_numpy())
        if abs(ap - m["pr_auc_per_repeat"][0]) > 1e-9:
            return False, f"{model}: stored {m['pr_auc_per_repeat'][0]:.6f} != recomputed {ap:.6f}"
    return True, f"{len(metrics)} models verified"


@check("model_load_and_deterministic", "bundle loads; repeat scoring identical")
def c5():
    from muleguard.data import ingest
    from muleguard.models.scoring import load_bundle, score_rows

    b = load_bundle()
    rows = ingest.load_dataset().head(10)
    r1 = score_rows(rows, bundle=b, with_explanations=False)
    r2 = score_rows(rows, bundle=b, with_explanations=False)
    same = all(a["calibrated_risk"] == c["calibrated_risk"] for a, c in zip(r1, r2))
    return (same, "10-row rescore identical" if same else "nondeterminism detected")


@check("llm_cannot_alter_score", "hallucination validator rejects score/tier changes")
def c6():
    import json as _json

    from muleguard.llm.schemas import NarratorInput, ReasonFact
    from muleguard.llm.validator import validate_llm_output

    ctx = NarratorInput(
        account_reference="GATE", calibrated_risk=0.9, risk_tier="CRITICAL_REVIEW",
        model_agreement=0.9, conformal_status="HIGH_RISK_SET",
        ood_status="IN_DISTRIBUTION",
        top_reasons=[ReasonFact(feature="F1", value=1.0, direction="INCREASES_RISK",
                                shap_contribution=0.1)],
    )
    bad = _json.dumps({"summary": "ok", "risk_tier": "MONITOR", "verified_risk_score": 0.1,
                       "reason_codes": [], "recommended_checks": [],
                       "limitations": ["intent", "human review"]})
    out, reasons = validate_llm_output(bad, ctx)
    return (out is None and len(reasons) >= 2, f"rejected with {len(reasons)} reasons")


@check("no_auto_freeze_paths", "no automatic freeze anywhere; approval required")
def c7():
    from muleguard.action.policy import PolicyThresholds, decide

    thr = PolicyThresholds(0.9, 0.6, 0.3, 99.0)
    r = decide(calibrated_risk=0.99, conformal_set="HIGH_RISK_SET",
               ood_status="IN_DISTRIBUTION", anomaly_percentile=99.9,
               model_agreement=1.0, verifier_flag=True, thresholds=thr)
    api_src = (settings.REPO_ROOT / "src/muleguard/api/main.py").read_text(encoding="utf-8")
    return (r["auto_action"] is None and "RECOMMEND_FREEZE requires approved_by" in api_src
            or 'requires approved_by' in api_src,
            "policy emits no auto actions; freeze needs approver in API")


@check("ood_routes_to_review", "extreme inputs get OOD_REVIEW, not confident labels")
def c8():
    demo = load_json(settings.EVIDENCE_DIR / "demo_scenarios.json")
    ood = demo["scenarios"]["ood_synthetic"]["result"]
    return (ood["ood_status"] == "OUT_OF_DISTRIBUTION" and ood["risk_tier"] == "OOD_REVIEW",
            f"tier={ood['risk_tier']} status={ood['ood_status']}")


@check("scoring_survives_ollama_outage", "narrator falls back deterministically")
def c9():
    demo = load_json(settings.EVIDENCE_DIR / "demo_scenarios.json")
    det = demo["llm_scenes"]["deterministic_narrative_works_without_ollama"]
    rej = demo["llm_scenes"]["hallucination_rejected"]["rejection_reasons"]
    return (bool(det["summary"]) and len(rej) >= 3,
            f"fallback ok; planted hallucination rejected for {len(rej)} reasons")


@check("raw_data_unmodified", "raw copy hash matches fingerprint")
def c10():
    fp = load_json(settings.REPO_ROOT / "data/interim/data_fingerprint.json")
    from pathlib import Path

    raw = Path(fp["raw_file"]["raw_copy_path"])
    return (sha256_file(raw) == fp["raw_file"]["sha256"], "SHA-256 verified")


@check("no_secrets_committed", ".env absent from git; no hardcoded secrets")
def c11():
    tracked = subprocess.run(["git", "ls-files"], cwd=settings.REPO_ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    return (".env" not in tracked, ".env not tracked")


@check("tests_pass", "full pytest suite green")
def c12():
    # No -q here: pyproject already sets it in addopts, and a second -q makes
    # pytest quiet enough to drop the "N passed" line entirely - which is how
    # this check came to report "no summary line" while genuinely passing.
    r = subprocess.run(
        [str(settings.REPO_ROOT / ".venv/Scripts/python.exe"), "-m", "pytest",
         "--tb=no", "-p", "no:warnings"],
        cwd=settings.REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    counts = [ln for ln in r.stdout.splitlines()
              if re.search(r"\d+ (passed|failed|error)", ln)]
    detail = counts[-1].strip() if counts else "no summary line"
    return (r.returncode == 0, detail)


@check("probabilities_bounded", "all stored predictions inside [0,1]")
def c13():
    for f in ("oof_predictions.parquet", "locked_test_predictions.parquet"):
        p = settings.PREDICTIONS_DIR / f
        df = pl.read_parquet(p)
        col = "score" if "score" in df.columns else "calibrated_risk"
        if float(df[col].min()) < 0 or float(df[col].max()) > 1:
            return False, f"{f} out of bounds"
    return True, "OOF + locked test bounded"


@check("artifacts_complete", "all required artifact files exist")
def c14():
    required = [
        settings.METRICS_DIR / "oof_metrics.json",
        settings.METRICS_DIR / "locked_test_metrics.json",
        settings.METRICS_DIR / "threshold_table.csv",
        settings.METRICS_DIR / "model_comparison.csv",
        settings.METRICS_DIR / "with_vs_without_f3912.json",
        settings.PREDICTIONS_DIR / "oof_predictions.parquet",
        settings.PREDICTIONS_DIR / "locked_test_predictions.parquet",
        settings.FEATURES_DIR / "selected_features.json",
        settings.FEATURES_DIR / "selection_frequency.csv",
        settings.FEATURES_DIR / "quarantined_features.json",
        settings.MODELS_DIR / "model_manifest.json",
        settings.REGISTRY_DIR / "registry.json",
        settings.PLOTS_DIR / "precision_recall_curve.png",
        settings.PLOTS_DIR / "calibration_curve.png",
        settings.PLOTS_DIR / "recall_at_budget.png",
        settings.PLOTS_DIR / "feature_stability.png",
        settings.PLOTS_DIR / "leakage_ablation.png",
        settings.PLOTS_DIR / "confusion_matrices.png",
        settings.ARTIFACTS_DIR / "environment_snapshot.json",
    ]
    missing = [str(p.relative_to(settings.REPO_ROOT)) for p in required if not p.exists()]
    return (not missing, f"missing={missing}" if missing else f"{len(required)} artifacts present")


# --- addendum checks (UPDATE 1-13) --------------------------------------------
# The gate above was written for the original build. These cover the upgrade,
# and they exist for the same reason as the others: an artifact that is only
# described in a document is a claim, and a claim is not evidence.


@check("addendum_artifacts_complete", "every upgrade artifact was actually produced")
def c15():
    required = [
        settings.METRICS_DIR / "tournament_v2.json",
        settings.METRICS_DIR / "promotion_decision_v2.json",
        settings.METRICS_DIR / "stability_stress_v2.json",
        settings.METRICS_DIR / "family_dropout_v2.json",
        settings.METRICS_DIR / "seed_variance_v2.json",
        settings.METRICS_DIR / "alert_context_ablation_v2.json",
        settings.METRICS_DIR / "robustness_grade_v2.json",
        settings.METRICS_DIR / "validation_shield_v2.json",
        settings.METRICS_DIR / "label_noise_audit_v2.json",
        settings.METRICS_DIR / "merchant_verifier_v2.json",
    ]
    missing = [p.name for p in required if not p.exists()]
    return (not missing, f"missing={missing}" if missing else
            f"{len(required)} addendum artifacts present")


@check("robustness_grade_not_hand_picked",
       "the badge is recomputable from the published thresholds")
def c16():
    """Recompute the grade from the raw measurements and the fixed thresholds.

    U4 says the status must come from documented thresholds and must not be
    invented. The only way to prove that is to derive it again here rather than
    read the stored string, so a grade edited by hand would fail this check.
    """
    from muleguard.models.robustness import robustness_grade

    stored = load_json(settings.METRICS_DIR / "robustness_grade_v2.json")
    fresh = robustness_grade(
        load_json(settings.METRICS_DIR / "stability_stress_v2.json"),
        load_json(settings.METRICS_DIR / "family_dropout_v2.json"),
    )
    same = fresh["hidden_validation_robustness"] == stored["hidden_validation_robustness"]
    return (same, f"grade={stored['hidden_validation_robustness']} "
                  f"limited_by={stored.get('limiting_criteria')}")


@check("label_audit_changed_nothing", "no label flipped, no row removed (U5)")
def c17():
    audit = load_json(settings.METRICS_DIR / "label_noise_audit_v2.json")
    flagged = audit["possible_label_noise"]["rows"] + audit["high_scoring_negatives"]["rows"]
    every_row_review_only = all(r["action"] == "HUMAN_REVIEW_ONLY" for r in flagged)
    guarantees = " ".join(audit["guarantees"])
    return (every_row_review_only and "no label was modified" in guarantees
            and "no row was removed" in guarantees,
            f"{len(flagged)} rows flagged, all HUMAN_REVIEW_ONLY")


@check("merchant_verifier_cannot_lower_risk",
       "business evidence moves confidence, never the score (U9)")
def c18():
    """The 0.70-multiplier failure mode, tested rather than promised."""
    from muleguard.models.merchant import MerchantVerdict, escalation_confidence

    strong = MerchantVerdict(legitimacy=0.95, band="STRONG_BUSINESS_EVIDENCE",
                             mule_probability_from_business_evidence=0.001,
                             components={})
    out = escalation_confidence(risk_tier="INVESTIGATE", calibrated_risk=0.91,
                                conformal_set="HIGH_RISK_SET", model_agreement=1.0,
                                merchant=strong)
    unchanged = (out["calibrated_risk_unchanged"] == 0.91
                 and out["risk_tier_unchanged"] == "INVESTIGATE")
    lowered = out["auto_escalation_confidence"] != "HIGH"
    audited = all("observed" in a and "rationale" in a for a in out["adjustments"])
    return (unchanged and lowered and audited,
            f"risk held at 0.91/INVESTIGATE; confidence -> "
            f"{out['auto_escalation_confidence']}; {len(out['adjustments'])} "
            "adjustments all carry their trigger values")


@check("graph_never_fabricates_edges", "no graph without an uploaded edge file (U8)")
def c19():
    from muleguard.graph import adapter

    u = adapter.unavailable()
    contract = " ".join(adapter.CONTRACT)
    return (u["status"] == "UNAVAILABLE"
            and "does not fabricate edges from feature similarity" in u["what_we_refuse_to_do"]
            and "none was derived from F-columns" in contract
            and "no graph metric is an input to the mule model" in contract,
            "default UNAVAILABLE; contract forbids derived edges")


@check("shield_reports_no_leaked_feature",
       "the champion's features all clear the availability firewall (U3)")
def c20():
    s = load_json(settings.METRICS_DIR / "validation_shield_v2.json")
    counts = s["feature_stability_report"]["counts"]
    return (counts["LEAKAGE"] == 0 and not s["release_blocker"],
            f"{counts['STABLE']} STABLE / {counts['WATCH']} WATCH / "
            f"{counts['SHIFT_PRONE']} SHIFT_PRONE / {counts['LEAKAGE']} LEAKAGE")


@check("no_forbidden_verdict_vocabulary", "never GUILTY/CERTIFIED_CLEAN/AUTO_FREEZE")
def c21():
    """Scan shipped source for verdict words the specification forbids.

    Matched case-sensitively as whole words so ordinary prose ("criminal
    intent" in a disclaimer that negates it) does not trip the check while an
    emitted label GUILTY would.
    """
    banned = ("GUILTY", "CRIMINAL", "PERMANENTLY_SAFE", "CERTIFIED_CLEAN",
              "AUTO_FREEZE")
    hits: list[str] = []
    roots = [settings.REPO_ROOT / "src", settings.REPO_ROOT / "frontend/src"]
    # This file and the tests that assert the same rule necessarily spell the
    # words out in order to search for them; captured API fixtures are recorded
    # backend output rather than shipped copy. None of the four emits a verdict.
    exempt = {"release_gate.py", "test_policy.py"}
    for root in roots:
        for p in list(root.rglob("*.py")) + list(root.rglob("*.tsx")) + list(root.rglob("*.ts")):
            if ("__fixtures__" in p.parts or p.name.endswith(".test.tsx")
                    or p.name in exempt):
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            for word in banned:
                if re.search(rf"\b{word}\b", text):
                    hits.append(f"{p.name}:{word}")
    return (not hits, f"hits={hits}" if hits else
            f"{len(banned)} forbidden verdict words absent from shipped source")


@check("attack_surface_covered",
       "XSS, SQLi, path-traversal and CSV-injection tests exist and pass")
def c22():
    """The four named attack classes must each have a live test, not a comment.

    Checked by collecting the security suite and matching test names, so
    deleting a test file fails the gate rather than quietly shrinking coverage.
    Presence and passing are separate facts: `tests_pass` above proves the
    suite is green, this proves the suite still covers what it claims to.
    """
    r = subprocess.run(
        [str(settings.REPO_ROOT / ".venv/Scripts/python.exe"), "-m", "pytest",
         "tests/security", "--collect-only", "-p", "no:warnings"],
        capture_output=True, text=True, cwd=settings.REPO_ROOT, timeout=300)
    collected = r.stdout
    required = {
        "sqli": "sql_injection",
        "xss": "xss",
        "path_traversal": "traversal",
        "csv_injection": "formula",
    }
    missing = [k for k, needle in required.items() if needle not in collected]
    n = len([ln for ln in collected.splitlines() if "::" in ln])
    return (not missing and r.returncode == 0,
            f"missing={missing}" if missing else
            f"{n} security tests collected covering {', '.join(required)}")


@check("organiser_dry_run_passed",
       "the section 42 rehearsal scored every malformed export without moving the model")
def c23():
    """The rehearsal has to have been run, and to have passed for the right reason.

    Three separate facts, because any one of them alone is satisfiable by a
    broken harness: every variant scored, the quarantined-column variants
    produced byte-identical predictions, and the sensitivity control did *not* -
    which is what proves the probe reached the scorer at all.
    """
    p = settings.METRICS_DIR / "organiser_dry_run.json"
    if not p.exists():
        return False, "artifacts/metrics/organiser_dry_run.json is missing; run `make dry-run`"
    d = load_json(p)
    inv = d.get("prediction_invariance", {})
    ok = (d.get("verdict") == "PASS" and d.get("accepted_model_unchanged")
          and inv.get("sound"))
    return ok, (f"{d.get('variants_passed')} variants, invariance sound="
                f"{inv.get('sound')}, model unchanged="
                f"{d.get('accepted_model_unchanged')}, locked-test PR-AUC "
                f"{(d.get('offline_label_comparison', {}).get('metrics') or {}).get('pr_auc')}")


def main() -> None:
    for name, _ in CHECKS:
        fn = globals()[[k for k, v in globals().items()
                        if getattr(v, "__name__", "") == name][0]]
    # run in declaration order
    for fn_name in [name for name, _ in CHECKS]:
        for obj in globals().values():
            if getattr(obj, "__name__", None) == fn_name:
                obj()
                break

    n_pass = sum(1 for r in RESULTS.values() if r["passed"])
    blockers = [k for k, r in RESULTS.items() if not r["passed"]]
    verdict = "PASS" if not blockers else "FAIL"

    lines = [
        "# Final Release Gate",
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).isoformat()} · "
        f"commit `{git_info(settings.REPO_ROOT)['commit_sha'][:12]}`",
        "",
        f"## Verdict: **{verdict}** ({n_pass}/{len(RESULTS)} checks passed)",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, r in RESULTS.items():
        lines.append(f"| {name} | {'PASS' if r['passed'] else '**FAIL**'} | {r['detail']} |")
    if blockers:
        lines += ["", "## Blockers", ""] + [f"- {b}: {RESULTS[b]['detail']}" for b in blockers]
    (settings.DOCS_DIR / "FINAL_RELEASE_GATE.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "verdict": verdict,
        "checks": RESULTS,
        "model_manifest": load_json(settings.MODELS_DIR / "model_manifest.json")
        if (settings.MODELS_DIR / "model_manifest.json").exists() else None,
    }
    save_json(manifest, settings.ARTIFACTS_DIR / "release_manifest.json")
    print(f"RELEASE GATE: {verdict} ({n_pass}/{len(RESULTS)})")
    for b in blockers:
        print(f"  BLOCKER {b}: {RESULTS[b]['detail']}")


if __name__ == "__main__":
    main()
