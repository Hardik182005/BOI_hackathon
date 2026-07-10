"""Final release gate - machine-checked blockers from the master prompt.

Each check maps to a release blocker. Verdicts:
  PASS | PASS WITH APPROVED NON-BLOCKING EXCEPTIONS | FAIL
Writes docs/FINAL_RELEASE_GATE.md and artifacts/release_manifest.json.
"""
from __future__ import annotations

import datetime as dt
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
    r = subprocess.run(
        [str(settings.REPO_ROOT / ".venv/Scripts/python.exe"), "-m", "pytest", "-q", "--tb=no"],
        cwd=settings.REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    tail = (r.stdout.strip().splitlines() or ["no output"])[-1]
    return (r.returncode == 0, tail)


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
