"""Locked-test evaluation - THE single touch.

Loads the frozen bundle, scores the locked test rows once, writes:
  artifacts/predictions/locked_test_predictions.parquet
  artifacts/metrics/locked_test_metrics.json
  artifacts/metrics/threshold_table.csv
  drift baseline (dev) + locked-test drift check

A sentinel file records that the locked test has been consumed; a second run
refuses without --force-retouch (which is itself recorded in the sentinel
history as a protocol deviation).
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.data import ingest, split as split_mod
from muleguard.evaluation.metrics import confusion_at_threshold, full_metric_report
from muleguard.features.preprocessing import candidate_feature_columns, encode_dataframe, load_quarantine_list
from muleguard.logging import get_logger
from muleguard.models.scoring import load_bundle, score_rows
from muleguard.monitoring.drift import drift_report, make_baseline
from muleguard.utils import load_json, save_json, set_global_seed

log = get_logger("cli.evaluate")

SENTINEL = settings.METRICS_DIR / "locked_test_touch_log.json"


def main(force_retouch: bool = False) -> None:
    set_global_seed(settings.GLOBAL_SEED)
    touches = load_json(SENTINEL) if SENTINEL.exists() else {"touches": []}
    if touches["touches"] and not force_retouch:
        raise SystemExit(
            f"locked test already evaluated at {touches['touches'][-1]['utc']} - "
            "re-evaluation is a protocol violation. Use --force-retouch only to "
            "reproduce the identical artifact; the deviation will be logged."
        )

    bundle = load_bundle()
    df = ingest.load_dataset()
    test_mask = split_mod.load_locked_test_mask()
    test_rows = df.filter(pl.Series(test_mask))
    y_test = test_rows[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()

    t0 = time.perf_counter()
    results = score_rows(test_rows, with_explanations=False)
    elapsed = time.perf_counter() - t0
    calibrated = np.array([r["calibrated_risk"] for r in results])
    raw_lgbm = np.array([r["raw_scores"]["lightgbm"] for r in results])
    tiers = [r["risk_tier"] for r in results]

    cfg = settings.load_config("train")
    report = full_metric_report(
        y_test, raw_lgbm, probs=calibrated,
        budgets=[int(b) for b in cfg["alert_budgets"]],
        fpr_targets=[float(f) for f in cfg["fpr_targets"]],
        n_boot=2000, seed=settings.GLOBAL_SEED,
        split_label="LOCKED TEST (single touch, natural prevalence)",
    )
    report["calibrated_pr_auc"] = full_metric_report(
        y_test, calibrated, n_boot=2000, seed=settings.GLOBAL_SEED,
        split_label="LOCKED TEST calibrated scores",
    )["pr_auc"]
    report["scoring_runtime_seconds"] = round(elapsed, 2)
    report["scoring_rows_per_second"] = round(len(y_test) / elapsed, 1)

    # tier confusion + per-tier precision
    thr = bundle["policy_thresholds"]
    threshold_rows = []
    for name, t in [("critical", thr["critical_risk"]), ("urgent", thr["urgent_risk"]),
                    ("standard", thr["standard_risk"])]:
        threshold_rows.append({"tier_threshold": name, **confusion_at_threshold(y_test, calibrated, t)})
    tier_arr = np.array(tiers)
    tier_stats = []
    for tier in ("CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW", "OOD_REVIEW", "MONITOR"):
        m = tier_arr == tier
        tier_stats.append({
            "tier": tier, "n": int(m.sum()),
            "n_true_mules": int(y_test[m].sum()),
            "precision_in_tier": float(y_test[m].mean()) if m.any() else None,
        })
    report["tier_distribution"] = tier_stats
    report["policy_thresholds"] = thr
    report["conformal"] = {
        "abstention_rate": float(np.mean([r["conformal_status"] == "UNCERTAIN_SET" for r in results])),
        "positive_coverage": float(np.mean(
            [r["conformal_status"] in ("HIGH_RISK_SET", "UNCERTAIN_SET")
             for r, yy in zip(results, y_test) if yy == 1])) if y_test.sum() else None,
    }
    report["ood_rate"] = float(np.mean([r["ood_status"] == "OUT_OF_DISTRIBUTION" for r in results]))

    save_json(report, settings.METRICS_DIR / "locked_test_metrics.json")
    pl.DataFrame(threshold_rows).write_csv(settings.METRICS_DIR / "threshold_table.csv")
    pl.DataFrame({
        "row_index": np.where(test_mask)[0],
        "target": y_test,
        "raw_lightgbm": raw_lgbm,
        "calibrated_risk": calibrated,
        "risk_tier": tiers,
        "conformal_status": [r["conformal_status"] for r in results],
        "ood_status": [r["ood_status"] for r in results],
        "anomaly_percentile": [r["anomaly_percentile"] for r in results],
        "model_agreement": [r["model_agreement"] for r in results],
    }).write_parquet(settings.PREDICTIONS_DIR / "locked_test_predictions.parquet")

    # ---- drift baseline (dev) + locked-test batch as first comparison -----
    quarantined = load_quarantine_list()
    feat_cols = candidate_feature_columns(df, quarantined)
    X_all, names, _ = encode_dataframe(df, feat_cols)
    selected = bundle["feature_list_selected"]
    idx = [names.index(f) for f in selected]
    Xsel = X_all[:, idx]
    prep = bundle["preprocessor"]
    Xp = prep.transform(Xsel)
    kept = bundle["feature_list_kept"]
    dev_scores = bundle["calibrator"].predict(
        bundle["models"]["lightgbm"].predict_proba(Xp[~test_mask])[:, 1]
    )
    baseline = make_baseline(Xp[~test_mask], kept, np.asarray(dev_scores))
    save_json({"summary": {k: v for k, v in baseline.items() if k != "features" and k != "score_reference"}},
              settings.METRICS_DIR / "drift_baseline_summary.json")
    import joblib

    joblib.dump(baseline, settings.METRICS_DIR / "drift_baseline.joblib", compress=3)
    drift = drift_report(baseline, Xp[test_mask], kept, calibrated)
    save_json(drift, settings.METRICS_DIR / "drift_locked_test.json")

    touches["touches"].append({
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "forced": force_retouch,
        "bundle_sha": load_json(settings.MODELS_DIR / "model_manifest.json")["bundle_sha256"][:16],
        "pr_auc": report["pr_auc"]["point"],
    })
    save_json(touches, SENTINEL)

    print(
        f"LOCKED TEST DONE pr_auc={report['pr_auc']['point']:.4f} "
        f"[{report['pr_auc']['ci_low']:.4f},{report['pr_auc']['ci_high']:.4f}] "
        f"roc={report['roc_auc']['point']:.4f} brier={report.get('brier'):.5f} "
        f"ood_rate={report['ood_rate']:.4f} drift={drift['status']} "
        f"rows/s={report['scoring_rows_per_second']}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-retouch", action="store_true")
    main(force_retouch=ap.parse_args().force_retouch)
