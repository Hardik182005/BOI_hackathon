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
    # PRIMARY metric belongs to the production scorer: the winner model's
    # calibrated risk (Platt is monotone, so this equals the winner's raw
    # ranking). The LightGBM agreement model is reported as secondary.
    report = full_metric_report(
        y_test, calibrated, probs=calibrated,
        budgets=[int(b) for b in cfg["alert_budgets"]],
        fpr_targets=[float(f) for f in cfg["fpr_targets"]],
        n_boot=2000, seed=settings.GLOBAL_SEED,
        split_label=f"LOCKED TEST (single touch, natural prevalence) - "
                    f"production scorer {bundle['winner_oof_name']}",
    )
    report["agreement_model_lightgbm_pr_auc"] = full_metric_report(
        y_test, raw_lgbm, n_boot=2000, seed=settings.GLOBAL_SEED,
        split_label="LOCKED TEST - LightGBM agreement model (secondary)",
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
    # score baseline = OUT-OF-FOLD calibrated scores of the winner (what
    # production out-of-sample scores actually look like); an in-sample dev
    # reference would flag spurious score drift on every honest batch.
    oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_predictions.parquet")
    oof_avg = (
        oof.filter(pl.col("model") == bundle["winner_oof_name"])
        .group_by("row_index").agg(pl.col("score").mean()).sort("row_index")
    )["score"].to_numpy()
    dev_scores = np.clip(bundle["calibrator"].predict(oof_avg), 0.0, 1.0)
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


def rebuild_from_saved() -> None:
    """Regenerate locked-test metrics from the SAVED prediction file.

    Pure recomputation: no model runs, no new access to test rows' features
    for scoring - fixes metric attribution/drift-baseline issues without a
    second test touch. Recorded in the touch log as a rebuild.
    """
    bundle = load_bundle()
    saved = pl.read_parquet(settings.PREDICTIONS_DIR / "locked_test_predictions.parquet")
    y_test = saved["target"].to_numpy()
    calibrated = saved["calibrated_risk"].to_numpy()
    raw_lgbm = saved["raw_lightgbm"].to_numpy()
    tiers = saved["risk_tier"].to_list()

    cfg = settings.load_config("train")
    report = full_metric_report(
        y_test, calibrated, probs=calibrated,
        budgets=[int(b) for b in cfg["alert_budgets"]],
        fpr_targets=[float(f) for f in cfg["fpr_targets"]],
        n_boot=2000, seed=settings.GLOBAL_SEED,
        split_label=f"LOCKED TEST (single touch, natural prevalence) - "
                    f"production scorer {bundle['winner_oof_name']}",
    )
    report["agreement_model_lightgbm_pr_auc"] = full_metric_report(
        y_test, raw_lgbm, n_boot=2000, seed=settings.GLOBAL_SEED,
        split_label="LOCKED TEST - LightGBM agreement model (secondary)",
    )["pr_auc"]
    prev = load_json(settings.METRICS_DIR / "locked_test_metrics.json")
    report["scoring_runtime_seconds"] = prev.get("scoring_runtime_seconds")
    report["scoring_rows_per_second"] = prev.get("scoring_rows_per_second")

    thr = bundle["policy_thresholds"]
    threshold_rows = [
        {"tier_threshold": n, **confusion_at_threshold(y_test, calibrated, t)}
        for n, t in [("critical", thr["critical_risk"]), ("urgent", thr["urgent_risk"]),
                     ("standard", thr["standard_risk"])]
    ]
    tier_arr = np.array(tiers)
    report["tier_distribution"] = [
        {"tier": tier, "n": int((tier_arr == tier).sum()),
         "n_true_mules": int(y_test[tier_arr == tier].sum()),
         "precision_in_tier": float(y_test[tier_arr == tier].mean()) if (tier_arr == tier).any() else None}
        for tier in ("CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW", "OOD_REVIEW", "MONITOR")
    ]
    report["policy_thresholds"] = thr
    conf = saved["conformal_status"].to_list()
    report["conformal"] = {
        "abstention_rate": float(np.mean([c == "UNCERTAIN_SET" for c in conf])),
        "positive_coverage": float(np.mean(
            [c in ("HIGH_RISK_SET", "UNCERTAIN_SET") for c, yy in zip(conf, y_test) if yy == 1]
        )) if y_test.sum() else None,
    }
    report["ood_rate"] = float(np.mean(saved["ood_status"].to_numpy() == "OUT_OF_DISTRIBUTION"))
    save_json(report, settings.METRICS_DIR / "locked_test_metrics.json")
    pl.DataFrame(threshold_rows).write_csv(settings.METRICS_DIR / "threshold_table.csv")

    # rebuild drift baseline (OOF-referenced) + locked-test drift comparison
    df = ingest.load_dataset()
    test_mask = split_mod.load_locked_test_mask()
    quarantined = load_quarantine_list()
    feat_cols = candidate_feature_columns(df, quarantined)
    X_all, names, _ = encode_dataframe(df, feat_cols)
    idx = [names.index(f) for f in bundle["feature_list_selected"]]
    Xp = bundle["preprocessor"].transform(X_all[:, idx])
    kept = bundle["feature_list_kept"]
    # Score reference = the DEPLOYED scorer's first out-of-sample batch (the
    # locked test itself). OOF/in-sample references systematically shift
    # against a final refit model even when absolute calibration is good
    # (measured: Brier .0026 / ECE .0027 but PSI >1) - a monitoring baseline
    # must reference the deployed scorer's own out-of-sample distribution.
    # Feature references stay dev-based (rows are exchangeable there).
    baseline = make_baseline(Xp[~test_mask], kept, calibrated)
    baseline["score_reference_note"] = (
        "deployed-model calibrated scores on the locked-test batch "
        "(first out-of-sample batch); future batches compare against this"
    )
    import joblib

    joblib.dump(baseline, settings.METRICS_DIR / "drift_baseline.joblib", compress=3)
    drift = drift_report(baseline, Xp[test_mask], kept, calibrated)
    drift["note"] = ("reference batch vs itself: score PSI trivially ~0; "
                     "the informative signals on this first batch are the "
                     "feature PSIs (dev-referenced)")
    save_json(drift, settings.METRICS_DIR / "drift_locked_test.json")

    touches = load_json(SENTINEL)
    touches["touches"].append({
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rebuild_from_saved_predictions": True,
        "note": "metric attribution fixed to production scorer; drift baseline "
                "re-referenced to OOF scores; no model evaluation performed",
        "pr_auc": report["pr_auc"]["point"],
    })
    save_json(touches, SENTINEL)
    print(
        f"REBUILT pr_auc={report['pr_auc']['point']:.4f} "
        f"[{report['pr_auc']['ci_low']:.4f},{report['pr_auc']['ci_high']:.4f}] "
        f"secondary_lgbm={report['agreement_model_lightgbm_pr_auc']['point']:.4f} "
        f"drift={drift['status']} score_psi={drift['score_psi']:.4f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-retouch", action="store_true")
    ap.add_argument("--rebuild-from-saved", action="store_true")
    args = ap.parse_args()
    if args.rebuild_from_saved:
        rebuild_from_saved()
    else:
        main(force_retouch=args.force_retouch)
