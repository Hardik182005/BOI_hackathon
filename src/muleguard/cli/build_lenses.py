"""Build the Trinetra lens stack and freeze the final scoring bundle.

Everything is fitted on DEV data / OOF predictions only:
  Lens 1: final base models (winner feature set) trained on dev
  Lens 2: calibrator selected+fit on OOF, Mondrian conformal on crossfit-
          calibrated OOF probabilities, hard-negative verifier from OOF FPs
  Lens 3: IsolationForest challenger + OOD detector on the reduced dev view
  Policy: tier thresholds derived from dev OOF calibrated-risk distribution

Output: artifacts/models/final_bundle.joblib (+ model_manifest.json,
model_registry/registry.json, policy_snapshot.json). Locked test remains
untouched; muleguard.cli.evaluate consumes this bundle exactly once.
"""
from __future__ import annotations

import datetime as dt

import joblib
import numpy as np
import polars as pl

from muleguard import settings
from muleguard.action.policy import PolicyThresholds
from muleguard.data import ingest, split as split_mod
from muleguard.evaluation.metrics import full_metric_report
from muleguard.explain.reason_codes import CohortReference
from muleguard.features.preprocessing import (
    FoldPreprocessor,
    candidate_feature_columns,
    encode_dataframe,
    load_quarantine_list,
)
from muleguard.logging import get_logger
from muleguard.models import core_models
from muleguard.models.anomaly import AnomalyChallenger, OODDetector
from muleguard.models.calibration import (
    crossfit_calibrated,
    fit_final_calibrator,
    select_calibrator,
)
from muleguard.models.conformal import MondrianConformal, empirical_coverage
from muleguard.models.hard_negative import HardNegativeVerifier, mine_hard_negatives
from muleguard.utils import git_info, load_json, save_json, set_global_seed, sha256_file

log = get_logger("cli.build_lenses")

WINNER_OOF = "lightgbm_tuned_top60"  # confirmed against oof_metrics at runtime
BASE_MODELS = ["lightgbm", "xgboost", "catboost"]


def main() -> None:
    set_global_seed(settings.GLOBAL_SEED)
    thr_cfg = settings.load_config("thresholds")

    # ---- data (dev only) ------------------------------------------------
    df = ingest.load_dataset()
    quarantined = load_quarantine_list()
    feat_cols = candidate_feature_columns(df, quarantined)
    X_all, names, _ = encode_dataframe(df, feat_cols)
    y_all = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    test_mask = split_mod.load_locked_test_mask()
    dev_mask = ~test_mask
    Xdev, ydev = X_all[dev_mask], y_all[dev_mask]

    # ---- winner + feature list ------------------------------------------
    oof_metrics = load_json(settings.METRICS_DIR / "oof_metrics.json")["models"]
    candidates = {k: v for k, v in oof_metrics.items() if not k.startswith("REJECTED")}
    winner_name = max(candidates, key=lambda k: candidates[k]["pr_auc_mean"])
    ens = load_json(settings.METRICS_DIR / "ensemble_decision.json")
    log.info("OOF winner: %s (PR-AUC %.4f); ensemble accepted=%s",
             winner_name, candidates[winner_name]["pr_auc_mean"], ens["accepted"])

    selected = load_json(settings.FEATURES_DIR / "selected_features.json")["compact_sets"]["top_60"]
    name_idx = {f: i for i, f in enumerate(names)}
    cols = np.array([name_idx[f] for f in selected])
    Xdev_sel = Xdev[:, cols]

    # ---- dev-level preprocessor (constants/dupes on full dev) -----------
    prep = FoldPreprocessor(mode="tree").fit(Xdev_sel, selected)
    Xdev_p = prep.transform(Xdev_sel)
    kept = prep.kept_features

    # ---- Lens 1: final base models on dev --------------------------------
    finals = {}
    for m in BASE_MODELS:
        _, model = core_models.SCORERS[m](
            Xdev_p, ydev, Xdev_p[:2], settings.GLOBAL_SEED, return_model=True
        )
        finals[m] = model
    log.info("final base models trained on dev (%d features kept)", len(kept))

    # ---- OOF winner scores for lens fitting ------------------------------
    oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_predictions.parquet")
    won = oof.filter(pl.col("model") == winner_name)
    # average score across repeats per row (stable OOF estimate)
    agg = won.group_by("row_index").agg(
        pl.col("score").mean().alias("score"), pl.col("target").first().alias("target")
    ).sort("row_index")
    oof_scores = agg["score"].to_numpy()
    oof_y = agg["target"].to_numpy()

    # ---- Lens 2: calibration ---------------------------------------------
    cal_sel = select_calibrator(oof_scores, oof_y, seed=settings.GLOBAL_SEED)
    calibrator = fit_final_calibrator(oof_scores, oof_y, cal_sel["winner"])
    oof_calibrated = crossfit_calibrated(oof_scores, oof_y, cal_sel["winner"],
                                         seed=settings.GLOBAL_SEED)
    log.info("calibrator: %s (%s)", cal_sel["winner"], cal_sel["comparison"])

    # ---- Lens 2: conformal ------------------------------------------------
    conformal = MondrianConformal(alpha=float(thr_cfg["conformal"]["alpha"]))
    conformal.fit(oof_calibrated, oof_y)
    coverage = empirical_coverage(conformal, oof_calibrated, oof_y)

    # ---- Lens 2: hard-negative verifier ------------------------------------
    mined = mine_hard_negatives(oof_scores, oof_y, seed=settings.GLOBAL_SEED)
    verifier = HardNegativeVerifier(seed=settings.GLOBAL_SEED).fit(Xdev_p, ydev, mined)

    # ---- Lens 3: anomaly + OOD ---------------------------------------------
    anom_cfg = thr_cfg["anomaly"]["isolation_forest"]
    anomaly = AnomalyChallenger(
        contamination=float(anom_cfg["contamination"]),
        n_estimators=int(anom_cfg["n_estimators"]), seed=settings.GLOBAL_SEED,
    ).fit(Xdev_p, ydev)
    ood = OODDetector(
        missingness_z=float(thr_cfg["ood"]["missingness_z_threshold"]),
        violation_share=float(thr_cfg["ood"]["range_violation_share_threshold"]),
        knn_quantile=float(thr_cfg["ood"]["knn_quantile"]),
    ).fit(Xdev_p)

    # ---- cohort reference for explanations ---------------------------------
    cohort = CohortReference(Xdev_p, ydev, kept)

    # ---- policy thresholds from dev OOF calibrated distribution ------------
    tiers = thr_cfg["tiers"]
    n_dev = len(oof_calibrated)
    crit_k = int(tiers["critical_review"]["daily_alert_capacity"])
    urg_k = int(tiers["urgent_review"]["daily_alert_capacity"])
    sorted_desc = np.sort(oof_calibrated)[::-1]
    critical_risk = float(sorted_desc[min(crit_k, n_dev) - 1])
    urgent_risk = float(sorted_desc[min(urg_k, n_dev) - 1])
    # standard band: dev-OOF threshold achieving the recall target on OOF labels
    rec_target = float(tiers["standard_review"]["recall_target"])
    pos_scores = np.sort(oof_calibrated[oof_y == 1])
    k = max(0, int(np.floor((1 - rec_target) * len(pos_scores))))
    standard_risk = float(pos_scores[k]) if len(pos_scores) else 0.5
    thresholds = PolicyThresholds(
        critical_risk=critical_risk, urgent_risk=urgent_risk,
        standard_risk=min(standard_risk, urgent_risk),  # bands must nest
        anomaly_escalation_pct=float(thr_cfg["anomaly"]["disagreement_percentile"]),
        policy_version=str(thr_cfg["policy_version"]),
    )
    log.info("policy thresholds: %s", thresholds.to_dict())

    # ---- OOF report for the full lens stack --------------------------------
    lens_report = full_metric_report(
        oof_y, oof_calibrated, probs=oof_calibrated,
        n_boot=1000, seed=settings.GLOBAL_SEED,
        split_label="dev OOF (repeat-averaged, crossfit-calibrated)",
    )
    save_json({
        "winner": winner_name,
        "calibration_selection": cal_sel,
        "conformal_coverage_oof": coverage,
        "hard_negative_mining": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                 for k, v in mined.items() if k == "band_threshold"},
        "n_hard_negatives": int(len(mined["hard_negative_idx"])),
        "policy_thresholds": thresholds.to_dict(),
        "oof_calibrated_report": lens_report,
    }, settings.METRICS_DIR / "lens_stack_oof.json")

    # ---- freeze bundle ------------------------------------------------------
    fingerprint = load_json(settings.REPO_ROOT / "data/interim/data_fingerprint.json")
    bundle = {
        "version": "1.0.0",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "winner_oof_name": winner_name,
        "ensemble_accepted": ens["accepted"],
        "feature_list_selected": selected,
        "feature_list_kept": kept,
        "preprocessor": prep,
        "models": finals,
        "calibrator": calibrator,
        "calibrator_method": cal_sel["winner"],
        "conformal": conformal.to_dict(),
        "verifier": verifier,
        "anomaly": anomaly,
        "ood": ood,
        "cohort": cohort,
        "policy_thresholds": thresholds.to_dict(),
        "data_fingerprint_sha256": fingerprint["raw_file"]["sha256"],
        "seed": settings.GLOBAL_SEED,
    }
    out = settings.MODELS_DIR / "final_bundle.joblib"
    joblib.dump(bundle, out, compress=3)
    bundle_sha = sha256_file(out)

    manifest = {
        "bundle_path": str(out),
        "bundle_sha256": bundle_sha,
        "created_utc": bundle["created_utc"],
        "git": bundle["git"],
        "winner": winner_name,
        "ensemble_accepted": ens["accepted"],
        "n_features": len(kept),
        "calibrator": cal_sel["winner"],
        "policy_thresholds": thresholds.to_dict(),
        "data_fingerprint_sha256": bundle["data_fingerprint_sha256"],
        "oof_pr_auc": candidates[winner_name]["pr_auc_mean"],
    }
    save_json(manifest, settings.MODELS_DIR / "model_manifest.json")
    registry_path = settings.REGISTRY_DIR / "registry.json"
    registry = load_json(registry_path) if registry_path.exists() else {"models": []}
    registry["models"].append({**manifest, "status": "champion"})
    for m in registry["models"][:-1]:
        if m.get("status") == "champion":
            m["status"] = "superseded"
    save_json(registry, registry_path)
    save_json({**thresholds.to_dict(), "derived_from": "dev OOF calibrated distribution"},
              settings.REGISTRY_DIR / "policy_snapshot.json")

    print(f"LENSES BUILT winner={winner_name} calibrator={cal_sel['winner']} "
          f"features={len(kept)} conformal_abstention={coverage['abstention_rate']:.3f} "
          f"thresholds=({thresholds.critical_risk:.3f},{thresholds.urgent_risk:.3f},"
          f"{thresholds.standard_risk:.3f}) bundle_sha={bundle_sha[:12]}")


if __name__ == "__main__":
    main()
