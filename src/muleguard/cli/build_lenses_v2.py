"""Freeze the scoring bundle from the leakage-free tournament (v2).

This replaces :mod:`muleguard.cli.build_lenses`, which read the pre-upgrade
artifacts. The difference is not cosmetic: the old pipeline selected features
with ``candidate_feature_columns`` and a hand-maintained quarantine list, and
the bundle it produced names F3898, F3913, F3914 and F3916 - all of which the
Feature Availability Firewall now classifies as post-resolution leakage. Its
headline 0.8077 OOF PR-AUC is therefore not a number that can survive a hidden
validation set, and the bundle it describes must not be served.

What is built here, all of it from development rows only:

  Lens 1  the three base families refit on the promoted feature set
  Lens 2  calibrator chosen and fitted on OOF, Mondrian conformal on crossfit-
          calibrated OOF probabilities, hard-negative verifier from OOF false
          positives
  Lens 3  IsolationForest challenger and OOD detector on the same reduced view
  Policy  tier thresholds read off the dev OOF calibrated distribution

The locked test is never loaded. The promoted model, its feature set and the
ensemble decision are read from the tournament artifacts rather than restated,
so this file cannot silently disagree with the leaderboard.

Usage:  python -m muleguard.cli.build_lenses_v2 [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt

import joblib
import numpy as np
import polars as pl

from muleguard import settings
from muleguard.action.policy import PolicyThresholds
from muleguard.evaluation.metrics import full_metric_report
from muleguard.explain.reason_codes import CohortReference
from muleguard.features import frame as mframe
from muleguard.features.preprocessing import FoldPreprocessor
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
from muleguard.models.harness import dev_split
from muleguard.utils import git_info, load_json, save_json, set_global_seed, sha256_file

log = get_logger("cli.build_lenses_v2")

BASE_MODELS = ["lightgbm", "xgboost", "catboost"]
BUNDLE_VERSION = "2.0.0"

PROMOTION_JSON = settings.METRICS_DIR / "promotion_decision_v2.json"
TOURNAMENT_JSON = settings.METRICS_DIR / "tournament_v2.json"
ENSEMBLE_JSON = settings.METRICS_DIR / "ensemble_v2.json"
SELECTION_JSON = settings.FEATURES_DIR / "selected_features_v2.json"
OOF_STORE = settings.PREDICTIONS_DIR / "oof_v2.parquet"


def _promoted() -> dict:
    """The champion, read from the artifact rather than hardcoded."""
    promo = load_json(PROMOTION_JSON)
    detail = promo["promoted_detail"]
    log.info("promoted: %s (%s, %s, %d features, OOF PR-AUC %.4f +/- %.4f)",
             promo["promoted"], detail["family"], detail["view"],
             detail["n_features"], detail["oof_pr_auc_mean"],
             detail["oof_pr_auc_std"])
    if promo.get("tie_break_applied"):
        log.info("tie-break applied: raw PR-AUC leader was %s",
                 promo["raw_pr_auc_leader"])
    return promo


def _feature_list(view: str, feature_set: str) -> list[str]:
    sel = load_json(SELECTION_JSON)
    pool = "ALL_ADMISSIBLE" if view in (None, "ALL_ADMISSIBLE") else view
    return list(sel["pools"][pool]["compact_sets"][feature_set])


def _repeat_avg(oof: pl.DataFrame, model: str) -> pl.DataFrame:
    """One score per row, averaged over CV repeats, in row order."""
    sub = oof.filter(pl.col("model") == model)
    if sub.is_empty():
        raise SystemExit(f"no OOF rows stored for {model!r}; rerun the tournament")
    return (sub.group_by("row_index")
            .agg(pl.col("score").mean().alias("score"),
                 pl.col("target").first().alias("target"))
            .sort("row_index"))


def main() -> None:
    ap = argparse.ArgumentParser(description="freeze the v2 scoring bundle")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit everything and report, but do not write the bundle")
    args = ap.parse_args()

    set_global_seed(settings.GLOBAL_SEED)
    thr_cfg = settings.load_config("thresholds")

    promo = _promoted()
    detail = promo["promoted_detail"]
    winner_name = promo["promoted"]
    winner_family = detail["family"]
    view = None if detail["view"] == "ALL_ADMISSIBLE" else detail["view"]
    selected = (_feature_list(detail["view"], detail["feature_set"])
                if detail["feature_set"] else None)

    ens = load_json(ENSEMBLE_JSON)
    ensemble_accepted = ens["decision"].startswith("ENSEMBLE_ACCEPTED")
    if ensemble_accepted:
        raise SystemExit(
            "the ensemble was accepted; this builder freezes a single-model "
            "bundle only. Extend it deliberately rather than shipping a "
            "bundle whose winner_family does not match the decision.")

    # ---- data: firewall-admitted frame, development rows only --------------
    mf = mframe.build_model_frame(view=view)
    if selected is not None:
        mf = mf.subset(selected)
    # `subset` returns matrix order, which is what every fitted object below
    # will expect at scoring time; the selection file's order is irrelevant.
    selected = list(mf.feature_names)
    dev = dev_split()
    Xdev, ydev = mf.X[dev.row_index], mf.y[dev.row_index]
    log.info("dev matrix: %d rows x %d features (%d positives), view=%s",
             Xdev.shape[0], Xdev.shape[1], int(ydev.sum()), detail["view"])

    raw = mframe.raw_with_meta()
    metas = set(mframe.meta_feature_names())
    # MG_* columns are numeric by construction; they are listed separately in
    # `meta_features_required` so the scoring path knows to derive them, but
    # their *schema* entry must stay "numeric" or canonicalisation would treat
    # them as free text.
    feature_schema = {
        c: ("numeric" if c in metas
            else "date" if raw.schema[c].is_temporal()
            else "numeric" if raw.schema[c].is_numeric()
            else "categorical")
        for c in selected
    }

    # ---- dev-level preprocessor -------------------------------------------
    prep = FoldPreprocessor(mode="tree").fit(Xdev, selected)
    Xdev_p = prep.transform(Xdev)
    kept = prep.kept_features
    log.info("preprocessor kept %d of %d features", len(kept), len(selected))

    # ---- Lens 1: base families refit on all dev rows -----------------------
    finals = {}
    for m in BASE_MODELS:
        _, model = core_models.SCORERS[m](
            Xdev_p, ydev, Xdev_p[:2], settings.GLOBAL_SEED, return_model=True)
        finals[m] = model
    log.info("lens 1: %s refit on dev", ", ".join(BASE_MODELS))

    # ---- OOF scores of the promoted model, for every downstream lens -------
    oof = pl.read_parquet(OOF_STORE)
    won = _repeat_avg(oof, winner_name)
    oof_scores = won["score"].to_numpy()
    oof_y = won["target"].to_numpy()
    if not np.array_equal(won["row_index"].to_numpy(), dev.row_index):
        raise SystemExit("OOF row order does not match the development split")

    # ---- Lens 2: calibration, conformal, hard-negative verifier ------------
    cal_sel = select_calibrator(oof_scores, oof_y, seed=settings.GLOBAL_SEED)
    calibrator = fit_final_calibrator(oof_scores, oof_y, cal_sel["winner"])
    oof_calibrated = crossfit_calibrated(oof_scores, oof_y, cal_sel["winner"],
                                         seed=settings.GLOBAL_SEED)
    log.info("lens 2: calibrator=%s (%s)", cal_sel["winner"], cal_sel["comparison"])

    conformal = MondrianConformal(alpha=float(thr_cfg["conformal"]["alpha"]))
    conformal.fit(oof_calibrated, oof_y)
    coverage = empirical_coverage(conformal, oof_calibrated, oof_y)

    mined = mine_hard_negatives(oof_scores, oof_y, seed=settings.GLOBAL_SEED)
    verifier = HardNegativeVerifier(seed=settings.GLOBAL_SEED).fit(Xdev_p, ydev, mined)
    log.info("lens 2: %d hard negatives mined", len(mined["hard_negative_idx"]))

    # ---- Lens 3: anomaly challenger + OOD ----------------------------------
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
    cohort = CohortReference(Xdev_p, ydev, kept)

    # ---- policy thresholds from the dev OOF calibrated distribution --------
    tiers = thr_cfg["tiers"]
    n_dev = len(oof_calibrated)
    crit_k = int(tiers["critical_review"]["daily_alert_capacity"])
    urg_k = int(tiers["urgent_review"]["daily_alert_capacity"])
    sorted_desc = np.sort(oof_calibrated)[::-1]
    critical_risk = float(sorted_desc[min(crit_k, n_dev) - 1])
    urgent_risk = float(sorted_desc[min(urg_k, n_dev) - 1])
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

    lens_report = full_metric_report(
        oof_y, oof_calibrated, probs=oof_calibrated, n_boot=1000,
        seed=settings.GLOBAL_SEED,
        split_label="dev OOF (repeat-averaged, crossfit-calibrated), leakage-free v2",
    )
    save_json({
        "winner": winner_name,
        "winner_family": winner_family,
        "view": detail["view"],
        "n_features_selected": len(selected),
        "n_features_kept": len(kept),
        "calibration_selection": cal_sel,
        "conformal_coverage_oof": coverage,
        "n_hard_negatives": int(len(mined["hard_negative_idx"])),
        "policy_thresholds": thresholds.to_dict(),
        "oof_calibrated_report": lens_report,
        "supersedes": "lens_stack_oof.json (pre-firewall, leaky feature set)",
    }, settings.METRICS_DIR / "lens_stack_oof_v2.json")

    if args.dry_run:
        log.info("dry run: bundle NOT written")
        return

    # ---- freeze -------------------------------------------------------------
    fingerprint = load_json(settings.REPO_ROOT / "data/interim/data_fingerprint.json")
    bundle = {
        "version": BUNDLE_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "winner_oof_name": winner_name,
        "winner_family": winner_family,
        "stacker": None,
        "ensemble_accepted": ensemble_accepted,
        "view": detail["view"],
        "feature_list_selected": selected,
        "feature_list_kept": kept,
        "meta_features_required": sorted(metas & set(selected)),
        "feature_schema": feature_schema,
        "cat_maps": mf.cat_maps,
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
        "oof_pr_auc_mean": detail["oof_pr_auc_mean"],
        "oof_pr_auc_std": detail["oof_pr_auc_std"],
        "leakage_status": "FIREWALL_ADMITTED_ONLY",
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
        "winner_family": winner_family,
        "view": detail["view"],
        "ensemble_accepted": ensemble_accepted,
        "ensemble_decision": ens["decision"],
        "n_features_selected": len(selected),
        "n_features": len(kept),
        "calibrator": cal_sel["winner"],
        "policy_thresholds": thresholds.to_dict(),
        "data_fingerprint_sha256": bundle["data_fingerprint_sha256"],
        "oof_pr_auc": detail["oof_pr_auc_mean"],
        "oof_pr_auc_std": detail["oof_pr_auc_std"],
        "leakage_status": "FIREWALL_ADMITTED_ONLY",
        "supersedes": {
            "winner": "catboost_tuned_top60",
            "oof_pr_auc": 0.8077340678700281,
            "why": ("the previous bundle selected F3898, F3913, F3914 and F3916, "
                    "which the Feature Availability Firewall classifies as "
                    "post-resolution leakage; its PR-AUC was inflated by them"),
        },
    }
    save_json(manifest, settings.MODELS_DIR / "model_manifest.json")

    registry_path = settings.REGISTRY_DIR / "registry.json"
    registry = load_json(registry_path) if registry_path.exists() else {"models": []}
    for entry in registry["models"]:
        if entry.get("status") == "champion":
            entry["status"] = "retired"
    registry["models"].append({**manifest, "status": "champion"})
    save_json(registry, registry_path)

    log.info("bundle frozen: %s (sha %s), winner=%s, %d features",
             out.name, bundle_sha[:12], winner_name, len(kept))


if __name__ == "__main__":
    main()
