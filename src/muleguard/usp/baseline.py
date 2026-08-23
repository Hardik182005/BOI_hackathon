"""The invariant snapshot - everything this upgrade is forbidden to change.

The Cohort Radar and Account-Control Ambiguity layers are *post-model*: they
read a score that has already been produced and never feed anything back into
the classifier. That is a design claim, and a design claim is worth exactly as
much as the measurement that checks it.

This module is the measurement. :func:`invariant_snapshot` records the champion's
identity, its selected features, its calibrator, its thresholds, the probability
it assigns to a fixed set of canonical rows, and the accuracy metrics recomputed
from the saved predictions. Taken before the upgrade it is the baseline; taken
after, it is the evidence. :func:`compare` diffs the two and says PASS or FAIL.

Three rules shape what is captured:

* **Recompute, never quote.** A metric read out of a report proves the report
  is self-consistent. Every ranking metric here is re-derived from the
  prediction parquet, so a stale artifact fails instead of being echoed.
* **Full float precision.** Probabilities are stored as Python floats and
  compared at 1e-12. Rounding for display would hide exactly the drift the
  comparison exists to catch.
* **Deterministic probes.** The canonical rows are chosen by a fixed seed from
  the development partition, so the before and after runs score the same
  accounts in the same order. The locked test is never read here; its stored
  artifact is hashed, not re-evaluated.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
    fbeta_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from muleguard import settings
from muleguard.evaluation.metrics import expected_calibration_error
from muleguard.logging import get_logger
from muleguard.utils import git_info, sha256_file

log = get_logger("usp.baseline")

BASELINE_DIR = settings.ARTIFACTS_DIR / "upgrade_baseline"
BASELINE_PATH = BASELINE_DIR / "usp_prechange_baseline.json"

#: How many development rows are scored through the live path as canonical
#: probes. Large enough that a change in any branch of the scoring code would
#: show up, small enough to run in seconds on the demo machine.
N_PROBE_ROWS = 400

#: The tolerance §24 sets for probability drift. Nothing in this upgrade touches
#: the model, so the honest expectation is exactly 0.0; the tolerance exists so
#: that a platform-level float difference is reported rather than crashing the
#: comparison, and any non-zero value is printed in full.
PROBABILITY_TOLERANCE = 1e-12


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _maybe_sha(path: Path) -> str | None:
    """File hash, or None when the file is legitimately absent.

    The merchant verifier and the original DataSet.xlsx are both optional on a
    fresh clone. A missing file is recorded as missing rather than as a hash of
    nothing, so the comparison can tell "absent in both runs" from "changed".
    """
    try:
        return sha256_file(path)
    except (FileNotFoundError, OSError):
        return None


# --------------------------------------------------------------------------
# champion identity
# --------------------------------------------------------------------------


def _feature_hash(names: list[str]) -> str:
    """Order-sensitive hash of a feature list.

    Order matters: the preprocessor's column order is part of the model, so a
    permutation is a change even though the set is identical.
    """
    return _sha256_text("\n".join(names))


def _calibrator_identity(calibrator: Any) -> dict[str, Any]:
    """Type and fitted parameters of the calibrator, as a comparable record.

    Platt calibration is two floats. Capturing them - not just the class name -
    is what makes "the calibrator is unchanged" a statement about behaviour
    instead of a statement about imports.
    """
    ident: dict[str, Any] = {"type": type(calibrator).__name__}
    params = {}
    for attr in ("a", "b", "coef_", "intercept_", "slope", "bias"):
        value = getattr(calibrator, attr, None)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            params[attr] = [float(v) for v in value.ravel()]
        elif isinstance(value, (int, float, np.floating, np.integer)):
            params[attr] = float(value)
    ident["parameters"] = params
    ident["parameters_hash"] = _sha256_text(repr(sorted(params.items())))
    return ident


def champion_identity() -> dict[str, Any]:
    """Who is serving, and by what exact configuration."""
    from muleguard.models.scoring import load_bundle

    b = load_bundle()
    selected = list(b["feature_list_selected"])
    kept = list(b["feature_list_kept"])
    return {
        "bundle_path": str((settings.MODELS_DIR / "final_bundle.joblib").relative_to(
            settings.REPO_ROOT)),
        "bundle_sha256": _maybe_sha(settings.MODELS_DIR / "final_bundle.joblib"),
        "merchant_verifier_sha256": _maybe_sha(
            settings.MODELS_DIR / "merchant_verifier.joblib"),
        "model_version": b.get("version"),
        "champion_model_id": b.get("winner_oof_name"),
        "winner_family": b.get("winner_family"),
        "base_models": sorted(b["models"].keys()),
        "ensemble_accepted": bool(b.get("ensemble_accepted", False)),
        "view": b.get("view"),
        "seed": b.get("seed"),
        "n_selected_features": len(selected),
        "n_kept_features": len(kept),
        "selected_feature_hash": _feature_hash(selected),
        "kept_feature_hash": _feature_hash(kept),
        "meta_features_required": list(b.get("meta_features_required") or []),
        "calibrator": _calibrator_identity(b["calibrator"]),
        "calibrator_method": b.get("calibrator_method"),
        "policy_thresholds": dict(b["policy_thresholds"]),
        "data_fingerprint_sha256": b.get("data_fingerprint_sha256"),
        "oof_pr_auc_mean_recorded_in_bundle": b.get("oof_pr_auc_mean"),
    }


# --------------------------------------------------------------------------
# canonical probe rows
# --------------------------------------------------------------------------


def probe_row_index(n: int = N_PROBE_ROWS) -> np.ndarray:
    """A fixed, reproducible slice of development rows.

    Seeded from ``settings.GLOBAL_SEED`` and sorted, so the before and after
    runs score identical accounts in identical order. Development rows only:
    the locked test is not an input to a regression check.
    """
    from muleguard.models.harness import dev_split

    dev = dev_split().row_index
    rng = np.random.default_rng(settings.GLOBAL_SEED)
    take = min(int(n), len(dev))
    return np.sort(rng.choice(dev, size=take, replace=False))


def probe_scores(rows: np.ndarray | None = None) -> dict[str, Any]:
    """Score the canonical rows through the live path and record everything.

    ``with_explanations=False`` because SHAP is an explanation of the score, not
    the score; including it would make the probe slow without making the
    invariance claim any stronger. Every number below is the scoring path's own
    output, at full precision.
    """
    from muleguard.features.frame import raw_with_meta
    from muleguard.models.scoring import score_rows

    idx = probe_row_index() if rows is None else np.asarray(rows)
    frame = raw_with_meta()
    subset = frame[idx.tolist()]
    scored = score_rows(subset, with_explanations=False, with_counterfactual=False)

    records = []
    for row_index, rec in zip(idx.tolist(), scored):
        records.append({
            "row_index": int(row_index),
            "raw_scores": {k: float(v) for k, v in sorted(rec["raw_scores"].items())},
            "calibrated_risk": float(rec["calibrated_risk"]),
            "risk_tier": rec["risk_tier"],
            "decision": rec.get("decision"),
            "auto_action": rec.get("auto_action"),
            "conformal_status": rec.get("conformal_status"),
            "ood_status": rec.get("ood_status"),
            "model_agreement": float(rec["model_agreement"]),
            "merchant_safeguard_applied": bool(
                rec.get("merchant_safeguard", {}).get("merchant_safeguard_applied", False)),
        })
    return {
        "n_rows": len(records),
        "selection": (f"seeded sample of development rows "
                      f"(seed={settings.GLOBAL_SEED}, sorted by row_index)"),
        "rows": records,
        "rows_hash": _sha256_text(repr([
            (r["row_index"], r["calibrated_risk"], r["risk_tier"]) for r in records])),
    }


# --------------------------------------------------------------------------
# accuracy, recomputed from saved predictions
# --------------------------------------------------------------------------


def _at_budget(y: np.ndarray, s: np.ndarray, k: int) -> tuple[float, float]:
    """Recall and precision inside the top-``k`` ranked accounts."""
    k = min(int(k), len(s))
    if k <= 0:
        return 0.0, 0.0
    order = np.argsort(-s, kind="stable")[:k]
    hits = float(y[order].sum())
    total = float(y.sum())
    return (hits / total if total else 0.0), hits / k


def classification_metrics(y: np.ndarray, scores: np.ndarray,
                           probs: np.ndarray | None = None,
                           threshold: float | None = None) -> dict[str, Any]:
    """The §25 battery, computed from labels and scores alone.

    ``scores`` rank; ``probs`` (defaulting to ``scores``) are the calibrated
    probabilities used for Brier, ECE and the thresholded confusion metrics. The
    threshold defaults to the policy's standard-review line, which is the
    operating point the product actually uses - a metric quoted at 0.5 would
    describe a system nobody runs.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    probs = scores if probs is None else np.asarray(probs, dtype=float)
    if threshold is None:
        from muleguard.models.scoring import load_bundle

        threshold = float(load_bundle()["policy_thresholds"]["standard_risk"])

    pred = (probs >= threshold).astype(int)
    n_legit = int((y == 0).sum())
    false_positives = int(((pred == 1) & (y == 0)).sum())

    out: dict[str, Any] = {
        "n": int(len(y)),
        "n_positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y, np.clip(probs, 0.0, 1.0))),
        "ece": float(expected_calibration_error(y, np.clip(probs, 0.0, 1.0))["ece"]),
        "false_positives_per_1000_legitimate": (
            1000.0 * false_positives / n_legit if n_legit else 0.0),
    }
    for k in (25, 50, 100):
        rec, prec = _at_budget(y, scores, k)
        out[f"recall_at_top{k}"] = rec
        out[f"precision_at_top{k}"] = prec
    return out


#: The registry stores its headline PR-AUC rounded to five decimals, so the
#: agreement check uses the same 1e-4 tolerance the repo's own verifier uses.
#: This is a check that the shipped number is the recomputed number - not a
#: bit-identity claim, which a rounded artifact could never satisfy.
PUBLISHED_METRIC_TOLERANCE = 1e-4


def _registry_champion() -> dict[str, Any] | None:
    """The active registry entry, read the way the repo's verifier reads it."""
    from muleguard.utils import load_json

    try:
        reg = load_json(settings.REGISTRY_DIR / "registry.json")
    except (FileNotFoundError, OSError, ValueError):
        return None
    active = [m for m in reg.get("models", []) if m.get("status") != "retired"]
    return active[-1] if active else None


def _registry_oof_pr_auc() -> float | None:
    """The champion PR-AUC as the model registry records it, if it is there."""
    entry = _registry_champion()
    value = None if entry is None else entry.get("oof_pr_auc")
    return None if value is None else float(value)


def champion_oof_metrics() -> dict[str, Any]:
    """Champion accuracy re-derived from the saved out-of-fold predictions.

    The stored ``score`` column is the model's **raw** probability, so the
    frozen Platt calibrator is applied here before any probability metric is
    taken. That is the production path exactly - raw, then calibrate, then
    threshold - and it is the only order under which Brier, ECE and the
    thresholded confusion metrics describe the system anyone actually runs.
    Ranking metrics are unaffected either way: Platt is monotone, so PR-AUC,
    ROC-AUC and the top-K budgets are identical before and after calibration.

    Averaged over repeats after computing each repeat separately: pooling the
    repeats into one vector would mix three predictions of the same row and
    quietly inflate the sample. The per-repeat values are kept so a reader can
    see the spread rather than trust a mean.
    """
    from muleguard.models.scoring import load_bundle

    bundle = load_bundle()
    champion = bundle["winner_oof_name"]
    path = settings.PREDICTIONS_DIR / "final_model_oof_predictions.parquet"
    preds = pl.read_parquet(path).filter(pl.col("model") == champion)
    if preds.is_empty():
        raise RuntimeError(
            f"no saved out-of-fold predictions for champion {champion!r} in "
            f"{path.name}; the accuracy baseline cannot be recomputed")

    per_repeat = []
    for repeat in sorted(preds["repeat"].unique().to_list()):
        r = preds.filter(pl.col("repeat") == repeat).sort("row_index")
        raw = r["score"].to_numpy()
        calibrated = np.clip(bundle["calibrator"].predict(raw), 0.0, 1.0)
        per_repeat.append(classification_metrics(
            r["target"].to_numpy(), raw, probs=calibrated))

    keys = [k for k, v in per_repeat[0].items() if isinstance(v, float)]
    mean = {k: float(np.mean([m[k] for m in per_repeat])) for k in keys}

    # The repeat-averaged view: one score per account, averaged across repeats,
    # then calibrated. This is the protocol behind the calibrated Brier and ECE
    # the repo publishes, and it is reported alongside - not instead of - the
    # per-repeat average, because the two answer different questions. Per-repeat
    # is the honest estimate of a single model run; repeat-averaged is the
    # ensemble-of-repeats the calibration report describes.
    averaged = (preds.group_by("row_index")
                .agg(pl.col("score").mean().alias("score"),
                     pl.col("target").first().alias("target"))
                .sort("row_index"))
    avg_raw = averaged["score"].to_numpy()
    avg_cal = np.clip(bundle["calibrator"].predict(avg_raw), 0.0, 1.0)
    repeat_averaged = classification_metrics(
        averaged["target"].to_numpy(), avg_raw, probs=avg_cal)

    # Does the number we just recomputed agree with the number the repo ships?
    # A baseline that only records itself proves nothing; this is the line that
    # would fail if the prediction store and the registry had drifted apart.
    registry_pr_auc = _registry_oof_pr_auc()
    bundle_pr_auc = bundle.get("oof_pr_auc_mean")
    agreement = {
        "recomputed_pr_auc_mean_over_repeats": mean["pr_auc"],
        "registry_oof_pr_auc": registry_pr_auc,
        "bundle_oof_pr_auc_mean": (
            None if bundle_pr_auc is None else float(bundle_pr_auc)),
        "tolerance": PUBLISHED_METRIC_TOLERANCE,
        "matches_registry": (
            registry_pr_auc is not None
            and abs(mean["pr_auc"] - registry_pr_auc) < PUBLISHED_METRIC_TOLERANCE),
        "matches_bundle": (
            bundle_pr_auc is not None
            and abs(mean["pr_auc"] - float(bundle_pr_auc))
            < PUBLISHED_METRIC_TOLERANCE),
    }
    if not agreement["matches_registry"]:
        log.warning("recomputed PR-AUC %.6f does not match registry %s",
                    mean["pr_auc"], registry_pr_auc)

    return {
        "champion": champion,
        "source": str(path.relative_to(settings.REPO_ROOT)),
        "source_sha256": _maybe_sha(path),
        "n_repeats": len(per_repeat),
        "protocol": ("each repeat evaluated separately, then averaged - the same "
                     "protocol the registry's oof_pr_auc uses; scores are "
                     "out-of-fold so no row is scored by a model that saw it"),
        "calibration": ("frozen Platt calibrator from the bundle applied to raw "
                        "out-of-fold scores before every probability metric, "
                        "matching the production path exactly"),
        "mean_over_repeats": mean,
        "per_repeat": per_repeat,
        "repeat_averaged": repeat_averaged,
        "agreement_with_published": agreement,
    }


def stored_artifact_hashes() -> dict[str, Any]:
    """Hashes of the artifacts this upgrade must leave untouched."""
    names = {
        "oof_predictions": "predictions/final_model_oof_predictions.parquet",
        "locked_test_predictions": "predictions/final_locked_test_predictions.parquet",
        "locked_test_metrics": "metrics/locked_test_metrics.json",
        "final_locked_test_metrics": "metrics/final_locked_test_metrics.json",
        "final_oof_metrics": "metrics/final_oof_metrics.json",
        "final_calibration": "metrics/final_calibration.json",
        "final_accuracy_table_csv": "metrics/final_accuracy_table.csv",
        "global_shap_importance": "metrics/global_shap_importance.json",
        "model_manifest": "models/model_manifest.json",
        "quarantined_features": "features/quarantined_features.json",
    }
    return {key: _maybe_sha(settings.ARTIFACTS_DIR / rel) for key, rel in names.items()}


def quarantine_state() -> dict[str, Any]:
    """The live firewall policy, as the radar will have to obey it.

    Read from the firewall rather than copied into a list here. §4 is explicit
    that the quarantine is whatever the config says today, not a set of column
    names someone typed into a prompt.
    """
    from muleguard.features import firewall

    cfg = firewall.config()
    quarantined = sorted(set(cfg.hard_quarantine)
                         | set(cfg.conditional_quarantine)
                         | set(cfg.fairness_excluded))
    return {
        "policy_version": cfg.policy_version,
        "n_quarantined": len(quarantined),
        "quarantined": quarantined,
        "hard_quarantine": sorted(cfg.hard_quarantine),
        "conditional_quarantine": sorted(cfg.conditional_quarantine),
        "fairness_excluded": sorted(cfg.fairness_excluded),
        "contextual_only": sorted(cfg.contextual_only),
        "quarantine_hash": _sha256_text("\n".join(quarantined)),
        "target_column": settings.TARGET_COLUMN,
    }


# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------


def invariant_snapshot(*, with_probes: bool = True) -> dict[str, Any]:
    """Everything that must be identical before and after the USP upgrade."""
    snap: dict[str, Any] = {
        "generated_utc": _utc(),
        "git": git_info(settings.REPO_ROOT),
        "inputs": {
            "dataset_sha256": _maybe_sha(settings.RAW_DIR / "DataSet.xlsx"),
            "dataset_repo_root_sha256": _maybe_sha(settings.REPO_ROOT / "DataSet.xlsx"),
            "description_sha256": _maybe_sha(settings.REPO_ROOT / "Description.xlsx"),
            "description_raw_dir_sha256": _maybe_sha(settings.RAW_DIR / "Description.xlsx"),
        },
        "champion": champion_identity(),
        "quarantine": quarantine_state(),
        "artifact_hashes": stored_artifact_hashes(),
        "accuracy": champion_oof_metrics(),
    }
    if with_probes:
        snap["probes"] = probe_scores()
    return snap


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _diff_scalar(before: Any, after: Any) -> dict[str, Any] | None:
    if before == after:
        return None
    return {"before": before, "after": after}


def compare(before: dict[str, Any], after: dict[str, Any],
            *, tolerance: float = PROBABILITY_TOLERANCE) -> dict[str, Any]:
    """Diff two snapshots and rule on whether the classifier survived intact.

    The verdict is FAIL on any identity change, any tier mismatch, any
    probability moving further than ``tolerance``, or any classification metric
    that is not bit-identical. §24 is explicit that a small difference is not to
    be explained away, so nothing here rounds before comparing.
    """
    findings: list[str] = []

    # --- identity -----------------------------------------------------------
    cb, ca = before["champion"], after["champion"]
    identity_keys = [
        "bundle_sha256", "merchant_verifier_sha256", "model_version",
        "champion_model_id", "winner_family", "selected_feature_hash",
        "kept_feature_hash", "n_selected_features", "calibrator_method",
        "data_fingerprint_sha256",
    ]
    identity: dict[str, Any] = {}
    for key in identity_keys:
        d = _diff_scalar(cb.get(key), ca.get(key))
        if d:
            identity[key] = d
            findings.append(f"champion.{key} changed")

    calib = _diff_scalar(cb["calibrator"]["parameters_hash"],
                         ca["calibrator"]["parameters_hash"])
    if calib:
        identity["calibrator_parameters_hash"] = calib
        findings.append("calibrator parameters changed")

    thresholds_equal = cb["policy_thresholds"] == ca["policy_thresholds"]
    if not thresholds_equal:
        identity["policy_thresholds"] = {"before": cb["policy_thresholds"],
                                         "after": ca["policy_thresholds"]}
        findings.append("policy thresholds changed")

    quarantine_equal = (before["quarantine"]["quarantine_hash"]
                        == after["quarantine"]["quarantine_hash"])
    if not quarantine_equal:
        findings.append("quarantine list changed")

    # --- probe rows ---------------------------------------------------------
    probe_report: dict[str, Any] = {"compared": False}
    if "probes" in before and "probes" in after:
        pb = {r["row_index"]: r for r in before["probes"]["rows"]}
        pa = {r["row_index"]: r for r in after["probes"]["rows"]}
        shared = sorted(set(pb) & set(pa))
        missing = sorted(set(pb) ^ set(pa))
        max_calibrated = 0.0
        max_raw = 0.0
        tier_mismatch: list[dict[str, Any]] = []
        action_mismatch: list[dict[str, Any]] = []
        for row in shared:
            b, a = pb[row], pa[row]
            max_calibrated = max(
                max_calibrated, abs(b["calibrated_risk"] - a["calibrated_risk"]))
            for fam in set(b["raw_scores"]) & set(a["raw_scores"]):
                max_raw = max(max_raw, abs(b["raw_scores"][fam] - a["raw_scores"][fam]))
            if b["risk_tier"] != a["risk_tier"]:
                tier_mismatch.append({"row_index": row, "before": b["risk_tier"],
                                      "after": a["risk_tier"]})
            if b.get("decision") != a.get("decision"):
                action_mismatch.append({"row_index": row, "before": b.get("decision"),
                                        "after": a.get("decision")})
        probe_report = {
            "compared": True,
            "n_rows_compared": len(shared),
            "n_rows_only_in_one_snapshot": len(missing),
            "raw_probability_max_abs_diff": max_raw,
            "probability_max_abs_diff": max_calibrated,
            "tolerance": tolerance,
            "tier_mismatch_count": len(tier_mismatch),
            "tier_mismatches": tier_mismatch[:20],
            "policy_action_mismatch_count": len(action_mismatch),
            "policy_action_mismatches": action_mismatch[:20],
        }
        if missing:
            findings.append(f"{len(missing)} probe row(s) present in only one snapshot")
        if max_calibrated > tolerance:
            findings.append(
                f"calibrated probability moved by {max_calibrated:.3e} "
                f"(tolerance {tolerance:.0e})")
        if max_raw > tolerance:
            findings.append(
                f"raw model probability moved by {max_raw:.3e} "
                f"(tolerance {tolerance:.0e})")
        if tier_mismatch:
            findings.append(f"{len(tier_mismatch)} risk-tier mismatch(es)")
        if action_mismatch:
            findings.append(f"{len(action_mismatch)} policy-action mismatch(es)")

    # --- accuracy -----------------------------------------------------------
    mb = before["accuracy"]["mean_over_repeats"]
    ma = after["accuracy"]["mean_over_repeats"]
    metric_differences = {}
    for key in sorted(set(mb) | set(ma)):
        b, a = mb.get(key), ma.get(key)
        if b is None or a is None:
            metric_differences[key] = {"before": b, "after": a, "difference": None}
            findings.append(f"metric {key} present in only one snapshot")
            continue
        diff = a - b
        metric_differences[key] = {"before": b, "after": a, "difference": diff}
        if diff != 0.0:
            findings.append(f"metric {key} changed by {diff:.3e}")

    verdict = "PASS" if not findings else "FAIL"
    return {
        "generated_utc": _utc(),
        "baseline_git_sha": before["git"].get("commit_sha"),
        "postchange_git_sha": after["git"].get("commit_sha"),
        "model_sha_before": cb.get("bundle_sha256"),
        "model_sha_after": ca.get("bundle_sha256"),
        "model_sha_equal": cb.get("bundle_sha256") == ca.get("bundle_sha256"),
        "feature_hash_before": cb.get("selected_feature_hash"),
        "feature_hash_after": ca.get("selected_feature_hash"),
        "feature_hash_equal": (cb.get("selected_feature_hash")
                               == ca.get("selected_feature_hash")),
        "calibrator_equal": not calib,
        "thresholds_equal": thresholds_equal,
        "quarantine_equal": quarantine_equal,
        "identity_differences": identity,
        "probability_max_abs_diff": probe_report.get("probability_max_abs_diff"),
        "raw_probability_max_abs_diff": probe_report.get("raw_probability_max_abs_diff"),
        "tier_mismatch_count": probe_report.get("tier_mismatch_count"),
        "probes": probe_report,
        "metrics_before": mb,
        "metrics_after": ma,
        "metric_differences": metric_differences,
        "findings": findings,
        "verdict": verdict,
    }
