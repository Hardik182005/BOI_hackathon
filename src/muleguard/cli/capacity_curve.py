"""Precompute the Analyst Capacity Optimizer curve.

    python -m muleguard.cli.capacity_curve [--n-boot 1000] [--dry-run]

Writes ``artifacts/metrics/capacity_curve.json``. The API serves that file and
the dashboard renders it, so no metric is ever computed inside a request or,
worse, typed into a page.

Everything the curve is built from already exists on disk:

  artifacts/predictions/oof_v2.parquet      stored development OOF predictions
  artifacts/metrics/lens_stack_oof_v2.json  the promoted champion + frozen
                                            policy thresholds (policy 1.0)
  artifacts/models/final_bundle.joblib      the frozen calibrator, used only to
                                            express a threshold on the same
                                            scale the scoring API emits

No model is fitted here and nothing is retrained. The locked test split is not
opened: capacity planning is a decision about the future, and the only split
allowed to inform a decision is the development one.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.models import capacity
from muleguard.utils import load_json, save_json

log = get_logger("cli.capacity")

OOF_STORE = settings.PREDICTIONS_DIR / "oof_v2.parquet"
LENS_JSON = settings.METRICS_DIR / "lens_stack_oof_v2.json"
BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
OUT = settings.METRICS_DIR / "capacity_curve.json"


def _champion_and_policy() -> tuple[str, dict[str, Any]]:
    """Read the served champion and the frozen thresholds from the artifact.

    Neither is typed in. If the promoted model ever changes, this file follows
    it without an edit, and if the artifact is missing the command stops rather
    than guessing.
    """
    lens = load_json(LENS_JSON)
    return str(lens["winner"]), dict(lens["policy_thresholds"])


def _repeat_averaged(oof: pl.DataFrame, model: str) -> pl.DataFrame:
    """One score per account, averaged over CV repeats - the served ranking.

    Identical to ``muleguard.cli.build_lenses_v2._repeat_avg`` on purpose: the
    capacity curve has to describe the ranking the bundle actually produces.
    """
    sub = oof.filter(pl.col("model") == model)
    if sub.is_empty():
        raise SystemExit(f"no OOF rows stored for {model!r} in {OOF_STORE}")
    return (sub.group_by("row_index")
            .agg(pl.col("score").mean().alias("score"),
                 pl.col("target").first().alias("target"))
            .sort("row_index"))


def _per_repeat(oof: pl.DataFrame, model: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    sub = oof.filter(pl.col("model") == model)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for r in sorted(sub["repeat"].unique().to_list()):
        s = sub.filter(pl.col("repeat") == r).sort("row_index")
        out[int(r)] = (s["target"].to_numpy(), s["score"].to_numpy())
    return out


def _calibrator():
    """The frozen calibrator from the shipped bundle, or None.

    Loading a serialised, already-fitted calibrator and calling ``predict`` is
    not training: it is the exact transform the scoring path applies. It is
    optional - without it the curve still answers every capacity question, it
    just cannot express the threshold on the calibrated scale.
    """
    if not BUNDLE.exists():
        log.warning("no final bundle at %s; thresholds will be reported on the "
                    "raw model-score scale only", BUNDLE)
        return None, None
    import joblib

    b = joblib.load(BUNDLE)
    cal = b.get("calibrator")
    if cal is None:
        return None, None
    return (lambda s: np.asarray(cal.predict(np.asarray(s, dtype=float)), dtype=float),
            str(b.get("calibrator_method", "unknown")))


def main() -> None:
    ap = argparse.ArgumentParser(description="precompute the analyst capacity curve")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="stratified bootstrap replicates for the intervals")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and log, but do not write the artifact")
    args = ap.parse_args()

    champion, frozen = _champion_and_policy()
    oof = pl.read_parquet(OOF_STORE)
    avg = _repeat_averaged(oof, champion)
    y = avg["target"].to_numpy()
    scores = avg["score"].to_numpy()
    repeats = _per_repeat(oof, champion)
    n, n_pos = len(y), int(y.sum())
    log.info("champion=%s  dev OOF rows=%d  positives=%d  repeats=%d",
             champion, n, n_pos, len(repeats))

    calibrate, cal_method = _calibrator()
    points = capacity.build_curve(
        y=y, scores=scores, repeats=repeats,
        calibrate=calibrate, n_boot=int(args.n_boot), seed=settings.GLOBAL_SEED,
    )

    position: list[dict[str, Any]] = []
    if calibrate is not None:
        position = capacity.locate_frozen_policy(y, calibrate(scores), frozen)
        for row in position:
            log.info("frozen %s sits at %d accounts (recall %.3f, precision %.3f)",
                     row["tier"], row["accounts_at_or_above"], row["recall"],
                     row["precision"])

    doc = capacity.make_curve_document(
        points=points,
        evaluation={
            "split": "development out-of-fold (repeat-averaged), leakage-free v2",
            "n_accounts": n,
            "n_positives": n_pos,
            "prevalence": float(y.mean()),
            "n_repeats": len(repeats),
            "recall_resolution_pct_points": 100.0 / n_pos,
            "leading_true_positive_run": capacity.leading_true_positive_run(y, scores),
            "locked_test_used": False,
        },
        frozen_policy=frozen,
        frozen_policy_position=position,
        provenance={
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "generator": "muleguard.cli.capacity_curve",
            "champion_model": champion,
            "champion_source": str(LENS_JSON.relative_to(settings.REPO_ROOT)),
            "predictions_source": str(OOF_STORE.relative_to(settings.REPO_ROOT)),
            "calibrator": cal_method,
            "calibrator_source": (str(BUNDLE.relative_to(settings.REPO_ROOT))
                                  if calibrate is not None else None),
            "bootstrap": {
                "n_boot": int(args.n_boot),
                "alpha": 0.05,
                "schemes": {
                    "stratified": "positives and negatives resampled separately; "
                                  "the mule count stays at 64, so this is "
                                  "uncertainty in the ranking alone",
                    "resampled_accounts": "the whole book resampled with "
                                          "replacement, so the number of mules "
                                          "in a replicate moves too; wider, and "
                                          "the one to quote when asked what a "
                                          "different month might look like",
                },
            },
            "seed": settings.GLOBAL_SEED,
            "retraining_performed": False,
            "caveats": [
                "Recall and precision at a budget are ranking statistics and are "
                "unaffected by calibration; only the threshold is on the "
                "calibrated scale.",
                "The frozen policy thresholds were originally read off a "
                "cross-fitted calibration of these same scores, while the "
                "positions reported here apply the shipped final calibrator. The "
                "two differ by a few accounts at each band; the band sizes below "
                "are therefore close to, not identical to, the sizes used when "
                "policy_version 1.0 was frozen.",
                "The stratified interval collapses onto the point estimate for "
                "budgets inside the leading run of true positives, because a "
                "scheme that holds the mule count fixed has nothing to vary "
                "there. That is a property of the ranking, not evidence of "
                "certainty; read the resampled-accounts interval beside it.",
                "Development out-of-fold only. No locked-test figure appears in "
                "this artifact.",
            ],
        },
    )

    log.info("leading run of true positives: %d accounts",
             doc["evaluation"]["leading_true_positive_run"])
    for p in doc["headline"]:
        log.info("budget %3d -> recall %.4f  strat CI [%.4f, %.4f]  "
                 "account CI [%.4f, %.4f]  precision %.4f  repeats %.4f-%.4f",
                 p["budget"], p["recall"], p["recall_ci_low"], p["recall_ci_high"],
                 p["recall_ci_low_resampled_accounts"],
                 p["recall_ci_high_resampled_accounts"],
                 p["precision"], p["recall_repeat_min"], p["recall_repeat_max"])

    if args.dry_run:
        log.info("dry run: %s NOT written", OUT)
        return
    save_json(doc, OUT)
    log.info("wrote %s (%d curve points)", OUT, len(points))


if __name__ == "__main__":
    main()
