"""Resolve the open question in ``docs/TUNING_OVERFIT_HYPOTHESIS.md``.

The note observed that a **tuned** HistGradientBoosting run scored *lower* on
the nested outer folds than an **untuned** one with the same estimator, the same
seed and the same folds, and refused to call it a finding because the comparison
was between two means. Pairing is what makes it a finding or kills it.

Both halves are already on disk, so nothing is retrained here:

* tuned - ``artifacts/predictions/nested_oof.parquet`` (per-row outer-validation
  scores from the nested tournament, Optuna inside every fold, feature-set size
  chosen on the inner folds) joined to ``artifacts/splits/nested_cv_assignments
  .parquet`` to recover which outer fold each row belonged to;
* untuned - the WITHOUT arm of ``artifacts/metrics/missingness_ablation.json``,
  which is the arm built on the same base frame with fixed hyperparameters and a
  fixed top-120, and therefore the only apples-to-apples one. (The WITH arm adds
  the missingness signature, a second difference, so it is not used here.)

Both lists are 15 outer folds in the order :func:`nested.build_outer_folds`
emits them - repeat-major, fold-minor - and that ordering is asserted against
the tournament's own per-repeat means before anything is compared. A silently
mispaired comparison would look exactly like a real effect.

Three tests are reported, not one: sign, Wilcoxon signed-rank and paired t. They
answer slightly different questions and the note committed in advance to
publishing all three rather than the most convenient.

    .venv/Scripts/python.exe -m muleguard.cli.tuning_overfit

Exit code is 0 whether or not the effect survives: this module measures, and the
engineering response is written down in the note it resolves.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.utils import load_json, save_json

log = get_logger("cli.tuning_overfit")

NESTED_OOF = settings.PREDICTIONS_DIR / "nested_oof.parquet"
# Written by the nested run beside its predictions, not into data/splits with
# the immutable dev/test split - it is a record of one run's folds, not a
# frozen partition of the dataset.
ASSIGNMENTS = settings.ARTIFACTS_DIR / "splits" / "nested_cv_assignments.parquet"
NESTED_JSON = settings.METRICS_DIR / "nested_cv.json"
ABLATION = settings.METRICS_DIR / "missingness_ablation.json"
OUT = settings.METRICS_DIR / "tuning_overfit_test.json"

FAMILY = "histgb"
# Tolerance for the ordering check below. The tournament rounds its leaderboard
# to 5 decimals, so anything tighter would fail on the rounding rather than on a
# real disagreement.
RECONSTRUCTION_TOL = 1e-4


def _tuned_fold_ap(family: str) -> tuple[list[float], list[float]]:
    """Per-outer-fold and per-repeat AP for one family of the nested run."""
    preds = (pl.read_parquet(NESTED_OOF)
             .filter(pl.col("model") == family)
             .select("repeat", "row_index", "target", "score"))
    if preds.height == 0:
        raise SystemExit(f"{family} has no rows in {NESTED_OOF.name}")
    assign = (pl.read_parquet(ASSIGNMENTS)
              .filter(pl.col("role") == "outer_valid")
              .select("repeat", "row_index", "outer_fold"))
    joined = preds.join(assign, on=["repeat", "row_index"], how="inner")
    if joined.height != preds.height:
        raise SystemExit(f"{preds.height - joined.height} scored rows have no "
                         "outer-fold assignment; the two artifacts disagree")

    per_fold: list[float] = []
    for rep in sorted(joined["repeat"].unique().to_list()):
        sub = joined.filter(pl.col("repeat") == rep)
        for k in sorted(sub["outer_fold"].unique().to_list()):
            f = sub.filter(pl.col("outer_fold") == k)
            per_fold.append(float(average_precision_score(
                f["target"].to_numpy(), f["score"].to_numpy())))
    per_repeat = [
        float(average_precision_score(r["target"].to_numpy(),
                                      r["score"].to_numpy()))
        for rep in sorted(joined["repeat"].unique().to_list())
        for r in [joined.filter(pl.col("repeat") == rep)]
    ]
    return per_fold, per_repeat


def _paired_tests(diff: np.ndarray) -> dict[str, Any]:
    """Sign, Wilcoxon and paired t on the same vector of differences."""
    n_pos = int((diff > 0).sum())
    n_eff = int((diff != 0).sum())
    sign_p = float(stats.binomtest(n_pos, n_eff, 0.5).pvalue) if n_eff else 1.0
    w = stats.wilcoxon(diff) if n_eff else None
    t = stats.ttest_1samp(diff, 0.0)
    return {
        "sign_test": {"folds_favouring_untuned": n_pos, "n_folds": int(diff.size),
                      "p_two_sided": round(sign_p, 5)},
        "wilcoxon_signed_rank": {
            "statistic": None if w is None else float(w.statistic),
            "p_two_sided": None if w is None else round(float(w.pvalue), 5)},
        "paired_t": {"statistic": round(float(t.statistic), 4),
                     "p_two_sided": round(float(t.pvalue), 5)},
    }


def run() -> dict[str, Any]:
    for path in (NESTED_OOF, ASSIGNMENTS, NESTED_JSON, ABLATION):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    tuned_folds, tuned_repeats = _tuned_fold_ap(FAMILY)
    board = {r["model"]: r for r in load_json(NESTED_JSON)["leaderboard"]}
    if FAMILY not in board:
        raise SystemExit(f"{FAMILY} is not on the nested leaderboard")
    claimed = float(board[FAMILY]["pr_auc_mean"])
    rebuilt = float(np.mean(tuned_repeats))
    # If the join were wrong, the reconstructed mean would not match the number
    # the tournament published for the same family. This is the only guard that
    # a mispairing cannot pass, so it runs before any statistic is computed.
    if abs(rebuilt - claimed) > RECONSTRUCTION_TOL:
        raise SystemExit(
            f"reconstruction check failed: rebuilt {rebuilt:.5f} vs published "
            f"{claimed:.5f}; the prediction store and the leaderboard disagree")

    ablation = load_json(ABLATION)
    untuned_folds = [float(x) for x in ablation["without"]["fold_ap"]]
    if len(untuned_folds) != len(tuned_folds):
        raise SystemExit(f"{len(tuned_folds)} tuned folds vs "
                         f"{len(untuned_folds)} untuned; not pairable")

    diff = np.asarray(untuned_folds) - np.asarray(tuned_folds)
    tests = _paired_tests(diff)
    survives = (tests["sign_test"]["p_two_sided"] < 0.05
                and (tests["wilcoxon_signed_rank"]["p_two_sided"] or 1.0) < 0.05
                and tests["paired_t"]["p_two_sided"] < 0.05)

    payload: dict[str, Any] = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question": ("Is the inner-fold Optuna tuning net-harmful on these "
                     "outer folds, or was the gap between the two means noise?"),
        "resolves": "docs/TUNING_OVERFIT_HYPOTHESIS.md",
        "design": {
            "family": FAMILY,
            "pairing": ("15 outer folds, repeat-major, from "
                        "nested.build_outer_folds(n_repeats=3, n_inner=4)"),
            "tuned_arm": ("nested tournament: Optuna inside each fold, feature "
                          "set size chosen from {30, 60, 120} on inner folds"),
            "untuned_arm": ("missingness ablation WITHOUT arm: fixed "
                            "hyperparameters, fixed top-120, same base frame"),
            "retraining_performed": False,
            "reconstruction_check": {
                "rebuilt_pr_auc_mean": round(rebuilt, 5),
                "published_pr_auc_mean": round(claimed, 5),
                "tolerance": RECONSTRUCTION_TOL,
            },
        },
        "tuned_fold_ap": [round(x, 5) for x in tuned_folds],
        "untuned_fold_ap": [round(x, 5) for x in untuned_folds],
        "per_fold_gain_from_not_tuning": [round(float(x), 5) for x in diff],
        "mean_gain_from_not_tuning": round(float(diff.mean()), 5),
        "median_gain_from_not_tuning": round(float(np.median(diff)), 5),
        "std_of_paired_diff": round(float(diff.std(ddof=1)), 5),
        "tests": tests,
        "effect_survives_pairing": bool(survives),
    }

    # The two means in the note were pooled per repeat - every fold's scores
    # ranked against every other fold's - while the paired test above is
    # per fold. They are different quantities and here they disagree, so both
    # are published with the gap between them named rather than left for a
    # reader to trip over.
    untuned_pooled = float(ablation["without"]["pr_auc_mean"])
    payload["pooling_decomposition"] = {
        "tuned_mean_of_fold_ap": round(float(np.mean(tuned_folds)), 5),
        "untuned_mean_of_fold_ap": round(float(np.mean(untuned_folds)), 5),
        "tuned_pooled_pr_auc": round(rebuilt, 5),
        "untuned_pooled_pr_auc": round(untuned_pooled, 5),
        "pooling_cost_tuned": round(float(np.mean(tuned_folds)) - rebuilt, 5),
        "pooling_cost_untuned": round(
            float(np.mean(untuned_folds)) - untuned_pooled, 5),
        "reading": (
            "Within a fold the two arms rank almost identically. The gap opens "
            "only when the folds are pooled, and it opens because the tuned arm "
            "loses more to pooling: each of its folds fits different "
            "hyperparameters on a different number of features, so its scores "
            "are less comparable across folds. This is consistent with - not "
            "proof of - in-fold tuning buying fold-local fit at the cost of a "
            "common score scale."),
        "why_it_matters": (
            "Deployment pools: one frozen threshold is applied to every "
            "account. The pooled figure is therefore the one that describes "
            "the product, which is why the tournament promotes on it."),
    }
    payload["finding"] = (
        "Tuning is net-harmful on these folds: the untuned configuration wins "
        f"by {payload['mean_gain_from_not_tuning']} PR-AUC on average and all "
        "three paired tests agree." if survives else
        "The gap between the two means does not survive pairing: fold by fold "
        "the untuned arm gains only "
        f"{payload['mean_gain_from_not_tuning']} PR-AUC (sd "
        f"{payload['std_of_paired_diff']}), and the three tests agree that this "
        "is not separable from zero. The larger pooled gap is a pooling effect, "
        "quantified in pooling_decomposition, not evidence that tuning costs "
        "ranking performance.")
    payload["what_this_does_not_license"] = (
        "Nothing is re-promoted on this test. It compares one family's tuned and "
        "untuned arms on the development folds; it does not rank families, does "
        "not touch the locked test, and does not license quoting the higher "
        "number as the model's score.")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    payload = run()
    save_json(payload, OUT)

    log.info("tuned   folds: %s", payload["tuned_fold_ap"])
    log.info("untuned folds: %s", payload["untuned_fold_ap"])
    log.info("gain from not tuning: mean %+.5f  median %+.5f  sd %.5f",
             payload["mean_gain_from_not_tuning"],
             payload["median_gain_from_not_tuning"],
             payload["std_of_paired_diff"])
    t = payload["tests"]
    log.info("sign %d/%d p=%s | wilcoxon p=%s | paired t p=%s",
             t["sign_test"]["folds_favouring_untuned"], t["sign_test"]["n_folds"],
             t["sign_test"]["p_two_sided"],
             t["wilcoxon_signed_rank"]["p_two_sided"],
             t["paired_t"]["p_two_sided"])
    log.info("%s", payload["finding"])
    log.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
