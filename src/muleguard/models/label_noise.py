"""Label-noise audit (addendum UPDATE 5).

With 81 mules in 9,082 accounts, a single mislabelled row is over one percent
of the entire positive class. That makes label quality a first-class question
rather than a footnote - but it also makes it a dangerous one, because the
cheapest way to improve any metric here is to quietly discard the mules the
model finds hard.

This module therefore only ever *reports*. It never flips a label, never drops
a row, and nothing downstream consumes its output as a filter. Two populations
are surfaced:

  POSSIBLE_LABEL_NOISE
      An account labelled mule that every competent model, across every CV
      repeat, ranks among the ordinary accounts. Worth a human looking at.

  HIGH_SCORING_NEGATIVE
      An account labelled legitimate that every competent model puts in the top
      fraction of a percent. This is deliberately NOT called a hidden mule: on
      this data it is at least as likely to be a genuine false positive, and
      that reading matters for the false-positive work.

The distinction the audit cannot make from data alone is between a wrong label
and a real mule whose behaviour simply is not in the feature set. Every flagged
row carries that caveat explicitly, because a reviewer who forgets it will
delete difficult examples and call it cleaning.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("models.label_noise")

# Fixed before the audit was first run, so the flagged set cannot be tuned to a
# convenient size.
NOISE_THRESHOLDS = {
    # A labelled mule below this percentile of all scored accounts is a
    # candidate. The median is a deliberately unambitious bar: it means the
    # model ranks the account as less suspicious than half the book.
    "positive_percentile_max": 50.0,
    # ... and it must hold under this share of (model, repeat) evaluations, so
    # one unlucky fold cannot flag an account.
    "positive_agreement_min": 0.80,
    # A labelled legitimate above this percentile is a candidate.
    "negative_percentile_min": 99.5,
    "negative_agreement_min": 0.80,
    # Models weaker than this are excluded from the consensus entirely: asking
    # a near-random ranker to vote on label quality adds noise, not evidence.
    "model_pr_auc_floor": 0.50,
}

FLAG_NOISE = "POSSIBLE_LABEL_NOISE"
FLAG_HIGH_NEG = "HIGH_SCORING_NEGATIVE"

POLICY = (
    "This audit reports only. Labels are never flipped, rows are never removed, "
    "and no model in this project is trained, calibrated or thresholded on a "
    "filtered version of the label set. A flagged positive is a request for "
    "human review, not a verdict about the account or the analyst who labelled it."
)

CAVEAT = (
    "A mule the model cannot rank is not necessarily mislabelled. The same "
    "evidence is equally consistent with a genuine mule whose behaviour is not "
    "represented in the available features, or with a pattern too rare for 64 "
    "training positives to teach. Removing these rows would raise every metric "
    "in this repository while making the detector strictly worse on the hidden "
    "validation set."
)


def _competent_models(tournament: dict[str, Any], floor: float) -> list[str]:
    """Models whose OOF PR-AUC clears the floor, best first."""
    rows = [
        (name, m.get("pr_auc_mean"))
        for name, m in tournament.get("models", {}).items()
        if isinstance(m, dict) and m.get("pr_auc_mean") is not None
    ]
    keep = sorted([r for r in rows if r[1] >= floor], key=lambda r: -r[1])
    return [name for name, _ in keep]


def _percentile_frame(oof: pl.DataFrame) -> pl.DataFrame:
    """Per (model, repeat), convert scores to within-run percentile ranks.

    Percentiles rather than raw scores because the models are not calibrated
    against each other - averaging an XGBoost probability with a LightGBM one
    would weight whichever model happens to be more confident, not whichever is
    more correct. Rank is the common currency.
    """
    return oof.with_columns(
        (pl.col("score").rank("average").over(["model", "repeat"])
         / pl.len().over(["model", "repeat"]) * 100.0).alias("pct")
    )


def audit_label_noise(
    oof: pl.DataFrame | None = None,
    tournament: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the audit over the stored OOF predictions.

    Args:
        oof: long frame with row_index / repeat / model / target / score.
        tournament: the tournament artifact, used to pick competent models.
    """
    from muleguard.utils import load_json

    t = {**NOISE_THRESHOLDS, **(thresholds or {})}
    if oof is None:
        oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_v2.parquet")
    if tournament is None:
        tournament = load_json(settings.METRICS_DIR / "tournament_v2.json")

    models = _competent_models(tournament, t["model_pr_auc_floor"])
    if not models:
        raise RuntimeError(
            "no model clears the PR-AUC floor; a consensus of weak rankers "
            "would be noise dressed as evidence")

    df = _percentile_frame(oof.filter(pl.col("model").is_in(models)))
    n_runs = df.select(pl.struct("model", "repeat").n_unique()).item()

    per_row = (
        df.group_by("row_index")
        .agg(
            pl.col("target").first().alias("target"),
            pl.col("pct").mean().alias("mean_percentile"),
            pl.col("pct").min().alias("min_percentile"),
            pl.col("pct").max().alias("max_percentile"),
            pl.col("score").mean().alias("mean_score"),
            pl.len().alias("n_evaluations"),
            (pl.col("pct") <= t["positive_percentile_max"]).mean().alias("share_low"),
            (pl.col("pct") >= t["negative_percentile_min"]).mean().alias("share_high"),
        )
        .sort("row_index")
    )

    pos = per_row.filter(pl.col("target") == 1)
    neg = per_row.filter(pl.col("target") == 0)

    flagged_pos = pos.filter(pl.col("share_low") >= t["positive_agreement_min"]) \
                     .sort("mean_percentile")
    flagged_neg = neg.filter(pl.col("share_high") >= t["negative_agreement_min"]) \
                     .sort("mean_percentile", descending=True)

    def rows(frame: pl.DataFrame, flag: str, limit: int = 60) -> list[dict[str, Any]]:
        out = []
        for r in frame.head(limit).iter_rows(named=True):
            out.append({
                "row_index": int(r["row_index"]),
                "flag": flag,
                "label": int(r["target"]),
                "mean_percentile": round(float(r["mean_percentile"]), 3),
                "min_percentile": round(float(r["min_percentile"]), 3),
                "max_percentile": round(float(r["max_percentile"]), 3),
                "mean_score": round(float(r["mean_score"]), 6),
                "agreement": round(float(r["share_low"] if flag == FLAG_NOISE
                                         else r["share_high"]), 3),
                "n_evaluations": int(r["n_evaluations"]),
                "action": "HUMAN_REVIEW_ONLY",
            })
        return out

    n_pos, n_neg = pos.height, neg.height
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "thresholds": t,
        "consensus_models": models,
        "n_consensus_models": len(models),
        "n_model_repeat_runs": int(n_runs),
        "n_rows_audited": per_row.height,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "possible_label_noise": {
            "flag": FLAG_NOISE,
            "definition": (
                f"labelled mule ranked at or below the "
                f"{t['positive_percentile_max']:.0f}th percentile in at least "
                f"{t['positive_agreement_min'] * 100:.0f}% of model-repeat runs"),
            "count": flagged_pos.height,
            "share_of_positives": round(flagged_pos.height / max(n_pos, 1), 4),
            "rows": rows(flagged_pos, FLAG_NOISE),
        },
        "high_scoring_negatives": {
            "flag": FLAG_HIGH_NEG,
            "definition": (
                f"labelled legitimate ranked at or above the "
                f"{t['negative_percentile_min']}th percentile in at least "
                f"{t['negative_agreement_min'] * 100:.0f}% of model-repeat runs"),
            "count": flagged_neg.height,
            "share_of_negatives": round(flagged_neg.height / max(n_neg, 1), 5),
            "reading": (
                "these are candidate false positives as much as candidate "
                "missed mules; the audit does not claim to distinguish them"),
            "rows": rows(flagged_neg, FLAG_HIGH_NEG),
        },
        "positive_percentile_distribution": {
            "p10": round(float(np.percentile(pos["mean_percentile"].to_numpy(), 10)), 2),
            "median": round(float(np.median(pos["mean_percentile"].to_numpy())), 2),
            "p90": round(float(np.percentile(pos["mean_percentile"].to_numpy(), 90)), 2),
        } if n_pos else None,
        "caveat": CAVEAT,
        "policy": POLICY,
        "guarantees": [
            "no label was modified",
            "no row was removed from any training, calibration or evaluation set",
            "no downstream component reads this artifact as a filter",
        ],
    }
    log.info("label-noise audit: %d/%d positives flagged, %d negatives flagged, "
             "consensus over %d models",
             payload["possible_label_noise"]["count"], n_pos,
             payload["high_scoring_negatives"]["count"], len(models))
    return payload
