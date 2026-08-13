"""Does the shipped champion survive the primary protocol?

The champion in ``artifacts/models/final_bundle.joblib`` was promoted on the
**flat** repeated-CV tournament: feature selection and hyperparameter tuning
happened once, outside the fold loop, and the folds then scored the result. That
protocol is optimistic by construction - the selection saw every row it was
later scored on.

The nested run fixes that: selection and tuning happen inside each outer fold,
against inner splits only, and the outer validation rows are predicted exactly
once. It is the estimate this project calls primary, and it is the one that
decides. This module does not retrain anything. It reads what the nested run
measured, applies the *same* promotion rule the flat tournament applied - the
serving-cost veto first, then the tie-band and the generalization score - and
says one of three things:

    CHAMPION_CONFIRMED   the nested protocol promotes the model already shipped
    CHAMPION_CHALLENGED  the nested protocol promotes a different model
    PENDING_EVIDENCE     the nested run is incomplete, so it decides nothing

A challenge is not a failure and it is not automatically acted on: swapping the
bundle means re-fitting the calibrator, re-freezing the thresholds and spending
the locked test again. What it is, is a fact that has to be stated rather than
buried under a flat number that reads better.

Exit codes: 0 confirmed, 1 challenged, 2 pending. A non-zero exit is the point -
a release script must not be able to walk past this.

    .venv/Scripts/python.exe -m muleguard.cli.nested_promotion
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.cli.tournament_v2 import (
    INTERACTIVE_BUDGET_SECONDS, RECALL_BONUS_WEIGHT, STABILITY_PENALTY_WEIGHT,
    TIE_BAND, TOPK_BUDGET, _promotable, _serving_cost_by_family,
    generalization_score,
)
from muleguard.logging import get_logger
from muleguard.utils import load_json, save_json

log = get_logger("cli.nested_promotion")

NESTED_JSON = settings.METRICS_DIR / "nested_cv.json"
NESTED_OOF = settings.PREDICTIONS_DIR / "nested_oof.parquet"
FLAT_PROMOTION = settings.METRICS_DIR / "promotion_decision_v2.json"
OUT = settings.METRICS_DIR / "nested_promotion_decision.json"

CONFIRMED = "CHAMPION_CONFIRMED"
CHALLENGED = "CHAMPION_CHALLENGED"
PENDING = "PENDING_EVIDENCE"

# Families that exist only to prove the metric is not degenerate. They are
# scored and reported, never promoted - a dummy that wins would mean the
# evaluation is broken, not that the dummy is good.
NON_CANDIDATE_FAMILIES = {"dummy_prevalence"}


def _recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if y.sum() == 0:
        return 0.0
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].sum() / y.sum())


def _nested_recall_at_k() -> dict[str, float]:
    """Mean Recall@TopK per family, from the stored nested predictions."""
    if not NESTED_OOF.exists():
        return {}
    preds = pl.read_parquet(NESTED_OOF)
    out: dict[str, float] = {}
    for (name,), sub in preds.group_by(["model"]):
        per_repeat = [
            _recall_at_k(rep["target"].to_numpy(), rep["score"].to_numpy(),
                         TOPK_BUDGET)
            for (_r,), rep in sub.group_by(["repeat"])
        ]
        out[str(name)] = float(np.mean(per_repeat)) if per_repeat else 0.0
    return out


def _paired_comparison(deployed: str, promoted: str,
                       n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Is the gap between two families bigger than the noise they share?

    The marginal bootstrap intervals of two families on 64 positives are wide
    and overlap almost completely, which invites the reading "the difference is
    not significant". That reading is wrong here, because the two families were
    scored on *the same rows in the same folds*: the honest comparison is
    paired, and the quantity with an interval around it is the difference, not
    each mean separately.

    Two paired views, because they can disagree and both matter:

    * per repeat - the sign of the gap in each of the three repeats. Three
      points is not a test, so it is reported as a count and never as a p-value.
    * account bootstrap - accounts are resampled once per replicate, stratified
      on the label to hold the 64 positives fixed, and *both* families are
      rescored on that same resample. The spread of the difference is the
      sampling noise the pairing does not remove.
    """
    if not NESTED_OOF.exists():
        return {}
    from sklearn.metrics import average_precision_score as ap

    preds = pl.read_parquet(NESTED_OOF)
    wide = {}
    for fam in (deployed, promoted):
        sub = preds.filter(pl.col("model") == fam)
        if sub.height == 0:
            return {}
        wide[fam] = {int(r): rep.sort("row_index")
                     for (r,), rep in sub.group_by(["repeat"])}
    repeats = sorted(set(wide[deployed]) & set(wide[promoted]))
    if not repeats:
        return {}

    per_repeat = []
    for r in repeats:
        d, p = wide[deployed][r], wide[promoted][r]
        y = d["target"].to_numpy()
        a_d = float(ap(y, d["score"].to_numpy()))
        a_p = float(ap(p["target"].to_numpy(), p["score"].to_numpy()))
        per_repeat.append({"repeat": r, "deployed_ap": round(a_d, 5),
                           "promoted_ap": round(a_p, 5),
                           "delta": round(a_p - a_d, 5)})

    y0 = wide[deployed][repeats[0]]["target"].to_numpy()
    pos = np.flatnonzero(y0 == 1)
    neg = np.flatnonzero(y0 == 0)
    scores = {fam: {r: wide[fam][r]["score"].to_numpy() for r in repeats}
              for fam in (deployed, promoted)}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        yb = y0[idx]
        deltas[b] = float(np.mean([
            ap(yb, scores[promoted][r][idx]) - ap(yb, scores[deployed][r][idx])
            for r in repeats]))
    lo, hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    return {
        "deployed_family": deployed,
        "promoted_family": promoted,
        "per_repeat": per_repeat,
        "repeats_favouring_promoted":
            f"{sum(1 for d in per_repeat if d['delta'] > 0)}/{len(per_repeat)}",
        "paired_delta_mean": round(float(np.mean(deltas)), 5),
        "paired_delta_ci95": [round(lo, 5), round(hi, 5)],
        "share_of_resamples_favouring_promoted":
            round(float(np.mean(deltas > 0)), 4),
        "n_boot": n_boot,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "reading": ("The interval is on the *difference* between two families "
                    "scored on identical rows. It is not comparable to the "
                    "single-family intervals in metric_battery.json, which are "
                    "marginal and much wider by construction."),
    }


def _deployed() -> dict[str, Any]:
    """The model that is actually shipped, read from the bundle itself.

    Read from the bundle rather than from the promotion artifact, because the
    artifact records an intent and the bundle records what happened. If they
    ever disagree, the bundle is the one serving traffic.
    """
    import joblib
    path = settings.MODELS_DIR / "final_bundle.joblib"
    if not path.exists():
        return {}
    b = joblib.load(path)
    return {"model": b.get("winner_oof_name"), "family": b.get("winner_family"),
            "bundle_version": b.get("version"),
            "flat_oof_pr_auc_mean": b.get("oof_pr_auc_mean")}


def _expected_repeats(design: dict[str, Any]) -> int | None:
    """Repeat count the nested design claims, parsed from its own description.

    The design is prose - "stratified 5-fold x 3 repeats (...)" - so the count
    is read as the token immediately before "repeats". Taken from the artifact
    rather than from a constant, so a run written by an older or differently
    configured invocation is still compared against what *it* claimed to do.
    """
    parts = str(design.get("outer") or "").split()
    for i, tok in enumerate(parts):
        if tok.startswith("repeat") and i:
            try:
                return int(parts[i - 1])
            except ValueError:
                return None
    return None


def decide() -> tuple[dict[str, Any], int]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    deployed = _deployed()
    base: dict[str, Any] = {
        "generated_utc": now,
        "question": ("Does the model promoted on the flat protocol survive the "
                     "nested protocol, which is the primary one?"),
        "deployed": deployed,
        "rule": (
            "The nested leaderboard is filtered by the serving-cost veto "
            f"({INTERACTIVE_BUDGET_SECONDS}s to score one account), then the "
            f"leader defines a {TIE_BAND} PR-AUC tie band, and inside that band "
            f"generalization_score = PR_AUC_mean - {STABILITY_PENALTY_WEIGHT} * "
            f"PR_AUC_std + {RECALL_BONUS_WEIGHT} * Recall@{TOPK_BUDGET} picks the "
            "more stable model. Identical to the flat promotion rule, so any "
            "difference in outcome comes from the protocol, not the tie-break."),
    }

    if not NESTED_JSON.exists():
        base |= {"verdict": PENDING,
                 "why": f"{NESTED_JSON.name} does not exist yet",
                 "fills_when": "muleguard.cli.nested_cv --repeats 3 --inner 4"}
        return base, 2

    nested = load_json(NESTED_JSON) or {}
    board = [r for r in (nested.get("leaderboard") or [])
             if r.get("model") not in NON_CANDIDATE_FAMILIES]
    design = nested.get("design") or {}
    want = _expected_repeats(design)
    got = sorted({int(r.get("n_repeats") or 0) for r in board}) or [0]

    base |= {
        "nested_run": {
            "generated_utc": nested.get("generated_utc"),
            "design": design,
            "families_scored": [r.get("model") for r in board],
            "repeats_per_family": {r.get("model"): r.get("n_repeats") for r in board},
        },
    }

    # An incomplete nested run cannot decide anything. Reporting the partial
    # leader as if it were the answer is precisely the failure mode this file
    # exists to prevent - the preliminary 1-repeat run had two families in it
    # and would have "confirmed" whichever one happened to be there.
    if want and (len(got) > 1 or got[0] != want):
        base |= {
            "verdict": PENDING,
            "why": (f"the design declares {want} repeats but the leaderboard "
                    f"carries {got}; this is a partial or superseded run"),
            "fills_when": "muleguard.cli.nested_cv --repeats 3 --inner 4",
        }
        return base, 2
    if len(board) < 3:
        base |= {
            "verdict": PENDING,
            "why": (f"only {len(board)} candidate famil(y/ies) scored; a "
                    "promotion needs a field to promote from"),
            "fills_when": "muleguard.cli.nested_cv --repeats 3 --inner 4",
        }
        return base, 2

    costs = _serving_cost_by_family()
    recalls = _nested_recall_at_k()
    rows = []
    for r in board:
        family = str(r.get("model"))
        cost = costs.get(family, {})
        rec = float(recalls.get(family, 0.0))
        rows.append({
            "model": family,
            "nested_pr_auc_mean": round(float(r["pr_auc_mean"]), 5),
            "nested_pr_auc_std": round(float(r.get("pr_auc_std") or 0.0), 5),
            "nested_roc_auc_mean": round(float(r.get("roc_auc_mean") or 0.0), 5),
            "feature_size_mode": r.get("feature_size_mode"),
            f"nested_recall_at_{TOPK_BUDGET}": round(rec, 5),
            "generalization_score": round(generalization_score(
                float(r["pr_auc_mean"]), float(r.get("pr_auc_std") or 0.0), rec), 5),
            "interactive_score_seconds": cost.get("single_row_seconds"),
            "promotion_eligible": _promotable(cost),
        })
    rows.sort(key=lambda d: -d["nested_pr_auc_mean"])

    servable = [r for r in rows if r["promotion_eligible"]]
    if not servable:
        base |= {"verdict": PENDING, "leaderboard": rows,
                 "why": ("no nested candidate can be served inside the "
                         f"{INTERACTIVE_BUDGET_SECONDS}s interactive budget")}
        return base, 2

    lead = servable[0]
    band = [r for r in servable
            if r["nested_pr_auc_mean"] >= lead["nested_pr_auc_mean"] - TIE_BAND]
    promoted = sorted(band, key=lambda d: (-d["generalization_score"],
                                           d["feature_size_mode"] or 0))[0]

    # The deployed model is named `xgboost_top_120`; the nested leaderboard is
    # per family, because the nested run chooses its own feature-set size inside
    # every fold. Comparing family to family is the honest comparison - the
    # feature count is an outcome of the protocol, not an input to it.
    deployed_family = str(deployed.get("family") or "")
    confirmed = deployed_family == promoted["model"]
    deployed_row = next((r for r in rows if r["model"] == deployed_family), None)

    base |= {
        "leaderboard": rows,
        "nested_leader": lead["model"],
        "models_within_tie_band": [r["model"] for r in band],
        "nested_promoted": promoted["model"],
        "nested_promoted_detail": promoted,
        "deployed_family_under_nested": deployed_row,
        "benched_on_serving_cost": [
            {k: r[k] for k in ("model", "nested_pr_auc_mean",
                               "interactive_score_seconds")}
            for r in rows if not r["promotion_eligible"]],
        "verdict": CONFIRMED if confirmed else CHALLENGED,
    }

    if confirmed:
        base["why"] = (f"the nested protocol promotes {promoted['model']}, which "
                       "is the family already shipped")
    else:
        gap = (promoted["nested_pr_auc_mean"]
               - (deployed_row or {}).get("nested_pr_auc_mean", float("nan")))
        base["why"] = (
            f"the nested protocol promotes {promoted['model']}, not the shipped "
            f"{deployed_family or 'unknown'}")
        base["nested_gap_over_deployed"] = (
            None if deployed_row is None else round(float(gap), 5))
        # A challenge that cannot survive its own noise is a coin flip, and
        # acting on one costs the locked test. The paired interval is what
        # separates the two cases, so it is computed before the finding is
        # written rather than argued about afterwards.
        paired = (_paired_comparison(deployed_family, promoted["model"])
                  if deployed_row is not None else {})
        if paired:
            base["paired_check"] = paired
        base["what_acting_on_this_costs"] = [
            "re-fit the calibrator and the conformal layer on the new scores",
            "re-freeze the policy thresholds against the new score distribution",
            "re-open the locked test, which is single-touch by construction",
            "re-run the release suite, because every QA number names the model",
        ]
        base["not_done_automatically"] = (
            "This file records the finding. Swapping the champion is a decision "
            "with a locked-test cost attached and is not taken by a report.")
    return base, (0 if confirmed else 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-challenge", action="store_true",
                    help="exit 0 even when the nested protocol promotes a "
                         "different model (the finding is still written)")
    args = ap.parse_args(argv)

    payload, rc = decide()
    save_json(payload, OUT)

    verdict = payload["verdict"]
    log.info("nested promotion verdict: %s", verdict)
    if verdict == PENDING:
        log.info("  %s", payload.get("why"))
        log.info("  fills when: %s", payload.get("fills_when", "-"))
    else:
        log.info("  nested leader   : %s", payload.get("nested_leader"))
        log.info("  nested promotes : %s", payload.get("nested_promoted"))
        log.info("  deployed family : %s", (payload.get("deployed") or {}).get("family"))
        for r in payload.get("leaderboard", []):
            log.info("    %-18s nested PR-AUC %.5f +/- %.5f  gen %.5f%s",
                     r["model"], r["nested_pr_auc_mean"], r["nested_pr_auc_std"],
                     r["generalization_score"],
                     "" if r["promotion_eligible"] else "  [benched: serving cost]")
    log.info("wrote %s", OUT)

    if args.allow_challenge and rc == 1:
        return 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
