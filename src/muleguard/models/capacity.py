"""Analyst Capacity Optimizer - what a fixed amount of analyst time actually buys.

A bank does not have an unlimited review team. It has a number: the accounts a
desk can work through in a day. This module turns that number into measured
expectations, and turns a tolerance for false alarms into the same thing from
the other direction:

    "we can review K accounts a day"        -> expected recall / precision
    "we accept at most F false alarms
     per 1,000 accounts"                    -> the K that stays inside it

Both answers come from one object, the *capacity curve*, computed once from the
stored development out-of-fold predictions of the promoted champion. Nothing is
refitted here and no model is trained: the curve is a re-reading of predictions
that already exist on disk.

Why a curve rather than a formula
---------------------------------
The alternative - a fixed cost function such as ``FN cost = 10, FP cost = 1`` -
requires the bank to accept two numbers it was never asked for. A capacity
curve asks for the one number a head of operations actually knows (how many
accounts the desk can work) and reports what the model does at it.

Honesty constraints baked into the maths
----------------------------------------
* The development split holds **64 positives in 7,264 rows**. Every recall
  figure is therefore a multiple of 1/64 = 1.56 percentage points, and a
  single-point estimate would be false precision. Every point on the curve
  carries a stratified bootstrap interval and the per-repeat spread.
* The headline ranking is the repeat-averaged one, because that is what the
  served bundle scores with. The three individual CV repeats are reported
  beside it: averaging three repeats denoises the ranking, so the averaged
  curve sits *above* the individual repeats. That gap is reported, not hidden.
* Recall and precision at a budget are pure ranking statistics, so they are
  unchanged by any monotone calibration. The recommended *threshold* is not,
  so it is reported on the calibrated-risk scale the served API actually emits,
  using the frozen calibrator from the shipped bundle.
* The recommendation is advisory. The frozen policy thresholds
  (``policy_version 1.0``) are read, compared against, and never written.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Status string used everywhere a recommendation leaves this module. Nothing
# here changes a threshold; a person does, deliberately, elsewhere.
ADVISORY_STATUS = "ADVISORY_REQUIRES_HUMAN_APPROVAL"

FP_BASES = ("legitimate", "screened")


# --------------------------------------------------------------------------
# pure computation
# --------------------------------------------------------------------------
def default_budget_grid(n: int) -> list[int]:
    """Review budgets to evaluate, dense where a real desk operates.

    Every K from 1 to 300 is exact because that is the range an analyst team
    plausibly lives in; above it the grid coarsens, and any lookup that lands
    between grid points is reported at the point it was answered on rather
    than interpolated into a number nobody measured.
    """
    grid: set[int] = set(range(1, min(300, n) + 1))
    grid.update(range(300, min(1000, n) + 1, 10))
    grid.update(range(1000, n + 1, 100))
    grid.add(n)
    return sorted(k for k in grid if 1 <= k <= n)


def _tp_at_budgets(y_sorted_desc: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """True positives inside the top-K, for every K at once.

    One cumulative sum answers the whole curve, which is what keeps the
    bootstrap affordable: a replicate costs one sort, not one sort per budget.
    """
    cum = np.cumsum(y_sorted_desc)
    return cum[budgets - 1]


def capacity_points(
    y: np.ndarray,
    scores: np.ndarray,
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    """Point estimates of the whole curve from one score vector.

    ``scores`` only has to rank; its scale is irrelevant to every field here
    except ``threshold_score``, which is by definition on the scale given.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0:
        raise ValueError("capacity curve needs at least one positive")

    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    s_sorted = scores[order]
    ks = np.asarray([k for k in budgets], dtype=int)
    tps = _tp_at_budgets(y_sorted, ks)

    out: list[dict[str, Any]] = []
    for k, tp in zip(ks.tolist(), tps.tolist()):
        tp = int(tp)
        fp = k - tp
        thr = float(s_sorted[k - 1])
        out.append({
            "budget": k,
            "true_positives": tp,
            "false_positives": fp,
            "mules_missed": n_pos - tp,
            "recall": tp / n_pos,
            "precision": tp / k,
            "fp_per_1000_legitimate": 1000.0 * fp / n_neg if n_neg else 0.0,
            "fp_per_1000_screened": 1000.0 * fp / n,
            "threshold_score": thr,
            # A threshold is only useful if applying it reproduces the budget.
            # Recomputed rather than assumed, because tied scores would break
            # the identity silently.
            "n_alerts_at_threshold": int((scores >= thr).sum()),
        })
    return out


def leading_true_positive_run(y: np.ndarray, scores: np.ndarray) -> int:
    """How many of the highest-ranked accounts are mules before the first miss.

    Reported because it explains the shape of the interval at small budgets:
    while the budget stays inside this run there is nothing for a resampling
    scheme that holds the mule count fixed to vary, so its interval collapses
    onto the point estimate. That is a property of the ranking, not a claim of
    certainty, and the panel has to say so.
    """
    y = np.asarray(y).astype(int)
    ordered = y[np.argsort(-np.asarray(scores, dtype=float), kind="stable")]
    miss = np.flatnonzero(ordered == 0)
    return int(miss[0]) if miss.size else int(len(ordered))


def bootstrap_budget_bands(
    y: np.ndarray,
    scores: np.ndarray,
    budgets: Sequence[int],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    scheme: str = "stratified",
) -> dict[int, dict[str, float]]:
    """Bootstrap interval for recall and precision at every budget.

    Two schemes, because they answer two different questions and neither one
    alone is honest here:

    ``stratified``
        Positives and negatives resampled separately, the same convention as
        :func:`muleguard.evaluation.metrics.bootstrap_ci`. Every replicate keeps
        exactly 64 mules in 7,264 rows, so this measures uncertainty in the
        *ranking* alone. Inside the leading run of true positives it is
        degenerate by construction - see :func:`leading_true_positive_run`.

    ``resample_accounts``
        The whole book resampled with replacement, so the number of mules in a
        replicate moves too. This measures "what if the bank's month had
        contained a different set of accounts", and it is the wider, more
        pessimistic of the two at small budgets because recall's denominator
        moves as well as its numerator.

    Both are reported. A reader who wants ranking stability reads the first; a
    reader who wants to know how much the whole exercise could move on a fresh
    sample reads the second.
    """
    if scheme not in ("stratified", "resample_accounts"):
        raise ValueError(f"unknown bootstrap scheme {scheme!r}")
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    ks = np.asarray(sorted(set(int(k) for k in budgets)), dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n, n_pos = len(y), len(pos_idx)
    rng = np.random.default_rng(seed)

    recall_draws = np.empty((n_boot, len(ks)), dtype=float)
    prec_draws = np.empty((n_boot, len(ks)), dtype=float)
    drawn = 0
    for _ in range(n_boot):
        if scheme == "stratified":
            idx = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ])
            pos_in_draw = n_pos
        else:
            idx = rng.integers(0, n, size=n)
            pos_in_draw = int(y[idx].sum())
            if pos_in_draw == 0:
                # A replicate with no mules has no recall to report. Skipping it
                # is the only option, and it is rare enough at 64 positives that
                # it does not bias the interval in any direction that matters.
                continue
        ordered = y[idx][np.argsort(-scores[idx], kind="stable")]
        tp = _tp_at_budgets(ordered, ks).astype(float)
        recall_draws[drawn] = tp / pos_in_draw
        prec_draws[drawn] = tp / ks
        drawn += 1

    recall_draws, prec_draws = recall_draws[:drawn], prec_draws[:drawn]
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    bands: dict[int, dict[str, float]] = {}
    for j, k in enumerate(ks.tolist()):
        rec_lo, rec_hi = np.quantile(recall_draws[:, j], [lo_q, hi_q])
        pre_lo, pre_hi = np.quantile(prec_draws[:, j], [lo_q, hi_q])
        bands[k] = {
            "recall_ci_low": float(rec_lo),
            "recall_ci_high": float(rec_hi),
            "precision_ci_low": float(pre_lo),
            "precision_ci_high": float(pre_hi),
        }
    return bands


def per_repeat_recall(
    repeats: dict[int, tuple[np.ndarray, np.ndarray]],
    budgets: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Recall at every budget for each CV repeat, kept separate on purpose.

    ``repeats`` maps repeat id -> (y, scores). A single repeat is a noisier
    estimator than the repeat-average the bundle serves; the spread across the
    three is the honest answer to "how much does this move if you reshuffle
    the folds".
    """
    ks = [int(k) for k in budgets]
    per_k: dict[int, dict[str, Any]] = {k: {"values": []} for k in ks}
    for _, (y_r, s_r) in sorted(repeats.items()):
        y_r = np.asarray(y_r).astype(int)
        n_pos = int(y_r.sum())
        ordered = y_r[np.argsort(-np.asarray(s_r, dtype=float), kind="stable")]
        tps = _tp_at_budgets(ordered, np.asarray(ks, dtype=int))
        for k, tp in zip(ks, tps.tolist()):
            per_k[k]["values"].append(float(tp) / n_pos)
    for k in ks:
        v = per_k[k]["values"]
        lo, hi = float(min(v)), float(max(v))
        # An arithmetic mean always lies inside [min, max]; only floating-point
        # rounding can push it a fraction of an ulp outside, which then reads
        # as "the average repeat beat every repeat". Clamp so the published
        # spread is never self-contradictory.
        per_k[k]["min"] = lo
        per_k[k]["max"] = hi
        per_k[k]["mean"] = float(min(max(sum(v) / len(v), lo), hi))
    return per_k


def build_curve(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    repeats: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    budgets: Sequence[int] | None = None,
    calibrate=None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Assemble the full curve: point estimate, interval, per-repeat spread.

    ``calibrate`` is an optional callable mapping raw scores to the calibrated
    risk the served API emits. It is applied to the *threshold* only; it cannot
    change recall or precision at a budget, because those depend on the
    ranking, and calibration here is monotone.
    """
    n = len(y)
    ks = list(budgets) if budgets is not None else default_budget_grid(n)
    points = capacity_points(y, scores, ks)
    bands = bootstrap_budget_bands(y, scores, ks, n_boot=n_boot, seed=seed,
                                   alpha=alpha, scheme="stratified")
    bands_acc = bootstrap_budget_bands(y, scores, ks, n_boot=n_boot, seed=seed + 1,
                                       alpha=alpha, scheme="resample_accounts")
    spread = per_repeat_recall(repeats, ks) if repeats else {}
    pure_run = leading_true_positive_run(y, scores)

    thr_raw = np.asarray([p["threshold_score"] for p in points], dtype=float)
    thr_cal = calibrate(thr_raw) if calibrate is not None else None

    for i, p in enumerate(points):
        k = p["budget"]
        p.update(bands[k])
        p["recall_ci_low_resampled_accounts"] = bands_acc[k]["recall_ci_low"]
        p["recall_ci_high_resampled_accounts"] = bands_acc[k]["recall_ci_high"]
        p["stratified_interval_degenerate"] = bool(
            p["recall_ci_low"] == p["recall_ci_high"])
        p["inside_leading_true_positive_run"] = bool(k <= pure_run)
        p["threshold_calibrated_risk"] = (
            float(thr_cal[i]) if thr_cal is not None else None)
        if spread:
            s = spread[k]
            p["recall_per_repeat"] = s["values"]
            p["recall_repeat_min"] = s["min"]
            p["recall_repeat_max"] = s["max"]
            p["recall_repeat_mean"] = s["mean"]
    return points


def make_curve_document(
    *,
    points: list[dict[str, Any]],
    evaluation: dict[str, Any],
    frozen_policy: dict[str, Any],
    provenance: dict[str, Any],
    frozen_policy_position: list[dict[str, Any]] | None = None,
    headline_budgets: Sequence[int] = (25, 50, 100),
) -> dict[str, Any]:
    """The one shape the artifact, the API and the tests all agree on.

    Kept here rather than in the CLI so that a change to the document cannot
    drift away from the functions that read it.
    """
    by_k = {p["budget"]: p for p in points}
    return {
        "schema_version": "1.0",
        "evaluation": evaluation,
        "frozen_policy": frozen_policy,
        "frozen_policy_position": frozen_policy_position or [],
        "provenance": provenance,
        "headline": [by_k[k] for k in headline_budgets if k in by_k],
        "points": points,
    }


# --------------------------------------------------------------------------
# answering a judge's question against a computed curve
# --------------------------------------------------------------------------
def _nearest_point(points: Sequence[dict[str, Any]], budget: int) -> dict[str, Any]:
    exact = [p for p in points if p["budget"] == budget]
    if exact:
        return exact[0]
    below = [p for p in points if p["budget"] <= budget]
    return below[-1] if below else points[0]


def answer_for_budget(
    curve: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    """"We can review K accounts a day" -> what that buys, with a threshold."""
    points: list[dict[str, Any]] = curve["points"]
    n = int(curve["evaluation"]["n_accounts"])
    if budget < 1:
        raise ValueError("review capacity must be at least 1 account")
    requested = int(budget)
    capped = min(requested, n)
    point = _nearest_point(points, capped)
    return _plan(curve, point, requested, mode="review_capacity",
                 capped=capped != requested)


def answer_for_fp_per_1000(
    curve: dict[str, Any],
    max_fp_per_1000: float,
    basis: str = "legitimate",
) -> dict[str, Any]:
    """"We accept at most F false alarms per 1,000" -> the largest budget inside it.

    ``basis`` picks the denominator, because the phrase is ambiguous and the
    two readings are not the same number:

      ``legitimate`` false alarms per 1,000 *legitimate* accounts. This is
                     1000 x FPR and matches ``fp_per_1000_legit`` everywhere
                     else in the repository.
      ``screened``   false alarms per 1,000 accounts *screened*, mules
                     included. At 0.88% prevalence it is the slightly smaller
                     number for the same threshold.
    """
    if basis not in FP_BASES:
        raise ValueError(f"basis must be one of {FP_BASES}, got {basis!r}")
    if max_fp_per_1000 < 0:
        raise ValueError("a false-alarm tolerance cannot be negative")
    field = ("fp_per_1000_legitimate" if basis == "legitimate"
             else "fp_per_1000_screened")
    points: list[dict[str, Any]] = curve["points"]
    inside = [p for p in points if p[field] <= max_fp_per_1000 + 1e-12]
    if not inside:
        raise ValueError(
            "no review budget on the measured curve stays inside "
            f"{max_fp_per_1000} false alarms per 1,000; the smallest budget "
            f"measured already produces {points[0][field]:.3f}")
    point = max(inside, key=lambda p: p["budget"])
    out = _plan(curve, point, point["budget"], mode="false_alarm_tolerance",
                capped=False)
    out["input"] = {"max_fp_per_1000": float(max_fp_per_1000), "basis": basis}
    out["binding_constraint"] = (
        f"largest measured review budget whose false alarms stay at or below "
        f"{max_fp_per_1000} per 1,000 {basis} accounts")
    return out


def _plan(
    curve: dict[str, Any],
    point: dict[str, Any],
    requested: int,
    *,
    mode: str,
    capped: bool,
) -> dict[str, Any]:
    ev = curve["evaluation"]
    frozen = curve["frozen_policy"]
    thr = point.get("threshold_calibrated_risk")
    return {
        "mode": mode,
        "input": {"review_capacity": requested},
        "answered_at_budget": point["budget"],
        "answered_exactly": point["budget"] == requested and not capped,
        "capped_to_evaluation_size": capped,
        "expected": {
            "recall": point["recall"],
            "recall_ci_low": point["recall_ci_low"],
            "recall_ci_high": point["recall_ci_high"],
            "recall_ci_low_resampled_accounts": point.get("recall_ci_low_resampled_accounts"),
            "recall_ci_high_resampled_accounts": point.get("recall_ci_high_resampled_accounts"),
            "stratified_interval_degenerate": point.get("stratified_interval_degenerate"),
            "inside_leading_true_positive_run": point.get("inside_leading_true_positive_run"),
            "recall_per_repeat": point.get("recall_per_repeat"),
            "recall_repeat_min": point.get("recall_repeat_min"),
            "recall_repeat_max": point.get("recall_repeat_max"),
            "precision": point["precision"],
            "precision_ci_low": point["precision_ci_low"],
            "precision_ci_high": point["precision_ci_high"],
            "mules_caught": point["true_positives"],
            "mules_missed": point["mules_missed"],
            "mules_available": ev["n_positives"],
            "false_alarms": point["false_positives"],
            "fp_per_1000_legitimate": point["fp_per_1000_legitimate"],
            "fp_per_1000_screened": point["fp_per_1000_screened"],
        },
        "recommended_threshold": {
            "status": ADVISORY_STATUS,
            "applied": False,
            "calibrated_risk": thr,
            "model_score": point["threshold_score"],
            "scale": ("calibrated risk, the same field the scoring API returns "
                      "as calibrated_risk"),
            "alerts_produced_on_evaluation_split": point["n_alerts_at_threshold"],
            "band_under_frozen_policy": (
                _band_of(thr, frozen) if thr is not None else None),
            "frozen_policy_version": frozen["policy_version"],
            "note": ("A recommendation for an authorised human to approve. It "
                     "does not change the frozen policy thresholds, and "
                     "nothing is actioned automatically."),
        },
        "uncertainty_note": (
            f"Measured on {ev['n_accounts']:,} development accounts containing "
            f"only {ev['n_positives']} confirmed mules. One mule is worth "
            f"{100.0 / ev['n_positives']:.2f} percentage points of recall, so "
            "read the interval, not the point."),
        "provenance": curve["provenance"],
    }


def _band_of(threshold: float, frozen: dict[str, Any]) -> str:
    """Which frozen review band a proposed threshold would sit in."""
    if threshold >= frozen["critical_risk"]:
        return "at or above CRITICAL_REVIEW"
    if threshold >= frozen["urgent_risk"]:
        return "between URGENT_REVIEW and CRITICAL_REVIEW"
    if threshold >= frozen["standard_risk"]:
        return "between STANDARD_REVIEW and URGENT_REVIEW"
    return "below STANDARD_REVIEW"


def locate_frozen_policy(
    y: np.ndarray,
    calibrated: np.ndarray,
    frozen: dict[str, Any],
) -> list[dict[str, Any]]:
    """Where today's frozen bands already sit on the capacity curve.

    This is the "you are here" marker: the bank's existing policy is itself a
    capacity decision (configs/thresholds.yaml sizes CRITICAL to 25 and URGENT
    to 100 accounts), and the panel should say so rather than pretend capacity
    planning starts now.
    """
    y = np.asarray(y).astype(int)
    calibrated = np.asarray(calibrated, dtype=float)
    n_pos = int(y.sum())
    rows = []
    for tier, key in (("CRITICAL_REVIEW", "critical_risk"),
                      ("URGENT_REVIEW", "urgent_risk"),
                      ("STANDARD_REVIEW", "standard_risk")):
        thr = float(frozen[key])
        mask = calibrated >= thr
        k = int(mask.sum())
        tp = int(y[mask].sum())
        rows.append({
            "tier": tier,
            "threshold_key": key,
            "calibrated_risk_threshold": thr,
            "accounts_at_or_above": k,
            "mules_caught": tp,
            "recall": tp / n_pos if n_pos else 0.0,
            "precision": tp / k if k else 0.0,
        })
    return rows
