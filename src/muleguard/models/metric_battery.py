"""The full metric battery with intervals (final-validation §24 - §27).

Why this module exists
---------------------
The repository already computes metrics in three or four places -
:mod:`muleguard.evaluation.metrics` for the bundle-freeze report,
:mod:`muleguard.models.capacity` for the analyst curve, the tournament for its
leaderboard - and each of them answers the question it was written for. None of
them answers §24 in one place, and that matters for a specific reason: §24 asks
for eleven classification metrics *at the frozen operational threshold*, twelve
analyst-budget figures, five banking-workload figures, four probability-quality
figures and five stability figures **for the same predictions**, so that a
reader can see whether a good ranking metric and a bad workload metric belong to
the same model. Assembling that from four artifacts invites the exact failure
§67 warns about: quoting whichever number reads best.

So this module computes all of them from one score vector, and it computes every
interval by resampling **accounts**, never by resampling repeats, folds or
metrics.

The resampling unit, stated once
--------------------------------
There are 64 confirmed mules in 7,264 development rows. One mule is 1.5625
percentage points of recall. Anything that moves the estimate has to move it in
those steps, and the only honest way to express that granularity is an interval.
Two questions are worth asking and they are not the same question, so both are
answered, following the convention :mod:`muleguard.models.capacity` established:

``stratified``
    Positives and negatives are resampled separately, so every replicate holds
    exactly 64 mules. This is uncertainty in the *ranking* - "if the model had
    ordered this book slightly differently". It is the narrower of the two, and
    it is degenerate wherever the statistic cannot move while the mule count is
    pinned (the leading run of true positives is the standard example).

``resample_accounts``
    The whole book is resampled with replacement, so the number of mules in a
    replicate moves as well. This answers "what might a different month look
    like". It is wider, and it is the interval to quote to a judge who asks how
    much of the headline is luck.

A third axis exists and is deliberately *not* used as an interval: the spread
across CV repeats. Three repeats give three numbers; the standard deviation of
three numbers is not a confidence interval and presenting it as one would
understate uncertainty by roughly the factor that 3 is smaller than 7,264. The
per-repeat spread is reported beside every interval as its own field, because
§24 asks for "fold AP std" and "seed AP std" and those are exactly what it is.

Bootstrapping the statistic that is actually reported
----------------------------------------------------
The headline point estimate for a repeated-CV design is the *mean over repeats*
of the metric - that is the number the tournament selected on and the number the
nested run reports. Its interval therefore has to be the interval of that mean,
not of one repeat. So a replicate draws one account sample and evaluates it
against **every** repeat's score vector, then averages. The replicate is a draw
of the reported statistic, which is the only way the interval is guaranteed to
be about the number printed next to it.

Cost control
-----------
Every ranking, budget and workload figure in §24 is a function of the labels
ordered by score. One ``argsort`` per repeat per replicate therefore answers all
of them, which is why average precision and ROC-AUC are computed here from the
sorted arrays rather than by calling scikit-learn 6,000 times. The
implementations handle tied scores the same way scikit-learn does - grouping
distinct score values before accumulating - and
``tests/unit/test_metric_battery.py`` asserts equality against
``sklearn.metrics`` on tie-heavy input rather than trusting the claim.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

SCHEMA_VERSION = "1.0"

#: Analyst budgets named by §24. 18 is included because the reference holdout
#: contains 17 mules, so "the top 18" is the smallest budget that could in
#: principle catch all of them there; it is reported on development too so the
#: two splits stay comparable.
BUDGETS: tuple[int, ...] = (10, 18, 25, 50, 100)

#: Percentage-of-book budgets, also from §24. Resolved with ``round`` to match
#: the budget grid already stored in ``artifacts/metrics/capacity_curve.json``.
PCT_BUDGETS: tuple[float, ...] = (0.01, 0.02)

#: Fractions of the mule population whose alert cost §24 asks for.
CATCH_FRACTIONS: tuple[float, ...] = (0.50, 0.75, 0.90)

ALPHA = 0.05
N_BOOT_DEFAULT = 2000
SCHEMES: tuple[str, ...] = ("stratified", "resample_accounts")

#: Confusion-matrix fields that get an interval at each frozen tier. §25 names F1
#: and MCC; precision and recall are added because a tier is reported to
#: operations as a pair of those two, and an interval on the summary without one
#: on its parts is not much use to a reviewer.
THRESHOLD_INTERVAL_FIELDS: tuple[str, ...] = ("f1", "mcc", "precision", "recall")

#: Everything that leaves this module as a recommendation carries this status.
#: Nothing here changes a frozen threshold; §27 is measured, not re-decided.
ADVISORY_STATUS = "ADVISORY_REQUIRES_HUMAN_APPROVAL"


# --------------------------------------------------------------------------
# shapes and guards
# --------------------------------------------------------------------------
def as_score_matrix(scores: np.ndarray) -> np.ndarray:
    """Coerce a score vector or per-repeat matrix to shape ``(n_repeats, n)``.

    A single evaluation (the reference holdout, say) is one repeat. Keeping one
    shape everywhere is what stops a caller from accidentally averaging over the
    wrong axis: after this call, axis 0 is always repeats and axis 1 is always
    accounts, and every function below documents which one it walks.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if s.ndim != 2:
        raise ValueError(f"scores must be 1-D or 2-D, got shape {s.shape}")
    return s


def _check(y: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    S = as_score_matrix(S)
    if S.shape[1] != len(y):
        raise ValueError(
            f"score matrix has {S.shape[1]} accounts but {len(y)} labels were "
            "given; axis 1 must be accounts")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y, S


def label_support(y: np.ndarray) -> dict[str, Any]:
    """What this label vector can and cannot support, decided before computing.

    Every metric below returns ``None`` rather than a number when its denominator
    is empty, and this block is the single place that says why. A battery that
    silently reports 0.0 for average precision on an all-negative split is worse
    than one that refuses: 0.0 looks like a measurement.
    """
    y = np.asarray(y).astype(int)
    n, n_pos = len(y), int(y.sum())
    n_neg = n - n_pos
    return {
        "n": n,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "prevalence": float(n_pos / n) if n else None,
        "recall_resolution_pct_points": (100.0 / n_pos) if n_pos else None,
        "ranking_metrics_defined": bool(n_pos > 0 and n_neg > 0),
        "why_not": (
            None if n_pos > 0 and n_neg > 0
            else "no positives: recall, average precision and ROC-AUC are undefined"
            if n_pos == 0
            else "no negatives: precision and ROC-AUC are undefined"),
        "degenerate": bool(n_pos == 0 or n_neg == 0 or n_pos == 1),
    }


# --------------------------------------------------------------------------
# ranking metrics from one sort
# --------------------------------------------------------------------------
def _ordered(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-s, kind="stable")
    return y[order], s[order]


def ap_roc_from_sorted(y_desc: np.ndarray, s_desc: np.ndarray) -> tuple[float, float]:
    """Average precision and ROC-AUC from labels already ordered by score desc.

    Ties are grouped by distinct score value before the cumulative counts are
    read, which is what makes this agree with scikit-learn exactly instead of
    approximately. It matters here: a dummy prevalence baseline produces
    thousands of identical scores, and a tie-blind implementation would report a
    flattering staircase for it.

    ROC-AUC uses the trapezoid over the same distinct-threshold points, which is
    the midrank treatment of ties.
    """
    n = len(y_desc)
    P = float(y_desc.sum())
    N = float(n - P)
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    # Boundaries of runs of equal score: the last index of each distinct value.
    distinct = np.flatnonzero(np.diff(s_desc)) if n > 1 else np.array([], dtype=int)
    idx = np.r_[distinct, n - 1]
    tp = np.cumsum(y_desc)[idx].astype(float)
    fp = (idx + 1).astype(float) - tp

    recall = tp / P
    precision = tp / (tp + fp)
    # AP as the step-function sum sklearn uses: sum_i (R_i - R_{i-1}) * P_i.
    d_recall = np.diff(np.r_[0.0, recall])
    ap = float(np.sum(d_recall * precision))

    tpr = recall
    fpr = fp / N
    roc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr])) if hasattr(
        np, "trapezoid") else float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    return ap, roc


def ranking_point_estimates(y: np.ndarray, S: np.ndarray) -> dict[str, Any]:
    """PR-AUC (primary) and ROC-AUC (secondary), per repeat and averaged.

    §24 names average precision as the primary ranking metric and ROC-AUC as
    secondary, and the ordering is not stylistic. At 0.89 % prevalence ROC-AUC
    is dominated by the vast negative mass: a model can hold ROC-AUC near 0.96
    while its precision at an affordable review budget collapses. PR-AUC moves
    when the top of the ranking changes, which is the only part of the ranking a
    bank can act on.
    """
    y, S = _check(y, S)
    sup = label_support(y)
    if not sup["ranking_metrics_defined"]:
        return {"pr_auc": None, "roc_auc": None, "defined": False,
                "why_not": sup["why_not"]}
    aps, rocs = [], []
    for s in S:
        ap, roc = ap_roc_from_sorted(*_ordered(y, s))
        aps.append(ap)
        rocs.append(roc)
    mean_rank = _mean_rank(S)
    ap_avg, roc_avg = ap_roc_from_sorted(*_ordered(y, mean_rank))
    return {
        "defined": True,
        "pr_auc": _spread(aps),
        "roc_auc": _spread(rocs),
        "pr_auc_rank_averaged": ap_avg,
        "roc_auc_rank_averaged": roc_avg,
        "aggregation_note": (
            "pr_auc.mean is the mean over CV repeats of average precision - the "
            "statistic the tournament and the nested run select on, and the one a "
            "single served fit corresponds to. pr_auc_rank_averaged pools the "
            "repeats into one ranking first; averaging independent fits denoises "
            "the ranking, so it is the larger number and it describes a "
            "three-model ensemble rather than the served single model."),
    }


def aggregation_reconciliation(
    y: np.ndarray, S: np.ndarray, calibrated: np.ndarray | None = None
) -> dict[str, Any]:
    """The four ways the same folds can be turned into one PR-AUC, side by side.

    This block exists because the repository contains two different headline
    numbers for the same champion on the same folds, and §67 forbids picking the
    nicer one without explaining the difference. There is no error involved:
    they are four distinct estimators.

    ``mean_of_per_repeat``
        Average precision computed separately in each repeat, then averaged.
        This is what the tournament selected on and what the nested run reports,
        and it is the one that corresponds to a *single* fitted model - which is
        what the served bundle contains. It is the honest headline.

    ``ap_of_mean_score`` / ``ap_of_mean_rank``
        Pool the repeats into one ranking first. Averaging several independent
        fits cancels part of each fit's noise, so the pooled ranking is genuinely
        better than any single fit - and correspondingly it describes an ensemble
        that is not what gets served.

    ``ap_of_calibrated_mean_score``
        The pooled ranking after cross-fitted calibration. Calibration is
        monotone within a fold but not across folds, so it moves the number
        slightly; this is the value the lens artifact records.

    The gap between the first and the rest is the price of honesty, and it is
    reported rather than resolved.
    """
    y, S = _check(y, S)
    if not label_support(y)["ranking_metrics_defined"]:
        return {"defined": False, "why_not": label_support(y)["why_not"]}
    per_repeat = []
    for s in S:
        ap, _ = ap_roc_from_sorted(*_ordered(y, s))
        per_repeat.append(float(ap))
    ap_score, _ = ap_roc_from_sorted(*_ordered(y, _repeat_mean(S)))
    ap_rank, _ = ap_roc_from_sorted(*_ordered(y, _mean_rank(S)))
    out: dict[str, Any] = {
        "defined": True,
        "mean_of_per_repeat": float(np.mean(per_repeat)),
        "per_repeat": [round(v, 6) for v in per_repeat],
        "std_of_per_repeat": float(np.std(per_repeat)),
        "ap_of_mean_score": float(ap_score),
        "ap_of_mean_rank": float(ap_rank),
        "ap_of_calibrated_mean_score": None,
        "headline_choice": "mean_of_per_repeat",
        "why": ("the served bundle is one fitted model, so the estimator that "
                "describes it is the per-repeat mean; the pooled variants "
                "describe a three-fit ensemble and read higher"),
    }
    if calibrated is not None:
        ap_cal, _ = ap_roc_from_sorted(*_ordered(y, np.asarray(calibrated, float)))
        out["ap_of_calibrated_mean_score"] = float(ap_cal)
    gaps = [v for k, v in out.items()
            if k.startswith("ap_of") and isinstance(v, float)]
    if gaps:
        out["max_gap_vs_headline"] = float(max(gaps) - out["mean_of_per_repeat"])
    return out


def _mean_rank(S: np.ndarray) -> np.ndarray:
    """Repeat-averaged ranking, over accounts (axis 1) for each repeat."""
    out = np.empty_like(S, dtype=float)
    n = S.shape[1]
    for i, s in enumerate(S):
        out[i] = np.argsort(np.argsort(s, kind="stable"), kind="stable") / max(n - 1, 1)
    return out.mean(axis=0)


def _spread(values: Sequence[float]) -> dict[str, Any]:
    """Point estimate plus the per-repeat spread, never labelled an interval."""
    v = [float(x) for x in values]
    lo, hi = min(v), max(v)
    return {
        "mean": float(min(max(sum(v) / len(v), lo), hi)),
        "std": float(np.std(v)),
        "min": lo,
        "max": hi,
        "per_repeat": [round(x, 6) for x in v],
        "n_repeats": len(v),
    }


# --------------------------------------------------------------------------
# analyst budgets and banking workload
# --------------------------------------------------------------------------
def resolve_budgets(n: int, extra_pct: Sequence[float] = PCT_BUDGETS) -> list[dict[str, Any]]:
    """The budget grid §24 asks for, with each entry's support recorded.

    §24 closes the budget list with "only show budgets supported by sample
    size". Two things can make a budget unsupportable and they are different:
    a budget larger than the split (arithmetically impossible) and a budget so
    small that recall is capped below any interesting value (K < the number of
    mules means recall cannot exceed K/64 however perfect the model is). The
    first is dropped; the second is reported with its ceiling stated, because
    Recall@Top10 is a legitimate question whose answer is bounded at 15.6 % and
    a reader who is not told that will read the number as a failure.
    """
    out: list[dict[str, Any]] = []
    for k in BUDGETS:
        if k <= n:
            out.append({"budget": int(k), "basis": "absolute"})
    for p in extra_pct:
        k = max(1, int(round(p * n)))
        if k <= n:
            out.append({"budget": int(k), "basis": f"top_{p * 100:g}pct"})
    seen: dict[int, dict[str, Any]] = {}
    for row in out:
        seen.setdefault(row["budget"], row)
    return [seen[k] for k in sorted(seen)]


def budget_point_estimates(
    y: np.ndarray, S: np.ndarray, budgets: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Recall, precision, alerts and review cost at each analyst budget.

    ``number_needed_to_review`` is the metric an operations manager recognises:
    how many accounts a desk opens per mule it actually finds. It is the
    reciprocal of precision, and it is reported alongside precision rather than
    instead of it because the two are read by different people.
    """
    y, S = _check(y, S)
    n_pos = int(y.sum())
    out: dict[int, dict[str, Any]] = {}
    per_repeat: dict[int, dict[str, list[float]]] = {
        int(k): {"recall": [], "precision": []} for k in budgets}
    tp_by_k: dict[int, list[int]] = {int(k): [] for k in budgets}
    for s in S:
        y_desc, _ = _ordered(y, s)
        cum = np.cumsum(y_desc)
        for k in budgets:
            k = int(k)
            tp = int(cum[k - 1])
            tp_by_k[k].append(tp)
            per_repeat[k]["recall"].append(tp / n_pos if n_pos else float("nan"))
            per_repeat[k]["precision"].append(tp / k)
    n_neg = len(y) - n_pos
    for k in budgets:
        k = int(k)
        tp_mean = float(np.mean(tp_by_k[k]))
        fp_mean = k - tp_mean
        out[k] = {
            "budget": k,
            "true_positives_mean": tp_mean,
            "true_positives_per_repeat": tp_by_k[k],
            "false_positives_mean": fp_mean,
            "recall": _spread(per_repeat[k]["recall"]) if n_pos else None,
            "precision": _spread(per_repeat[k]["precision"]),
            "recall_ceiling_at_this_budget": (min(1.0, k / n_pos) if n_pos else None),
            "number_needed_to_review_per_mule": (k / tp_mean) if tp_mean > 0 else None,
            "fp_per_1000_legitimate": (1000.0 * fp_mean / n_neg) if n_neg else None,
        }
    return out


def workload_point_estimates(y: np.ndarray, S: np.ndarray) -> dict[str, Any]:
    """The alert budget required to catch a stated share of the mules.

    This is §24's banking-workload block read the other way round from the
    budget block: not "what does 25 alerts buy" but "what does catching 90 % of
    the mules cost". Reported as a count of alerts and as a share of the book,
    because the first is a staffing number and the second is what survives a
    change of portfolio size.

    ``None`` is returned for a target the ranking never reaches - which cannot
    happen when the whole book is scored, but does happen on a truncated score
    list, and returning ``None`` is better than returning ``n``.
    """
    y, S = _check(y, S)
    n, n_pos = len(y), int(y.sum())
    if n_pos == 0:
        return {"defined": False, "why_not": "no positives"}
    rows: dict[str, Any] = {"defined": True}
    for frac in CATCH_FRACTIONS:
        need = int(np.ceil(frac * n_pos))
        per_repeat_k, per_repeat_nnr = [], []
        for s in S:
            y_desc, _ = _ordered(y, s)
            cum = np.cumsum(y_desc)
            hit = np.flatnonzero(cum >= need)
            if hit.size == 0:
                per_repeat_k.append(float("nan"))
                per_repeat_nnr.append(float("nan"))
                continue
            k = int(hit[0]) + 1
            per_repeat_k.append(float(k))
            per_repeat_nnr.append(float(k) / need)
        key = f"alerts_to_catch_{int(frac * 100)}pct"
        rows[key] = {
            "mules_required": need,
            "alerts": _spread(per_repeat_k),
            "share_of_book": _spread([v / n for v in per_repeat_k]),
            "number_needed_to_review_per_mule": _spread(per_repeat_nnr),
        }
    return rows


# --------------------------------------------------------------------------
# classification at a threshold
# --------------------------------------------------------------------------
def threshold_metrics(y: np.ndarray, probs: np.ndarray, thr: float) -> dict[str, Any]:
    """Every classification metric §24 lists, at one operating threshold.

    Accuracy and balanced accuracy are both here because §24 asks for both, and
    the pair is the clearest single illustration of why this project never
    optimises accuracy: at 0.89 % prevalence, alerting on nothing scores 0.9911
    accuracy and 0.5 balanced accuracy. Accuracy is reported so a reader can see
    that it carries no information at this prevalence, not because it is a
    quality signal.

    F2 is reported beside F1 because a missed mule costs a bank more than a
    reviewed innocent account; F2 weights recall four times as heavily as
    precision, which is closer to the real loss than F1's equal weighting.

    MCC is the one summary here that degrades gracefully under imbalance: it
    uses all four cells of the confusion matrix, so it cannot be inflated by the
    true-negative mass the way accuracy is.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(probs, dtype=float)
    if len(p) != len(y):
        raise ValueError("probs and labels must have the same length")
    alert = p >= thr
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    fn = int(np.sum(~alert & (y == 1)))
    tn = int(np.sum(~alert & (y == 0)))
    n = tp + fp + fn + tn
    pos, neg = tp + fn, fp + tn

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / pos if pos else None
    specificity = tn / neg if neg else None
    npv = tn / (tn + fn) if tn + fn else None
    f1 = _fbeta(precision, recall, 1.0)
    f2 = _fbeta(precision, recall, 2.0)
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom > 0 else None
    return {
        "threshold": float(thr),
        "alerts": int(alert.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": ((tp + tn) / n) if n else None,
        "balanced_accuracy": (((recall + specificity) / 2.0)
                             if recall is not None and specificity is not None
                             else None),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "mcc": float(mcc) if mcc is not None else None,
        "npv": npv,
        "false_positive_rate": (fp / neg) if neg else None,
        "false_negative_rate": (fn / pos) if pos else None,
        "fp_per_1000_legitimate": (1000.0 * fp / neg) if neg else None,
        "number_needed_to_review_per_mule": ((tp + fp) / tp) if tp else None,
    }


def _fbeta(precision: float | None, recall: float | None, beta: float) -> float | None:
    if precision is None or recall is None:
        return None
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


# --------------------------------------------------------------------------
# probability quality (§24, §26)
# --------------------------------------------------------------------------
def calibration_error(
    y: np.ndarray, probs: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> dict[str, Any]:
    """ECE, maximum calibration error and the reliability curve.

    Both binning strategies are computed because at this prevalence they answer
    different questions and only reporting one is misleading:

    ``uniform``
        Equal-width bins on [0, 1]. This is the convention the rest of the
        repository uses, so the number is comparable to
        ``lens_stack_oof_v2.json``. It is also nearly vacuous here - roughly
        99 % of accounts land in the first bin, so the weighted average is
        dominated by one bin whose predicted risk is already near zero.

    ``quantile``
        Equal-mass bins. Every bin carries the same number of accounts, so the
        top decile of predicted risk gets a bin of its own and a miscalibration
        confined to the accounts an analyst actually sees can no longer hide
        inside bin 0.

    Maximum calibration error is the largest per-bin gap. §24 flags it "if
    useful"; it is reported with each bin's count attached, because with 64
    positives a bin holding two accounts can produce a large gap that means
    nothing.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    n = len(y)
    if n == 0:
        return {"ece": None, "mce": None, "bins": [], "strategy": strategy}
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    elif strategy == "quantile":
        qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        idx = np.clip(np.digitize(p, np.unique(qs)), 0, n_bins - 1)
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")

    ece, mce, bins = 0.0, 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(p[m].mean())
        obs = float(y[m].mean())
        w = float(m.mean())
        gap = abs(obs - conf)
        ece += w * gap
        mce = max(mce, gap)
        bins.append({
            "bin": int(b), "count": int(m.sum()), "positives": int(y[m].sum()),
            "mean_predicted": conf, "observed_rate": obs, "gap": gap,
        })
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins,
            "strategy": strategy, "bins": bins}


def probability_quality(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Brier, log loss and both flavours of calibration error.

    Brier is reported without apology and with a warning attached: at 0.89 %
    prevalence a model that predicts the base rate for every account scores
    about 0.0087, so a Brier of 0.0031 is *not* evidence of good probabilities
    on its own. The base-rate reference is therefore computed and reported
    beside it, which turns an unfalsifiable small number into a comparison.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-15, 1 - 1e-15)
    base = float(y.mean()) if len(y) else None
    brier = float(np.mean((p - y) ** 2)) if len(y) else None
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else None
    ref_brier = (float(np.mean((np.full_like(p, base) - y) ** 2))
                 if base is not None else None)
    ref_ll = None
    if base is not None and 0.0 < base < 1.0:
        ref_ll = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    return {
        "brier": brier,
        "log_loss": logloss,
        "brier_base_rate_reference": ref_brier,
        "log_loss_base_rate_reference": ref_ll,
        "brier_skill_vs_base_rate": (
            1.0 - brier / ref_brier if brier is not None and ref_brier else None),
        "calibration_uniform_bins": calibration_error(y, p, strategy="uniform"),
        "calibration_quantile_bins": calibration_error(y, p, strategy="quantile"),
        "note": ("A base-rate predictor scores Brier ~= prevalence * (1 - "
                 "prevalence). Read brier_skill_vs_base_rate, not brier alone."),
    }


def calibration_comparison(
    y: np.ndarray,
    raw_scores: np.ndarray,
    *,
    seed: int = 42,
    n_folds: int = 5,
) -> dict[str, Any]:
    """§26: uncalibrated vs Platt vs isotonic, all cross-fitted on OOF only.

    Cross-fitting is the whole point. Fitting a calibrator on the same OOF
    predictions it is then scored on would make isotonic look excellent - it can
    interpolate 64 positives almost exactly - so each account's calibrated
    probability comes from a calibrator that never saw that account. The
    machinery is :mod:`muleguard.models.calibration`, reused rather than
    reimplemented so that the comparison here cannot drift from the selection
    the shipped bundle actually made.

    "Uncalibrated" means the raw model score read as a probability, which is what
    a caller gets if they skip calibration. It is included because §26 asks for
    the comparison, and because a boosted tree trained with
    ``scale_pos_weight`` emits scores that are wildly over-confident at this
    prevalence - the log loss shows it immediately.
    """
    from muleguard.models.calibration import crossfit_calibrated, select_calibrator

    y = np.asarray(y).astype(int)
    raw = np.asarray(raw_scores, dtype=float)
    variants: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {"uncalibrated": np.clip(raw, 0.0, 1.0)}
    if int(y.sum()) >= n_folds and int((y == 0).sum()) >= n_folds:
        for method in ("platt", "isotonic"):
            vectors[method] = crossfit_calibrated(raw, y, method, seed=seed,
                                                  n_folds=n_folds)
    for name, vec in vectors.items():
        q = probability_quality(y, vec)
        ap, roc = ap_roc_from_sorted(*_ordered(y, vec))
        variants[name] = {
            "brier": q["brier"],
            "log_loss": q["log_loss"],
            "ece_uniform": q["calibration_uniform_bins"]["ece"],
            "ece_quantile": q["calibration_quantile_bins"]["ece"],
            "mce_quantile": q["calibration_quantile_bins"]["mce"],
            "pr_auc": None if np.isnan(ap) else ap,
            "roc_auc": None if np.isnan(roc) else roc,
            "reliability_quantile_bins": q["calibration_quantile_bins"]["bins"],
        }
    selection = None
    if "platt" in vectors:
        selection = select_calibrator(raw, y, seed=seed)
    return {
        "variants": variants,
        "selection": selection,
        "fitted_on": "development out-of-fold predictions only, cross-fitted "
                     f"{n_folds}-fold; no locked-test label is involved",
        "ranking_invariance_note": (
            "Platt scaling is monotone, so it cannot change PR-AUC or ROC-AUC. "
            "Cross-fitted calibration is monotone only within a fold, so small "
            "ranking differences between the variants are an artefact of "
            "cross-fitting, not a calibration effect."),
    }


# --------------------------------------------------------------------------
# threshold candidates (§27)
# --------------------------------------------------------------------------
def threshold_candidates(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    target_precision: float = 0.50,
    target_recall: float = 0.90,
    alert_budgets: Sequence[int] = (25, 100),
    fpr_targets: Sequence[float] = (0.001, 0.005, 0.01),
) -> list[dict[str, Any]]:
    """The six threshold families §27 asks to compare, all read off dev OOF.

    §27's first sentence is the binding one: *do not optimise the threshold on
    the final evaluation fold*. Every candidate here is computed from
    development out-of-fold predictions, and every one is advisory. The frozen
    policy (``policy_version 1.0``) is compared against, never rewritten - this
    function returns rows, and the caller reports them.

    The families answer different questions and land in different places, which
    is itself the argument for the tiered policy §27 prefers over one cutoff:
    max-F1 buys purity, max-F2 buys recall, a fixed alert budget buys a
    predictable staffing cost, and a fixed FPR buys a predictable nuisance rate
    for legitimate customers.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(probs, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return []
    order = np.argsort(-p, kind="stable")
    y_desc, p_desc = y[order], p[order]
    cum_tp = np.cumsum(y_desc)
    ks = np.arange(1, len(y) + 1)
    precision = cum_tp / ks
    recall = cum_tp / n_pos
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                   out=np.zeros_like(precision), where=(4 * precision + recall) > 0)

    rows: list[dict[str, Any]] = []

    def add(rule: str, k: int, detail: str) -> None:
        k = int(min(max(k, 1), len(y)))
        thr = float(p_desc[k - 1])
        m = threshold_metrics(y, p, thr)
        rows.append({"rule": rule, "detail": detail, "status": ADVISORY_STATUS,
                     "applied": False, **m})

    add("max_f1", int(np.argmax(f1)) + 1,
        "threshold at the top-K that maximises F1 on dev OOF")
    add("max_f2", int(np.argmax(f2)) + 1,
        "threshold at the top-K that maximises F2 (recall weighted 4x)")

    ok = np.flatnonzero(precision >= target_precision)
    if ok.size:
        add(f"target_precision_{target_precision:g}", int(ok[-1]) + 1,
            "largest budget whose precision still meets the target")
    ok = np.flatnonzero(recall >= target_recall)
    if ok.size:
        add(f"target_recall_{target_recall:g}", int(ok[0]) + 1,
            "smallest budget that reaches the recall target")
    for k in alert_budgets:
        if k <= len(y):
            add(f"fixed_alert_budget_{int(k)}", int(k),
                "fixed analyst capacity per scoring batch")
    n_neg = len(y) - n_pos
    for f in fpr_targets:
        allowed = int(np.floor(f * n_neg))
        fp_desc = ks - cum_tp
        ok = np.flatnonzero(fp_desc <= allowed)
        if ok.size:
            add(f"fixed_fpr_{f:g}", int(ok[-1]) + 1,
                f"largest budget with at most {allowed} false alerts "
                f"({f:g} of {n_neg} legitimate accounts)")
    return rows


def locate_thresholds(
    y: np.ndarray, probs: np.ndarray, frozen: dict[str, Any]
) -> list[dict[str, Any]]:
    """The frozen tiers evaluated where they actually sit, read-only.

    The tier names come from §27's review ladder. ``OOD_REVIEW`` and ``MONITOR``
    are deliberately absent: they are not risk-score thresholds at all - one is
    driven by the out-of-distribution detector and the other is the residual
    bucket - so reporting a precision for them here would invent a number.
    """
    rows = []
    for tier, key in (("CRITICAL_REVIEW", "critical_risk"),
                      ("URGENT_REVIEW", "urgent_risk"),
                      ("STANDARD_REVIEW", "standard_risk")):
        if key not in frozen:
            continue
        m = threshold_metrics(y, probs, float(frozen[key]))
        rows.append({"tier": tier, "threshold_key": key,
                     "policy_version": frozen.get("policy_version"),
                     "frozen": True, "modified_by_this_run": False, **m})
    return rows


# --------------------------------------------------------------------------
# the interval engine
# --------------------------------------------------------------------------
def _draw(rng: np.random.Generator, y: np.ndarray, pos_idx: np.ndarray,
          neg_idx: np.ndarray, scheme: str) -> np.ndarray | None:
    """One resample of ACCOUNTS. Never of repeats, folds or metric values."""
    if scheme == "stratified":
        return np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
    idx = rng.integers(0, len(y), size=len(y))
    if int(y[idx].sum()) == 0 or int((y[idx] == 0).sum()) == 0:
        # A replicate with no mules (or no legitimate accounts) has no recall or
        # no precision to report. Dropping it is the only option; at 64
        # positives it is rare enough not to move the interval.
        return None
    return idx


#: A bundle maps one resample to every statistic that resample can answer.
#: Bundling rather than looping over independent closures is what lets a single
#: ``argsort`` per repeat serve sixteen metrics, and it is also what makes two
#: metrics from the same replicate comparable.
StatBundle = Callable[[np.ndarray, np.ndarray], dict[str, float | None]]


def bundle_from_statistics(
    statistics: dict[str, Callable[[np.ndarray, np.ndarray], float | None]]
) -> StatBundle:
    """Adapt a dict of single-value closures into a bundle.

    Kept because it is the readable form for a one-off statistic and for tests
    that want to pin one metric in isolation. The shipped battery uses the fused
    bundles instead, which compute the same numbers from shared sorts - and the
    unit tests assert the two agree, so the fast path cannot drift from the
    obvious one.
    """
    def bundle(y: np.ndarray, S: np.ndarray) -> dict[str, float | None]:
        return {name: fn(y, S) for name, fn in statistics.items()}
    bundle.stat_names = tuple(statistics)          # type: ignore[attr-defined]
    return bundle


def bootstrap_intervals(
    y: np.ndarray,
    S: np.ndarray,
    bundle: StatBundle | dict[str, Callable[[np.ndarray, np.ndarray], float | None]],
    *,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 42,
    alpha: float = ALPHA,
    scheme: str = "stratified",
) -> dict[str, dict[str, Any]]:
    """95 % intervals for many statistics from one set of account resamples.

    ``bundle`` maps ``(y_resampled, S_resampled)`` to a dict of scalars, where
    ``S_resampled`` still has repeats on axis 0. The bundle is therefore
    responsible for averaging over repeats, and by construction it averages over
    the *same* accounts in every repeat - which is what makes the replicate a
    draw of the reported mean-over-repeats statistic rather than of some other
    quantity that happens to be nearby.

    Sharing one set of draws across all statistics is not only cheaper, it is
    more informative: two statistics from the same replicate are comparable, so
    a caller can ask whether recall and precision moved together.

    Degeneracy is flagged, not hidden. Where an interval has zero width the
    ``degenerate`` field says so and the reason is almost always structural (the
    statistic cannot move while the mule count is pinned), which is a property
    of the ranking rather than evidence of precision.
    """
    y, S = _check(y, S)
    if isinstance(bundle, dict):
        bundle = bundle_from_statistics(bundle)
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        names = getattr(bundle, "stat_names", ()) or tuple(bundle(y, S))
        return {name: {"point": None, "ci_low": None, "ci_high": None,
                       "n_boot_effective": 0, "scheme": scheme,
                       "why_not": label_support(y)["why_not"]}
                for name in names}

    point_values = bundle(y, S)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point_values}
    for _ in range(n_boot):
        idx = _draw(rng, y, pos_idx, neg_idx, scheme)
        if idx is None:
            continue
        for name, v in bundle(y[idx], S[:, idx]).items():
            if v is not None and np.isfinite(v):
                draws.setdefault(name, []).append(float(v))

    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    out: dict[str, dict[str, Any]] = {}
    for name, point in point_values.items():
        vals = np.asarray(draws[name], dtype=float)
        if vals.size == 0:
            out[name] = {"point": point, "ci_low": None, "ci_high": None,
                         "n_boot_effective": 0, "scheme": scheme,
                         "why_not": "every replicate was undefined"}
            continue
        lo, hi = np.quantile(vals, [lo_q, hi_q])
        out[name] = {
            "point": None if point is None else float(point),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_width": float(hi - lo),
            "bootstrap_mean": float(vals.mean()),
            "bootstrap_std": float(vals.std()),
            "n_boot_effective": int(vals.size),
            "n_draws_attempted": int(n_boot),
            "scheme": scheme,
            "resample_unit": "account",
            "degenerate": bool(hi - lo == 0.0),
            "alpha": alpha,
        }
    return out


# --------------------------------------------------------------------------
# statistic closures - the only place repeats get averaged
# --------------------------------------------------------------------------
def _mean_over_repeats(fn: Callable[[np.ndarray, np.ndarray], float | None]):
    """Lift a per-repeat statistic to the mean over repeats.

    Axis discipline lives here and nowhere else: ``S[i]`` is one repeat's scores
    for the accounts in the current resample, so the loop is over axis 0 and the
    resample is over axis 1. Anything that reversed the two would produce a
    number of the right magnitude and the wrong meaning, which is why the unit
    tests pin it with an independent reimplementation.
    """
    def wrapped(y: np.ndarray, S: np.ndarray) -> float | None:
        vals = [fn(y, S[i]) for i in range(S.shape[0])]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None
    return wrapped


def _stat_ap(y: np.ndarray, s: np.ndarray) -> float | None:
    ap, _ = ap_roc_from_sorted(*_ordered(y, s))
    return None if np.isnan(ap) else ap


def _stat_roc(y: np.ndarray, s: np.ndarray) -> float | None:
    _, roc = ap_roc_from_sorted(*_ordered(y, s))
    return None if np.isnan(roc) else roc


def _stat_recall_at(k: int):
    def f(y: np.ndarray, s: np.ndarray) -> float | None:
        n_pos = int(y.sum())
        if n_pos == 0 or k > len(y):
            return None
        y_desc, _ = _ordered(y, s)
        return float(np.cumsum(y_desc)[k - 1]) / n_pos
    return f


def _stat_precision_at(k: int):
    def f(y: np.ndarray, s: np.ndarray) -> float | None:
        if k > len(y):
            return None
        y_desc, _ = _ordered(y, s)
        return float(np.cumsum(y_desc)[k - 1]) / k
    return f


def _stat_at_threshold(field: str, thr: float):
    """A confusion-matrix statistic at a fixed threshold, on the score scale given.

    Used for F1 and MCC, which §25 names explicitly. The threshold is held fixed
    across replicates on purpose: the frozen policy is a constant, so the
    interval must describe uncertainty in the *outcome* of applying it, not
    uncertainty in where a re-optimised cutoff would land.
    """
    def f(y: np.ndarray, s: np.ndarray) -> float | None:
        v = threshold_metrics(y, s, thr)[field]
        return None if v is None else float(v)
    return f


def ranking_interval_statistics(
    budgets: Sequence[int],
) -> dict[str, Callable[[np.ndarray, np.ndarray], float | None]]:
    """The threshold-free half of §25, as independent mean-over-repeat closures.

    §25's minimum list is PR-AUC, ROC-AUC, Recall@TopK, Precision@TopK, F1 and
    MCC. The first four are ranking statistics that need no threshold, so they
    are computed on the per-repeat score matrix and averaged over repeats -
    which makes their intervals intervals *of the reported point estimate*.

    This is the readable, obviously-correct implementation: one closure per
    metric, each doing its own sort. :func:`ranking_bundle` computes the same
    numbers about ten times faster by sharing sorts, and the unit tests assert
    the two agree on identical draws. Keeping both is deliberate - the slow one
    is the specification of what the fast one is supposed to mean.
    """
    stats: dict[str, Callable[[np.ndarray, np.ndarray], float | None]] = {
        "pr_auc": _mean_over_repeats(_stat_ap),
        "roc_auc": _mean_over_repeats(_stat_roc),
    }
    for k in budgets:
        k = int(k)
        stats[f"recall_at_top_{k}"] = _mean_over_repeats(_stat_recall_at(k))
        stats[f"precision_at_top_{k}"] = _mean_over_repeats(_stat_precision_at(k))
    return stats


def ranking_bundle(
    budgets: Sequence[int],
    *,
    catch_fractions: Sequence[float] = CATCH_FRACTIONS,
) -> StatBundle:
    """Every threshold-free §24/§25 figure for one resample, from one sort each.

    The whole ranking half of §24 - average precision, ROC-AUC, recall and
    precision at each analyst budget, and the alert count needed to catch a
    stated share of the mules - is a function of the labels ordered by score.
    One ``argsort`` and one ``cumsum`` per repeat therefore answers all of them,
    which is what makes a 2,000-draw bootstrap over 7,264 accounts affordable on
    a machine whose cores are already busy.

    The alert-count figures get intervals here that the point-estimate block
    reports without them; that is a free addition rather than a required one, and
    it is the interval a staffing decision actually depends on.
    """
    ks = [int(k) for k in budgets]
    fracs = [float(f) for f in catch_fractions]

    def bundle(y: np.ndarray, S: np.ndarray) -> dict[str, float | None]:
        n, n_pos = len(y), int(y.sum())
        acc: dict[str, list[float]] = {}

        def push(name: str, value: float | None) -> None:
            if value is not None and np.isfinite(value):
                acc.setdefault(name, []).append(float(value))

        for s in S:
            y_desc, s_desc = _ordered(y, s)
            ap, roc = ap_roc_from_sorted(y_desc, s_desc)
            push("pr_auc", None if np.isnan(ap) else ap)
            push("roc_auc", None if np.isnan(roc) else roc)
            cum = np.cumsum(y_desc)
            for k in ks:
                if k > n:
                    continue
                tp = float(cum[k - 1])
                push(f"recall_at_top_{k}", tp / n_pos if n_pos else None)
                push(f"precision_at_top_{k}", tp / k)
            for f in fracs:
                if n_pos == 0:
                    continue
                need = int(np.ceil(f * n_pos))
                hit = np.flatnonzero(cum >= need)
                if hit.size:
                    push(f"alerts_to_catch_{int(f * 100)}pct", float(hit[0] + 1))

        names = ["pr_auc", "roc_auc"]
        names += [f"recall_at_top_{k}" for k in ks if k <= n]
        names += [f"precision_at_top_{k}" for k in ks if k <= n]
        names += [f"alerts_to_catch_{int(f * 100)}pct" for f in fracs]
        return {name: (float(np.mean(acc[name])) if acc.get(name) else None)
                for name in names}

    bundle.stat_names = tuple(                     # type: ignore[attr-defined]
        ["pr_auc", "roc_auc"]
        + [f"recall_at_top_{k}" for k in ks]
        + [f"precision_at_top_{k}" for k in ks]
        + [f"alerts_to_catch_{int(f * 100)}pct" for f in fracs])
    return bundle


def threshold_interval_statistics(
    tier_thresholds: dict[str, float],
) -> dict[str, Callable[[np.ndarray, np.ndarray], float | None]]:
    """F1 and MCC intervals at the frozen tiers, on the calibrated scale.

    These have to be bootstrapped on the *calibrated probability vector*,
    because that is the scale ``policy_version 1.0`` is expressed on - a
    threshold of 0.09774 means nothing against a raw margin. They are therefore
    a separate call to the interval engine from the ranking statistics, on a
    one-row matrix, with the same seed so the account draws are identical and
    the two families remain comparable replicate by replicate.

    The threshold is held fixed across replicates on purpose: the frozen policy
    is a constant, so the interval describes uncertainty in the *outcome* of
    applying it, not uncertainty about where a re-optimised cutoff would land.
    """
    stats: dict[str, Callable[[np.ndarray, np.ndarray], float | None]] = {}
    for tier, thr in tier_thresholds.items():
        for field in THRESHOLD_INTERVAL_FIELDS:
            stats[f"{field}_at_{tier}"] = _mean_over_repeats(
                _stat_at_threshold(field, float(thr)))
    return stats


def threshold_bundle(tier_thresholds: dict[str, float]) -> StatBundle:
    """The same figures as :func:`threshold_interval_statistics`, fused.

    One confusion matrix per tier per repeat serves all four fields instead of
    four, which matters less than the sorting saving but keeps the two interval
    families symmetric in shape.
    """
    tiers = {str(t): float(v) for t, v in tier_thresholds.items()}

    def bundle(y: np.ndarray, S: np.ndarray) -> dict[str, float | None]:
        acc: dict[str, list[float]] = {}
        for s in S:
            for tier, thr in tiers.items():
                m = threshold_metrics(y, s, thr)
                for field in THRESHOLD_INTERVAL_FIELDS:
                    v = m[field]
                    if v is not None and np.isfinite(v):
                        acc.setdefault(f"{field}_at_{tier}", []).append(float(v))
        return {f"{field}_at_{tier}":
                (float(np.mean(acc[f"{field}_at_{tier}"]))
                 if acc.get(f"{field}_at_{tier}") else None)
                for tier in tiers for field in THRESHOLD_INTERVAL_FIELDS}

    bundle.stat_names = tuple(                     # type: ignore[attr-defined]
        f"{field}_at_{tier}" for tier in tiers for field in THRESHOLD_INTERVAL_FIELDS)
    return bundle


# --------------------------------------------------------------------------
# stability (§24)
# --------------------------------------------------------------------------
def fold_level_ap(
    y: np.ndarray, S: np.ndarray, fold_ids: np.ndarray
) -> dict[str, Any]:
    """Average precision computed inside each outer fold separately.

    §24 asks for "fold AP std", and it is a much larger number than the
    repeat-to-repeat spread for a reason worth stating: a fold holds about 13
    mules, so its AP is estimated from a fifth of the evidence and swings
    accordingly. It is reported because it is the honest scale of "how much does
    this depend on which accounts happened to be held out", and it is *not* the
    headline, because the headline is estimated on all 64.

    ``fold_ids`` has shape ``(n_repeats, n)`` - one fold label per account per
    repeat - matching the score matrix.
    """
    y, S = _check(y, S)
    fold_ids = np.asarray(fold_ids)
    if fold_ids.ndim == 1:
        fold_ids = fold_ids.reshape(1, -1)
    if fold_ids.shape != S.shape:
        raise ValueError(
            f"fold_ids shape {fold_ids.shape} does not match scores {S.shape}")
    per_fold = []
    for r in range(S.shape[0]):
        for k in np.unique(fold_ids[r]):
            m = fold_ids[r] == k
            ap, _ = ap_roc_from_sorted(*_ordered(y[m], S[r][m]))
            per_fold.append({"repeat": int(r), "fold": int(k),
                             "n": int(m.sum()), "n_positives": int(y[m].sum()),
                             "pr_auc": None if np.isnan(ap) else float(ap)})
    vals = [f["pr_auc"] for f in per_fold if f["pr_auc"] is not None]
    return {
        "n_folds": len(per_fold),
        "pr_auc_mean": float(np.mean(vals)) if vals else None,
        "pr_auc_std": float(np.std(vals)) if vals else None,
        "pr_auc_min": float(np.min(vals)) if vals else None,
        "pr_auc_max": float(np.max(vals)) if vals else None,
        "positives_per_fold_mean": float(np.mean([f["n_positives"] for f in per_fold])),
        "per_fold": per_fold,
        "note": ("Each fold holds about a fifth of the positives, so this std is "
                 "the spread of a much noisier estimator than the pooled figure "
                 "and must not be compared with it as if they measured the same "
                 "thing."),
    }


def rank_stability(S: np.ndarray, budgets: Sequence[int] = (25, 50, 100)) -> dict[str, Any]:
    """How much the ranking itself moves when the fold shuffle changes.

    Two numbers, because the first one alone is misleading at this prevalence:

    * Spearman correlation over **all** accounts. About 99 % of them are
      legitimate accounts whose scores sit in a narrow band, so this figure is
      largely a measurement of how a model orders noise and it reads low even
      when the top of the ranking is stable.
    * Jaccard overlap of the top-K sets at each analyst budget. This is the
      figure that matters operationally: it answers whether the same accounts
      would be reviewed if the folds had been shuffled differently.

    Needs at least two repeats; with one it returns ``None`` rather than 1.0,
    because a single ranking trivially agrees with itself.
    """
    S = as_score_matrix(S)
    if S.shape[0] < 2:
        return {"measurable": False,
                "why_not": "rank stability needs at least two repeats"}
    pairs = [(i, j) for i in range(S.shape[0]) for j in range(i + 1, S.shape[0])]
    spearman = []
    for i, j in pairs:
        a = np.argsort(np.argsort(S[i], kind="stable"), kind="stable").astype(float)
        b = np.argsort(np.argsort(S[j], kind="stable"), kind="stable").astype(float)
        sa, sb = a.std(), b.std()
        spearman.append(float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else float("nan"))
    top = {}
    for k in budgets:
        k = int(k)
        if k > S.shape[1]:
            continue
        js = []
        for i, j in pairs:
            A = set(np.argsort(-S[i], kind="stable")[:k].tolist())
            B = set(np.argsort(-S[j], kind="stable")[:k].tolist())
            js.append(len(A & B) / len(A | B))
        top[f"top_{k}_jaccard"] = float(np.mean(js))
    vals = [v for v in spearman if np.isfinite(v)]
    return {
        "measurable": True,
        "n_repeat_pairs": len(pairs),
        "spearman_all_accounts_mean": float(np.mean(vals)) if vals else None,
        "top_budget_jaccard": top,
        "note": ("Spearman over all accounts is dominated by the ~99 % of rows "
                 "that are legitimate and closely scored; the top-K Jaccard "
                 "figures are the operationally meaningful ones."),
    }


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build_battery(
    *,
    y: np.ndarray,
    S: np.ndarray,
    calibrated: np.ndarray | None = None,
    fold_ids: np.ndarray | None = None,
    frozen_thresholds: dict[str, Any] | None = None,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 42,
    alpha: float = ALPHA,
    split_label: str = "unspecified",
    protocol: str = "unspecified",
    allow_threshold_search: bool = True,
) -> dict[str, Any]:
    """The whole of §24 - §27 for one set of predictions, in one document.

    ``S`` carries repeats on axis 0. ``calibrated`` is the single probability
    vector the frozen policy is expressed on - for development that is the
    cross-fitted calibration of the repeat-averaged score, because that is the
    vector ``policy_version 1.0`` was read off. Passing ``None`` drops the
    threshold, probability-quality and §26/§27 blocks and says so, rather than
    quietly substituting raw scores for probabilities.

    ``protocol`` is mandatory in spirit even though it defaults: every number
    this function returns is only interpretable next to the protocol that
    produced it (nested, flat, or reference holdout), and the caller is the only
    one that knows.
    """
    y, S = _check(y, S)
    sup = label_support(y)
    budget_rows = resolve_budgets(len(y))
    budgets = [r["budget"] for r in budget_rows]
    frozen = dict(frozen_thresholds or {})
    tier_keys = {"critical": "critical_risk", "urgent": "urgent_risk",
                 "standard": "standard_risk"}
    tiers = {name: float(frozen[key]) for name, key in tier_keys.items()
             if key in frozen} if calibrated is not None else {}

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split_label,
        "protocol": protocol,
        "support": sup,
        "ranking": ranking_point_estimates(y, S),
        "aggregation_reconciliation": aggregation_reconciliation(y, S, calibrated),
        "analyst_budgets": {
            "grid": budget_rows,
            "points": {str(k): v for k, v in
                       budget_point_estimates(y, S, budgets).items()},
        },
        "banking_workload": workload_point_estimates(y, S),
    }

    if calibrated is not None:
        cal = np.asarray(calibrated, dtype=float)
        if len(cal) != len(y):
            raise ValueError("calibrated vector length does not match labels")
        doc["probability_quality"] = probability_quality(y, cal)
        doc["calibration_comparison"] = calibration_comparison(
            y, _repeat_mean(S), seed=seed)
        doc["frozen_policy"] = {
            **{k: frozen.get(k) for k in ("critical_risk", "urgent_risk",
                                          "standard_risk", "policy_version")},
            "modified_by_this_run": False,
            "tiers_not_score_based": ["OOD_REVIEW", "MONITOR"],
        }
        doc["at_frozen_thresholds"] = locate_thresholds(y, cal, frozen)
        if allow_threshold_search:
            doc["threshold_candidates"] = threshold_candidates(y, cal)
        else:
            doc["threshold_candidates"] = None
            doc["threshold_candidates_withheld"] = (
                "Searching for a best threshold requires reading the labels of "
                "this split, which is exactly the tuning §9 forbids here. The "
                "frozen thresholds are still *evaluated* on this split, because "
                "applying a constant decided elsewhere reads no label to make a "
                "choice; searching for a new one would.")
    else:
        doc["probability_quality"] = None
        doc["calibration_comparison"] = None
        doc["at_frozen_thresholds"] = None
        doc["threshold_candidates"] = None
        doc["calibration_note"] = (
            "no calibrated probability vector was supplied, so §24's "
            "probability-quality block, §26 and §27 are not reported; raw "
            "scores were NOT substituted for probabilities")

    rank_stats = ranking_bundle(budgets)
    thr_stats = threshold_bundle(tiers) if tiers else None
    doc["intervals"] = {}
    for i, scheme in enumerate(SCHEMES):
        block = bootstrap_intervals(y, S, rank_stats, n_boot=n_boot,
                                    seed=seed + i, alpha=alpha, scheme=scheme)
        for row in block.values():
            row["computed_on"] = "per_repeat_score_matrix"
        if thr_stats is not None:
            cal_block = bootstrap_intervals(
                y, as_score_matrix(np.asarray(calibrated, dtype=float)), thr_stats,
                n_boot=n_boot, seed=seed + i, alpha=alpha, scheme=scheme)
            for row in cal_block.values():
                row["computed_on"] = "calibrated_probability_vector"
            block.update(cal_block)
        doc["intervals"][scheme] = block
    doc["interval_method"] = {
        "estimator": "percentile bootstrap over accounts",
        "resample_unit": "account (row)",
        "statistic_bootstrapped": "mean over CV repeats of the metric",
        "n_boot": int(n_boot),
        "alpha": alpha,
        "confidence": 1.0 - alpha,
        "seed": int(seed),
        "basis": {
            "per_repeat_score_matrix": "ranking and budget statistics: each "
                                       "replicate is evaluated against every "
                                       "repeat and averaged, matching the "
                                       "per-repeat mean point estimate",
            "calibrated_probability_vector": "threshold statistics: one vector, "
                                             "because a frozen threshold is only "
                                             "meaningful on the scale it was "
                                             "frozen on",
        },
        "schemes": {
            "stratified": "positives and negatives resampled separately, so every "
                          "replicate keeps the observed positive count; measures "
                          "uncertainty in the ranking alone",
            "resample_accounts": "the whole book resampled with replacement, so "
                                 "the positive count moves too; wider, and the "
                                 "one to quote for out-of-sample uncertainty",
        },
        "not_used_as_an_interval": (
            "the standard deviation across CV repeats. Three numbers are not a "
            "confidence interval; the per-repeat spread is reported separately."),
    }

    stability: dict[str, Any] = {
        "per_repeat_pr_auc": doc["ranking"].get("pr_auc"),
        "rank_stability": rank_stability(S, budgets=[k for k in (25, 50, 100)
                                                    if k <= len(y)]),
    }
    if fold_ids is not None:
        stability["fold_level_pr_auc"] = fold_level_ap(y, S, fold_ids)
    doc["stability"] = stability
    return doc


def _repeat_mean(S: np.ndarray) -> np.ndarray:
    """Mean score per account over repeats - the served ranking, on axis 0."""
    return as_score_matrix(S).mean(axis=0)


