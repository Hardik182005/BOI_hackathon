"""Stability, ensembling and distribution-shift experiments on NESTED folds.

Final-validation sections 15-23, 53 and 54 were originally answered under a
**flat** repeated-CV protocol (``artifacts/metrics/ensemble_v2.json``,
``stability_stress_v2.json``, ``seed_variance_v2.json``). Those numbers are not
comparable with the nested tournament, and worse, several of them were computed
with a statistic - a blend weight, a stacker, a noise flag - fitted on the same
out-of-fold vector they were then scored against. This module re-runs them on
the nested outer folds so that every arm is paired against every other arm on
byte-identical partitions, and so that **every fitted quantity is fitted inside
the outer-training partition only**.

What "fold-local" means here, concretely
---------------------------------------
For one outer fold the only inputs a fitting step may read are ``fold.Xtr``,
``fold.ytr`` and quantities derived from them (including ``fold.inner_ids`` and
``fold.ranked_features``, both of which were themselves built from outer-train
rows by :func:`muleguard.models.nested.build_outer_folds`). ``fold.yva`` is
never an input to anything: it appears exactly once per arm, in the scoring
call. That is why every combiner in this module has the signature
``(inner_oof, ytr, val) -> scores`` and none of them can even name the
validation labels.

The consequence is visible in the tests: shuffling ``fold.yva`` must leave every
arm's validation *predictions* bit-identical. An implementation that fitted a
blend weight, a calibration curve or a noise flag on the validation partition
would fail that test immediately.

Base predictions
----------------
:func:`compute_bases` produces, for one outer fold and each model family:

* ``inner_oof`` - out-of-fold predictions for the outer-**training** rows,
  obtained from the fold's own inner 4-fold split. This is the only surface a
  meta-model, a blend weight or a noise flag is allowed to learn from.
* ``val`` - one prediction per outer-validation row from a model refitted on
  the whole outer-training partition.

Everything downstream (sections 19, 20, 22, 53, and the calibration part of 54)
is a function of those two arrays plus ``ytr``.

Hyperparameters are held at each family's default configuration rather than
tuned per arm, following ``muleguard.cli.missingness_ablation``: tuning inside
an arm lets it win on a luckier hyperparameter draw instead of on the thing the
experiment is meant to isolate. ``docs/TUNING_OVERFIT_HYPOTHESIS.md`` records a
separate, unresolved question about whether the tuning helps at all at this
number of positives; nothing here depends on the answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from muleguard.logging import get_logger
from muleguard.models import harness
from muleguard.models.nested import OuterFold
from muleguard.models.selection import _selector_importances

log = get_logger("models.nested_experiments")

#: Per-fold analyst budgets. The project's dev-wide budgets are 25/50/100 rows
#: out of 7,264 (0.34 %, 0.69 %, 1.38 %); an outer-validation fold holds 1,453
#: rows, so the equivalent budgets are 5/10/20. Reporting recall on a per-fold
#: budget keeps the 15 numbers paired and comparable; reporting it on a pooled
#: OOF vector would require probabilities from 5 different models to share a
#: scale, which for a rank ensemble they deliberately do not.
FOLD_BUDGETS: tuple[int, ...] = (5, 10, 20)

Combiner = Callable[[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]],
                    tuple[np.ndarray, dict[str, Any]]]


# ==========================================================================
# small numerics
# ==========================================================================
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def _pct_rank(s: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1]; ties broken by position, which is stable."""
    order = np.argsort(np.argsort(np.asarray(s), kind="stable"), kind="stable")
    return order / max(len(s) - 1, 1)


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """Share of positives inside the k highest scores. Ties resolved by order."""
    pos = float(np.asarray(y).sum())
    if pos == 0:
        return float("nan")
    idx = np.argsort(-np.asarray(s), kind="stable")[:k]
    return float(np.asarray(y)[idx].sum() / pos)


def fold_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    out = {"ap": float(average_precision_score(y, s))}
    if 0 < y.sum() < len(y):
        out["roc"] = float(roc_auc_score(y, s))
    for k in FOLD_BUDGETS:
        out[f"recall_at_{k}"] = recall_at_k(y, s, k)
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width expected calibration error. Empty bins contribute nothing."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


# ==========================================================================
# fold-local column pools (sections 15, 16, 17)
# ==========================================================================
def pool_columns(fold: OuterFold, pool: Sequence[str] | None, n_feat: int) -> np.ndarray:
    """The fold's ``n_feat`` best columns, restricted to a named pool.

    ``fold.ranked_features`` is a ranking produced inside the outer-training
    partition, so filtering it by a fixed, label-independent set of column names
    stays fold-local. If the ranking yields fewer than ``n_feat`` members of the
    pool - which happens whenever the pool is small, because the ranking only
    keeps columns the selector gave non-zero gain - the remainder is filled from
    the pool in the fold's own column order, deterministically.

    Known bias, stated rather than hidden: the ranking was fitted with the
    *whole* candidate pool competing, so a column that a restricted arm relies
    on may have been out-ranked by a correlated column the arm excludes. This
    biases restricted arms **downwards**. Re-running the selector inside each
    arm removes the bias at the cost of ~45 s per outer fold per arm; where that
    was affordable (small pools) the caller does it, and where it was not, the
    result is labelled with this caveat.
    """
    if pool is None:
        return fold.top(n_feat)
    pos = {n: i for i, n in enumerate(fold.kept_features)}
    allowed = {pos[n] for n in pool if n in pos}
    if not allowed:
        raise ValueError("pool has no columns surviving this fold's preprocessing")
    chosen = [c for c in fold.ranked_features if c in allowed][:n_feat]
    if len(chosen) < n_feat:
        seen = set(chosen)
        for c in sorted(allowed):
            if len(chosen) >= n_feat:
                break
            if c not in seen:
                chosen.append(c)
    return np.sort(np.asarray(chosen, dtype=int))


# ==========================================================================
# base predictions (sections 19, 20, 22, 53, 54)
# ==========================================================================
@dataclass
class FoldBases:
    """Inner-OOF and outer-validation predictions for one outer fold."""

    repeat: int
    fold: int
    cols: np.ndarray
    ytr: np.ndarray
    yva: np.ndarray
    inner_oof: dict[str, np.ndarray] = field(default_factory=dict)
    val: dict[str, np.ndarray] = field(default_factory=dict)
    inner_ap: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def families(self) -> list[str]:
        return sorted(self.val)


def compute_bases(fold: OuterFold, families: dict[str, Callable], *,
                  n_feat: int, pool: Sequence[str] | None = None) -> FoldBases:
    """Inner-OOF + outer-validation predictions for every family, one fold.

    The inner split is the one already stored on the fold, so the meta-level
    sees exactly the partition the tuner saw and no new randomness enters.
    """
    t0 = time.time()
    cols = pool_columns(fold, pool, n_feat)
    seed = harness.fold_seed(fold.repeat, fold.fold)
    out = FoldBases(repeat=fold.repeat, fold=fold.fold, cols=cols,
                    ytr=fold.ytr, yva=fold.yva)
    Xtr_c = fold.Xtr[:, cols]
    Xva_c = fold.Xva[:, cols]
    for name, fp in families.items():
        oof = np.full(len(fold.ytr), np.nan)
        for j in np.unique(fold.inner_ids):
            itr, iva = fold.inner_ids != j, fold.inner_ids == j
            oof[iva] = fp(Xtr_c[itr], fold.ytr[itr], Xtr_c[iva], seed + int(j))
        if np.isnan(oof).any():
            raise RuntimeError(f"{name}: inner CV left outer-train rows unscored")
        out.inner_oof[name] = oof
        out.inner_ap[name] = float(average_precision_score(fold.ytr, oof))
        out.val[name] = np.asarray(fp(Xtr_c, fold.ytr, Xva_c, seed), dtype=float)
    out.seconds = time.time() - t0
    log.info("bases r%d f%d: %s (%.0fs)", fold.repeat, fold.fold,
             "  ".join(f"{k}={v:.4f}" for k, v in sorted(out.inner_ap.items())),
             out.seconds)
    return out


# ==========================================================================
# combiners - sections 19 (E1/E2/E3) and 20 (rank / Borda)
# ==========================================================================
def _ordered(d: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = sorted(d)
    return names, np.column_stack([d[n] for n in names])


def best_single_inner(inner_oof, ytr, val):
    """The single family with the best INNER average precision.

    This is the honest reference for an ensemble: picking the best single model
    by its outer-validation score would hand the baseline a winner's-curse
    advantage and make every ensemble look worse than it is.
    """
    names = sorted(inner_oof)
    aps = {n: float(average_precision_score(ytr, inner_oof[n])) for n in names}
    pick = max(names, key=lambda n: (aps[n], n))
    return val[pick], {"picked": pick, "inner_ap": round(aps[pick], 5)}


def prob_mean(inner_oof, ytr, val):
    names, M = _ordered(val)
    return M.mean(axis=1), {"weights": {n: round(1 / len(names), 4) for n in names}}


def logit_mean(inner_oof, ytr, val):
    names, M = _ordered(val)
    return _logit(M).mean(axis=1), {"weights": {n: round(1 / len(names), 4) for n in names}}


def perf_weighted(inner_oof, ytr, val):
    """E1 - weighted probability average, weights from inner-fold skill.

    ``w_m`` is proportional to the family's inner average precision in excess of
    the prevalence baseline, which is the part of its score that is not free.
    Closed form, no search, so it cannot overfit the inner folds the way an
    optimised weight vector can - that failure mode is E3's to demonstrate.
    """
    names = sorted(val)
    prev = float(np.mean(ytr))
    lift = np.array([max(float(average_precision_score(ytr, inner_oof[n])) - prev, 0.0)
                     for n in names])
    w = np.full(len(names), 1.0 / len(names)) if lift.sum() <= 0 else lift / lift.sum()
    M = np.column_stack([val[n] for n in names])
    return M @ w, {"weights": {n: round(float(x), 4) for n, x in zip(names, w)}}


def logistic_stacker(inner_oof, ytr, val):
    """E2 - regularized logistic stacker on inner-OOF logits.

    The meta-model never sees an in-sample base prediction: its training matrix
    is the inner **out-of-fold** probability of each family for each
    outer-training row.
    """
    from sklearn.linear_model import LogisticRegression

    names, Mtr = _ordered(inner_oof)
    Mva = np.column_stack([val[n] for n in names])
    clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                             max_iter=2000, random_state=0)
    clf.fit(_logit(Mtr), ytr)
    coef = {n: round(float(c), 4) for n, c in zip(names, clf.coef_[0])}
    return clf.predict_proba(_logit(Mva))[:, 1], {"coef": coef,
                                                  "intercept": round(float(clf.intercept_[0]), 4)}


def _simplex_grid(m: int, step: int | None = None) -> np.ndarray:
    """All weight vectors on the ``step``-resolution simplex. Deterministic.

    ``step`` defaults to the smallest multiple of ``m`` that is at least 10, so
    that the **uniform** vector is always a grid point. A fixed 0.1 resolution
    would have excluded it for four members - 0.25 is not a multiple of 0.1 -
    which quietly meant the equal-weight blend was not in E3's search space and
    the tie-break towards uniform had no exact target to aim at. That was a real
    defect, found by a unit test, and this is the fix.
    """
    if m == 1:
        return np.ones((1, 1))
    if step is None:
        step = m * -(-10 // m)          # smallest multiple of m that is >= 10
    rows = []

    def rec(prefix: list[int], left: int, slots: int):
        if slots == 1:
            rows.append(prefix + [left])
            return
        for v in range(left + 1):
            rec(prefix + [v], left - v, slots - 1)

    rec([], step, m)
    return np.asarray(rows, dtype=float) / step


def constrained_blend(inner_oof, ytr, val):
    """E3 - non-negative weights summing to 1, chosen on the inner folds.

    Average precision is piecewise-constant in the weights, so a gradient
    optimiser would stall on a plateau and its answer would depend on its
    starting point. A fixed-resolution grid is searched exhaustively instead:
    455 points for four families at a 1/12 resolution, fully deterministic, and
    ties are broken towards the uniform vector so that a plateau returns the
    least committed member of it rather than an arbitrary corner.
    """
    names, Mtr = _ordered(inner_oof)
    grid = _simplex_grid(len(names))
    scores = np.array([average_precision_score(ytr, Mtr @ w) for w in grid])
    best = scores.max()
    cand = np.flatnonzero(scores >= best - 1e-12)
    uni = np.full(len(names), 1.0 / len(names))
    pick = cand[int(np.argmin(((grid[cand] - uni) ** 2).sum(axis=1)))]
    w = grid[pick]
    Mva = np.column_stack([val[n] for n in names])
    return Mva @ w, {"weights": {n: round(float(x), 4) for n, x in zip(names, w)},
                     "inner_ap": round(float(best), 5),
                     "n_grid_points": int(len(grid))}


def rank_mean(inner_oof, ytr, val):
    """Section 20 - mean percentile rank across families, within the fold."""
    names, _ = _ordered(val)
    R = np.column_stack([_pct_rank(val[n]) for n in names])
    return R.mean(axis=1), {"members": names}


def rank_median(inner_oof, ytr, val):
    names, _ = _ordered(val)
    R = np.column_stack([_pct_rank(val[n]) for n in names])
    return np.median(R, axis=1), {"members": names}


def borda(inner_oof, ytr, val):
    """Section 20 - Borda count.

    With complete rankings and no missing members, a Borda count is the sum of
    per-model ranks, which is the mean rank times a constant, and average
    precision is invariant to that constant. This function is kept because the
    section asks for it explicitly, and its near-identity with ``rank_mean`` is
    reported as a finding rather than disguised by an arbitrary tweak.
    """
    names, _ = _ordered(val)
    n = len(next(iter(val.values())))
    B = np.zeros(n)
    for name in names:
        B += np.argsort(np.argsort(val[name], kind="stable"), kind="stable")
    return B, {"members": names,
               "note": "sum of ranks; monotone-equivalent to rank_mean"}


COMBINERS: dict[str, Combiner] = {
    "best_single_inner": best_single_inner,
    "prob_mean": prob_mean,
    "logit_mean": logit_mean,
    "E1_perf_weighted": perf_weighted,
    "E2_logistic_stacker": logistic_stacker,
    "E3_constrained_blend": constrained_blend,
    "rank_mean": rank_mean,
    "rank_median": rank_median,
    "borda": borda,
}


# ==========================================================================
# section 18 - model-seed bagging
# ==========================================================================
def seed_bag(fold: OuterFold, fit_predict: Callable, cols: np.ndarray,
             seeds: Sequence[int]) -> dict[str, Any]:
    """Fit one family under several seeds; return single-seed and bagged scores.

    ``probability_mean/std`` and ``rank_mean/std`` are the quantities section 18
    asks for. The single-seed arm uses ``seeds[0]``, which is the fold's
    canonical seed, so the paired comparison is "the model we would have shipped"
    against "the same model averaged".
    """
    P = np.column_stack([
        np.asarray(fit_predict(fold.Xtr[:, cols], fold.ytr, fold.Xva[:, cols], s),
                   dtype=float) for s in seeds])
    R = np.column_stack([_pct_rank(P[:, i]) for i in range(P.shape[1])])
    return {
        "single": P[:, 0],
        "prob_mean": P.mean(axis=1),
        "rank_mean": R.mean(axis=1),
        "probability_std": float(np.mean(P.std(axis=1))),
        "rank_std": float(np.mean(R.std(axis=1))),
        "per_seed_ap": [float(average_precision_score(fold.yva, P[:, i]))
                        for i in range(P.shape[1])],
    }


# ==========================================================================
# section 21 - positive-removal stability
# ==========================================================================
def positive_removal(fold: OuterFold, fit_predict: Callable, cols: np.ndarray, *,
                     rounds: int, fraction: float, n_jobs: int,
                     reference: np.ndarray,
                     reference_importance: np.ndarray) -> dict[str, Any]:
    """Drop a share of the fold's TRAINING positives, refit, rescore.

    The validation partition is untouched in every round, so the spread is
    attributable to label scarcity in training and to nothing else. Feature-rank
    stability is measured with the project's own selector
    (:func:`_selector_importances`) rather than with the classifier's internal
    importances, because the selector is the instrument that actually decides
    which columns a fold uses.
    """
    from scipy import stats

    pos = np.flatnonzero(fold.ytr == 1)
    n_drop = max(1, int(round(fraction * len(pos))))
    seed0 = harness.fold_seed(fold.repeat, fold.fold)
    aps, recalls, pred_rho, feat_rho = [], {k: [] for k in FOLD_BUDGETS}, [], []
    for r in range(rounds):
        rng = np.random.default_rng(seed0 * 1000 + r)
        drop = rng.choice(pos, size=n_drop, replace=False)
        keep = np.ones(len(fold.ytr), dtype=bool)
        keep[drop] = False
        Xk, yk = fold.Xtr[np.ix_(keep, cols)], fold.ytr[keep]
        s = np.asarray(fit_predict(Xk, yk, fold.Xva[:, cols], seed0), dtype=float)
        aps.append(float(average_precision_score(fold.yva, s)))
        for k in FOLD_BUDGETS:
            recalls[k].append(recall_at_k(fold.yva, s, k))
        pred_rho.append(float(stats.spearmanr(s, reference).statistic))
        imp = _selector_importances(Xk, yk, seed0, n_jobs)
        feat_rho.append(float(stats.spearmanr(imp, reference_importance).statistic))
    return {
        "ap": aps, "recall": {k: recalls[k] for k in FOLD_BUDGETS},
        "prediction_rank_correlation": pred_rho,
        "feature_rank_correlation": feat_rho,
        "n_train_positives": int(len(pos)), "n_dropped": int(n_drop),
    }


# ==========================================================================
# section 22 - label-noise audit
# ==========================================================================
def noise_flags(scores_by_family: dict[str, np.ndarray], y: np.ndarray, *,
                percentile: float = 0.5) -> tuple[np.ndarray, dict[str, Any]]:
    """Positives that EVERY family ranks below ``percentile`` of all rows.

    Consensus is required deliberately. A positive that one family misses is a
    model failure; a positive that every family independently ranks below the
    median of a 99 %-negative population is at least worth looking at. The
    function returns flags only - nothing is relabelled and nothing is deleted.

    ``scores_by_family`` must come from predictions the flagged rows did not
    train on. Feeding it a globally-pooled out-of-fold vector while the flags are
    then used inside a training fold is a real leak, and the CLI runs that arm
    explicitly, labelled as rejected evidence, to measure how large it is.
    """
    names = sorted(scores_by_family)
    ranks = {n: _pct_rank(scores_by_family[n]) for n in names}
    low = np.ones(len(y), dtype=bool)
    for n in names:
        low &= ranks[n] < percentile
    flags = low & (np.asarray(y) == 1)
    mean_rank = np.mean([ranks[n] for n in names], axis=0)
    return flags, {
        "n_flagged": int(flags.sum()),
        "n_positives": int(np.asarray(y).sum()),
        "flagged_mean_rank": (round(float(mean_rank[flags].mean()), 4)
                              if flags.any() else None),
        "positive_mean_rank": round(float(mean_rank[np.asarray(y) == 1].mean()), 4),
        "consensus_families": names,
    }


# ==========================================================================
# section 23 - within-dataset adversarial validation and feature shift
# ==========================================================================
def adversarial_auc(fold: OuterFold, cols: np.ndarray, fit_predict: Callable,
                    n_splits: int = 4) -> float:
    """Can a classifier tell this fold's training rows from its validation rows?

    Under a correct random stratified split the answer must be no, and an AUC
    materially above 0.5 would mean the folds are not exchangeable - which would
    invalidate every paired comparison built on them. This is a check on the
    harness, not on the model.
    """
    from sklearn.model_selection import StratifiedKFold

    X = np.vstack([fold.Xtr[:, cols], fold.Xva[:, cols]])
    z = np.concatenate([np.zeros(len(fold.ytr)), np.ones(len(fold.yva))])
    seed = harness.fold_seed(fold.repeat, fold.fold)
    oof = np.full(len(z), np.nan)
    for tr, va in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed).split(X, z):
        oof[va] = fit_predict(X[tr], z[tr], X[va], seed)
    return float(roc_auc_score(z, oof))


def feature_shift(fold: OuterFold, cols: np.ndarray) -> list[dict[str, Any]]:
    """Per-column train-vs-validation shift inside one outer fold."""
    from scipy import stats

    rows = []
    for c in cols:
        a, b = fold.Xtr[:, c], fold.Xva[:, c]
        fa, fb = a[~np.isnan(a)], b[~np.isnan(b)]
        ks = (float(stats.ks_2samp(fa, fb).statistic)
              if len(fa) > 1 and len(fb) > 1 else float("nan"))
        uniq = np.unique(fa)
        unseen = (float(np.mean(~np.isin(fb, uniq))) if len(uniq) <= 20 and len(fb)
                  else 0.0)
        rows.append({
            "feature": fold.kept_features[c],
            "ks": ks,
            "missing_rate_shift": float(abs(np.isnan(a).mean() - np.isnan(b).mean())),
            "unseen_category_rate": unseen,
            "low_cardinality": bool(len(uniq) <= 20),
        })
    return rows


def ood_rate(fold: OuterFold, cols: np.ndarray) -> float:
    """Share of validation rows with a finite value outside the train range.

    A blunt extrapolation proxy, chosen because it needs no fitted density model
    and cannot itself overfit. It is not a substitute for a real OOD detector
    and is reported as what it is.
    """
    lo = np.nanmin(fold.Xtr[:, cols], axis=0)
    hi = np.nanmax(fold.Xtr[:, cols], axis=0)
    V = fold.Xva[:, cols]
    out = (V < lo) | (V > hi)
    out &= ~np.isnan(V)
    return float(np.mean(out.any(axis=1)))


# ==========================================================================
# calibration (section 8 step 5, feeding section 54)
# ==========================================================================
def platt_calibrate(inner_oof: np.ndarray, ytr: np.ndarray,
                    val: np.ndarray) -> np.ndarray:
    """Platt scaling fitted on inner-OOF predictions of the TRAINING rows only."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(_logit(inner_oof).reshape(-1, 1), ytr)
    return clf.predict_proba(_logit(val).reshape(-1, 1))[:, 1]
