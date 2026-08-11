"""Repeated NESTED stratified cross-validation (final-validation §8).

Why this module exists
----------------------
The tournament in :mod:`muleguard.cli.tournament_v2` reports an honest *flat*
repeated-CV number, but its compact feature sets (``top_60``, ``top_120``, ...)
were chosen by :func:`muleguard.models.selection.stability_select`, which pools
selection frequencies across **every** training fold of **every** repeat. A row
therefore helps decide which columns exist before that same row is scored as a
held-out example. The optimism this introduces is usually small, but with 64
development positives it cannot be assumed small - it has to be measured.

This module measures it. The outer folds are byte-identical to the tournament's
folds, so the *only* thing that changes between the two estimates is the
protocol:

======================  ==========================  ==========================
step                    flat tournament             nested (here)
======================  ==========================  ==========================
preprocessing           fitted on outer-train       fitted on outer-train
feature selection       pooled over ALL dev folds   inner folds of outer-train
hyperparameters         one hand-set config         tuned on inner folds only
feature-set size        chosen by comparing OOF     chosen on inner folds only
outer-val labels        never used to fit           never used to fit
======================  ==========================  ==========================

Holding the folds fixed is deliberate. A difference between the two numbers is
then attributable to the protocol and to nothing else - not to a luckier split,
not to a different seed.

Cost control
------------
The selector is the expensive step (~7-9 s per inner fold over ~2,850 surviving
columns), so it runs **once per outer fold** and its result is shared by every
model family. That is not a shortcut: the selection procedure is a property of
the outer-train partition, not of the estimator, so every family must see the
same candidate sets for the comparison to be fair.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from muleguard import settings
from muleguard.features.frame import ModelFrame
from muleguard.features.preprocessing import FoldPreprocessor
from muleguard.logging import get_logger
from muleguard.models import harness
from muleguard.models.selection import _selector_importances

log = get_logger("models.nested")

#: Candidate compact-set sizes offered to the inner loop. The inner CV picks
#: one; the outer fold never votes on it.
FEATURE_SET_SIZES: tuple[int, ...] = (30, 60, 120)


@dataclass
class OuterFold:
    """Everything the inner loop is allowed to see for one outer fold."""

    repeat: int
    fold: int
    train_idx: np.ndarray          # positions within the dev array
    valid_idx: np.ndarray
    Xtr: np.ndarray                # preprocessed outer-train
    Xva: np.ndarray                # preprocessed outer-val (transform only)
    ytr: np.ndarray
    yva: np.ndarray
    kept_features: list[str]
    inner_ids: np.ndarray          # inner fold id per outer-train row
    ranked_features: list[int]     # column positions, best first, from inner folds
    selection_frequency: dict[int, float]
    seconds_selection: float = 0.0

    def top(self, n: int) -> np.ndarray:
        """The n best column positions according to the inner-fold ranking."""
        return np.sort(np.asarray(self.ranked_features[:n], dtype=int))


@dataclass
class NestedResult:
    """Outer-validation predictions for one model family."""

    name: str
    row_index: np.ndarray
    y: np.ndarray
    scores: np.ndarray             # (n_repeats, n_dev_rows)
    chosen: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    def per_repeat_ap(self) -> list[float]:
        return [float(average_precision_score(self.y, s)) for s in self.scores]

    def summary(self) -> dict[str, Any]:
        from sklearn.metrics import roc_auc_score
        aps = self.per_repeat_ap()
        rocs = [float(roc_auc_score(self.y, s)) for s in self.scores]
        sizes = [c["n_features"] for c in self.chosen]
        return {
            "model": self.name,
            "protocol": "NESTED",
            "n_repeats": int(self.scores.shape[0]),
            "pr_auc_mean": float(np.mean(aps)),
            "pr_auc_std": float(np.std(aps)),
            "pr_auc_min": float(np.min(aps)),
            "pr_auc_per_repeat": [round(a, 5) for a in aps],
            "roc_auc_mean": float(np.mean(rocs)),
            "feature_sizes_chosen": sorted(set(sizes)),
            "feature_size_mode": int(max(set(sizes), key=sizes.count)) if sizes else 0,
            "seconds": round(self.seconds, 1),
        }


def build_outer_folds(
    frame: ModelFrame,
    *,
    n_repeats: int = 3,
    n_inner: int = 4,
    selector_top_k: int = 200,
) -> list[OuterFold]:
    """Materialise every outer fold, with selection done inside outer-train.

    The returned objects contain no information derived from an outer-validation
    label. ``ranked_features`` is produced by fitting the selector separately on
    each inner-train partition of the outer-train rows and pooling the results,
    which is the nested analogue of the flat procedure.
    """
    dev = harness.dev_split(n_repeats)
    Xdev = frame.X[dev.row_index]
    ydev = frame.y[dev.row_index]
    names = frame.feature_names
    n_jobs = int(settings.load_config("train")["n_jobs"])

    folds: list[OuterFold] = []
    for rep in range(dev.n_repeats):
        ids = dev.fold_ids[rep]
        for k in np.unique(ids):
            tr = np.flatnonzero(ids != k)
            va = np.flatnonzero(ids == k)

            prep = FoldPreprocessor(mode="tree")
            Xtr = prep.fit_transform(Xdev[tr], names)
            Xva = prep.transform(Xdev[va])
            ytr, yva = ydev[tr], ydev[va]

            inner = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                    random_state=harness.fold_seed(rep, int(k)))
            inner_ids = np.empty(len(tr), dtype=np.int16)
            for j, (_, vi) in enumerate(inner.split(Xtr, ytr)):
                inner_ids[vi] = j

            t0 = time.time()
            hits: dict[int, int] = {}
            rank_sum: dict[int, float] = {}
            for j in range(n_inner):
                itr = inner_ids != j
                imp = _selector_importances(
                    Xtr[itr], ytr[itr], harness.fold_seed(rep, int(k)) + 100 + j, n_jobs)
                order = [int(c) for c in np.argsort(-imp, kind="stable")[:selector_top_k]
                         if imp[c] > 0]
                for pos, c in enumerate(order):
                    hits[c] = hits.get(c, 0) + 1
                    rank_sum[c] = rank_sum.get(c, 0.0) + pos + 1
            ranked = sorted(hits, key=lambda c: (-hits[c], rank_sum[c] / hits[c]))
            sel_secs = time.time() - t0

            folds.append(OuterFold(
                repeat=rep, fold=int(k), train_idx=tr, valid_idx=va,
                Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva,
                kept_features=list(prep.kept_features),
                inner_ids=inner_ids,
                ranked_features=ranked,
                selection_frequency={c: hits[c] / n_inner for c in ranked},
                seconds_selection=sel_secs,
            ))
            log.info("outer r%d f%d: train=%d(+%d) valid=%d(+%d) cols=%d "
                     "ranked=%d selection %.0fs",
                     rep, int(k), len(tr), int(ytr.sum()), len(va), int(yva.sum()),
                     Xtr.shape[1], len(ranked), sel_secs)
    return folds


def inner_cv_score(
    fold: OuterFold,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray],
    cols: np.ndarray,
    seed: int,
) -> float:
    """Average precision over the inner folds of one outer fold.

    This is the only objective the tuner is permitted to optimise. It never
    touches ``fold.Xva`` or ``fold.yva``.
    """
    oof = np.full(len(fold.ytr), np.nan)
    for j in np.unique(fold.inner_ids):
        itr, iva = fold.inner_ids != j, fold.inner_ids == j
        oof[iva] = fit_predict(fold.Xtr[np.ix_(itr, cols)], fold.ytr[itr],
                               fold.Xtr[np.ix_(iva, cols)], seed + int(j))
    if np.isnan(oof).any():
        raise RuntimeError("inner CV left rows unscored")
    return float(average_precision_score(fold.ytr, oof))


def run_nested(
    name: str,
    folds: Sequence[OuterFold],
    make_fit_predict: Callable[[dict[str, Any]], Callable[..., np.ndarray]],
    suggest: Callable[[Any], dict[str, Any]] | None,
    *,
    n_trials: int = 25,
    n_repeats: int = 3,
    n_dev: int,
    row_index: np.ndarray,
    y: np.ndarray,
) -> NestedResult:
    """Tune on inner folds, fit on outer-train, predict each outer-val once."""
    t0 = time.time()
    scores = np.full((n_repeats, n_dev), np.nan)
    chosen: list[dict[str, Any]] = []

    for f in folds:
        best_params: dict[str, Any]
        best_size: int
        if suggest is None:
            best_params, best_size, best_inner = {}, FEATURE_SET_SIZES[-1], float("nan")
            best_inner = inner_cv_score(f, make_fit_predict({}), f.top(best_size),
                                        harness.fold_seed(f.repeat, f.fold))
        else:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            seed = harness.fold_seed(f.repeat, f.fold)

            def objective(trial: Any) -> float:
                params = suggest(trial)
                size = trial.suggest_categorical("n_features", list(FEATURE_SET_SIZES))
                return inner_cv_score(f, make_fit_predict(params), f.top(size), seed)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = {k: v for k, v in study.best_params.items() if k != "n_features"}
            best_size = int(study.best_params["n_features"])
            best_inner = float(study.best_value)

        cols = f.top(best_size)
        fp = make_fit_predict(best_params)
        scores[f.repeat, f.valid_idx] = fp(
            f.Xtr[:, cols], f.ytr, f.Xva[:, cols], harness.fold_seed(f.repeat, f.fold))
        chosen.append({
            "repeat": f.repeat, "fold": f.fold, "n_features": best_size,
            "inner_pr_auc": round(best_inner, 5), "params": best_params,
        })
        log.info("%-22s r%d f%d  inner AP %.4f  n_feat %d",
                 name, f.repeat, f.fold, best_inner, best_size)

    if np.isnan(scores).any():
        raise RuntimeError(f"{name}: nested CV left rows unscored")
    res = NestedResult(name=name, row_index=row_index, y=y, scores=scores,
                       chosen=chosen, seconds=time.time() - t0)
    s = res.summary()
    log.info("%-22s NESTED PR-AUC %.5f +/- %.5f  (%.0fs)",
             name, s["pr_auc_mean"], s["pr_auc_std"], s["seconds"])
    return res
