"""Stability feature selection over the firewall-admitted pool.

With 81 positives and ~3,900 candidate columns, any single importance ranking
is mostly noise: refit on a different fold and a different set of columns rises
to the top. The defence is to rank inside every training fold separately and
keep only the columns that keep reappearing.

`stability_select()` therefore:

  1. fits a gradient-boosted ranker on each training fold of each repeat
     (never on the held-out fold, never on the locked test),
  2. records the top-K columns of that fold,
  3. reports, for every column, the fraction of folds in which it appeared.

Selection frequency is the number a judge can interrogate: "this feature was
chosen in 24 of 25 independent training folds" is a far stronger statement
than "this feature had the highest gain once".

Nothing here looks at the target of a held-out fold, so the resulting compact
sets can be re-evaluated by the OOF harness without optimistic bias beyond the
usual (small, documented) selection-inside-CV caveat, which the accepted model
also reports as a nested-selection check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.features.frame import ModelFrame
from muleguard.features.preprocessing import FoldPreprocessor
from muleguard.logging import get_logger
from muleguard.models import harness

log = get_logger("models.selection")


@dataclass
class StabilityResult:
    """Per-feature selection frequency across every training fold."""

    feature: list[str]
    frequency: list[float]
    mean_rank: list[float]
    mean_gain: list[float]
    gain_cv: list[float]
    n_folds: int
    top_k: int
    pool: str

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame({
            "feature": self.feature,
            "selection_frequency": self.frequency,
            "mean_rank_when_selected": self.mean_rank,
            "mean_gain_share": self.mean_gain,
            "gain_cv_across_folds": self.gain_cv,
        }).sort(["selection_frequency", "mean_gain_share"], descending=[True, True])

    def top(self, n: int) -> list[str]:
        return self.to_frame().head(n)["feature"].to_list()

    def at_least(self, freq: float) -> list[str]:
        f = self.to_frame().filter(pl.col("selection_frequency") >= freq)
        return f["feature"].to_list()

    def compact_sets(self, sizes: Sequence[int], freqs: Sequence[float]) -> dict[str, list[str]]:
        sets: dict[str, list[str]] = {f"top_{n}": self.top(n) for n in sizes}
        for f in freqs:
            key = f"freq_ge_{f:.2f}".replace(".", "_")
            sel = self.at_least(f)
            if sel:
                sets[key] = sel
        return sets


def _selector_importances(Xtr: np.ndarray, ytr: np.ndarray, seed: int,
                          n_jobs: int) -> np.ndarray:
    """Gain importances from a small, heavily regularised LightGBM.

    Deliberately small: the selector must not memorise 65 positives. Depth and
    leaf count are capped and a fixed 400 trees are grown without early
    stopping, so every fold contributes a comparable ranking.
    """
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        colsample_bytree=0.3,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        scale_pos_weight=(ytr == 0).sum() / max(ytr.sum(), 1),
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
        importance_type="gain",
    )
    clf.fit(Xtr, ytr)
    return np.asarray(clf.feature_importances_, dtype=float)


def stability_select(
    frame: ModelFrame,
    *,
    top_k: int = 60,
    n_repeats: int = 2,
    pool_label: str | None = None,
) -> StabilityResult:
    """Rank features inside every training fold; report how often each survives."""
    dev = harness.dev_split(n_repeats)
    Xdev, ydev = frame.X[dev.row_index], frame.y[dev.row_index]
    names = frame.feature_names
    n_jobs = int(settings.load_config("train")["n_jobs"])

    hits: dict[str, int] = {}
    ranks: dict[str, list[float]] = {}
    gains: dict[str, list[float]] = {}
    n_folds = 0

    for rep in range(dev.n_repeats):
        ids = dev.fold_ids[rep]
        for k in np.unique(ids):
            tr = ids != k
            prep = FoldPreprocessor(mode="tree")
            Xtr = prep.fit_transform(Xdev[tr], names)
            kept = prep.kept_features
            imp = _selector_importances(Xtr, ydev[tr], harness.fold_seed(rep, int(k)), n_jobs)
            total = imp.sum()
            share = imp / total if total > 0 else imp
            order = np.argsort(-imp, kind="stable")[:top_k]
            n_folds += 1
            for rank, j in enumerate(order):
                if imp[j] <= 0:
                    break
                f = kept[j]
                hits[f] = hits.get(f, 0) + 1
                ranks.setdefault(f, []).append(float(rank + 1))
                gains.setdefault(f, []).append(float(share[j]))

    feats = sorted(hits, key=lambda f: (-hits[f], np.mean(ranks[f])))

    def _cv(f: str) -> float:
        """Dispersion of a feature's gain share across the folds that chose it.

        Padded with zeros for the folds where the feature did not appear at
        all: a column that dominates one fold and is absent from the next is
        exactly what this number must expose.
        """
        g = gains[f] + [0.0] * (n_folds - len(gains[f]))
        m = float(np.mean(g))
        return float(np.std(g) / m) if m > 0 else 0.0

    result = StabilityResult(
        feature=feats,
        frequency=[hits[f] / n_folds for f in feats],
        mean_rank=[float(np.mean(ranks[f])) for f in feats],
        mean_gain=[float(np.mean(gains[f])) for f in feats],
        gain_cv=[_cv(f) for f in feats],
        n_folds=n_folds,
        top_k=top_k,
        pool=pool_label or frame.view,
    )
    log.info("stability selection on %s: %d folds, %d columns ever selected, "
             "%d with frequency >= 0.5",
             result.pool, n_folds, len(feats), len(result.at_least(0.5)))
    return result


def nested_selection_check(
    frame: ModelFrame,
    scorer,
    *,
    top_k: int = 60,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """Re-run selection INSIDE each fold to measure the optimism of reusing
    one global compact set.

    The accepted model uses a compact set chosen from training folds pooled
    across repeats, which technically lets information from a row's own fold
    influence which columns exist. This routine repeats the whole procedure
    strictly inside the fold - select on the training part only, then train and
    score the held-out part - and reports the gap. A small gap means the
    compact set is a property of the data, not of the split.
    """
    from sklearn.metrics import average_precision_score

    dev = harness.dev_split(n_repeats)
    Xdev, ydev = frame.X[dev.row_index], frame.y[dev.row_index]
    names = frame.feature_names
    n_jobs = int(settings.load_config("train")["n_jobs"])

    oof = np.full(len(dev.row_index), np.nan)
    per_fold_sets: list[list[str]] = []
    for rep in range(dev.n_repeats):
        ids = dev.fold_ids[rep]
        for k in np.unique(ids):
            tr, va = ids != k, ids == k
            prep = FoldPreprocessor(mode="tree")
            Xtr = prep.fit_transform(Xdev[tr], names)
            Xva = prep.transform(Xdev[va])
            seed = harness.fold_seed(rep, int(k))
            imp = _selector_importances(Xtr, ydev[tr], seed, n_jobs)
            order = np.argsort(-imp, kind="stable")[:top_k]
            keep = np.sort(order)
            per_fold_sets.append([prep.kept_features[j] for j in keep])
            oof[va] = scorer(Xtr[:, keep], ydev[tr], Xva[:, keep], seed)
    ap = float(average_precision_score(ydev, oof))
    union = sorted(set().union(*per_fold_sets))
    inter = sorted(set(per_fold_sets[0]).intersection(*per_fold_sets[1:]))
    return {
        "nested_pr_auc": ap,
        "n_folds": len(per_fold_sets),
        "top_k": top_k,
        "union_size": len(union),
        "always_selected": inter,
        "jaccard_mean": float(np.mean([
            len(set(a) & set(b)) / len(set(a) | set(b))
            for i, a in enumerate(per_fold_sets) for b in per_fold_sets[i + 1:]
        ])) if len(per_fold_sets) > 1 else 1.0,
    }
