"""Section 18: offline validation of the Cohort Radar, outer-fold-safe.

The radar is a retrieval layer, not a classifier, so it cannot be scored with
PR-AUC. The question it has to answer is narrower and more useful: **when an
analyst opens the cohort panel on a confirmed mule, how often does a confirmed
mule appear in it?** That is Hit@k, and it is what this module measures.

The protocol is the one section 18 sets out, and every step of it exists to
stop a number that flatters the feature:

1. the similarity statistics are refit on outer-training rows only, per fold -
   a transform fitted on everything would have seen the validation rows'
   medians, and the retrieval would be quietly optimistic;
2. the reference index holds outer-training rows only, so a query can never
   retrieve itself or anything fitted on it;
3. queries come from outer-validation;
4. **no label touches retrieval** - the neighbours are chosen, ranked and
   returned before any label is read;
5. labels enter afterwards, once, to count what was retrieved.

The headline is the lift over base prevalence. Hit@10 of 60% sounds impressive
until you notice the portfolio is 50% positive; reported as a multiple of what
random neighbours would give, the number means something.

**This is a retrieval-quality diagnostic. It is not the mule classifier's
accuracy, and cohort similarity does not establish common ownership.**
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.usp import cohort_radar as cr
from muleguard.utils import git_info

log = get_logger("usp.cohort_eval")

OUT_PATH = settings.METRICS_DIR / "cohort_radar_retrieval.json"

#: How many outer folds to evaluate. Every fold of repeat 0 by default: three
#: repeats would triple the runtime to re-measure the same quantity, and the
#: fold-to-fold spread within one repeat already shows the stability.
DEFAULT_REPEAT = 0

#: Neighbour counts reported. 5 and 10 are the section-18 requirement; 25 is
#: the API's own ceiling, included so the curve's shape is visible rather than
#: two points that could be joined any way a reader likes.
K_VALUES = (5, 10, 25)

DISCLAIMER = (
    "Retrieval-quality diagnostic for a post-model similarity layer. These "
    "figures describe how often a behaviourally similar reference account "
    "shares the query's label. They are not classifier accuracy, they are not "
    "used to score anything, and they do not establish common ownership, "
    "common control or a shared network."
)


#: The model's own fold map. ``settings.SPLITS_DIR`` points at ``data/splits``;
#: the nested-CV writer puts its assignments under ``artifacts/splits``, and the
#: retrieval evaluation must use *that* file - reusing the classifier's folds is
#: the whole point of calling the protocol outer-fold-safe.
ASSIGNMENTS_PATH = settings.ARTIFACTS_DIR / "splits" / "nested_cv_assignments.parquet"


def _fold_assignments(repeat: int) -> pl.DataFrame:
    path = ASSIGNMENTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run the nested CV split before evaluating "
            "cohort retrieval - the folds must be the model's own folds")
    return pl.read_parquet(path).filter(pl.col("repeat") == repeat)


def _labels(frame: pl.DataFrame) -> np.ndarray:
    """The target column, read once, and only for scoring the retrieval.

    Deliberately a separate function with a name that says what it is. Every
    other function in this module must be callable without it.
    """
    return frame[settings.TARGET_COLUMN].to_numpy().astype(int)


def _retrieve(transform: cr.CohortTransform, ref_num: np.ndarray,
              ref_cat: np.ndarray, q_num: np.ndarray, q_cat: np.ndarray,
              k: int) -> np.ndarray:
    """Top-k reference positions for one query, by similarity alone.

    No labels are in scope here, by construction: this function is not given
    any. Ties break on reference position so the ordering is total.
    """
    sims = transform.similarity(q_num, q_cat, ref_num, ref_cat)
    order = np.lexsort((np.arange(sims.size), -sims))
    return order[:k]


def _evaluate_fold(*, frame: pl.DataFrame, train_rows: np.ndarray,
                   valid_rows: np.ndarray, k_values=K_VALUES) -> dict[str, Any]:
    """One outer fold, start to finish, with labels read only at the end."""
    transform = cr.fit(frame=frame, rows=train_rows)
    ref = frame[train_rows.tolist()].select(transform.features)
    ref_num, ref_cat = transform.encode(ref)

    qry = frame[valid_rows.tolist()].select(transform.features)
    q_num, q_cat = transform.encode(qry)

    kmax = max(k_values)
    retrieved = np.vstack([
        _retrieve(transform, ref_num, ref_cat, q_num[i], q_cat[i], kmax)
        for i in range(q_num.shape[0])
    ])

    # --- labels enter here, and not one line earlier -----------------------
    y_ref = _labels(frame[train_rows.tolist()])
    y_qry = _labels(frame[valid_rows.tolist()])
    base_prevalence = float(y_ref.mean())

    out: dict[str, Any] = {
        "n_reference": int(len(train_rows)),
        "n_queries": int(len(valid_rows)),
        "n_positive_queries": int(y_qry.sum()),
        "n_legitimate_queries": int((y_qry == 0).sum()),
        "reference_positive_prevalence": base_prevalence,
        "by_query_class": {},
    }
    for name, mask in (("positive", y_qry == 1), ("legitimate", y_qry == 0)):
        if not mask.any():
            continue
        hits = y_ref[retrieved[mask]]          # (n_queries, kmax) of 0/1
        stats: dict[str, Any] = {"n_queries": int(mask.sum())}
        for k in k_values:
            top = hits[:, :k]
            prevalence = float(top.mean())
            stats[f"hit_at_{k}"] = float((top.sum(axis=1) > 0).mean())
            stats[f"neighbor_positive_prevalence_at_{k}"] = prevalence
            stats[f"mean_positive_neighbors_at_{k}"] = float(top.sum(axis=1).mean())
            stats[f"lift_at_{k}"] = (
                prevalence / base_prevalence if base_prevalence > 0 else None)
        out["by_query_class"][name] = stats
    return out


def _pool(folds: list[dict[str, Any]], cls: str, key: str) -> float | None:
    """Query-count-weighted mean of one statistic across folds.

    Weighted rather than a plain mean of fold means: the folds hold different
    numbers of positive queries, and an unweighted average would let the
    smallest fold speak loudest.
    """
    pairs = [(f["by_query_class"][cls]["n_queries"], f["by_query_class"][cls][key])
             for f in folds if cls in f["by_query_class"]
             and f["by_query_class"][cls].get(key) is not None]
    if not pairs:
        return None
    total = sum(n for n, _ in pairs)
    return float(sum(n * v for n, v in pairs) / total) if total else None


def evaluate(*, repeat: int = DEFAULT_REPEAT, k_values=K_VALUES,
             folds: int | None = None) -> dict[str, Any]:
    """Run the section-18 protocol over every outer fold of one repeat."""
    from muleguard.features.frame import raw_with_meta

    frame = raw_with_meta()
    assignments = _fold_assignments(repeat)
    fold_ids = sorted(assignments["outer_fold"].unique().to_list())
    if folds is not None:
        fold_ids = fold_ids[:folds]

    results = []
    for fold in fold_ids:
        part = assignments.filter(pl.col("outer_fold") == fold)
        train = np.sort(part.filter(pl.col("role") == "train")["row_index"]
                        .unique().to_numpy())
        valid = np.sort(part.filter(pl.col("role") == "outer_valid")["row_index"]
                        .unique().to_numpy())
        overlap = np.intersect1d(train, valid)
        if overlap.size:
            raise RuntimeError(
                f"fold {fold}: {overlap.size} rows are in both the reference "
                "index and the query set; the evaluation would be scoring "
                "self-matches")
        log.info("fold %d: %d reference rows, %d queries", fold,
                 len(train), len(valid))
        r = _evaluate_fold(frame=frame, train_rows=train, valid_rows=valid,
                           k_values=k_values)
        r["outer_fold"] = int(fold)
        results.append(r)

    pooled = {}
    for cls in ("positive", "legitimate"):
        stats = {"n_queries": sum(f["by_query_class"].get(cls, {}).get("n_queries", 0)
                                  for f in results)}
        for k in k_values:
            for key in (f"hit_at_{k}", f"neighbor_positive_prevalence_at_{k}",
                        f"mean_positive_neighbors_at_{k}", f"lift_at_{k}"):
                stats[key] = _pool(results, cls, key)
        pooled[cls] = stats

    prevalences = [f["reference_positive_prevalence"] for f in results]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "radar_version": str(cr._config()["radar_version"]),
        "what_this_is": DISCLAIMER,
        "protocol": {
            "repeat": repeat,
            "n_outer_folds": len(results),
            "assignments": str(ASSIGNMENTS_PATH.relative_to(settings.REPO_ROOT)),
            "similarity_fitted_on": "outer-training rows of each fold, separately",
            "reference_index": "outer-training rows only",
            "queries": "outer-validation rows only",
            "labels_used_for_retrieval": False,
            "labels_used_after_retrieval": True,
            "locked_test_used": False,
            "k_values": list(k_values),
            "tie_break": "reference position, so the ranking is total",
        },
        "reference_positive_prevalence": {
            "mean": float(np.mean(prevalences)),
            "min": float(np.min(prevalences)),
            "max": float(np.max(prevalences)),
        },
        "pooled": pooled,
        "per_fold": results,
    }


def write(report: dict[str, Any] | None = None, **kw) -> dict[str, Any]:
    """Run (unless handed a report) and persist to the section-39 path."""
    from muleguard.utils import save_json

    report = evaluate(**kw) if report is None else report
    save_json(report, OUT_PATH)
    log.info("wrote %s", OUT_PATH)
    return report
