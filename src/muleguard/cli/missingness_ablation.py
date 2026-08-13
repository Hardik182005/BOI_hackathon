"""Decide whether the missingness signature earns a place in the model.

``updates1`` upgrade #1 asks for missingness-derived indicators, built **inside
CV only**, and keeps them only if "repeated CV improves and the indicators
remain stable across folds". This module is the deciding experiment, and it is
written so that it can return NO.

Design
------
Two arms, WITHOUT and WITH, on **byte-identical outer folds**:

* the same 3x5 outer fold assignments the nested tournament uses;
* the same preprocessing, the same inner-fold selector, the same model family,
  the same hyperparameters, the same seeds;
* the only difference is that the WITH arm calls
  :meth:`MissingnessSignature.fit` on the outer-training rows and appends the
  block before preprocessing.

Because the arms are paired, the comparison is far more sensitive than the
project's published 0.0905 unpaired seed-noise floor: the paired difference has
its own, much smaller, spread, and that spread is what the decision is measured
against. The unpaired floor is the right yardstick for "is model A better than
model B across the board"; it is the wrong - and needlessly blunt -yardstick for
"do these extra columns help, holding everything else fixed".

Hyperparameters are held **fixed** rather than tuned per arm. Tuning both arms
would let the WITH arm win on a luckier hyperparameter draw rather than on the
new columns, which is exactly the confound this experiment exists to avoid.

What "stable across folds" means here
-------------------------------------
Two things are recorded, because a mean improvement built on one fold is not a
finding:

* ``n_folds_improved`` - a sign test over the 15 outer folds.
* ``indicator_stability`` - how often each missingness column survives the
  inner-fold selector, across all outer folds. A column that appears in one
  fold and never again is noise even if the mean moved.

Writes ``artifacts/metrics/missingness_ablation.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from muleguard import settings
from muleguard.features import frame as frame_mod
from muleguard.features.missingness import PREFIX, MissingnessSignature
from muleguard.logging import configure, get_logger
from muleguard.models import harness, nested

log = get_logger("cli.missingness_ablation")


def _spw(y: np.ndarray) -> float:
    pos = float(y.sum())
    return float((len(y) - pos) / max(pos, 1.0))


def hgb_fit_predict(Xtr, ytr, Xva, seed):
    """The nested leader's family, at its default configuration.

    Held fixed across both arms - see the module docstring.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    w = np.where(ytr == 1, _spw(ytr), 1.0)
    m = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=20,
        l2_regularization=1.0, random_state=seed, early_stopping=False)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict_proba(Xva)[:, 1]


def xgb_fit_predict(Xtr, ytr, Xva, seed):
    """The served champion's family, at its default configuration."""
    import xgboost as xgb

    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=1.0,
        subsample=0.8, colsample_bytree=0.6, gamma=0.0, reg_alpha=0.0,
        reg_lambda=1.0, scale_pos_weight=_spw(ytr), tree_method="hist",
        n_jobs=int(settings.load_config("train")["n_jobs"]), random_state=seed,
        eval_metric="aucpr", verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


FAMILIES = {"histgb": hgb_fit_predict, "xgboost": xgb_fit_predict}


def _make_augmentor(registry: dict[str, Any], *, max_flags: int,
                    exclude: tuple[str, ...] = ()) -> Any:
    """Build the ``augment`` hook for :func:`nested.build_outer_folds`.

    Args:
        exclude: substrings; any generated column whose name contains one is
            dropped before the block is appended. This exists for the
            suspect-family arm - if the WITH arm only wins because of a column
            whose provenance could not be established, that is a reason to
            reject it, and the only way to find out is to run the comparison
            again without it. With an empty tuple the block is byte-identical
            to :meth:`MissingnessSignature.augment`.
    """
    fitted: list[MissingnessSignature] = []

    def augment(Xtr_raw, Xva_raw, names):
        sig = MissingnessSignature.fit(
            Xtr_raw, list(names), registry, max_flags=max_flags)
        fitted.append(sig)
        if not exclude:
            return (sig.augment(Xtr_raw), sig.augment(Xva_raw),
                    sig.augmented_names())
        keep = [j for j, n in enumerate(sig.names)
                if not any(tok in n for tok in exclude)]
        block_names = [sig.names[j] for j in keep]
        return (np.hstack([Xtr_raw, sig.transform(Xtr_raw)[:, keep]]),
                np.hstack([Xva_raw, sig.transform(Xva_raw)[:, keep]]),
                list(names) + block_names)

    augment.fitted = fitted  # type: ignore[attr-defined]
    return augment


def _score_folds(folds, fit_predict, *, n_feat: int, n_repeats: int,
                 n_dev: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Outer-validation scores for a fixed feature-set size.

    The size is fixed rather than inner-selected so that both arms spend the
    same feature budget; if the WITH arm needs 120 columns to fit 120 useful
    ones, it has to displace base columns to do it, which is the honest test.
    """
    scores = np.full((n_repeats, n_dev), np.nan)
    per_fold: list[dict] = []
    for f in folds:
        cols = f.top(n_feat)
        pred = fit_predict(f.Xtr[:, cols], f.ytr, f.Xva[:, cols], seed)
        scores[f.repeat, f.valid_idx] = pred
        chosen = [f.kept_features[c] for c in cols]
        per_fold.append({
            "repeat": f.repeat, "fold": f.fold,
            "fold_ap": float(average_precision_score(f.yva, pred)),
            "n_missingness_cols_selected": sum(
                1 for c in chosen if c.startswith(PREFIX)),
            "missingness_cols_selected": [c for c in chosen if c.startswith(PREFIX)],
        })
    if np.isnan(scores).any():
        raise RuntimeError("outer folds left rows unscored")
    return scores, per_fold


def _summarise(name: str, scores: np.ndarray, y: np.ndarray,
               per_fold: list[dict]) -> dict[str, Any]:
    aps = [float(average_precision_score(y, s)) for s in scores]
    rocs = [float(roc_auc_score(y, s)) for s in scores]
    return {
        "arm": name,
        "pr_auc_mean": float(np.mean(aps)),
        "pr_auc_std": float(np.std(aps)),
        "pr_auc_per_repeat": [round(a, 5) for a in aps],
        "roc_auc_mean": float(np.mean(rocs)),
        "fold_ap": [round(p["fold_ap"], 5) for p in per_fold],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="histgb", choices=sorted(FAMILIES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-feat", type=int, default=120)
    ap.add_argument("--max-flags", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--exclude-contains", default="",
                    help="comma-separated substrings; generated columns whose "
                         "name contains one are dropped from the WITH arm")
    ap.add_argument("--out", default="missingness_ablation.json",
                    help="artifact filename under artifacts/metrics/")
    args = ap.parse_args(argv)

    exclude = tuple(t for t in (s.strip() for s in args.exclude_contains.split(","))
                    if t)

    configure()
    seed = args.seed if args.seed is not None else settings.GLOBAL_SEED
    fit_predict = FAMILIES[args.family]

    frame = frame_mod.build_model_frame()
    registry = frame_mod.augmented_registry()["features"]
    dev = harness.dev_split(args.repeats)
    y = np.asarray(frame.y)[dev.row_index]
    n_dev = len(dev.row_index)

    log.info("ARM 1/2: WITHOUT missingness (%s, top_%d, %d repeats)",
             args.family, args.n_feat, args.repeats)
    base_folds = nested.build_outer_folds(frame, n_repeats=args.repeats)
    base_scores, base_per_fold = _score_folds(
        base_folds, fit_predict, n_feat=args.n_feat,
        n_repeats=args.repeats, n_dev=n_dev, seed=seed)
    without = _summarise("WITHOUT", base_scores, y, base_per_fold)
    log.info("WITHOUT: PR-AUC %.5f +/- %.5f",
             without["pr_auc_mean"], without["pr_auc_std"])

    log.info("ARM 2/2: WITH missingness signature, fitted inside each outer fold")
    if exclude:
        log.info("WITH arm excludes generated columns containing: %s",
                 ", ".join(exclude))
    augment = _make_augmentor(registry, max_flags=args.max_flags, exclude=exclude)
    with_folds = nested.build_outer_folds(
        frame, n_repeats=args.repeats, augment=augment)
    with_scores, with_per_fold = _score_folds(
        with_folds, fit_predict, n_feat=args.n_feat,
        n_repeats=args.repeats, n_dev=n_dev, seed=seed)
    with_arm = _summarise("WITH", with_scores, y, with_per_fold)
    log.info("WITH:    PR-AUC %.5f +/- %.5f",
             with_arm["pr_auc_mean"], with_arm["pr_auc_std"])

    # --- paired comparison -------------------------------------------------
    a = np.asarray(without["fold_ap"], dtype=float)
    b = np.asarray(with_arm["fold_ap"], dtype=float)
    diff = b - a
    n_improved = int((diff > 0).sum())
    # Two-sided exact sign test against p=0.5. With 15 folds this is coarse,
    # which is the point: it cannot manufacture significance.
    from math import comb
    k = max(n_improved, len(diff) - n_improved)
    p_sign = min(1.0, 2.0 * sum(comb(len(diff), i) for i in range(k, len(diff) + 1))
                 / 2 ** len(diff))

    # The sign test ignores magnitude, so it is deliberately hard to please: a
    # fold that improves by 0.0003 counts the same as one that improves by 0.20.
    # Two tests that do use magnitude are recorded alongside it, not to find a
    # kinder p-value but so that a disagreement between them is visible in the
    # artifact instead of being resolved silently in favour of whichever number
    # suits. Where they disagree, the write-up has to explain the disagreement.
    from scipy import stats

    t_stat, p_t = stats.ttest_rel(b, a)
    try:
        w_stat, p_w = stats.wilcoxon(diff)
    except ValueError:  # pragma: no cover - only if every difference is zero
        w_stat, p_w = float("nan"), float("nan")

    # --- indicator stability ----------------------------------------------
    counts: dict[str, int] = {}
    for p in with_per_fold:
        for c in p["missingness_cols_selected"]:
            counts[c] = counts.get(c, 0) + 1
    n_folds = len(with_per_fold)
    stability = {c: round(v / n_folds, 4) for c, v in
                 sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}
    survived_all = [c for c, v in counts.items() if v == n_folds]
    survived_most = [c for c, v in counts.items() if v >= 0.8 * n_folds]

    per_fold_selected = [p["n_missingness_cols_selected"] for p in with_per_fold]

    # --- verdict ------------------------------------------------------------
    # Three conditions, all required. Any one failing means the columns do not
    # go into the model - a mean gain alone is not enough.
    mean_gain = float(diff.mean())
    cond_mean = mean_gain > 0
    cond_sign = n_improved >= int(np.ceil(0.7 * n_folds))
    cond_stable = len(survived_most) >= 3
    keep = bool(cond_mean and cond_sign and cond_stable)

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question": "Does the missingness signature improve the model?",
        "design": {
            "protocol": "NESTED, paired on byte-identical outer folds",
            "family": args.family,
            "hyperparameters": "FIXED across both arms - not tuned per arm",
            "feature_set_size": args.n_feat,
            "n_repeats": args.repeats,
            "n_outer_folds": n_folds,
            "seed": seed,
            "signature_fitted": "inside each outer fold, on training rows only",
            "max_flags": args.max_flags,
            "excluded_from_with_arm": list(exclude),
        },
        "without": without,
        "with": with_arm,
        "paired": {
            "mean_gain": round(mean_gain, 5),
            "median_gain": round(float(np.median(diff)), 5),
            "std_of_paired_diff": round(float(diff.std()), 5),
            "per_fold_gain": [round(d, 5) for d in diff],
            "n_folds_improved": n_improved,
            "n_folds": n_folds,
            "sign_test_p_two_sided": round(p_sign, 5),
            "wilcoxon_p_two_sided": round(float(p_w), 5),
            "paired_t_p_two_sided": round(float(p_t), 5),
            "paired_t_statistic": round(float(t_stat), 4),
            "note": ("Paired on identical folds, so the relevant yardstick is the "
                     "spread of the paired difference, not the 0.0905 unpaired "
                     "seed-noise floor."),
            "test_disagreement_note": (
                "The sign test discards magnitude and is the most conservative of "
                "the three. If it disagrees with the magnitude-aware tests, the "
                "cause is a few large positive gains against several near-zero "
                "differences, and the write-up must say so rather than quote the "
                "most favourable p-value."),
        },
        "indicator_stability": {
            "missingness_cols_selected_per_fold": per_fold_selected,
            "mean_selected_per_fold": round(float(np.mean(per_fold_selected)), 2),
            "n_distinct_selected": len(counts),
            "survived_all_folds": sorted(survived_all),
            "survived_80pct_folds": sorted(survived_most),
            "selection_frequency": stability,
        },
        "decision": {
            "keep": keep,
            "conditions": {
                "mean_gain_positive": cond_mean,
                "improved_in_at_least_70pct_of_folds": cond_sign,
                "at_least_3_indicators_stable_in_80pct_of_folds": cond_stable,
            },
            "rule": ("All three conditions are required. A positive mean with "
                     "unstable indicators is rejected as noise."),
        },
    }

    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.METRICS_DIR / args.out
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote %s", path)

    log.info("PAIRED GAIN %+0.5f  improved %d/%d folds  sign-test p=%.4f",
             mean_gain, n_improved, n_folds, p_sign)
    log.info("DECISION: %s", "KEEP" if keep else "REJECT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
