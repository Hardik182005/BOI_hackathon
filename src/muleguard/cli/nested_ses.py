"""Stability / Ensemble / Shift on nested folds (final-validation 15-23, 53, 54).

Run::

    .venv/Scripts/python.exe -m muleguard.cli.nested_ses --stages all --n-jobs 2

Every stage in this programme is a **paired** comparison on the 15 outer folds
produced by :func:`muleguard.models.nested.build_outer_folds` with
``harness.dev_split(3)``, ``n_repeats=3``, ``n_inner=4``. The folds are built
once, in this process, and shared by every stage: two arms that were scored on
different folds are not comparable and this CLI gives no way to produce such a
pair.

Stages
------
``bases``       inner-OOF + outer-val predictions for 4 families (feeds 19/20/22/54)
``ensemble``    sections 19, 20, 53 - probability, stacked and rank ensembles
``seedbag``     section 18 - model-seed bagging
``posremoval``  section 21 - rare-positive removal stress
``labelnoise``  section 22 - label-noise audit, plus a fold-local down-weight arm
``shift``       section 23 - adversarial validation, feature shift, OOD, calibration
``families``    sections 15, 16, 17 - bank-prior, meta-feature and alert-context pools
``score``       section 54 - the generalization / shift report, composed from the above

Each stage writes its own JSON under ``artifacts/metrics/`` as soon as it
finishes, so a long run that is interrupted still leaves usable evidence.

What this CLI will not do
-------------------------
It never reads the locked test set, it never reads ``fold.yva`` outside a
scoring call, and it never tunes hyperparameters inside an arm. The decision
rules are fixed in ``docs/NESTED_STABILITY_ENSEMBLE_SHIFT.md`` before the run and
are evaluated here mechanically; the code does not choose between them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from typing import Any

import numpy as np

from muleguard import settings
from muleguard.features import firewall
from muleguard.features import frame as frame_mod
from muleguard.logging import configure, get_logger
from muleguard.models import harness, nested
from muleguard.models import nested_experiments as nx
from muleguard.models.paired import paired_report
from muleguard.models.robustness import ROBUSTNESS_THRESHOLDS
from muleguard.models.selection import _selector_importances

log = get_logger("cli.nested_ses")

#: The 13 columns that must never reach a model. The firewall removes them; the
#: artifacts record the verification rather than assuming it.
HARD_QUARANTINE = (
    "F2230", "F3892", "F3898", "F3899", "F3912", "F3913", "F3914",
    "F3915", "F3916", "F3917", "F3918", "F3924", "__UNNAMED__0",
)

#: Seed offsets for section 18, matching artifacts/metrics/seed_variance_v2.json
#: so that the nested numbers sit next to the flat ones without a second
#: unexplained difference between them.
SEED_OFFSETS = (0, 7919, 15838, 23757, 31676)

STAGES = ("bases", "ensemble", "seedbag", "posremoval", "labelnoise",
          "shift", "families", "score")


# ==========================================================================
# shared helpers
# ==========================================================================
def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write(name: str, payload: dict[str, Any]) -> None:
    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.METRICS_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def _read(name: str) -> dict[str, Any] | None:
    path = settings.METRICS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def firewall_check(frame, folds) -> dict[str, Any]:
    """Verify the quarantine actually held, in the matrix the run used."""
    in_frame = [c for c in HARD_QUARANTINE if c in set(frame.feature_names)]
    in_folds = sorted({c for f in folds for c in HARD_QUARANTINE
                       if c in set(f.kept_features)})
    if in_frame or in_folds:
        raise RuntimeError(f"quarantined columns reached the model: "
                           f"{in_frame or in_folds}")
    return {
        "hard_quarantine_checked": list(HARD_QUARANTINE),
        "quarantined_columns_in_model_frame": in_frame,
        "quarantined_columns_in_any_fold_matrix": in_folds,
        "n_model_features": len(frame.feature_names),
        "n_columns_surviving_fold_preprocessing": sorted(
            {len(f.kept_features) for f in folds}),
        "target_column_excluded": settings.TARGET_COLUMN not in set(frame.feature_names),
    }


def design_block(args, folds) -> dict[str, Any]:
    return {
        "protocol": "NESTED repeated CV, paired on byte-identical outer folds",
        "outer": f"stratified 5-fold x {args.repeats} repeats "
                 f"= {len(folds)} outer folds",
        "inner": f"stratified {args.inner}-fold within outer-train",
        "fold_source": "muleguard.models.nested.build_outer_folds "
                       "(the same function the nested tournament uses)",
        "fitted_inside_outer_train_only": [
            "preprocessing (constant/duplicate removal, imputation, scaling)",
            "feature ranking and feature-set choice",
            "ensemble weights, stacker coefficients, calibration",
            "label-noise flags (fold-local arm)",
        ],
        "outer_validation_labels": "used once per arm, for scoring only",
        "feature_set_size": args.n_feat,
        "hyperparameters": "each family at its default configuration, "
                           "identical across arms - not tuned per arm",
        "yardstick": (
            "paired differences on fixed folds. The project's 0.0905 PR-AUC "
            "seed-noise floor is an UNPAIRED figure and does not apply here; "
            "each comparison is judged against its own std_of_paired_diff."),
        "power_warning": (
            "each outer-validation fold holds ~13 positives, so fold-level "
            "average precision is extremely coarse and almost nothing in this "
            "programme can reach significance. Null results mean 'no effect of "
            "this size', never 'no effect'."),
    }


def _families(n_jobs: int) -> dict[str, Any]:
    """The four gradient-boosting families at their default configurations.

    Imported from :mod:`muleguard.cli.nested_cv` rather than redefined, so the
    estimator built here is the same object the nested tournament builds.
    """
    from muleguard.cli import nested_cv as nc

    # nested_cv exposes its thread budget as a module global that the factory
    # closures read at call time. Two heavy jobs share this machine, so the
    # budget is lowered explicitly rather than inherited.
    nc.N_JOBS = n_jobs
    log.info("model thread budget: n_jobs=%d", nc.N_JOBS)
    return {
        "lightgbm": nc.lgbm_factory({}),
        "xgboost": nc.xgb_factory({}),
        "catboost": nc.cat_factory({}),
        "histgb": nc.hgb_factory({}),
    }


def _fold_key(f) -> str:
    return f"r{f.repeat}f{f.fold}"


# ==========================================================================
# stage: bases
# ==========================================================================
def stage_bases(ctx: dict[str, Any], args) -> list[nx.FoldBases]:
    if "bases" in ctx:
        return ctx["bases"]
    fams = _families(args.n_jobs)
    t0 = time.time()
    bases = [nx.compute_bases(f, fams, n_feat=args.n_feat) for f in ctx["folds"]]
    log.info("base predictions for %d folds x %d families in %.0fs",
             len(bases), len(fams), time.time() - t0)
    ctx["bases"] = bases
    return bases


# ==========================================================================
# stage: ensemble  (sections 19, 20, 53)
# ==========================================================================
def stage_ensemble(ctx: dict[str, Any], args) -> dict[str, Any]:
    bases = stage_bases(ctx, args)
    per_fold: dict[str, list[dict[str, float]]] = {k: [] for k in nx.COMBINERS}
    detail: list[dict[str, Any]] = []
    single_ap: dict[str, list[float]] = {}

    for b in bases:
        row: dict[str, Any] = {"repeat": b.repeat, "fold": b.fold,
                               "inner_ap": {k: round(v, 5) for k, v in b.inner_ap.items()}}
        for fam, s in b.val.items():
            single_ap.setdefault(fam, []).append(
                float(nx.fold_metrics(b.yva, s)["ap"]))
        for name, fn in nx.COMBINERS.items():
            s, meta = fn(b.inner_oof, b.ytr, b.val)
            per_fold[name].append(nx.fold_metrics(b.yva, s))
            row[name] = {"ap": round(per_fold[name][-1]["ap"], 5), **{
                k: v for k, v in meta.items() if k in
                ("picked", "weights", "coef", "intercept")}}
        detail.append(row)

    base_ap = [m["ap"] for m in per_fold["best_single_inner"]]
    comparisons = {}
    for name in nx.COMBINERS:
        if name == "best_single_inner":
            continue
        arm_ap = [m["ap"] for m in per_fold[name]]
        pr = paired_report(base_ap, arm_ap, name=f"{name} vs best_single_inner",
                           baseline_name="best_single_inner", arm_name=name)
        comparisons[name] = pr.to_dict()

    def summ(name: str) -> dict[str, Any]:
        ms = per_fold[name]
        out = {"fold_ap_mean": round(float(np.mean([m["ap"] for m in ms])), 5),
               "fold_ap_std": round(float(np.std([m["ap"] for m in ms])), 5),
               "fold_ap": [round(m["ap"], 5) for m in ms]}
        for k in nx.FOLD_BUDGETS:
            out[f"recall_at_{k}_mean"] = round(
                float(np.mean([m[f"recall_at_{k}"] for m in ms])), 5)
        return out

    arms = {name: summ(name) for name in nx.COMBINERS}
    base = arms["best_single_inner"]

    # --- section 53, applied mechanically ---------------------------------
    hardest = list(np.argsort(base_ap)[:5])
    verdicts = {}
    for name, comp in comparisons.items():
        diff = np.asarray(comp["per_fold_diff"], dtype=float)
        c1 = comp["mean_paired_diff"] > 0 or (
            comp["ci95_of_mean"][0] <= 0 <= comp["ci95_of_mean"][1])
        c2 = (arms[name]["fold_ap_std"] < base["fold_ap_std"]
              or arms[name][f"recall_at_10_mean"] > base["recall_at_10_mean"])
        c3 = bool(diff.min() >= -0.05 and float(diff[hardest].mean()) >= 0.0)
        verdicts[name] = {
            "accept": bool(c1 and c2 and c3),
            "c1_ap_improves_or_tied": bool(c1),
            "c2_variance_down_or_recall_up": bool(c2),
            "c3_stress_proxy_ok": c3,
            "worst_fold_diff": round(float(diff.min()), 5),
            "mean_diff_on_5_hardest_folds": round(float(diff[hardest].mean()), 5),
        }

    accepted = [n for n, v in verdicts.items() if v["accept"]]
    best_arm = max(arms, key=lambda n: arms[n]["fold_ap_mean"])
    return {
        "generated_utc": _now(),
        "question": "Do probability, stacked or rank ensembles beat the best "
                    "single model when weights are fitted inside the fold?",
        "sections": ["19", "20", "53"],
        "design": design_block(args, ctx["folds"]) | {
            "members": sorted(bases[0].val),
            "reference_arm": "best_single_inner - the family with the best INNER "
                             "average precision in that fold, so the baseline is "
                             "chosen without seeing outer-validation labels",
            "meta_inputs": "inner out-of-fold probabilities of the outer-training "
                           "rows; no in-sample base prediction is ever a stacker input",
            "rank_scope": "percentile ranks are computed within an outer-validation "
                          "fold and use no labels",
        },
        "firewall": ctx["firewall"],
        "single_family_fold_ap_mean": {k: round(float(np.mean(v)), 5)
                                       for k, v in single_ap.items()},
        "arms": arms,
        "paired_vs_best_single_inner": comparisons,
        "section_53_rule": {
            "text": "accept an ensemble only if (mean AP improves OR is "
                    "statistically tied) AND (variance decreases OR Recall@TopK "
                    "improves) AND external-like stress does not degrade materially",
            "operationalised": {
                "c1": "mean paired diff > 0, or the 95% CI of the mean contains 0",
                "c2": "std of the 15 fold APs decreases, or mean fold Recall@10 rises",
                "c3": "worst fold diff >= -0.05 AND mean diff over the 5 "
                      "lowest-baseline folds >= 0 (a within-dataset PROXY for "
                      "external stress - no external set exists here)",
            },
            "verdicts": verdicts,
            "accepted": accepted,
            "decision": ("ADOPT " + best_arm if best_arm in accepted and
                         arms[best_arm]["fold_ap_mean"] > base["fold_ap_mean"]
                         else "KEEP THE STRONGEST SIMPLE SINGLE MODEL"),
        },
        "per_fold_detail": detail,
    }


# ==========================================================================
# stage: seedbag  (section 18)
# ==========================================================================
def stage_seedbag(ctx: dict[str, Any], args) -> dict[str, Any]:
    fams = _families(args.n_jobs)
    out: dict[str, Any] = {
        "generated_utc": _now(), "sections": ["18"],
        "question": "Does averaging a family over 5 seeds improve or stabilise it?",
        "design": design_block(args, ctx["folds"]) | {
            "seeds_per_fold": [f"fold_seed + {o}" for o in SEED_OFFSETS],
            "single_seed_arm": "seed offset 0 - the model that would have shipped",
        },
        "firewall": ctx["firewall"], "families": {},
    }
    for fam in args.bag_families.split(","):
        fp = fams[fam]
        single, pmean, rmean = [], [], []
        pstd, rstd, per_seed = [], [], []
        for f in ctx["folds"]:
            cols = f.top(args.n_feat)
            seeds = [harness.fold_seed(f.repeat, f.fold) + o for o in SEED_OFFSETS]
            r = nx.seed_bag(f, fp, cols, seeds)
            single.append(float(nx.fold_metrics(f.yva, r["single"])["ap"]))
            pmean.append(float(nx.fold_metrics(f.yva, r["prob_mean"])["ap"]))
            rmean.append(float(nx.fold_metrics(f.yva, r["rank_mean"])["ap"]))
            pstd.append(r["probability_std"])
            rstd.append(r["rank_std"])
            per_seed.append([round(a, 5) for a in r["per_seed_ap"]])
            log.info("seedbag %-9s %s single=%.4f bag=%.4f", fam, _fold_key(f),
                     single[-1], pmean[-1])
        cmp_prob = paired_report(single, pmean, baseline_name="single_seed",
                                 arm_name="seed_bag_prob_mean").to_dict()
        cmp_rank = paired_report(single, rmean, baseline_name="single_seed",
                                 arm_name="seed_bag_rank_mean").to_dict()
        std_single = float(np.std(single))
        std_bag = float(np.std(pmean))
        adopt = (cmp_prob["mean_paired_diff"] > 0
                 and cmp_prob["n_tests_below_0.05"] >= 2)
        tied = (cmp_prob["ci95_of_mean"][0] <= 0 <= cmp_prob["ci95_of_mean"][1])
        stability_only = (not adopt) and tied and std_bag < std_single
        # A family with no stochastic component fits the identical model five
        # times, so the arm cannot move and NO_CHANGE would be read as evidence
        # that seed averaging was tried and did not help. It was not tried:
        # there was nothing to average. histgb is the case here - sklearn's
        # HistGradientBoosting draws no random subsample, so random_state does
        # not reach the fit.
        deterministic = float(np.mean(pstd)) == 0.0 and float(np.mean(rstd)) == 0.0
        out["families"][fam] = {
            "fold_ap_single_mean": round(float(np.mean(single)), 5),
            "fold_ap_bag_mean": round(float(np.mean(pmean)), 5),
            "fold_ap_rankbag_mean": round(float(np.mean(rmean)), 5),
            "fold_ap_std_single": round(std_single, 5),
            "fold_ap_std_bag": round(std_bag, 5),
            "mean_probability_std_across_seeds": round(float(np.mean(pstd)), 6),
            "mean_rank_std_across_seeds": round(float(np.mean(rstd)), 6),
            "per_fold_per_seed_ap": per_seed,
            "paired_prob_mean_vs_single": cmp_prob,
            "paired_rank_mean_vs_single": cmp_rank,
            "deterministic_under_reseeding": deterministic,
            "decision": ("NOT_APPLICABLE" if deterministic else
                         "ADOPT" if adopt else
                         "ADOPT_FOR_STABILITY_ONLY" if stability_only else "NO_CHANGE"),
        }
        if deterministic:
            out["families"][fam]["why_not_applicable"] = (
                f"every seed produced byte-identical predictions "
                f"(across-seed probability sd 0.0 on all {len(ctx['folds'])} folds), "
                f"so the five fits are one fit. This is a property of the estimator, "
                f"not a result: read it as 'seed averaging does not apply to "
                f"{fam}', never as 'seed averaging was tried and did not help'.")
    out["rule"] = {
        "ADOPT": "mean paired gain > 0 AND at least 2 of the 3 paired tests p < 0.05",
        "ADOPT_FOR_STABILITY_ONLY": "not ADOPT, the 95% CI contains 0 (tied), and "
                                    "the across-fold AP spread decreases",
        "NO_CHANGE": "otherwise - keep the single-seed model, which is 5x cheaper",
        "NOT_APPLICABLE": "the family is deterministic under reseeding, so the "
                          "comparison has no content - checked before the others",
    }
    return out


# ==========================================================================
# stage: posremoval  (section 21)
# ==========================================================================
def stage_posremoval(ctx: dict[str, Any], args) -> dict[str, Any]:
    fams = _families(args.n_jobs)
    fam = args.stress_family
    fp = fams[fam]
    bases = stage_bases(ctx, args)
    by_key = {_fold_key(b): b for b in bases}

    ref_ap, mean_ap, std_ap, pred_rho, feat_rho = [], [], [], [], []
    rec_ref: dict[int, list[float]] = {k: [] for k in nx.FOLD_BUDGETS}
    rec_mean: dict[int, list[float]] = {k: [] for k in nx.FOLD_BUDGETS}
    rec_std: dict[int, list[float]] = {k: [] for k in nx.FOLD_BUDGETS}
    detail = []
    for f in ctx["folds"]:
        cols = f.top(args.n_feat)
        b = by_key[_fold_key(f)]
        if not np.array_equal(cols, b.cols):
            raise RuntimeError("stress and base arms disagree on columns")
        reference = b.val[fam]                      # identical config and seed
        ref_imp = _selector_importances(
            f.Xtr[:, cols], f.ytr, harness.fold_seed(f.repeat, f.fold), args.n_jobs)
        r = nx.positive_removal(f, fp, cols, rounds=args.stress_rounds,
                                fraction=args.stress_fraction, n_jobs=args.n_jobs,
                                reference=reference, reference_importance=ref_imp)
        m = nx.fold_metrics(f.yva, reference)
        ref_ap.append(m["ap"])
        mean_ap.append(float(np.mean(r["ap"])))
        std_ap.append(float(np.std(r["ap"])))
        pred_rho.append(float(np.mean(r["prediction_rank_correlation"])))
        feat_rho.append(float(np.mean(r["feature_rank_correlation"])))
        for k in nx.FOLD_BUDGETS:
            rec_ref[k].append(m[f"recall_at_{k}"])
            rec_mean[k].append(float(np.mean(r["recall"][k])))
            rec_std[k].append(float(np.std(r["recall"][k])))
        detail.append({"repeat": f.repeat, "fold": f.fold,
                       "reference_ap": round(m["ap"], 5),
                       "stressed_ap_mean": round(mean_ap[-1], 5),
                       "stressed_ap_std": round(std_ap[-1], 5),
                       "n_train_positives": r["n_train_positives"],
                       "n_dropped": r["n_dropped"]})
        log.info("posremoval %s ref=%.4f stressed=%.4f+/-%.4f rho=%.3f",
                 _fold_key(f), m["ap"], mean_ap[-1], std_ap[-1], pred_rho[-1])

    cmp_ = paired_report(ref_ap, mean_ap, baseline_name="all_positives",
                         arm_name="positives_removed").to_dict()
    rel_drop = float((np.mean(ref_ap) - np.mean(mean_ap)) / max(np.mean(ref_ap), 1e-9))
    grade_inputs = {
        "positive_removal_pr_auc_relative_drop": round(rel_drop, 5),
        "positive_removal_pr_auc_std": round(float(np.mean(std_ap)), 5),
        "prediction_rank_stability": round(float(np.mean(pred_rho)), 5),
    }
    return {
        "generated_utc": _now(), "sections": ["21"],
        "question": "Does the model collapse if a handful of known mules change?",
        "design": design_block(args, ctx["folds"]) | {
            "family": fam, "rounds_per_fold": args.stress_rounds,
            "removal_fraction": args.stress_fraction,
            "validation_partition": "UNCHANGED in every round - only training "
                                    "positives are removed",
            "feature_rank_instrument": "muleguard.models.selection."
                                       "_selector_importances, the same selector "
                                       "build_outer_folds uses",
        },
        "firewall": ctx["firewall"],
        "reference_fold_ap_mean": round(float(np.mean(ref_ap)), 5),
        "stressed_fold_ap_mean": round(float(np.mean(mean_ap)), 5),
        "stressed_fold_ap_std_within_fold_mean": round(float(np.mean(std_ap)), 5),
        "stressed_fold_ap_std_across_folds": round(float(np.std(mean_ap)), 5),
        "recall_reference_mean": {str(k): round(float(np.mean(rec_ref[k])), 5)
                                  for k in nx.FOLD_BUDGETS},
        "recall_stressed_mean": {str(k): round(float(np.mean(rec_mean[k])), 5)
                                 for k in nx.FOLD_BUDGETS},
        "recall_stressed_std": {str(k): round(float(np.mean(rec_std[k])), 5)
                                for k in nx.FOLD_BUDGETS},
        "prediction_rank_correlation_mean": round(float(np.mean(pred_rho)), 5),
        "prediction_rank_correlation_per_fold": [round(x, 4) for x in pred_rho],
        "feature_rank_correlation_mean": round(float(np.mean(feat_rho)), 5),
        "feature_rank_correlation_per_fold": [round(x, 4) for x in feat_rho],
        "paired_degradation": cmp_,
        "grade_inputs": grade_inputs,
        "note": ("prediction_rank_correlation here is Spearman over the 1,453 rows "
                 "of ONE outer-validation fold, not over all development rows as in "
                 "the flat artifact. The two definitions are not interchangeable and "
                 "the flat caveat still applies: most of these rows are negatives "
                 "whose order carries no operational meaning."),
        "per_fold": detail,
    }


# ==========================================================================
# stage: labelnoise  (section 22)
# ==========================================================================
def stage_labelnoise(ctx: dict[str, Any], args) -> dict[str, Any]:
    bases = stage_bases(ctx, args)
    fams = _families(args.n_jobs)
    dev = harness.dev_split(args.repeats)

    # --- audit: consensus over outer-validation predictions ----------------
    n_dev = len(dev.row_index)
    y = np.asarray(ctx["frame"].y)[dev.row_index]
    fam_names = sorted(bases[0].val)
    low_count = np.zeros(n_dev)          # times ranked below median
    seen = np.zeros(n_dev)
    for b in bases:
        idx = ctx["valid_idx"][_fold_key(b)]
        for fam in fam_names:
            r = nx._pct_rank(b.val[fam])
            low_count[idx] += (r < 0.5)
            seen[idx] += 1
    share_low = np.divide(low_count, np.maximum(seen, 1))
    pos = np.flatnonzero(y == 1)
    flagged = pos[share_low[pos] == 1.0]
    audit = [{
        "row_index": int(dev.row_index[i]),
        "share_of_model_repeat_pairs_ranking_it_below_median": 1.0,
        "n_model_repeat_pairs": int(seen[i]),
    } for i in flagged]

    # --- controlled ablation: fold-local flags vs globally-pooled flags -----
    def weighted_hgb(flag_idx: set[int]):
        from sklearn.ensemble import HistGradientBoostingClassifier

        def fp(Xtr, ytr, Xva, seed, rows=None):
            spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
            w = np.where(ytr == 1, spw, 1.0)
            if rows is not None:
                mask = np.array([r in flag_idx for r in rows])
                w[mask & (ytr == 1)] *= args.noise_weight
            m = HistGradientBoostingClassifier(
                max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=20,
                l2_regularization=1.0, random_state=seed, early_stopping=False)
            m.fit(Xtr, ytr, sample_weight=w)
            return m.predict_proba(Xva)[:, 1]
        return fp

    global_flags = set(int(dev.row_index[i]) for i in flagged)
    base_ap, local_ap, leak_ap = [], [], []
    n_local, n_leak = [], []
    for f, b in zip(ctx["folds"], bases):
        cols = f.top(args.n_feat)
        seed = harness.fold_seed(f.repeat, f.fold)
        rows = dev.row_index[f.train_idx]
        base_ap.append(float(nx.fold_metrics(f.yva, b.val["histgb"])["ap"]))

        # fold-local: flags derived from THIS fold's inner-OOF only
        lf, _ = nx.noise_flags(b.inner_oof, f.ytr)
        local_rows = set(int(r) for r in rows[lf])
        n_local.append(int(lf.sum()))
        fp = weighted_hgb(local_rows)
        local_ap.append(float(nx.fold_metrics(
            f.yva, fp(f.Xtr[:, cols], f.ytr, f.Xva[:, cols], seed, rows))["ap"]))

        # leaky: flags derived from the globally pooled outer-val predictions,
        # which for this fold's training rows were produced by models fitted on
        # rows that are in THIS fold's validation partition. Rejected evidence.
        n_leak.append(int(sum(1 for r in rows if r in global_flags)))
        fp2 = weighted_hgb(global_flags)
        leak_ap.append(float(nx.fold_metrics(
            f.yva, fp2(f.Xtr[:, cols], f.ytr, f.Xva[:, cols], seed, rows))["ap"]))
        log.info("labelnoise %s base=%.4f local=%.4f leaky=%.4f (flags %d/%d)",
                 _fold_key(f), base_ap[-1], local_ap[-1], leak_ap[-1],
                 n_local[-1], n_leak[-1])

    cmp_local = paired_report(base_ap, local_ap, baseline_name="baseline",
                              arm_name="fold_local_downweight").to_dict()
    cmp_leak = paired_report(base_ap, leak_ap, baseline_name="baseline",
                             arm_name="globally_flagged_downweight").to_dict()
    cmp_gap = paired_report(local_ap, leak_ap, baseline_name="fold_local_downweight",
                            arm_name="globally_flagged_downweight").to_dict()
    keep = (cmp_local["mean_paired_diff"] > 0
            and cmp_local["n_tests_below_0.05"] >= 2)
    return {
        "generated_utc": _now(), "sections": ["22"],
        "question": "Are there positives every model contradicts, and does "
                    "down-weighting them help?",
        "design": design_block(args, ctx["folds"]) | {
            "flag_rule": "a positive that EVERY family ranks below the median of "
                         "all rows, in every repeat it is validated in",
            "arms": {
                "baseline": "histgb, balanced weights",
                "fold_local_downweight": f"flagged training positives weighted "
                                         f"x{args.noise_weight}; flags from this "
                                         f"fold's inner-OOF only",
                "globally_flagged_downweight": "IDENTICAL except the flags come "
                    "from the pooled outer-validation predictions - REJECTED "
                    "EVIDENCE, run only to measure the size of that leak",
            },
        },
        "firewall": ctx["firewall"],
        "audit": {
            "n_positives": int((y == 1).sum()),
            "n_flagged_possible_label_noise": len(audit),
            "flagged_rows": audit,
            "mean_share_below_median_positives": round(
                float(share_low[pos].mean()), 4),
            "policy": "FLAG ONLY. Nothing is relabelled and no positive is deleted.",
        },
        "arms": {
            "baseline_fold_ap_mean": round(float(np.mean(base_ap)), 5),
            "fold_local_fold_ap_mean": round(float(np.mean(local_ap)), 5),
            "globally_flagged_fold_ap_mean": round(float(np.mean(leak_ap)), 5),
            "mean_flags_per_fold_local": round(float(np.mean(n_local)), 2),
            "mean_flags_per_fold_global": round(float(np.mean(n_leak)), 2),
        },
        "paired_fold_local_vs_baseline": cmp_local,
        "paired_globally_flagged_vs_baseline": cmp_leak,
        "paired_leak_minus_local": cmp_gap,
        "decision": {
            "keep_downweighting": bool(keep),
            "rule": "KEEP only if the FOLD-LOCAL arm beats the baseline with a "
                    "positive mean paired gain and at least 2 of 3 tests p < 0.05. "
                    "The globally-flagged arm can never justify a change no matter "
                    "what it scores.",
        },
    }


# ==========================================================================
# stage: shift  (section 23, and the calibration input to 54)
# ==========================================================================
def stage_shift(ctx: dict[str, Any], args) -> dict[str, Any]:
    fams = _families(args.n_jobs)
    bases = stage_bases(ctx, args)
    adv, ood = [], []
    shift_rows: dict[str, list[dict[str, Any]]] = {}
    for f in ctx["folds"]:
        cols = f.top(args.n_feat)
        adv.append(nx.adversarial_auc(f, cols, fams["lightgbm"]))
        ood.append(nx.ood_rate(f, cols))
        for r in nx.feature_shift(f, cols):
            shift_rows.setdefault(r["feature"], []).append(r)
        log.info("shift %s adversarial_auc=%.4f ood_rate=%.4f",
                 _fold_key(f), adv[-1], ood[-1])

    n_folds = len(ctx["folds"])
    features = []
    for name, rows in shift_rows.items():
        ks = float(np.nanmean([r["ks"] for r in rows]))
        ms = float(np.mean([r["missing_rate_shift"] for r in rows]))
        uc = float(np.mean([r["unseen_category_rate"] for r in rows]))
        freq = len(rows) / n_folds
        if ks >= 0.10 or ms >= 0.05 or uc >= 0.02:
            cls = "SHIFT_PRONE"
        elif ks >= 0.05 or freq < 0.5:
            cls = "WATCH"
        else:
            cls = "STABLE"
        features.append({"feature": name, "selected_in_folds": len(rows),
                         "selection_frequency": round(freq, 4),
                         "mean_ks": round(ks, 4),
                         "mean_missing_rate_shift": round(ms, 4),
                         "mean_unseen_category_rate": round(uc, 4),
                         "class": cls})
    features.sort(key=lambda r: (-r["mean_ks"], r["feature"]))
    counts: dict[str, int] = {}
    for r in features:
        counts[r["class"]] = counts.get(r["class"], 0) + 1

    # --- calibration fitted on inner-OOF only ------------------------------
    cal = []
    for b in bases:
        raw = b.val["histgb"]
        cald = nx.platt_calibrate(b.inner_oof["histgb"], b.ytr, raw)
        cal.append({
            "repeat": b.repeat, "fold": b.fold,
            "ece_raw": round(nx.ece(b.yva, raw), 5),
            "ece_calibrated": round(nx.ece(b.yva, cald), 5),
            "brier_raw": round(float(np.mean((raw - b.yva) ** 2)), 6),
            "brier_calibrated": round(float(np.mean((cald - b.yva) ** 2)), 6),
        })
    ece_cal = [c["ece_calibrated"] for c in cal]
    ece_raw = [c["ece_raw"] for c in cal]
    cmp_cal = paired_report(ece_raw, ece_cal, baseline_name="uncalibrated",
                            arm_name="platt_on_inner_oof").to_dict()

    mean_adv = float(np.mean(adv))
    return {
        "generated_utc": _now(), "sections": ["23", "54 (inputs)"],
        "question": "Are the outer folds exchangeable, which selected features "
                    "shift between them, and does fold-local calibration hold?",
        "design": design_block(args, ctx["folds"]) | {
            "adversarial_classifier": "lightgbm, 4-fold CV, outer-train rows "
                                      "labelled 0 and outer-validation rows 1, "
                                      "on the fold's own selected columns",
            "external_upload_arm": "NOT RUN - no judge upload exists in this "
                                   "session. The within-dataset arm below cannot "
                                   "substitute for it.",
            "calibration": "Platt scaling fitted on the inner-OOF predictions of "
                           "the outer-training rows, applied once to outer-val",
        },
        "firewall": ctx["firewall"],
        "adversarial_validation": {
            "auc_per_fold": [round(a, 4) for a in adv],
            "auc_mean": round(mean_adv, 5),
            "expected": "0.5 - the outer split is random and stratified",
            "check_band": [0.45, 0.55],
            "within_band": bool(0.45 <= mean_adv <= 0.55),
            "interpretation": "an AUC materially above 0.55 would mean the outer "
                              "folds are not exchangeable, which would invalidate "
                              "every paired comparison in this programme",
        },
        "ood": {
            "rate_per_fold": [round(o, 4) for o in ood],
            "rate_mean": round(float(np.mean(ood)), 5),
            "definition": "share of validation rows with at least one selected "
                          "feature outside the training min-max range; a blunt "
                          "extrapolation proxy, not a density model",
        },
        "feature_shift": {
            "thresholds": {
                "SHIFT_PRONE": "mean KS >= 0.10 or mean missing-rate shift >= 0.05 "
                               "or unseen-category rate >= 0.02",
                "WATCH": "mean KS >= 0.05 or selected in fewer than half the folds",
                "STABLE": "otherwise",
            },
            "class_counts": counts,
            "n_distinct_features_selected": len(features),
            "top_20_by_ks": features[:20],
        },
        "calibration": {
            "per_fold": cal,
            "ece_raw_mean": round(float(np.mean(ece_raw)), 5),
            "ece_calibrated_mean": round(float(np.mean(ece_cal)), 5),
            "ece_calibrated_std_across_folds": round(float(np.std(ece_cal)), 5),
            "paired_ece_change": cmp_cal,
            "note": "lower ECE is better, so a NEGATIVE mean paired difference "
                    "means calibration helped",
        },
    }


# ==========================================================================
# stage: families  (sections 15, 16, 17)
# ==========================================================================
def _pools(frame) -> dict[str, list[str]]:
    from muleguard.features import dictionary as fd

    reg = frame_mod.augmented_registry()
    cols = list(frame.feature_names)
    by_class: dict[str, list[str]] = {}
    bank: list[str] = []
    for c in cols:
        rec = fd.describe(c, reg)
        by_class.setdefault(rec["availability_class"], []).append(c)
        if rec.get("bank_finalized"):
            bank.append(c)
    beh = by_class.get(fd.BEHAVIORAL, [])
    prof = by_class.get(fd.PROFILE, [])
    alert = by_class.get(fd.ALERT_CONTEXT, [])
    meta = list(frame.meta_features)
    return {
        "full_clean": cols,
        "bank_prior": bank,
        "behavior_only": beh,
        "behavior_profile": beh + prof,
        "alert_only": alert,
        "no_meta_features": [c for c in cols if c not in set(meta)],
        "meta_features_only": meta,
    }


def stage_families(ctx: dict[str, Any], args) -> dict[str, Any]:
    fams = _families(args.n_jobs)
    fp = fams[args.arm_family]
    frame = ctx["frame"]
    pools = _pools(frame)

    arms: dict[str, dict[str, Any]] = {}
    ap_by_arm: dict[str, list[float]] = {}
    for arm, pool in pools.items():
        for size in (args.n_feat,) if arm != "full_clean" else (args.n_feat, 60, 30):
            key = arm if size == args.n_feat else f"{arm}_top{size}"
            aps, sizes = [], []
            for f in ctx["folds"]:
                cols = nx.pool_columns(f, None if arm == "full_clean" else pool, size)
                seed = harness.fold_seed(f.repeat, f.fold)
                s = fp(f.Xtr[:, cols], f.ytr, f.Xva[:, cols], seed)
                aps.append(float(nx.fold_metrics(f.yva, s)["ap"]))
                sizes.append(len(cols))
            ap_by_arm[key] = aps
            arms[key] = {
                "pool_size": len(pool),
                "columns_used_per_fold": sorted(set(sizes)),
                "fold_ap_mean": round(float(np.mean(aps)), 5),
                "fold_ap_std": round(float(np.std(aps)), 5),
                "fold_ap": [round(a, 5) for a in aps],
            }
            log.info("arm %-22s pool=%-5d used=%-4d AP=%.5f", key, len(pool),
                     sizes[0], arms[key]["fold_ap_mean"])

    base = ap_by_arm["full_clean"]
    comparisons = {k: paired_report(base, v, baseline_name="full_clean",
                                    arm_name=k).to_dict()
                   for k, v in ap_by_arm.items() if k != "full_clean"}
    return {
        "generated_utc": _now(), "sections": ["15", "16", "17"],
        "question": "Do the bank's prior, the MG_* meta-features, or the "
                    "alert-context block change the model?",
        "design": design_block(args, ctx["folds"]) | {
            "family": args.arm_family,
            "arm_construction": "the fold's outer-train feature ranking is "
                                "filtered to the arm's column pool; no arm ever "
                                "re-ranks using validation rows",
            "known_bias": "the ranking was fitted with the WHOLE candidate pool "
                          "competing, so a restricted arm may inherit a weaker "
                          "ordering than it would get from a selector re-run "
                          "inside the arm. This biases restricted arms DOWNWARDS "
                          "and is the price of not spending ~45s x 15 folds per "
                          "arm on re-selection while two other jobs hold the CPU.",
            "size_note": "section 15 asks for top-60 and top-100; the harness "
                         "offers 30/60/120, so top-120 stands in for top-100 and "
                         "the substitution is recorded rather than hidden.",
        },
        "firewall": ctx["firewall"],
        "arms": arms,
        "paired_vs_full_clean": comparisons,
        "rule": {
            "text": "an arm replaces full_clean only if its mean paired difference "
                    "is positive and at least 2 of the 3 paired tests give p < 0.05. "
                    "A pool is not kept because a bank marked it, and the "
                    "alert-context block is not accepted on a large jump without a "
                    "timing review.",
            "verdicts": {k: ("REPLACES_FULL_CLEAN"
                             if v["mean_paired_diff"] > 0
                             and v["n_tests_below_0.05"] >= 2 else "NO_CHANGE")
                         for k, v in comparisons.items()},
        },
    }


# ==========================================================================
# stage: score  (section 54)
# ==========================================================================
def stage_score(ctx: dict[str, Any], args) -> dict[str, Any]:
    parts = {
        "ensemble": _read("nested_ensemble.json"),
        "seedbag": _read("nested_seed_bagging.json"),
        "posremoval": _read("nested_positive_removal.json"),
        "shift": _read("nested_shift_shield.json"),
    }
    missing = [k for k, v in parts.items() if v is None]
    comp: dict[str, Any] = {}

    if parts["ensemble"]:
        base = parts["ensemble"]["arms"]["best_single_inner"]
        comp["cv_stability"] = {
            "fold_ap_mean": base["fold_ap_mean"],
            "fold_ap_std_across_15_folds": base["fold_ap_std"],
            "note": "fold-level AP on ~13 positives is intrinsically noisy; this "
                    "spread is mostly sampling, not model instability",
        }
    if parts["seedbag"]:
        fam = args.stress_family if args.stress_family in parts["seedbag"]["families"] \
            else sorted(parts["seedbag"]["families"])[0]
        s = parts["seedbag"]["families"][fam]
        comp["seed_stability"] = {
            "family": fam,
            "mean_probability_std_across_seeds": s["mean_probability_std_across_seeds"],
            "mean_rank_std_across_seeds": s["mean_rank_std_across_seeds"],
            "fold_ap_std_single": s["fold_ap_std_single"],
            "fold_ap_std_bag": s["fold_ap_std_bag"],
        }
    if parts["posremoval"]:
        p = parts["posremoval"]
        comp["positive_removal_stability"] = p["grade_inputs"] | {
            "reference_fold_ap_mean": p["reference_fold_ap_mean"],
            "stressed_fold_ap_mean": p["stressed_fold_ap_mean"],
        }
        comp["feature_stability"] = {
            "feature_rank_correlation_mean": p["feature_rank_correlation_mean"],
        }
    if parts["shift"]:
        s = parts["shift"]
        comp["adversarial_validation_auc"] = s["adversarial_validation"]["auc_mean"]
        comp["ood_rate"] = s["ood"]["rate_mean"]
        comp["calibration_stability"] = {
            "ece_calibrated_mean": s["calibration"]["ece_calibrated_mean"],
            "ece_calibrated_std_across_folds":
                s["calibration"]["ece_calibrated_std_across_folds"],
        }
        comp["feature_stability"] = (comp.get("feature_stability") or {}) | {
            "class_counts": s["feature_shift"]["class_counts"],
        }

    # --- documented thresholds --------------------------------------------
    grade: str | None = None
    criteria: dict[str, Any] = {}
    pr = comp.get("positive_removal_stability")
    if pr:
        for level in ("HIGH", "MEDIUM"):
            t = ROBUSTNESS_THRESHOLDS[level]
            criteria[level] = {
                "positive_removal_pr_auc_rel_drop":
                    pr["positive_removal_pr_auc_relative_drop"]
                    <= t["positive_removal_pr_auc_rel_drop_max"],
                "positive_removal_pr_auc_std":
                    pr["positive_removal_pr_auc_std"]
                    <= t["positive_removal_pr_auc_std_max"],
                "prediction_rank_stability":
                    pr["prediction_rank_stability"]
                    >= t["prediction_rank_stability_min"],
            }
            if "adversarial_validation_auc" in comp:
                criteria[level]["folds_exchangeable"] = bool(
                    0.45 <= comp["adversarial_validation_auc"] <= 0.55)
        grade = ("HIGH" if all(criteria["HIGH"].values())
                 else "MEDIUM" if all(criteria["MEDIUM"].values()) else "LOW")

    return {
        "generated_utc": _now(), "sections": ["54"],
        "question": "How robust is this model to the things a hidden validation "
                    "set will change?",
        "design": design_block(args, ctx["folds"]),
        "firewall": ctx["firewall"],
        "missing_inputs": missing,
        "components_reported_separately": comp,
        "thresholds_used": ROBUSTNESS_THRESHOLDS,
        "extra_criterion": {"folds_exchangeable": "adversarial AUC in [0.45, 0.55]"},
        "criteria_by_level": criteria,
        "generalization_robustness": grade,
        "rule": "the components above are the report. The single HIGH/MEDIUM/LOW "
                "label is a summary of published thresholds only, it is not a new "
                "score, and it is never used to select a model.",
        "not_covered": [
            "no external / judge-uploaded dataset exists in this session, so the "
            "adversarial-validation component measures fold exchangeability "
            "WITHIN one dataset and says nothing about a distribution the model "
            "has not seen",
            "the locked test set is a HISTORICAL HOLDOUT and was not read here",
        ],
    }


# ==========================================================================
# main
# ==========================================================================
STAGE_FN = {
    "ensemble": (stage_ensemble, "nested_ensemble.json"),
    "seedbag": (stage_seedbag, "nested_seed_bagging.json"),
    "posremoval": (stage_posremoval, "nested_positive_removal.json"),
    "labelnoise": (stage_labelnoise, "nested_label_noise.json"),
    "shift": (stage_shift, "nested_shift_shield.json"),
    "families": (stage_families, "nested_feature_family_arms.json"),
    "score": (stage_score, "nested_generalization_score.json"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="all",
                    help="comma-separated: " + ",".join(STAGE_FN) + " or 'all'")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--n-feat", type=int, default=120)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--bag-families", default="xgboost,histgb")
    ap.add_argument("--stress-family", default="xgboost")
    ap.add_argument("--stress-rounds", type=int, default=12)
    ap.add_argument("--stress-fraction", type=float, default=0.125)
    ap.add_argument("--noise-weight", type=float, default=0.5)
    ap.add_argument("--arm-family", default="xgboost")
    args = ap.parse_args(argv)

    configure()
    wanted = list(STAGE_FN) if args.stages == "all" else args.stages.split(",")
    unknown = [s for s in wanted if s not in STAGE_FN]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}")

    t0 = time.time()
    frame = frame_mod.build_model_frame()
    dev = harness.dev_split(args.repeats)
    log.info("dev=%d rows (+%d, %.6f%%)  features=%d", len(dev.row_index),
             int(frame.y[dev.row_index].sum()),
             100 * float(frame.y[dev.row_index].mean()), len(frame.feature_names))

    folds = nested.build_outer_folds(frame, n_repeats=args.repeats,
                                     n_inner=args.inner)
    ctx: dict[str, Any] = {
        "frame": frame, "folds": folds,
        "valid_idx": {_fold_key(f): f.valid_idx for f in folds},
    }
    ctx["firewall"] = firewall_check(frame, folds)
    log.info("firewall verified: 0 of %d quarantined columns present",
             len(HARD_QUARANTINE))

    for name in wanted:
        fn, out_name = STAGE_FN[name]
        log.info("=== stage %s ===", name)
        t = time.time()
        payload = fn(ctx, args)
        payload["runtime_seconds"] = round(time.time() - t, 1)
        _write(out_name, payload)
        log.info("stage %s done in %.0fs", name, time.time() - t)

    log.info("total %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
