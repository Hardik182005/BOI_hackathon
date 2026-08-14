"""Section 31: what do the sensitive attributes actually buy the model?

`docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md` establishes that none of the
four sensitive columns is among the champion's 120 features, and it measures
the disparity that remains anyway. What it does not establish is the
counterfactual: **would the model be better if it had them?** That question
decides whether the gender exclusion costs detection, and it had never been
run. Answering it is the difference between a policy and a preference.

Design
------
The arms are scored on the **same 15 nested outer folds** as every other paired
comparison in the programme, so these numbers sit beside the family, subset and
resampling arms without a second unexplained difference between them.

An arm that forces a sensitive column in takes the fold's top 120 and adds the
column, giving a pool of 121-124 rather than 120. Substituting it for the
120th-ranked feature would have been the alternative, and it was rejected: that
design removes a feature the selector *did* want at the same time as it adds one
it did not, so a null result could be read either way. Adding one column to 120
is a confound worth one column; a swap is a confound worth two.

Gender
------
`F3892` never enters `build_model_frame()` - the firewall removes it, and
`tests/unit/test_evidence_bookkeeping.py` asserts it never reaches the final
selection. Testing it therefore requires deliberately re-admitting it, which
the frame builder supports for exactly this purpose
(``build_model_frame(include_gender=True)``, documented "ONLY for the fairness
ablation").

That re-admitted frame is used for one thing only: to lift the single encoded
`F3892` column and append it to the fold matrices. The fold ranking is never
rebuilt, so the pool the selector chose is held byte-identical and the only
difference between ``baseline`` and ``gender_forced`` is the presence of the
gender column. The run asserts that the re-admitted frame differs from the
shipped frame in that one column and nowhere else, and records the assertion.

The price of that control is a question this run does **not** answer: whether
stability selection would have *chosen* gender if it had been offered. It
answers the stronger operational question instead - whether a model that has
gender beats a model that does not.

    .venv/Scripts/python.exe -m muleguard.cli.fairness_ablation

Nothing here selects a model, and the locked test set is never read. The output
is a table, a slice audit, and a verdict rule fixed before the run: a sensitive
attribute is re-admitted only if its arm's mean paired difference is positive
AND the sign test rejects at 0.05. Anything else keeps the exclusion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.features import frame as frame_mod
from muleguard.logging import configure, get_logger
from muleguard.models import harness, nested
from muleguard.models import nested_experiments as nx
from muleguard.models.paired import paired_report
from muleguard.utils import save_json

log = get_logger("cli.fairness_ablation")

OUT = settings.METRICS_DIR / "fairness_ablation.json"
BASELINE = "baseline"

GENDER = "F3892"
AREA = "F3890"
OCCUPATION = "F3891"
AGE = "F3894"

#: The four columns Description.xlsx marks sensitive, with the kind §31 names.
SENSITIVE: dict[str, dict[str, str]] = {
    GENDER: {"variable": "GENDER", "kind": "gender",
             "meaning": "customer gender"},
    AREA: {"variable": "AREA_CATEGORY", "kind": "geography",
           "meaning": "area category of the customer"},
    OCCUPATION: {"variable": "CUST_OCCP", "kind": "occupation",
                 "meaning": "occupation code"},
    AGE: {"variable": "AGE_IN_YRS", "kind": "age",
          "meaning": "customer age as of alert date"},
}

IN_FRAME = (AREA, OCCUPATION, AGE)   # gender is removed upstream; see module docstring

ARMS: dict[str, dict[str, Any]] = {
    BASELINE: {
        "drop": (), "force": (),
        "what": "the shipped pool: the fold's own top 120, untouched"},
    "sensitive_excluded": {
        "drop": IN_FRAME, "force": (),
        "what": "geography, occupation and age struck from the fold ranking "
                "before the top 120 is taken - a no-op in any fold that did "
                "not rank one of them inside 120, which is the finding"},
    "geography_forced": {
        "drop": (), "force": (AREA,),
        "what": "top 120 plus F3890 AREA_CATEGORY"},
    "occupation_forced": {
        "drop": (), "force": (OCCUPATION,),
        "what": "top 120 plus F3891 CUST_OCCP"},
    "age_forced": {
        "drop": (), "force": (AGE,),
        "what": "top 120 plus F3894 AGE_IN_YRS"},
    "gender_forced": {
        "drop": (), "force": (GENDER,),
        "what": "top 120 plus F3892 GENDER, re-admitted through the frame "
                "builder's fairness-ablation switch"},
    "sensitive_forced": {
        "drop": (), "force": IN_FRAME,
        "what": "top 120 plus all three admissible demographics"},
    "all_four_forced": {
        "drop": (), "force": IN_FRAME + (GENDER,),
        "what": "top 120 plus every column Description.xlsx marks sensitive"},
}


# --- slice audit --------------------------------------------------------------

#: A selection rate over fewer rows than this is one alert wide or worse, so it
#: describes the rounding and not the group.
MIN_ROWS_FOR_RATE = 50
#: Matches the companion audit's rule. Below three positives a single case moves
#: recall by more than 0.33, which is larger than the whole between-group spread
#: the number would be used to discuss.
MIN_POSITIVES_FOR_RECALL = 3

AGE_BINS = ((-np.inf, 25, "< 25"), (25, 35, "25-34"), (35, 45, "35-44"),
            (45, 55, "45-54"), (55, 65, "55-64"), (65, np.inf, "65+"))
NOT_RECORDED = "(not recorded)"


def _age_groups(v: np.ndarray) -> np.ndarray:
    """Bin ages the way the companion audit binned them, so the two compare.

    Negative ages exist in this extract and land in ``< 25``. They are counted
    separately in the payload rather than quietly dropped, because a bin that
    silently absorbs impossible values is a bin that hides a data-quality
    problem behind a fairness number.
    """
    out = np.full(len(v), NOT_RECORDED, dtype=object)
    ok = np.isfinite(v)
    for lo, hi, label in AGE_BINS:
        out[ok & (v >= lo) & (v < hi)] = label
    return out


def _slice_table(y: np.ndarray, score: np.ndarray, groups: np.ndarray, *,
                 budget: int) -> dict[str, Any]:
    """Selection rate and recall@budget per group, with suppression recorded.

    A suppressed group still reports its size. Hiding the fact that a group was
    too small to measure is a different failure from reporting a number that
    cannot carry weight, and this reports neither.
    """
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-score, kind="stable")[:budget]] = True

    rows: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    labels = sorted(set(groups.tolist()), key=lambda g: (g == NOT_RECORDED, str(g)))
    for label in labels:
        m = groups == label
        n = int(m.sum())
        n_pos = int(y[m].sum())
        if n < MIN_ROWS_FOR_RATE:
            suppressed.append({
                "group": label, "n": n, "positives": n_pos,
                "reason": f"n < {MIN_ROWS_FOR_RATE}; a selection rate here is "
                          f"one alert wide"})
            continue
        entry: dict[str, Any] = {
            "group": label, "n": n, "positives": n_pos,
            "n_flagged": int(flagged[m].sum()),
            "selection_rate": round(float(flagged[m].mean()), 5),
            "group_prevalence": round(float(y[m].mean()), 5),
        }
        if n_pos >= MIN_POSITIVES_FOR_RECALL:
            entry[f"recall_at_{budget}"] = round(
                float(flagged[m & (y == 1)].mean()), 5)
        else:
            entry[f"recall_at_{budget}"] = None
            entry["recall_suppressed"] = (
                f"{n_pos} positives < {MIN_POSITIVES_FOR_RECALL}")
        rows.append(entry)

    rows.sort(key=lambda r: -r["selection_rate"])
    return {"groups": rows, "suppressed": suppressed}


def _attribute_groups(dev_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Group labels per dev row, read from the raw frame, not from the model.

    Deliberately taken from the source columns rather than the feature matrix:
    an audit of who gets flagged has to be able to slice on an attribute the
    model was never given, and gender is exactly that attribute.
    """
    df: pl.DataFrame = frame_mod.raw_with_meta()
    out: dict[str, np.ndarray] = {}
    for col in (GENDER, AREA, OCCUPATION):
        s = df[col].to_numpy()[dev_rows]
        out[SENSITIVE[col]["kind"]] = np.array(
            [NOT_RECORDED if v is None or v != v else str(v) for v in s],
            dtype=object)
    age = df[AGE].cast(pl.Float64).to_numpy()[dev_rows]
    out["age"] = _age_groups(age)
    out["_n_impossible_age"] = int((np.isfinite(age) & (age < 0)).sum())
    return out


# --- the run ------------------------------------------------------------------

def _fit_predict(n_jobs: int):
    from muleguard.cli import nested_cv as nc

    nc.N_JOBS = n_jobs
    return nc.xgb_factory({})


def _gender_column(frame: frame_mod.ModelFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """The encoded F3892 column, plus proof that lifting it changed nothing else.

    The isolation check is the whole licence for injecting a column into a fold
    matrix whose ranking was computed without it: if the re-admitted frame is
    identical to the shipped frame everywhere except this one column, then the
    injected arm differs from the baseline in that column and in nothing else.
    """
    g = frame_mod.build_model_frame(include_gender=True)
    j = g.feature_names.index(GENDER)
    shared = [i for i in range(len(g.feature_names)) if i != j]
    isolated = (
        [n for i, n in enumerate(g.feature_names) if i != j] == frame.feature_names
        and np.array_equal(g.X[:, shared], frame.X, equal_nan=True))
    if not isolated:
        raise RuntimeError(
            "re-admitting F3892 perturbed other columns; the injected gender "
            "arm would no longer be a controlled comparison")
    col = np.asarray(g.X[:, j], dtype=np.float32)
    return col, {
        "source": "muleguard.features.frame.build_model_frame(include_gender=True)",
        "isolation_asserted": True,
        "n_distinct_encoded_values": int(len(np.unique(col[np.isfinite(col)]))),
        "n_missing": int((~np.isfinite(col)).sum()),
    }


def run(repeats: int = 3, inner: int = 4, n_feat: int = 120,
        budget: int = 100, n_jobs: int = 2) -> dict[str, Any]:
    configure()
    t0 = time.time()
    frame = frame_mod.build_model_frame()
    dev = harness.dev_split(repeats)
    y_dev = np.asarray(frame.y)[dev.row_index]
    log.info("dev=%d rows (+%d)", len(dev.row_index), int(y_dev.sum()))

    gender_dev, gender_note = _gender_column(frame)
    gender_dev = gender_dev[dev.row_index]

    folds = nested.build_outer_folds(frame, n_repeats=repeats, n_inner=inner)
    fp = _fit_predict(n_jobs)

    # Where each sensitive column sits in each fold's own ranking. This is the
    # number that decides whether `sensitive_excluded` can differ from baseline
    # at all, so it is recorded before any arm is scored.
    ranks: dict[str, list[int | None]] = {c: [] for c in IN_FRAME}
    for f in folds:
        pos = {n: i for i, n in enumerate(f.kept_features)}
        order = {c: r for r, c in enumerate(f.ranked_features)}
        for c in IN_FRAME:
            j = pos.get(c)
            ranks[c].append(None if j is None else order.get(j))

    ap_by_arm: dict[str, list[float]] = {}
    oof_by_arm: dict[str, np.ndarray] = {}
    detail: dict[str, dict[str, Any]] = {}

    for name, spec in ARMS.items():
        aps: list[float] = []
        sizes: list[int] = []
        changed = 0
        oof = np.full((repeats, len(dev.row_index)), np.nan)

        for f in folds:
            pos = {n: i for i, n in enumerate(f.kept_features)}
            drop = {pos[c] for c in spec["drop"] if c in pos}
            ranked = [c for c in f.ranked_features if c not in drop]
            cols = set(ranked[:n_feat])
            forced_in_frame = {pos[c] for c in spec["force"] if c in pos}
            base_cols = set(f.ranked_features[:n_feat])
            cols = np.sort(np.fromiter(cols | forced_in_frame, dtype=int))

            Xtr, Xva = f.Xtr[:, cols], f.Xva[:, cols]
            injected = [c for c in spec["force"] if c not in pos]
            for c in injected:
                if c != GENDER:      # nothing else is ever outside the frame
                    raise RuntimeError(f"{c} is neither in the frame nor injectable")
                Xtr = np.column_stack([Xtr, gender_dev[f.train_idx]])
                Xva = np.column_stack([Xva, gender_dev[f.valid_idx]])

            if set(cols) != base_cols or injected:
                changed += 1
            s = fp(Xtr, f.ytr, Xva, harness.fold_seed(f.repeat, f.fold))
            aps.append(float(nx.fold_metrics(f.yva, s)["ap"]))
            sizes.append(int(Xtr.shape[1]))
            oof[f.repeat, f.valid_idx] = s

        if np.isnan(oof).any():
            raise RuntimeError(f"{name}: outer folds left dev rows unscored")

        ap_by_arm[name] = aps
        oof_by_arm[name] = oof.mean(axis=0)
        detail[name] = {
            "what": spec["what"],
            "columns_struck_from_the_ranking": list(spec["drop"]),
            "columns_forced_into_the_pool": list(spec["force"]),
            "n_features_per_fold": sorted(set(sizes)),
            "folds_whose_pool_differs_from_baseline": changed,
            "fold_ap_mean": round(float(np.mean(aps)), 5),
            "fold_ap_std": round(float(np.std(aps)), 5),
            "fold_ap": [round(a, 5) for a in aps],
        }
        log.info("arm %-20s AP=%.5f (+-%.5f)  n_feat=%s  pools_changed=%d/%d",
                 name, detail[name]["fold_ap_mean"], detail[name]["fold_ap_std"],
                 detail[name]["n_features_per_fold"], changed, len(folds))

    base = ap_by_arm[BASELINE]
    comparisons = {k: paired_report(base, v, baseline_name=BASELINE,
                                    arm_name=k).to_dict()
                   for k, v in ap_by_arm.items() if k != BASELINE}

    readmit = [k for k, c in comparisons.items()
               if k != "sensitive_excluded"
               and c.get("mean_paired_diff", 0) > 0
               and (c.get("sign_test_p_two_sided") or 1) < 0.05]

    # --- slice audit -------------------------------------------------------
    groups = _attribute_groups(dev.row_index)
    n_bad_age = groups.pop("_n_impossible_age")
    slices = {
        arm: {attr: _slice_table(y_dev, oof_by_arm[arm], g, budget=budget)
              for attr, g in groups.items()}
        for arm in ARMS
    }
    overall = {arm: {
        "selection_rate": round(budget / len(y_dev), 5),
        f"recall_at_{budget}": round(
            float(nx.recall_at_k(y_dev, oof_by_arm[arm], budget)), 5)}
        for arm in ARMS}

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec": "section 31 - sensitive attribute testing (model ablations)",
        "question": "Does giving the model a protected or demographic attribute "
                    "improve detection enough to justify using it?",
        "sensitive_columns": [
            {"column": c, **SENSITIVE[c],
             "in_model_frame": c in IN_FRAME,
             "how_tested": ("ablation on the frame column" if c in IN_FRAME
                            else "re-admitted and injected; see gender_handling")}
            for c in (GENDER, AREA, OCCUPATION, AGE)
        ],
        "design": {
            "protocol": "nested outer folds",
            "outer_folds": len(folds),
            "n_repeats": repeats, "n_inner_folds": inner,
            "family": "xgboost at default configuration (no tuning inside an arm)",
            "n_features": n_feat,
            "feature_ranking": "the fold's own outer-train ranking, computed "
                               "once and shared by every arm; no arm re-ranks",
            "forcing_rule": "a forced column is ADDED to the top 120 rather than "
                            "swapped for the 120th, so a null result is not "
                            "confounded by the loss of a wanted feature",
            "locked_test_read": False,
            "seed": settings.GLOBAL_SEED,
        },
        "gender_handling": {
            **gender_note,
            "why_injection": "F3892 is removed by the firewall before the frame "
                             "exists, so it cannot be ranked. Rebuilding the "
                             "ranking with it admitted would change every "
                             "column's rank and stop the arms being paired on "
                             "identical pools; injecting the single column "
                             "holds everything else fixed.",
            "not_answered_by_this_run": "whether stability selection would have "
                                        "CHOSEN gender if offered. This run "
                                        "answers whether a model that has it "
                                        "beats one that does not, which is the "
                                        "question the policy turns on.",
        },
        "rank_of_sensitive_column_in_fold_ranking": {
            c: {"per_fold": ranks[c],
                "folds_ranked_inside_top_120": sum(
                    1 for r in ranks[c] if r is not None and r < n_feat),
                "note": "null means the selector gave it zero importance in "
                        "every inner fold, so it never entered the ranking"}
            for c in IN_FRAME
        },
        "arms": detail,
        "paired_vs_baseline": comparisons,
        "decision_rule": {
            "text": "a sensitive attribute is re-admitted to the accepted "
                    "primary model only if its arm's mean paired difference is "
                    "positive AND the sign test rejects at 0.05. Fixed before "
                    "the run. Anything else - including a positive mean that "
                    "the sign test cannot separate from noise - is "
                    "KEEP_EXCLUSION.",
            "arms_meeting_the_rule": readmit,
            "note": "sensitive_excluded is scored and reported but is not a "
                    "candidate for re-admission; it is the control that shows "
                    "what removing the attributes costs.",
        },
        "verdict": ("READMIT " + ", ".join(readmit)) if readmit else "KEEP_EXCLUSION",
        "slice_audit": {
            "budget": budget,
            "score": f"mean of the {repeats} outer-fold OOF score matrices, "
                     f"which is how the companion audit ranks dev rows",
            "group_source": "raw columns, not model features - so gender can be "
                            "sliced on even though the model never sees it",
            "reporting_rule": {
                "selection_rate": f"reported only when the group has >= "
                                  f"{MIN_ROWS_FOR_RATE} rows",
                "recall": f"reported only when the group has >= "
                          f"{MIN_POSITIVES_FOR_RECALL} positives, matching "
                          f"docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md so the "
                          f"two documents can be read together",
                "suppressed_groups_still_report_their_size": True,
            },
            "impossible_age_rows": n_bad_age,
            "impossible_age_note": "ages below zero exist in this extract and "
                                   "fall in the '< 25' bin; counted here so the "
                                   "bin does not hide them",
            "overall": overall,
            "by_arm": slices,
        },
        "interpretation": (
            "the sensitive_excluded arm is the control that matters most: if it "
            "is identical to baseline on every fold, then the exclusion policy "
            "costs nothing because the selector never wanted the attributes in "
            "the first place, and no amount of paired statistics is needed to "
            "say so"),
        "runtime_s": round(time.time() - t0, 1),
    }
    save_json(payload, OUT)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--n-feat", type=int, default=120)
    ap.add_argument("--budget", type=int, default=100,
                    help="alert budget for the slice audit")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    p = run(a.repeats, a.inner, a.n_feat, a.budget, a.n_jobs)

    print(f"\n{'arm':<22} {'PR-AUC':>8} {'+-':>7} {'delta':>9} {'sign p':>8} "
          f"{'changed':>8}")
    b = p["arms"][BASELINE]["fold_ap_mean"]
    for name, d in p["arms"].items():
        c = p["paired_vs_baseline"].get(name, {})
        dl = "" if name == BASELINE else f"{d['fold_ap_mean'] - b:+.5f}"
        sp = c.get("sign_test_p_two_sided")
        print(f"{name:<22} {d['fold_ap_mean']:>8.5f} {d['fold_ap_std']:>7.5f} "
              f"{dl:>9} {sp if sp is not None else '-':>8} "
              f"{d['folds_whose_pool_differs_from_baseline']:>8}")
    print(f"\nverdict: {p['verdict']}  ({p['runtime_s']}s)")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
