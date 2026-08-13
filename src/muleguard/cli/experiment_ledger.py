"""Experiment ledger (final-validation §60) - one row per experiment, no forgotten runs.

Run::

    .venv/Scripts/python.exe -m muleguard.cli.experiment_ledger

Writes ``artifacts/experiments/experiment_ledger.csv`` with the fifteen columns
§60 names, and nothing else - the column set is fixed by the spec and is not
extended here.

Why a registry and not a scanner
--------------------------------
The 30-odd JSON files under ``artifacts/metrics/`` have thirty-odd different
shapes. A generic scanner that guessed which number was "the" metric would be
wrong silently and often - it would read a threshold as an AP, or an inner score
as an outer one - and a ledger that quietly misreports a number is worse than no
ledger. So every source is registered explicitly with the path to its own
metric, and the registry is the thing under review.

The completeness guarantee is mechanical rather than clerical: every ``*.json``
under ``artifacts/metrics/`` must be claimed by some entry below, either as an
experiment or as an explicit non-experiment with a reason. An unclaimed file is
reported as ``UNREGISTERED_REVIEW`` and the command exits non-zero. That is the
"no forgotten experiments" rule enforced by the code rather than by memory.

Artifacts that are registered but do not exist yet are emitted as ``PENDING_RUN``
rather than dropped, so a run that has not happened is visible as a gap instead
of being invisible.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.utils import git_info, load_json

log = get_logger("cli.experiment_ledger")

OUT_DIR = settings.ARTIFACTS_DIR / "experiments"
OUT_CSV = OUT_DIR / "experiment_ledger.csv"

COLUMNS = [
    "experiment_id", "timestamp", "git_sha", "dataset_sha", "description_sha",
    "model", "feature_set", "quarantines", "cv_scheme", "seed",
    "primary_metric", "metric_value", "status", "reason", "artifact_path",
]

#: The 13 columns the feature-availability firewall removes before any model
#: sees the frame. Spelled out rather than abbreviated so a reader of the CSV
#: never has to come back to this file to learn what was excluded.
FIREWALL_13 = ("F2230|F3892|F3898|F3899|F3912|F3913|F3914|F3915|F3916|F3917|"
               "F3918|F3924|__UNNAMED__0")

# Cross-validation schemes, named once so two rows never describe the same
# protocol in two different ways.
CV_FLAT = "flat stratified 5-fold x 3 repeats"
CV_NESTED = "nested: outer stratified 5-fold x 3 repeats, inner 4-fold"
CV_NESTED_PRELIM = "nested: outer stratified 5-fold x 1 repeat, inner 4-fold"
CV_HOLDOUT = "single held-out split (historical)"
CV_NONE = "not a cross-validated experiment"

Row = dict[str, Any]


@dataclass(frozen=True)
class Source:
    """One artifact file and the rows it contributes to the ledger."""

    artifact: str                       # relative to the repo root
    expand: Callable[[Any], list[Row]]  # loaded json -> partial rows
    kind: str = "EXPERIMENT"            # EXPERIMENT | NOT_AN_EXPERIMENT
    defaults: dict[str, Any] = field(default_factory=dict)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _point(value: Any) -> Any:
    """Some artifacts store a metric as a float, others as {point, ci_low, ...}."""
    if isinstance(value, dict):
        return value.get("point", value.get("mean"))
    return value


def _mean(values: Any) -> Any:
    if isinstance(values, list) and values:
        return sum(values) / len(values)
    return None


def _timestamp(payload: Any, path: Path) -> str:
    """The artifact's own recorded time, or its mtime when it records none."""
    if isinstance(payload, dict):
        for key in ("generated_utc", "written_utc", "graded_utc", "updated_utc",
                    "created_utc"):
            if isinstance(payload.get(key), str):
                return payload[key]
        prov = payload.get("provenance")
        if isinstance(prov, dict) and isinstance(prov.get("generated_utc"), str):
            return prov["generated_utc"]
    if path.exists():
        return (dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
                .isoformat(timespec="seconds"))
    return ""


# --------------------------------------------------------------------------
# expanders - each returns the rows its artifact is responsible for
# --------------------------------------------------------------------------
#: Generation-1 ran before the availability firewall existed. Anything it
#: produced is evidence about a model that could see quarantined columns.
PRE_FIREWALL_MARKERS = ("retired", "pre_firewall", "catboost_tuned_top60")


def _is_pre_firewall(name: str) -> bool:
    low = (name or "").lower()
    return any(marker in low for marker in PRE_FIREWALL_MARKERS)


def _tournament_v2(d: Any) -> list[Row]:
    rows = []
    for name, m in d["models"].items():
        rows.append({
            "model": name,
            "feature_set": m.get("feature_set", name.split("_top_")[-1]),
            "primary_metric": "OOF PR-AUC (mean over repeats)",
            "metric_value": m.get("pr_auc_mean"),
            "status": "SUPERSEDED",
            "reason": "flat protocol; feature selection pooled across dev folds. "
                      "Superseded by the nested tournament as the selection estimate",
        })
    return rows


def _nested_cv(d: Any) -> list[Row]:
    design = d.get("design", {})
    outer = str(design.get("outer", ""))
    prelim = "x 1 repeats" in outer
    rows = []
    for m in d.get("leaderboard", []):
        rows.append({
            "model": m["model"],
            "feature_set": f"top_{m.get('feature_size_mode', '')}",
            "cv_scheme": CV_NESTED_PRELIM if prelim else CV_NESTED,
            "primary_metric": "nested outer-fold PR-AUC (mean)",
            "metric_value": m.get("pr_auc_mean"),
            "status": "SUPERSEDED" if prelim else "ACCEPTED",
            "reason": ("1 repeat only - under-powered, kept as the optimism-gap "
                       "reference" if prelim else
                       "primary unbiased estimate for this research cycle"),
        })
    return rows


def _metric_battery(d: Any) -> list[Row]:
    rows = []
    for key, run in d.get("runs", {}).items():
        protocol, _, model = key.partition(":")
        # ranking.pr_auc is {mean, std, min, max, per_repeat, n_repeats}; the
        # mean over repeats is the statistic the tournament ranked on.
        headline = run.get("ranking", {}).get("pr_auc", {})
        value = headline.get("mean") if isinstance(headline, dict) else None
        status = {"FLAT": "SUPERSEDED", "NESTED": "ACCEPTED",
                  "NESTED_PRELIMINARY": "SUPERSEDED",
                  "HOLDOUT_REFERENCE": "REFERENCE_ONLY"}.get(protocol, "REVIEW")
        if _is_pre_firewall(model):
            # A high number from a model that could see the quarantined columns
            # must never sit in the ledger looking like current behaviour.
            status = "REJECTED_LEAKAGE"
        rows.append({
            "model": model,
            "feature_set": model.split("_top_")[-1] if "_top_" in model else "",
            "cv_scheme": {"FLAT": CV_FLAT, "NESTED": CV_NESTED,
                          "NESTED_PRELIMINARY": CV_NESTED_PRELIM,
                          "HOLDOUT_REFERENCE": CV_HOLDOUT}.get(protocol, ""),
            "primary_metric": f"PR-AUC with bootstrap 95% CI ({protocol})",
            "metric_value": value,
            "status": status,
            "reason": "full metric battery, bootstrap CIs; "
                      + ("primary" if key == d.get("primary_run_key")
                         else "secondary reading"),
        })
    return rows


def _f3912(d: Any) -> list[Row]:
    return [
        {"model": d["without_f3912"]["model"], "feature_set": "clean baseline",
         "primary_metric": "OOF PR-AUC",
         "metric_value": d["without_f3912"]["pr_auc_mean"],
         "status": "ACCEPTED",
         "reason": "leakage-free control arm for the F3912 probe"},
        {"model": d["with_f3912"]["model"], "feature_set": "clean baseline + F3912",
         "quarantines": FIREWALL_13.replace("F3912|", "") + " (F3912 deliberately re-admitted)",
         "primary_metric": "OOF PR-AUC",
         "metric_value": d["with_f3912"]["pr_auc_mean"],
         "status": "REJECTED_LEAKAGE",
         "reason": "REJECTED LEAKAGE / AVAILABILITY EXPERIMENT - NOT A VALID "
                   "COMPETITION RESULT. Post-resolution field; exists only as "
                   "evidence of the inflation it causes"},
    ]


def _alert_context(d: Any) -> list[Row]:
    rows = []
    for name, v in d.get("variants", {}).items():
        rows.append({
            "model": d.get("model_family", ""),
            "feature_set": name,
            "primary_metric": "OOF PR-AUC (mean over repeats)",
            "metric_value": v.get("pr_auc_mean") if isinstance(v, dict) else v,
            "status": "ACCEPTED",
            "reason": f"alert-context ablation; delta {_fmt(d.get('delta'))}. "
                      + str(d.get("verdict", "")),
        })
    return rows


def _family_dropout(d: Any) -> list[Row]:
    return [{
        "model": d.get("model", ""),
        "feature_set": f"champion set minus {f['family_removed']}",
        "primary_metric": "OOF PR-AUC after family removal",
        "metric_value": f.get("pr_auc"),
        "status": "DIAGNOSTIC",
        "reason": f"semantic-family dropout, relative drop {_fmt(f.get('relative_drop'))}"
                  + (" - worst family" if f["family_removed"] == d.get("worst_family") else ""),
    } for f in d.get("per_family", [])]


def _advanced(d: Any) -> list[Row]:
    rows = []
    for name, c in d.get("challengers", {}).items():
        c = c if isinstance(c, dict) else {}
        ran = c.get("pr_auc_mean") is not None
        rows.append({
            "model": name, "feature_set": "",
            "primary_metric": "OOF PR-AUC" if ran else "not scored",
            "metric_value": c.get("pr_auc_mean"),
            "status": "CHALLENGER_ONLY" if ran else "NOT_RUN",
            "reason": (f"{c.get('status', '')}: {c.get('reason', '')}"[:150]
                       if not ran else
                       f"{c.get('n_repeats')} repeat(s), "
                       f"{_fmt(c.get('runtime_seconds'))}s; challenger only - "
                       "promotion needs repeated nested evidence (§52)"),
        })
    return rows


def _missingness(d: Any) -> list[Row]:
    dec = d.get("decision", {})
    return [
        {"model": d["design"]["family"], "feature_set": "top_120, no missingness signature",
         "cv_scheme": CV_NESTED, "primary_metric": "nested PR-AUC (mean)",
         "metric_value": d["without"]["pr_auc_mean"], "status": "DIAGNOSTIC",
         "reason": "control arm of the missingness ablation"},
        {"model": d["design"]["family"], "feature_set": "top_120 + missingness signature",
         "cv_scheme": CV_NESTED, "primary_metric": "nested PR-AUC (mean)",
         "metric_value": d["with"]["pr_auc_mean"],
         "status": "PROVISIONAL" if dec.get("keep") else "REJECTED",
         "reason": f"paired gain {_fmt(d['paired'].get('mean_gain'))} on identical "
                   f"folds, sign-test p {_fmt(d['paired'].get('sign_test_p_two_sided'))}; "
                   "provisional pending the no-FEES confirmation arm (§9.4)"},
    ]


def _missingness_nofees(d: Any) -> list[Row]:
    rows = _missingness(d)
    for r in rows:
        r["feature_set"] += ", FEES_AND_CHARGES excluded"
        r["reason"] = ("confirmation arm: does the gain survive without the "
                       "untraceable FEES_AND_CHARGES cohort columns")
    return rows


def _seed_variance(d: Any) -> list[Row]:
    return [{
        "model": d.get("model", ""), "feature_set": "champion set",
        "primary_metric": "PR-AUC mean over seeds (spread reported)",
        "metric_value": d.get("pr_auc_mean"),
        "status": "DIAGNOSTIC",
        "reason": f"{d.get('n_seeds')} seeds, unpaired spread {_fmt(d.get('spread'))} "
                  "- this is the unpaired noise floor, not a yardstick for paired tests",
    }]


def _stability_stress(d: Any) -> list[Row]:
    return [{
        "model": d.get("model", ""), "feature_set": "champion set",
        "primary_metric": "PR-AUC under positive removal (mean)",
        "metric_value": d.get("positive_removal_pr_auc_mean"),
        "status": "DIAGNOSTIC",
        "reason": f"{d.get('rounds')} rounds removing {_fmt(d.get('removal_fraction'))} "
                  f"of training positives; relative drop "
                  f"{_fmt(d.get('positive_removal_pr_auc_relative_drop'))}",
    }]


def _ensemble(d: Any) -> list[Row]:
    """One row per blend. mean_pr_auc is keyed by blend, including best_single."""
    best = d.get("best_single_model", "")
    wins = d.get("wins_over_best_single", {})
    decision = str(d.get("decision", ""))[:110]
    rows = []
    for blend, value in (d.get("mean_pr_auc") or {}).items():
        baseline = blend == "best_single"
        rows.append({
            "model": best if baseline else f"{blend} of {', '.join(d.get('members', []))}",
            "feature_set": "member feature sets",
            "primary_metric": "OOF PR-AUC (mean over repeats)",
            "metric_value": value,
            "status": "REFERENCE_ONLY" if baseline else "REJECTED",
            "reason": ("baseline for the ensemble comparison - the best single model"
                       if baseline else
                       f"wins {wins.get(blend)}/{d.get('n_repeats')} repeats vs "
                       f"{best}. Flat protocol; re-judged on nested folds by "
                       f"nested_ses (§53). {decision}"),
        })
    return rows


def _ensemble_decision(d: Any) -> list[Row]:
    return [{
        "model": "probability ensemble (gen-1)",
        "feature_set": ", ".join(d.get("base_models", [])),
        "primary_metric": "OOF PR-AUC (mean over repeats)",
        "metric_value": _mean(d.get("ensemble_ap_by_repeat")),
        "status": "ACCEPTED" if d.get("accepted") else "REJECTED",
        "reason": "generation-1 ensemble decision, superseded by ensemble_v2",
    }]


def _promotion(d: Any) -> list[Row]:
    det = d.get("promoted_detail", {})
    return [{
        "model": d.get("promoted", ""), "feature_set": det.get("feature_set", ""),
        "primary_metric": "generalization score (see rule)",
        "metric_value": det.get("generalization_score"),
        "status": "ACCEPTED",
        "reason": "promoted champion under " + str(d.get("rule", ""))[:120],
    }]


def _challenger_review(d: Any) -> list[Row]:
    return [{
        "model": f"{d['challenger']['model']} vs {d['champion']['model']}",
        "feature_set": "champion vs challenger sets",
        "primary_metric": "OOF PR-AUC (challenger)",
        "metric_value": d["challenger"].get("pr_auc_mean"),
        "status": "ACCEPTED" if d.get("all_gates_passed") else "REJECTED",
        "reason": "challenger review gates; " + str(d.get("rule", ""))[:120],
    }]


def _holdout(d: Any) -> list[Row]:
    rows = []
    cur = d.get("current_champion", {})
    if cur:
        rows.append({
            "model": cur.get("model", ""), "feature_set": "champion set",
            "cv_scheme": CV_HOLDOUT, "primary_metric": "holdout PR-AUC",
            "metric_value": _point(cur.get("pr_auc")), "status": "REFERENCE_ONLY",
            "reason": "HISTORICAL HOLDOUT - already viewed, no longer an unbiased "
                      "selection set (§9)",
        })
    ret = d.get("retired_run", {}).get("metrics", {})
    if ret:
        rows.append({
            "model": "retired_gen1_pre_firewall_stack", "feature_set": "pre-firewall",
            "cv_scheme": CV_HOLDOUT, "primary_metric": "holdout PR-AUC",
            "metric_value": _point(ret.get("pr_auc")), "status": "REJECTED_LEAKAGE",
            "reason": "pre-firewall generation; " + str(d["retired_run"].get("warning", ""))[:120],
        })
    return rows


def _locked_test(d: Any) -> list[Row]:
    # The split string names the scorer that produced the file; both spec-named
    # locked-test artifacts still hold the retired generation-1 run.
    split = str(d.get("split", ""))
    model = split.split("production scorer")[-1].strip(" -") if "production scorer" in split \
        else d.get("model", "champion at time of run")
    retired = _is_pre_firewall(model) or _is_pre_firewall(split)
    return [{
        "model": model, "feature_set": "pre-firewall" if retired else "champion set",
        "cv_scheme": CV_HOLDOUT, "primary_metric": "locked-test PR-AUC",
        "metric_value": _point(d.get("pr_auc")),
        "status": "REJECTED_LEAKAGE" if retired else "REFERENCE_ONLY",
        "reason": ("RETIRED generation-1 run on a pre-firewall feature pool - not a "
                   "description of current behaviour, and not a competition result"
                   if retired else
                   "HISTORICAL HOLDOUT - FOR REFERENCE ONLY, never tuned against (§9)"),
    }]


def _lens_stack(d: Any) -> list[Row]:
    report = d.get("oof_calibrated_report") or {}
    winner = d.get("winner", "")
    retired = _is_pre_firewall(winner)
    return [{
        "model": winner, "feature_set": f"{d.get('n_features_selected', '')} selected",
        "primary_metric": "calibrated OOF PR-AUC (policy artifact)",
        "metric_value": _point(report.get("pr_auc")),
        "status": "SUPERSEDED" if retired else "ACCEPTED",
        "reason": (f"generation-1 policy stack, superseded by {d.get('supersedes') or 'v2'} "
                   "- pre-firewall winner" if retired else
                   f"calibration {d.get('calibration_selection', '')}, policy version "
                   f"{d.get('policy_thresholds', {}).get('policy_version', '')}"),
    }]


def _shield(d: Any) -> list[Row]:
    return [{
        "model": d.get("champion", ""), "feature_set": "champion set",
        "primary_metric": "champion OOF PR-AUC (shield reference)",
        "metric_value": d.get("champion_oof_pr_auc"),
        "status": "RELEASE_BLOCKER" if d.get("release_blocker") else "ACCEPTED",
        "reason": "validation shield: feature stability, shift-prone ablation, "
                  "leakage flags. " + str(d.get("release_blocker_reason") or "no blocker"),
    }]


def _robustness_grade(d: Any) -> list[Row]:
    return [{
        "model": "champion", "feature_set": "champion set",
        "primary_metric": "hidden-validation robustness grade",
        "metric_value": d.get("hidden_validation_robustness"),
        "status": "DIAGNOSTIC",
        "reason": "graded against published thresholds; limited by "
                  + ", ".join(d.get("limiting_criteria", []) or ["nothing"]),
    }]


def _label_noise(d: Any) -> list[Row]:
    pln = d.get("possible_label_noise", {})
    return [{
        "model": ", ".join(d.get("consensus_models", []))[:80],
        "feature_set": "champion set",
        "primary_metric": "rows flagged POSSIBLE_LABEL_NOISE",
        "metric_value": pln.get("count"),
        "status": "DIAGNOSTIC",
        "reason": "observation only - nothing relabelled, nothing deleted (§22)",
    }]


def _capacity(d: Any) -> list[Row]:
    prov = d.get("provenance", {})
    return [{
        "model": prov.get("champion_model", ""), "feature_set": "champion set",
        "primary_metric": "recall at analyst budget (curve)",
        "metric_value": (d.get("headline") or [{}])[0].get("recall"),
        "status": "ACCEPTED",
        "reason": "capacity curve from stored OOF predictions; retraining_performed="
                  + _fmt(prov.get("retraining_performed")),
    }]


def _error_atlas(d: Any) -> list[Row]:
    return [{
        "model": "champion", "feature_set": "champion set",
        "primary_metric": "missed mules classified",
        "metric_value": len(d.get("misses") or []), "status": "DIAGNOSTIC",
        "reason": "read-only miss taxonomy; may not be used to write per-case rules",
    }]


def _oof_metrics(d: Any) -> list[Row]:
    return [{
        "model": name, "feature_set": name.split("_top_")[-1] if "_top_" in name else "",
        "primary_metric": "OOF PR-AUC",
        "metric_value": (m or {}).get("pr_auc_mean") if isinstance(m, dict) else None,
        "status": "SUPERSEDED",
        "reason": "generation-1 flat OOF metrics, kept for reconciliation",
    } for name, m in (d.get("models") or {}).items()]


def _simple(model: str, metric: str, key: str, status: str, reason: str,
            cv: str = CV_FLAT) -> Callable[[Any], list[Row]]:
    def fn(d: Any) -> list[Row]:
        return [{"model": model, "feature_set": "", "cv_scheme": cv,
                 "primary_metric": metric,
                 "metric_value": d.get(key) if isinstance(d, dict) else None,
                 "status": status, "reason": reason}]
    return fn


def _none(reason: str) -> Callable[[Any], list[Row]]:
    """A file that is not an experiment - claimed, with the reason recorded."""
    def fn(_: Any) -> list[Row]:
        return [{"model": "", "feature_set": "", "cv_scheme": CV_NONE,
                 "primary_metric": "", "metric_value": None,
                 "status": "NOT_AN_EXPERIMENT", "reason": reason}]
    return fn


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------
M = "artifacts/metrics"

SOURCES: list[Source] = [
    Source(f"{M}/tournament_v2.json", _tournament_v2),
    Source(f"{M}/nested_cv.json", _nested_cv),
    Source(f"{M}/metric_battery.json", _metric_battery),
    Source(f"{M}/with_vs_without_f3912.json", _f3912),
    Source(f"{M}/alert_context_ablation_v2.json", _alert_context),
    Source(f"{M}/family_dropout_v2.json", _family_dropout),
    Source(f"{M}/advanced_models.json", _advanced),
    Source(f"{M}/missingness_ablation.json", _missingness),
    Source(f"{M}/missingness_ablation_no_fees.json", _missingness_nofees),
    Source(f"{M}/seed_variance_v2.json", _seed_variance),
    Source(f"{M}/stability_stress_v2.json", _stability_stress),
    Source(f"{M}/ensemble_v2.json", _ensemble),
    Source(f"{M}/ensemble_decision.json", _ensemble_decision),
    Source(f"{M}/promotion_decision_v2.json", _promotion),
    Source(f"{M}/challenger_review_v2.json", _challenger_review),
    Source(f"{M}/holdout_metrics.json", _holdout),
    Source(f"{M}/locked_test_metrics.json", _locked_test),
    Source(f"{M}/final_locked_test_metrics.json", _locked_test),
    Source(f"{M}/lens_stack_oof.json", _lens_stack),
    Source(f"{M}/lens_stack_oof_v2.json", _lens_stack),
    Source(f"{M}/validation_shield_v2.json", _shield),
    Source(f"{M}/robustness_grade_v2.json", _robustness_grade),
    Source(f"{M}/label_noise_audit_v2.json", _label_noise),
    Source(f"{M}/capacity_curve.json", _capacity),
    Source(f"{M}/error_atlas.json", _error_atlas),
    Source(f"{M}/oof_metrics.json", _oof_metrics),
    Source(f"{M}/final_oof_metrics.json", _oof_metrics),
    Source(f"{M}/merchant_verifier_v2.json",
           _simple("merchant legitimacy verifier", "verifier OOF PR-AUC "
                   "(business-evidence view)", "oof_pr_auc", "ACCEPTED",
                   "hard-negative safeguard: measures business look-alikes, never "
                   "overrides the champion score (§29)")),
    Source(f"{M}/missingness_probe.json",
           _simple("missingness signature (probe)", "probe status (in-sample only)",
                   "status", "DIAGNOSTIC",
                   "in-sample maxima over 358 candidate indicators - a shortlist, "
                   "not evidence of gain; the ablation is the evidence")),
    Source(f"{M}/organiser_dry_run.json",
           _simple("shipped bundle", "dry-run variants passed", "variants_passed",
                   "ACCEPTED", "organiser dry run: predictions invariant across "
                   "input variants, model unchanged", CV_NONE)),
    Source(f"{M}/drift_locked_test.json",
           _simple("drift monitor", "score PSI", "score_psi", "DIAGNOSTIC",
                   "population stability on the historical holdout", CV_NONE)),
    Source(f"{M}/temporal_stress_metrics.json",
           _simple("temporal stress", "temporal stress status", "status", "DIAGNOSTIC",
                   "temporal ordering unavailable in this dataset", CV_NONE)),
    # ---- registered as not-an-experiment ---------------------------------
    Source(f"{M}/drift_baseline_summary.json",
           _none("reference distribution for the drift monitor, not a result"),
           kind="NOT_AN_EXPERIMENT"),
    Source(f"{M}/locked_test_touch_log.json",
           _none("audit log of every locked-test read, not a result"),
           kind="NOT_AN_EXPERIMENT"),
    Source(f"{M}/tabpfn_latency.json",
           _none("operational timing measurement, not a model comparison"),
           kind="NOT_AN_EXPERIMENT"),
    # ---- registered ahead of their runs (§15-23, 53, 54) ------------------
    Source(f"{M}/nested_ensemble.json",
           _simple("nested ensembles", "paired mean AP difference vs best single",
                   "mean_gain", "PENDING_RUN", "sections 19, 20, 53", CV_NESTED),),
    Source(f"{M}/nested_seed_bagging.json",
           _simple("seed-bagged champion", "paired mean AP gain", "mean_gain",
                   "PENDING_RUN", "section 18", CV_NESTED)),
    Source(f"{M}/nested_positive_removal.json",
           _simple("positive-removal stress (nested)", "AP under removal", "ap_mean",
                   "PENDING_RUN", "section 21", CV_NESTED)),
    Source(f"{M}/nested_label_noise.json",
           _simple("label-noise down-weighting", "paired mean AP gain", "mean_gain",
                   "PENDING_RUN", "section 22", CV_NESTED)),
    Source(f"{M}/nested_shift_shield.json",
           _simple("adversarial validation", "adversarial AUC", "adversarial_auc",
                   "PENDING_RUN", "section 23", CV_NESTED)),
    Source(f"{M}/nested_feature_family_arms.json",
           _simple("feature-pool arms", "paired mean AP gain", "mean_gain",
                   "PENDING_RUN", "sections 15, 16, 17", CV_NESTED)),
    Source(f"{M}/nested_generalization_score.json",
           _simple("generalization report", "components reported separately", "grade",
                   "PENDING_RUN", "section 54", CV_NESTED)),
]


def _fingerprints() -> tuple[str, str]:
    dataset_sha = description_sha = ""
    fp = settings.REPO_ROOT / "data/interim/data_fingerprint.json"
    if fp.exists():
        dataset_sha = load_json(fp).get("raw_file", {}).get("sha256", "")
    fd = settings.FEATURES_DIR / "feature_dictionary.json"
    if fd.exists():
        description_sha = load_json(fd).get("source_sha256", "")
    return dataset_sha, description_sha


def build_rows() -> tuple[list[Row], list[str]]:
    """Return (rows, unregistered_files)."""
    git = git_info(settings.REPO_ROOT)
    git_sha = (git.get("commit_sha") or "")[:12]
    dataset_sha, description_sha = _fingerprints()
    seen: set[Path] = set()
    rows: list[Row] = []

    for idx, src in enumerate(SOURCES, start=1):
        path = settings.REPO_ROOT / src.artifact
        seen.add(path.resolve())
        stem = Path(src.artifact).stem
        if not path.exists():
            rows.append({
                "experiment_id": f"EXP{idx:03d}", "timestamp": "",
                "git_sha": git_sha, "dataset_sha": dataset_sha[:12],
                "description_sha": description_sha[:12], "model": "",
                "feature_set": "", "quarantines": FIREWALL_13,
                "cv_scheme": "", "seed": settings.GLOBAL_SEED,
                "primary_metric": "", "metric_value": "",
                "status": "PENDING_RUN",
                "reason": "registered; the run has not produced this artifact yet",
                "artifact_path": src.artifact,
            })
            continue

        payload = load_json(path)
        ts = _timestamp(payload, path)
        try:
            parts = src.expand(payload)
        except Exception as exc:                       # noqa: BLE001
            log.warning("%s: expander failed (%s)", src.artifact, exc)
            parts = [{"model": "", "feature_set": "", "primary_metric": "",
                      "metric_value": None, "status": "UNREADABLE",
                      "reason": f"registered but the expander failed: {exc}"}]
        for sub, part in enumerate(parts, start=1):
            suffix = f".{sub}" if len(parts) > 1 else ""
            row = {
                "experiment_id": f"EXP{idx:03d}{suffix}",
                "timestamp": ts, "git_sha": git_sha,
                "dataset_sha": dataset_sha[:12],
                "description_sha": description_sha[:12],
                "quarantines": FIREWALL_13, "cv_scheme": CV_FLAT,
                "seed": settings.GLOBAL_SEED, "artifact_path": src.artifact,
            }
            row.update(src.defaults)
            row.update(part)
            row["metric_value"] = _fmt(row.get("metric_value"))
            rows.append({c: _fmt(row.get(c, "")) for c in COLUMNS})

    unregistered = sorted(
        str(p.relative_to(settings.REPO_ROOT)).replace("\\", "/")
        for p in (settings.METRICS_DIR).glob("*.json")
        if p.resolve() not in seen)
    for idx, rel in enumerate(unregistered, start=len(SOURCES) + 1):
        rows.append({c: _fmt({
            "experiment_id": f"EXP{idx:03d}", "git_sha": git_sha,
            "dataset_sha": dataset_sha[:12], "description_sha": description_sha[:12],
            "quarantines": FIREWALL_13, "seed": settings.GLOBAL_SEED,
            "status": "UNREGISTERED_REVIEW", "artifact_path": rel,
            "timestamp": _timestamp(None, settings.REPO_ROOT / rel),
            "reason": "artifact exists but no ledger entry claims it - register it "
                      "in cli/experiment_ledger.py or mark it NOT_AN_EXPERIMENT",
        }.get(c, "")) for c in COLUMNS})
    return rows, unregistered


def main() -> int:
    configure()
    rows, unregistered = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    log.info("wrote %s (%d rows)", OUT_CSV, len(rows))
    for status, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        log.info("  %-22s %d", status, n)
    if unregistered:
        log.error("%d unregistered artifact(s) - the ledger is incomplete:",
                  len(unregistered))
        for rel in unregistered:
            log.error("    %s", rel)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
