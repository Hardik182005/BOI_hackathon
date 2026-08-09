"""Run the robustness suite against the promoted champion (section 21, U4).

    python -m muleguard.cli.robustness_v2 stress     addendum UPDATE 4
    python -m muleguard.cli.robustness_v2 dropout    semantic family dropout
    python -m muleguard.cli.robustness_v2 seeds      seed variance
    python -m muleguard.cli.robustness_v2 ablation   alert-context ablation
    python -m muleguard.cli.robustness_v2 grade      combine into the badge
    python -m muleguard.cli.robustness_v2 all        every stage, in order

The champion is read from artifacts/metrics/promotion_decision_v2.json, never
hardcoded, so this suite always tests whatever the tournament actually
promoted. Each stage writes its own artifact and ``grade`` combines them, which
means a stage that fails or is interrupted costs only itself - the same
restartability the tournament has, for the same reason: these fits are slow and
the machine is not.

Nothing here consults the locked test set or any label the model was not
already trained on. Every experiment re-measures the SAME development split
with a modified model, so the yardstick never moves.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import Any

# Same thread hygiene as the tournament, and for the same reason: this suite
# fits LightGBM, XGBoost and CatBoost in one process, each with its own bundled
# OpenMP runtime. Set before those libraries are imported anywhere below.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from muleguard import settings
from muleguard.features.frame import build_model_frame
from muleguard.logging import get_logger
from muleguard.models import robustness as rb
from muleguard.utils import load_json, save_json, set_global_seed

log = get_logger("cli.robustness_v2")

PROMOTION_JSON = settings.METRICS_DIR / "promotion_decision_v2.json"
SELECTED_JSON = settings.FEATURES_DIR / "selected_features_v2.json"

STRESS_JSON = settings.METRICS_DIR / "stability_stress_v2.json"
DROPOUT_JSON = settings.METRICS_DIR / "family_dropout_v2.json"
SEEDS_JSON = settings.METRICS_DIR / "seed_variance_v2.json"
ABLATION_JSON = settings.METRICS_DIR / "alert_context_ablation_v2.json"
GRADE_JSON = settings.METRICS_DIR / "robustness_grade_v2.json"


def _champion() -> dict[str, Any]:
    if not PROMOTION_JSON.exists():
        raise SystemExit(f"{PROMOTION_JSON} missing - run the tournament first")
    return load_json(PROMOTION_JSON)["promoted_detail"]


def _spec(champ: dict[str, Any]) -> rb.RunSpec:
    """Rebuild the promoted configuration as a RunSpec.

    Imported lazily from the tournament module so the scorer definitions have
    exactly one home; a second copy here would drift the first time someone
    retuned a model.
    """
    from muleguard.cli.tournament_v2 import _scorers

    scorer, mode = _scorers()[champ["family"]]
    view = champ["view"] if champ["view"] != "ALL_ADMISSIBLE" else None
    frame = build_model_frame(view=view)

    feats = None
    if champ.get("feature_set"):
        sel = load_json(SELECTED_JSON)
        pool = champ["view"] or "ALL_ADMISSIBLE"
        feats = sel["pools"][pool]["compact_sets"][champ["feature_set"]]

    return rb.RunSpec(name=champ["model"], scorer=scorer, frame=frame,
                      mode=mode, features=feats, family=champ["family"])


def _stamp(payload: dict[str, Any], champ: dict[str, Any], seconds: float) -> dict[str, Any]:
    payload["generated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload["champion"] = champ["model"]
    payload["champion_oof_pr_auc"] = champ["oof_pr_auc_mean"]
    payload["runtime_seconds"] = round(seconds, 1)
    return payload


def run_stress(rounds: int = 15, removal_fraction: float = 0.125,
               n_repeats: int = 1) -> dict[str, Any]:
    """Addendum UPDATE 4 - the Mule Stability Stress Test."""
    champ = _champion()
    spec = _spec(champ)
    log.info("stress: %s, %d rounds, removing %.1f%% of training positives",
             spec.name, rounds, removal_fraction * 100)
    t0 = time.perf_counter()
    out = rb.positive_removal_stress(spec, rounds=rounds,
                                     removal_fraction=removal_fraction,
                                     n_repeats=n_repeats)
    save_json(_stamp(out, champ, time.perf_counter() - t0), STRESS_JSON)
    log.info("stress: PR-AUC %.5f +/- %.5f (reference %.5f), rank stability %.4f",
             out["positive_removal_pr_auc_mean"], out["positive_removal_pr_auc_std"],
             out["reference_pr_auc"], out["prediction_rank_stability"])
    return out


def run_dropout(n_repeats: int = 2) -> dict[str, Any]:
    champ = _champion()
    spec = _spec(champ)
    log.info("family dropout: %s", spec.name)
    t0 = time.perf_counter()
    out = rb.family_dropout(spec, n_repeats=n_repeats)
    save_json(_stamp(out, champ, time.perf_counter() - t0), DROPOUT_JSON)
    log.info("family dropout: worst family %s, relative drop %s",
             out["worst_family"], out["worst_relative_drop"])
    return out


def run_seeds(seeds: int = 5, n_repeats: int = 2) -> dict[str, Any]:
    champ = _champion()
    spec = _spec(champ)
    log.info("seed variance: %s over %d seeds", spec.name, seeds)
    t0 = time.perf_counter()
    out = rb.seed_variance(spec, seeds=seeds, n_repeats=n_repeats)
    save_json(_stamp(out, champ, time.perf_counter() - t0), SEEDS_JSON)
    log.info("seed variance: %.5f +/- %.5f, spread %.5f",
             out["pr_auc_mean"], out["pr_auc_std"], out["spread"])
    return out


def run_ablation(n_repeats: int = 3, top_k: int = 60) -> dict[str, Any]:
    """Does the model relay an analyst's prior suspicion, or find it itself?"""
    champ = _champion()
    from muleguard.cli.tournament_v2 import _scorers

    scorer, mode = _scorers()[champ["family"]]
    log.info("alert-context ablation with the %s family", champ["family"])
    t0 = time.perf_counter()
    out = rb.alert_context_ablation(scorer, n_repeats=n_repeats, top_k=top_k,
                                    mode=mode)
    out["model_family"] = champ["family"]
    save_json(_stamp(out, champ, time.perf_counter() - t0), ABLATION_JSON)
    log.info("ablation: %s", out["verdict"])
    return out


def run_grade() -> dict[str, Any]:
    """Combine the measured spreads into the HIGH/MEDIUM/LOW badge.

    Refuses to grade without the stress test, because the badge's thresholds are
    mostly defined on it; a badge computed from family dropout alone would be a
    different statement wearing the same name.
    """
    if not STRESS_JSON.exists():
        raise SystemExit(f"{STRESS_JSON} missing - run stage 'stress' first")
    stress = load_json(STRESS_JSON)
    dropout = load_json(DROPOUT_JSON) if DROPOUT_JSON.exists() else None

    out = rb.robustness_grade(stress, dropout)
    out["champion"] = stress.get("champion")
    out["inputs"] = {
        "stress": STRESS_JSON.name,
        "family_dropout": DROPOUT_JSON.name if dropout else None,
        "seed_variance": SEEDS_JSON.name if SEEDS_JSON.exists() else None,
        "alert_context_ablation": ABLATION_JSON.name if ABLATION_JSON.exists() else None,
    }
    if SEEDS_JSON.exists():
        s = load_json(SEEDS_JSON)
        out["seed_spread"] = s["spread"]
        out["differences_below_this_are_noise"] = s["spread"]
    if ABLATION_JSON.exists():
        a = load_json(ABLATION_JSON)
        out["alert_context_relative_gain"] = a.get("relative_gain")
        out["alert_context_verdict"] = a.get("verdict")
    out["feature_rank_stability"] = stress.get("feature_rank_stability")
    out["display_rule"] = (
        "Show as 'Hidden Validation Robustness: <grade>'. The grade is read off "
        "fixed thresholds published in muleguard.models.robustness; it is never "
        "chosen, and it is never inferred from a metric the thresholds do not name."
    )
    save_json(out, GRADE_JSON)
    log.info("Hidden Validation Robustness: %s", out["hidden_validation_robustness"])
    return out


def run_all(rounds: int, seeds: int) -> dict[str, Any]:
    """Every stage in dependency order, tolerating individual failures.

    A failed stage is recorded and skipped rather than aborting the run: on a
    16 GB CPU box the whole suite is an hour of fitting, and losing the stress
    test because the ablation could not build a view would be a poor trade.
    """
    results: dict[str, Any] = {}
    stages = [
        ("stress", lambda: run_stress(rounds=rounds)),
        ("family_dropout", run_dropout),
        ("seed_variance", lambda: run_seeds(seeds=seeds)),
        ("alert_context_ablation", run_ablation),
    ]
    for name, fn in stages:
        try:
            fn()
            results[name] = "OK"
        except Exception as e:  # noqa: BLE001 - one stage must not kill the suite
            log.exception("stage %s failed", name)
            results[name] = f"FAILED: {type(e).__name__}: {e}"
    try:
        grade = run_grade()
        results["grade"] = grade["hidden_validation_robustness"]
    except Exception as e:  # noqa: BLE001
        log.exception("grading failed")
        results["grade"] = f"FAILED: {e}"
    log.info("robustness suite: %s", results)
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Robustness suite for the promoted model")
    p.add_argument("stage", choices=["stress", "dropout", "seeds", "ablation",
                                     "grade", "all"])
    p.add_argument("--rounds", type=int, default=15,
                   help="stress rounds (addendum UPDATE 4 asks for 10-20)")
    p.add_argument("--removal-fraction", type=float, default=0.125,
                   help="share of training positives removed per round (0.10-0.15)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args(argv)

    if not 0.10 <= a.removal_fraction <= 0.15:
        raise SystemExit("--removal-fraction must stay within the 0.10-0.15 band "
                         "the addendum specifies")
    set_global_seed(settings.GLOBAL_SEED)
    np.seterr(all="ignore")

    if a.stage == "stress":
        run_stress(rounds=a.rounds, removal_fraction=a.removal_fraction)
    elif a.stage == "dropout":
        run_dropout(n_repeats=a.repeats)
    elif a.stage == "seeds":
        run_seeds(seeds=a.seeds, n_repeats=a.repeats)
    elif a.stage == "ablation":
        run_ablation()
    elif a.stage == "grade":
        run_grade()
    else:
        run_all(rounds=a.rounds, seeds=a.seeds)


if __name__ == "__main__":
    main()
