"""Freeze the pre-change invariants, or check the post-change ones against them.

Run before touching a single production file::

    .venv/Scripts/python.exe -m muleguard.cli.usp_baseline --save

and again after the Cohort Radar and Account-Control layers are in::

    .venv/Scripts/python.exe -m muleguard.cli.usp_baseline --check

``--save`` writes ``artifacts/upgrade_baseline/usp_prechange_baseline.json``:
the champion's identity, its calibrator's fitted constants, its thresholds, the
live quarantine, the hashes of every frozen artifact, the accuracy recomputed
from the saved out-of-fold predictions, and the full-precision probability the
live scoring path assigns to 400 fixed development accounts.

``--check`` recomputes all of it and diffs. The exit code is the verdict: 0 if
every probability, tier and metric is identical, 1 otherwise. It writes
``artifacts/testing/usp_accuracy_regression.json`` for the record.

Why the same module does both
-----------------------------
Before and after must be measured by identical code. If the "after" run used a
newer snapshot function than the "before" run, a difference in the report could
be a difference in the measurement rather than in the model - which is exactly
the failure a regression check exists to rule out. One module, one code path,
two invocations.

``--save`` refuses to overwrite an existing baseline unless ``--force`` is
given. A baseline that can be silently regenerated after a change is not a
baseline; it is a description of the change agreeing with itself.
"""
from __future__ import annotations

import argparse
import sys

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.usp import baseline as B
from muleguard.utils import load_json, save_json

log = get_logger("cli.usp_baseline")

REGRESSION_PATH = settings.ARTIFACTS_DIR / "testing" / "usp_accuracy_regression.json"

#: Metrics printed in the console summary. The full battery goes to JSON; this
#: is the subset a human reads to decide whether to look at the JSON at all.
HEADLINE = ("pr_auc", "roc_auc", "recall", "precision", "f1", "f2", "mcc",
            "brier", "ece", "false_positives_per_1000_legitimate")


def _rel(path) -> str:
    try:
        return str(path.relative_to(settings.REPO_ROOT))
    except ValueError:
        return str(path)


def do_save(*, force: bool, with_probes: bool) -> int:
    if B.BASELINE_PATH.exists() and not force:
        log.error("baseline already exists: %s", _rel(B.BASELINE_PATH))
        log.error("refusing to overwrite - pass --force only if you are certain "
                  "no production file has changed since it was taken")
        return 1

    snap = B.invariant_snapshot(with_probes=with_probes)
    B.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(snap, B.BASELINE_PATH)

    champ = snap["champion"]
    acc = snap["accuracy"]["mean_over_repeats"]
    agree = snap["accuracy"]["agreement_with_published"]
    log.info("wrote %s", _rel(B.BASELINE_PATH))
    log.info("  champion            %s (%s)", champ["champion_model_id"],
             champ["winner_family"])
    log.info("  bundle sha256       %s", champ["bundle_sha256"])
    log.info("  selected features   %d (hash %s)", champ["n_selected_features"],
             champ["selected_feature_hash"][:16])
    log.info("  calibrator          %s a=%.12f b=%.12f", champ["calibrator_method"],
             champ["calibrator"]["parameters"].get("a", float("nan")),
             champ["calibrator"]["parameters"].get("b", float("nan")))
    log.info("  quarantined         %d feature(s), policy %s",
             snap["quarantine"]["n_quarantined"],
             snap["quarantine"]["policy_version"])
    if with_probes:
        log.info("  probe rows          %d (hash %s)", snap["probes"]["n_rows"],
                 snap["probes"]["rows_hash"][:16])
    for key in HEADLINE:
        log.info("  %-34s %.10f", key, acc[key])
    log.info("  PR-AUC vs registry  recomputed %.8f, registry %s -> %s",
             agree["recomputed_pr_auc_mean_over_repeats"],
             agree["registry_oof_pr_auc"],
             "AGREES" if agree["matches_registry"] else "DISAGREES")
    if not agree["matches_registry"]:
        log.error("the shipped PR-AUC does not reproduce from saved predictions; "
                  "fix that before starting the upgrade")
        return 1
    return 0


def do_check(*, with_probes: bool) -> int:
    if not B.BASELINE_PATH.exists():
        log.error("no baseline at %s - run --save first (and do it before "
                  "changing anything)", _rel(B.BASELINE_PATH))
        return 1

    before = load_json(B.BASELINE_PATH)
    after = B.invariant_snapshot(with_probes=with_probes)
    report = B.compare(before, after)
    REGRESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, REGRESSION_PATH)

    log.info("wrote %s", _rel(REGRESSION_PATH))
    log.info("  model sha equal     %s", report["model_sha_equal"])
    log.info("  feature hash equal  %s", report["feature_hash_equal"])
    log.info("  calibrator equal    %s", report["calibrator_equal"])
    log.info("  thresholds equal    %s", report["thresholds_equal"])
    log.info("  quarantine equal    %s", report["quarantine_equal"])
    if report["probes"].get("compared"):
        log.info("  probes compared     %d row(s)", report["probes"]["n_rows_compared"])
        log.info("  max |Δprobability|  %.3e (tolerance %.0e)",
                 report["probability_max_abs_diff"], B.PROBABILITY_TOLERANCE)
        log.info("  tier mismatches     %d", report["tier_mismatch_count"])
    for key in HEADLINE:
        d = report["metric_differences"].get(key)
        if d is None:
            continue
        log.info("  %-34s %.10f -> %.10f  (Δ %+.3e)", key, d["before"], d["after"],
                 d["difference"] or 0.0)

    if report["verdict"] == "PASS":
        log.info("VERDICT: PASS - the classifier is untouched")
        return 0
    log.error("VERDICT: FAIL - %d finding(s):", len(report["findings"]))
    for f in report["findings"]:
        log.error("    %s", f)
    log.error("do not explain these away as small; find the cause.")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Freeze or verify the pre-change model invariants.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save", action="store_true",
                      help="write the pre-change baseline")
    mode.add_argument("--check", action="store_true",
                      help="recompute and diff against the saved baseline")
    ap.add_argument("--force", action="store_true",
                    help="allow --save to overwrite an existing baseline")
    ap.add_argument("--no-probes", action="store_true",
                    help="skip live scoring of the probe rows (identity and "
                         "recomputed accuracy only)")
    args = ap.parse_args(argv)

    configure()
    with_probes = not args.no_probes
    if args.save:
        return do_save(force=args.force, with_probes=with_probes)
    return do_check(with_probes=with_probes)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
