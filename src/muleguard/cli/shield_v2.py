"""Development-time half of the Trinetra Validation Shield (addendum UPDATE 3).

    python -m muleguard.cli.shield_v2            report + ablation
    python -m muleguard.cli.shield_v2 --report-only

The upload-time half - adversarial validation of a judge's file against our
training distribution - already runs inside the Validation Lab. This CLI covers
the other half: which of the champion's own features look like they will not
survive a differently-drawn sample, and does removing them actually help?

The order matters. The report flags SHIFT_PRONE features, and then the ablation
*measures* the model with and without them. A flag alone is not a reason to
delete a feature: on 64 training positives, dropping a genuinely predictive
column because its distribution wobbles across folds is a reliable way to lose
recall on the hidden validation set. The decision rule is written in the shield
module and applied here without discretion - drop only if PR-AUC does not fall
and fold variance does not rise.

Reads no label the model was not already trained on, and never touches the
locked test set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import Any

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from muleguard import settings  # noqa: E402
from muleguard.logging import get_logger  # noqa: E402
from muleguard.models import shield  # noqa: E402
from muleguard.utils import load_json, save_json, set_global_seed  # noqa: E402

log = get_logger("cli.shield_v2")

REPORT_JSON = settings.METRICS_DIR / "validation_shield_v2.json"


def run(*, n_repeats: int = 2, ablation_repeats: int = 3,
        report_only: bool = False) -> dict[str, Any]:
    from muleguard.cli.robustness_v2 import _champion, _spec

    set_global_seed(settings.GLOBAL_SEED)
    champ = _champion()
    spec = _spec(champ)
    features = spec.features or list(spec.frame.feature_names)
    log.info("shield: champion %s, %d features", champ["model"], len(features))

    t0 = time.time()
    report = shield.feature_stability_report(spec.frame, features,
                                             n_repeats=n_repeats)
    counts = report["counts"]
    log.info("shield report: %d STABLE, %d WATCH, %d SHIFT_PRONE, %d LEAKAGE",
             counts["STABLE"], counts["WATCH"], counts["SHIFT_PRONE"],
             counts["LEAKAGE"])

    # A LEAKAGE flag on a feature the champion is actually using is a release
    # blocker, not a note. It means the firewall and the selected feature list
    # disagree about what is admissible, and until that is resolved no number
    # produced by this model can be trusted.
    leaked = [r["feature"] for r in report["features"] if r["flag"] == shield.LEAKAGE]

    ablation: dict[str, Any] | None = None
    if not report_only and not report["shift_prone"]:
        # Nothing to ablate. Fitting the model twice to compare it against
        # itself would burn an hour to produce a difference of exactly zero.
        ablation = {
            "n_shift_prone": 0,
            "decision": "NO_ABLATION_NEEDED",
            "reason": ("no feature in the champion's set was flagged "
                       "SHIFT_PRONE, so there is no candidate removal to "
                       "measure"),
        }
    elif not report_only:
        ablation = shield.shift_prone_ablation(
            spec.scorer, spec.frame, features, report["shift_prone"],
            n_repeats=ablation_repeats, mode=spec.mode)
    if ablation:
        log.info("shift-prone ablation: %s", ablation["decision"])

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "champion": champ["model"],
        "champion_oof_pr_auc": champ["oof_pr_auc_mean"],
        "n_features": len(features),
        "feature_stability_report": report,
        "shift_prone_ablation": ablation,
        "leakage_flagged_features": leaked,
        "release_blocker": bool(leaked),
        "release_blocker_reason": (
            "the champion uses features the availability firewall rejects; the "
            "feature list and the firewall must be reconciled before release"
        ) if leaked else None,
        "what_this_does_not_do": [
            "it does not delete a feature because a flag was raised",
            "it does not read the locked test set or any hidden validation label",
            "it does not modify any prediction, threshold or calibrated risk",
        ],
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(payload, REPORT_JSON)
    log.info("written %s", REPORT_JSON.name)
    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Trinetra Validation Shield (dev-time)")
    p.add_argument("--repeats", type=int, default=2,
                   help="CV repeats for the stability report")
    p.add_argument("--ablation-repeats", type=int, default=3)
    p.add_argument("--report-only", action="store_true",
                   help="skip the with/without ablation fits")
    a = p.parse_args(argv)
    run(n_repeats=a.repeats, ablation_repeats=a.ablation_repeats,
        report_only=a.report_only)


if __name__ == "__main__":
    main()
