"""Emit the artifact and report names §57/§58 require, from the real sources.

Run::

    .venv/Scripts/python.exe -m muleguard.cli.reconcile_artifacts

The project grew its own names before the final-validation prompt fixed a list of
them. Two dishonest ways to close that gap would be to rename the real evidence
so the list matches, or to write placeholder files with plausible numbers in
them. This does neither. Every spec-named artifact here is **derived from a named
source file**, carries a ``__provenance__`` block naming that source and its
sha256, and is regenerated on every run so it cannot drift away from the source.

Where the evidence does not exist yet - the nested stability arms, for instance -
the requirement is reported ``PENDING`` with the run that will produce it. A gap
is recorded as a gap.

Outputs:

* the spec-named files themselves, under ``artifacts/``
* ``artifacts/testing/artifact_manifest.json`` - every §57 requirement, its
  status, its source and the sha256 of both
* ``docs/ARTIFACT_AND_REPORT_INDEX.md`` - the §57/§58 name -> real file map
* pointer documents for the §58 report names that exist under another name
"""
from __future__ import annotations

import csv
import datetime as dt
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.utils import sha256_file, save_json, load_json

log = get_logger("cli.reconcile_artifacts")

ROOT = settings.REPO_ROOT
TESTING = settings.ARTIFACTS_DIR / "testing"
MANIFEST = TESTING / "artifact_manifest.json"
INDEX_DOC = settings.DOCS_DIR / "ARTIFACT_AND_REPORT_INDEX.md"

PRESENT, DERIVED, PENDING, MISSING = "PRESENT", "DERIVED", "PENDING", "MISSING"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _exists(rel: str) -> Path | None:
    p = ROOT / rel
    return p if p.exists() else None


@dataclass
class Req:
    """One §57 requirement: where it must land, and how it is produced."""

    spec: str                                   # required path, relative to root
    build: Callable[[], dict[str, Any]] | None  # None => must already exist
    sources: tuple[str, ...] = ()
    note: str = ""
    pending_run: str = ""                       # command that would produce it


def _copy_of(source_rel: str, section: str) -> Callable[[], dict[str, Any]]:
    """The spec name holds the source's own content, plus its provenance."""
    def build() -> dict[str, Any]:
        src = ROOT / source_rel
        payload = load_json(src)
        if not isinstance(payload, dict):
            payload = {"content": payload}
        return {**payload, "__provenance__": {
            "spec_section": section,
            "spec_name_of": source_rel,
            "source_sha256": sha256_file(src),
            "regenerated_utc": _now(),
            "note": "Content is the source file's, unmodified. This file exists "
                    "because the final-validation prompt names it; the source is "
                    "the working copy the pipeline writes.",
        }}
    return build


# --------------------------------------------------------------------------
# derivations that are more than a rename
# --------------------------------------------------------------------------
def _description_integrity() -> dict[str, Any]:
    fd = load_json(settings.FEATURES_DIR / "feature_dictionary.json")
    quar = load_json(settings.FEATURES_DIR / "quarantined_features.json")
    fingerprint = load_json(ROOT / "data/interim/data_fingerprint.json")
    features = fd.get("features") or fd.get("entries") or []
    return {
        "description_workbook_sha256": fd.get("source_sha256", ""),
        "dataset_workbook_sha256": fingerprint.get("raw_file", {}).get("sha256", ""),
        "n_described_features": len(features) if isinstance(features, list) else fd.get("n_features"),
        "n_dataset_columns": fingerprint.get("n_cols"),
        "quarantined": quar.get("quarantined") or quar.get("columns") or quar,
        "__provenance__": {
            "spec_section": "57",
            "derived_from": ["artifacts/features/feature_dictionary.json",
                             "artifacts/features/quarantined_features.json",
                             "data/interim/data_fingerprint.json"],
            "regenerated_utc": _now(),
        },
    }


def _nested_cv_results() -> dict[str, Any]:
    d = load_json(settings.METRICS_DIR / "nested_cv.json")
    outer = str(d.get("design", {}).get("outer", ""))
    preliminary = "x 1 repeats" in outer
    return {**d, "__provenance__": {
        "spec_section": "57",
        "spec_name_of": "artifacts/metrics/nested_cv.json",
        "source_sha256": sha256_file(settings.METRICS_DIR / "nested_cv.json"),
        "regenerated_utc": _now(),
        "status": "PRELIMINARY_SINGLE_REPEAT" if preliminary else "FULL_REPEATED",
        "warning": ("This is the 1-repeat run. It is under-powered and must not be "
                    "quoted as the headline nested result." if preliminary else ""),
    }}


def _positive_removal() -> dict[str, Any]:
    nested = _exists("artifacts/metrics/nested_positive_removal.json")
    if nested:
        return _copy_of("artifacts/metrics/nested_positive_removal.json", "57")()
    flat = "artifacts/metrics/stability_stress_v2.json"
    payload = _copy_of(flat, "57")()
    payload["protocol"] = "FLAT_FALLBACK"
    payload["__provenance__"]["fallback"] = (
        "FLAT PROTOCOL. The nested positive-removal arm has not run yet; this is "
        "the flat-CV measurement, which cannot be placed beside nested numbers.")
    return payload


def _calibration_results() -> dict[str, Any]:
    lens = load_json(settings.METRICS_DIR / "lens_stack_oof_v2.json")
    battery = load_json(settings.METRICS_DIR / "metric_battery.json")
    run = battery.get("runs", {}).get(battery.get("primary_run_key", ""), {})
    return {
        "calibration_selection": lens.get("calibration_selection"),
        "conformal_coverage_oof": lens.get("conformal_coverage_oof"),
        "oof_calibrated_report": lens.get("oof_calibrated_report"),
        "probability_quality": run.get("probability_quality"),
        "calibration_comparison": run.get("calibration_comparison"),
        "measured_on": run.get("protocol"),
        "__provenance__": {
            "spec_section": "57",
            "derived_from": ["artifacts/metrics/lens_stack_oof_v2.json",
                             "artifacts/metrics/metric_battery.json"],
            "regenerated_utc": _now(),
        },
    }


def _threshold_results() -> dict[str, Any]:
    lens = load_json(settings.METRICS_DIR / "lens_stack_oof_v2.json")
    battery = load_json(settings.METRICS_DIR / "metric_battery.json")
    run = battery.get("runs", {}).get(battery.get("primary_run_key", ""), {})
    return {
        "frozen_policy": lens.get("policy_thresholds"),
        "threshold_candidates": run.get("threshold_candidates"),
        "at_frozen_thresholds": run.get("at_frozen_thresholds"),
        "analyst_budgets": run.get("analyst_budgets"),
        "measured_on": run.get("protocol"),
        "__provenance__": {
            "spec_section": "57",
            "derived_from": ["artifacts/metrics/lens_stack_oof_v2.json",
                             "artifacts/metrics/metric_battery.json"],
            "regenerated_utc": _now(),
            "note": "The frozen policy is the shipped one; the candidate table is "
                    "what it was chosen from.",
        },
    }


def _final_calibration() -> dict[str, Any]:
    return _calibration_results()


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _budget_csv() -> dict[str, Any]:
    d = load_json(settings.METRICS_DIR / "capacity_curve.json")
    curve = d.get("curve") or d.get("headline") or []
    cols = ["budget", "true_positives", "false_positives", "mules_missed", "recall",
            "precision", "fp_per_1000_screened", "threshold_score",
            "recall_ci_low", "recall_ci_high"]
    _write_csv(ROOT / "artifacts/metrics/final_budget_metrics.csv", cols,
               [[row.get(c) for c in cols] for row in curve])
    return {"rows": len(curve)}


def _alert_context_csv() -> dict[str, Any]:
    d = load_json(settings.METRICS_DIR / "alert_context_ablation_v2.json")
    rows = [[name, v.get("pr_auc_mean") if isinstance(v, dict) else v,
             v.get("pr_auc_std") if isinstance(v, dict) else None,
             v.get("n_features") if isinstance(v, dict) else None]
            for name, v in (d.get("variants") or {}).items()]
    _write_csv(ROOT / "artifacts/metrics/alert_context_ablation.csv",
               ["variant", "pr_auc_mean", "pr_auc_std", "n_features"], rows)
    return {"rows": len(rows), "verdict": d.get("verdict")}


def _feature_subset_csv() -> dict[str, Any]:
    nested = _exists("artifacts/metrics/nested_feature_family_arms.json")
    if nested:
        d = load_json(nested)
        rows = [[a.get("arm"), a.get("mean_gain"), a.get("std_of_paired_diff"),
                 a.get("sign_test_p_two_sided"), a.get("wilcoxon_p"),
                 a.get("paired_t_p"), a.get("verdict"), "NESTED"]
                for a in (d.get("arms") or [])]
    else:
        d = load_json(settings.METRICS_DIR / "family_dropout_v2.json")
        rows = [[f"drop:{f.get('family_removed')}", f.get("pr_auc"), None, None,
                 None, None, f"relative_drop={f.get('relative_drop')}", "FLAT"]
                for f in (d.get("per_family") or [])]
    _write_csv(ROOT / "artifacts/metrics/feature_subset_ablation.csv",
               ["arm", "metric", "std_of_paired_diff", "sign_test_p", "wilcoxon_p",
                "paired_t_p", "verdict", "protocol"], rows)
    return {"rows": len(rows), "protocol": "NESTED" if nested else "FLAT_FALLBACK"}


def _copy_parquet(source_rel: str, target_rel: str) -> Callable[[], dict[str, Any]]:
    def build() -> dict[str, Any]:
        src, dst = ROOT / source_rel, ROOT / target_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return {"copied_from": source_rel, "sha256": sha256_file(dst)}
    return build


# --------------------------------------------------------------------------
# §57 requirements
# --------------------------------------------------------------------------
REQUIREMENTS: list[Req] = [
    Req("artifacts/testing/environment.json",
        _copy_of("artifacts/environment_snapshot.json", "57"),
        ("artifacts/environment_snapshot.json",)),
    Req("artifacts/testing/data_integrity.json",
        _copy_of("artifacts/testing/data_integrity_results.json", "57"),
        ("artifacts/testing/data_integrity_results.json",)),
    Req("artifacts/testing/description_integrity.json", _description_integrity,
        ("artifacts/features/feature_dictionary.json",)),
    Req("artifacts/testing/leakage_results.json",
        _copy_of("artifacts/testing/leakage_test_results.json", "57"),
        ("artifacts/testing/leakage_test_results.json",)),
    Req("artifacts/testing/nested_cv_results.json", _nested_cv_results,
        ("artifacts/metrics/nested_cv.json",),
        pending_run="muleguard.cli.nested_cv --repeats 3 --inner 4"),
    Req("artifacts/testing/positive_removal_results.json", _positive_removal,
        ("artifacts/metrics/nested_positive_removal.json",
         "artifacts/metrics/stability_stress_v2.json"),
        pending_run="muleguard.cli.nested_ses --stages posremoval"),
    Req("artifacts/testing/adversarial_validation.json",
        _copy_of("artifacts/metrics/nested_shift_shield.json", "57"),
        ("artifacts/metrics/nested_shift_shield.json",),
        pending_run="muleguard.cli.nested_ses --stages shift"),
    Req("artifacts/testing/calibration_results.json", _calibration_results,
        ("artifacts/metrics/lens_stack_oof_v2.json",)),
    Req("artifacts/testing/threshold_results.json", _threshold_results,
        ("artifacts/metrics/lens_stack_oof_v2.json",)),
    Req("artifacts/testing/backend_results.json",
        _copy_of("artifacts/testing/backend_test_results.json", "57"),
        ("artifacts/testing/backend_test_results.json",)),
    Req("artifacts/testing/ui_metric_consistency.json",
        _copy_of("artifacts/testing/api_frontend_consistency.json", "57"),
        ("artifacts/testing/api_frontend_consistency.json",)),
    Req("artifacts/testing/offline_results.json", None, (),
        note="written by scripts/test_offline.sh",
        pending_run="bash scripts/test_offline.sh"),
    Req("artifacts/testing/security_results.json", None, (),
        note="written directly by the security suite"),

    Req("artifacts/metrics/model_comparison.csv", None, (),
        note="generation-1 tournament table; model_comparison_v2.csv is the "
             "post-firewall one and the ledger is the full record"),
    Req("artifacts/metrics/final_nested_cv.json", _nested_cv_results,
        ("artifacts/metrics/nested_cv.json",),
        pending_run="muleguard.cli.nested_cv --repeats 3 --inner 4"),
    Req("artifacts/metrics/final_budget_metrics.csv", _budget_csv,
        ("artifacts/metrics/capacity_curve.json",)),
    Req("artifacts/metrics/final_calibration.json", _final_calibration,
        ("artifacts/metrics/lens_stack_oof_v2.json",)),
    Req("artifacts/metrics/feature_subset_ablation.csv", _feature_subset_csv,
        ("artifacts/metrics/nested_feature_family_arms.json",
         "artifacts/metrics/family_dropout_v2.json"),
        pending_run="muleguard.cli.nested_ses --stages families"),
    Req("artifacts/metrics/alert_context_ablation.csv", _alert_context_csv,
        ("artifacts/metrics/alert_context_ablation_v2.json",)),
    Req("artifacts/metrics/leakage_ablation.csv", None, (),
        note="written by the leakage ablation run"),

    Req("artifacts/predictions/all_outer_oof_predictions.parquet",
        _copy_parquet("artifacts/predictions/nested_oof.parquet",
                      "artifacts/predictions/all_outer_oof_predictions.parquet"),
        ("artifacts/predictions/nested_oof.parquet",),
        pending_run="muleguard.cli.nested_cv --repeats 3 --inner 4"),
    # The champion's own dev OOF vector. oof_v2.parquet is the post-firewall
    # champion's; final_oof_predictions.parquet is generation-1 and is not it.
    Req("artifacts/predictions/final_model_oof_predictions.parquet",
        _copy_parquet("artifacts/predictions/oof_v2.parquet",
                      "artifacts/predictions/final_model_oof_predictions.parquet"),
        ("artifacts/predictions/oof_v2.parquet",),
        note="champion dev OOF; re-copied whenever the champion changes"),

    Req("artifacts/features/feature_dictionary.json", None, ()),
    Req("artifacts/features/selected_features.json", None, ()),
    Req("artifacts/features/selection_frequency.csv", None, ()),
    Req("artifacts/features/quarantined_features.json", None, ()),
]

# --------------------------------------------------------------------------
# §58 reports: required name -> (authoritative file, note)
# --------------------------------------------------------------------------
REPORTS: dict[str, tuple[str | None, str]] = {
    "docs/FINAL_XHIGH_VALIDATION_REPORT.md":
        (None, "written last, from the completed nested evidence"),
    "docs/DESCRIPTION_VALIDATION_REPORT.md":
        ("docs/FEATURE_DICTIONARY_REPORT.md",
         "Description.xlsx parsing, coverage and the availability ruling per field"),
    "docs/FEATURE_AVAILABILITY_FIREWALL.md":
        ("docs/FEATURE_AVAILABILITY_AUDIT.md",
         "the firewall itself: what was quarantined and on what evidence"),
    "docs/FINAL_LEAKAGE_FORENSICS.md":
        ("docs/FINAL_DATA_AND_LEAKAGE_AUDIT.md",
         "post-outcome field forensics, including the F3912 probe"),
    "docs/HISTORICAL_METRIC_RECONCILIATION.md":
        ("docs/HISTORICAL_METRIC_RECONCILIATION.md", "already under the spec name"),
    "docs/NESTED_CV_MODEL_TOURNAMENT.md":
        ("docs/MODEL_TOURNAMENT_REPORT.md",
         "the tournament; the nested results section is filled by the nested run"),
    "docs/HIDDEN_VALIDATION_READINESS.md":
        ("docs/HIDDEN_VALIDATION_STRATEGY.md",
         "what happens when the organiser's file arrives"),
    "docs/POSITIVE_REMOVAL_STABILITY.md":
        ("docs/ROBUSTNESS_REPORT.md", "positive-removal rounds and the grade they feed"),
    "docs/ADVERSARIAL_VALIDATION_REPORT.md":
        ("docs/NESTED_STABILITY_ENSEMBLE_SHIFT.md",
         "section 23 of the pre-registered nested programme"),
    "docs/CALIBRATION_AND_THRESHOLDS.md":
        ("docs/FINAL_CALIBRATION_AND_THRESHOLD_REPORT.md",
         "calibration choice, threshold freeze and the policy version"),
    "docs/FALSE_POSITIVE_VALIDATION.md":
        ("docs/FALSE_POSITIVE_CONTROL_REPORT.md",
         "false-positive control, including the merchant verifier"),
    "docs/VALIDATION_LAB_TEST_REPORT.md":
        ("docs/VALIDATION_LAB_REPORT.md", "the judge-facing validation lab"),
    "docs/SEALED_VALIDATION_PROTOCOL.md":
        ("docs/LOCKED_TEST_RULING.md",
         "the seal, the single touch, and why the locked test is reference only"),
    "docs/UI_METRIC_CONSISTENCY.md":
        ("docs/FINAL_FRONTEND_UI_REPORT.md",
         "every number shown in the UI traced to the artifact it came from"),
    "docs/OFFLINE_RUNTIME_TEST.md":
        ("docs/NO_MCP_NO_BROWSER_AGENT_COMPLIANCE.md",
         "offline behaviour with the narrator disabled and no network"),
    "docs/SECURITY_TEST_REPORT.md":
        ("docs/FINAL_SECURITY_REPORT.md", "security suite results"),
    "docs/FINAL_MODEL_CARD.md":
        ("docs/MODEL_CARD.md", "the model card"),
}

POINTER_TEMPLATE = """# {title}

_Required by section 58 of the final-validation prompt._

**The report lives at [`{target}`]({target_name}).** That file is the
authoritative copy - it was written first, it is the one the pipeline updates,
and duplicating its contents here would only let the two drift apart.

{note}

<sub>Generated by `muleguard.cli.reconcile_artifacts`; regenerated on every
validation run. Do not edit this file - edit the authoritative report.</sub>
"""


def _write_pointers() -> list[str]:
    written = []
    for spec, (target, note) in REPORTS.items():
        if target is None or target == spec:
            continue
        spec_path, target_path = ROOT / spec, ROOT / target
        if not target_path.exists():
            continue
        title = Path(spec).stem.replace("_", " ").title()
        spec_path.write_text(POINTER_TEMPLATE.format(
            title=title, target=target, target_name=Path(target).name,
            note=f"It covers {note}."), encoding="utf-8")
        written.append(spec)
    return written


def _index_doc(entries: list[dict[str, Any]], pointers: list[str]) -> None:
    lines = [
        "# Artifact and report index",
        "",
        "Sections 57 and 58 fix a list of file names. This project had already "
        "written most of that evidence under its own names. Rather than rename "
        "the evidence - which would break every document that cites it - each "
        "required name is produced from the real source and says so in its own "
        "`__provenance__` block.",
        "",
        "Regenerate with `make reconcile-artifacts`. Nothing here is written by "
        "hand, so the table cannot claim a file that is not there.",
        "",
        "## Section 57 - artifacts",
        "",
        "| required path | status | source | note |",
        "|---|---|---|---|",
    ]
    for e in entries:
        src = ", ".join(f"`{s}`" for s in e["sources"]) if e["sources"] else "-"
        note = e.get("note") or ""
        protocol = (e.get("detail") or {}).get("protocol")
        if protocol and "FALLBACK" in str(protocol):
            # Say it in the table, not only inside the file: a flat-protocol
            # stand-in must not be read as the nested arm it is standing in for.
            note = (f"**{protocol}** - the nested arm has not run; "
                    f"flat-CV stand-in, not comparable to nested numbers")
        if e["status"] == PENDING and e.get("pending_run"):
            note = f"waiting on `{e['pending_run']}`"
        lines.append(f"| `{e['spec']}` | {e['status']} | {src} | {note} |")

    lines += [
        "",
        "`PRESENT` - the pipeline already writes this exact path. "
        "`DERIVED` - regenerated here from the named source. "
        "`PENDING` - the run that produces it has not finished. "
        "`MISSING` - neither the file nor its source exists.",
        "",
        "## Section 58 - reports",
        "",
        "| required name | authoritative file | covers |",
        "|---|---|---|",
    ]
    for spec, (target, note) in REPORTS.items():
        if target is None:
            lines.append(f"| `{spec}` | _not written yet_ | {note} |")
        elif target == spec:
            lines.append(f"| `{spec}` | itself | {note} |")
        else:
            marker = " (pointer written)" if spec in pointers else ""
            lines.append(f"| `{spec}` | `{target}`{marker} | {note} |")

    lines += [
        "",
        "A pointer file carries no numbers of its own. It names the report that "
        "does, so a reader looking for the required name finds the real thing "
        "one hop away rather than a second copy that can go stale.",
        "",
    ]
    INDEX_DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    configure()
    TESTING.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for req in REQUIREMENTS:
        target = ROOT / req.spec
        have_source = any(_exists(s) for s in req.sources)
        entry = {"spec": req.spec, "sources": list(req.sources),
                 "note": req.note, "pending_run": req.pending_run}

        if req.build is None:
            entry["status"] = PRESENT if target.exists() else MISSING
            if not target.exists() and req.pending_run:
                entry["status"] = PENDING
        elif not have_source:
            entry["status"] = PENDING if req.pending_run else MISSING
        else:
            try:
                result = req.build()
                if req.spec.endswith(".json"):
                    save_json(result, target)
                entry["status"] = DERIVED
                entry["detail"] = {k: v for k, v in result.items()
                                   if k in {"rows", "protocol", "verdict",
                                            "copied_from", "sha256"}
                                   and not isinstance(v, (dict, list))}
            except Exception as exc:                       # noqa: BLE001
                log.warning("%s: %s", req.spec, exc)
                entry["status"] = MISSING
                entry["note"] = f"build failed: {exc}"

        if target.exists():
            entry["sha256"] = sha256_file(target)
            entry["bytes"] = target.stat().st_size
        entries.append(entry)

    pointers = _write_pointers()
    _index_doc(entries, pointers)

    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    save_json({
        "generated_utc": _now(),
        "spec_sections": ["57", "58"],
        "counts": counts,
        "artifacts": entries,
        "report_pointers_written": pointers,
        "index": _rel(INDEX_DOC),
    }, MANIFEST)

    log.info("%d requirements: %s", len(entries),
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    log.info("manifest %s", _rel(MANIFEST))
    log.info("index    %s", _rel(INDEX_DOC))
    for e in entries:
        if e["status"] in (PENDING, MISSING):
            log.info("  %-8s %s", e["status"], e["spec"])
    # A pending requirement is a truthful state, not a failure; a missing one
    # with no run behind it is the thing worth failing on.
    return 1 if counts.get(MISSING) else 0


if __name__ == "__main__":
    raise SystemExit(main())
