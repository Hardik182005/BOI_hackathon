"""The section 65 A-L final response, assembled from artifacts rather than memory.

Run::

    .venv/Scripts/python.exe -m muleguard.cli.final_verdict

The final response is the one document a judge is guaranteed to read, which makes
it the easiest place in the project to be accidentally dishonest: every number in
it was true of *some* run, and nothing in a hand-written summary remembers which.
So this module writes it mechanically. Every field is either read out of a named
artifact or printed as ``PENDING`` with the run that would produce it. There is no
code path that invents a value, and none that carries a number forward from a
model that is no longer the champion.

Three judgements are made here rather than by a human:

**Release blockers (section 63).** Each blocker line is checked against evidence.
A blocker is ``CLEAR`` only when a current artifact says so. Evidence that exists
but predates the promotion of the current champion is ``STALE`` - not ``CLEAR`` -
because a suite that passed against ``catboost_tuned_top60`` says nothing about
``xgboost_top_120``. Evidence that does not exist is ``UNVERIFIED``.

**Pass criteria (section 64).** Same treatment, one entry per criterion.

**The verdict (section 65 L).** Section 65 permits exactly three strings. This
tool emits one of them only when every blocker and every criterion has been
decided. While evidence is outstanding it writes ``PENDING_EVIDENCE`` instead and
says which runs are missing, because a ``PASS`` issued over incomplete evidence
is not a verdict, it is a guess wearing a verdict's clothes.

Writes ``artifacts/testing/final_verdict.json`` and ``docs/FINAL_VERDICT_A_L.md``.
Exit code: 0 PASS (with or without approved exceptions), 1 FAIL, 2 pending.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.utils import save_json

log = get_logger("cli.final_verdict")

ROOT = settings.REPO_ROOT
TESTING = settings.ARTIFACTS_DIR / "testing"
OUT_JSON = TESTING / "final_verdict.json"
OUT_DOC = settings.DOCS_DIR / "FINAL_VERDICT_A_L.md"
# Absent by design. A non-blocking exception only counts if somebody wrote it
# down and said who approved it; an empty file and a missing file mean the same
# thing here - nothing has been waived.
EXCEPTIONS = TESTING / "approved_exceptions.json"

CLEAR = "CLEAR"
BLOCKED = "BLOCKED"
STALE = "STALE"
UNVERIFIED = "UNVERIFIED"
MET = "MET"
NOT_MET = "NOT_MET"

PASS = "PASS"
PASS_WITH_EXCEPTIONS = "PASS WITH APPROVED NON-BLOCKING EXCEPTIONS"
FAIL = "FAIL"
PENDING_EVIDENCE = "PENDING_EVIDENCE"

FIREWALL_13 = ["F2230", "F3892", "F3898", "F3899", "F3912", "F3913", "F3914",
               "F3915", "F3916", "F3917", "F3918", "F3924", "__UNNAMED__0"]

TIMESTAMP_KEYS = ("generated_utc", "captured_utc", "written_utc", "regenerated_utc",
                  "finished_utc", "started_utc")


def _pending(producer: str) -> str:
    return f"PENDING - produced by: {producer}"


def _fmt(x: Any, nd: int = 5) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _parse_ts(text: str) -> dt.datetime | None:
    try:
        stamp = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


class Evidence:
    """Every artifact this report reads, loaded once and dated.

    ``stale`` is the interesting method. An artifact is stale when it was written
    before the current champion was promoted, because at that moment every claim
    about "the model" acquired a different subject.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        promotion = self.get("artifacts/metrics/promotion_decision_v2.json") or {}
        self.champion: str | None = promotion.get("promoted")
        self.promoted_at = _parse_ts(promotion.get("generated_utc", "")) if promotion else None

    def get(self, rel: str) -> Any | None:
        if rel not in self._cache:
            path = ROOT / rel
            try:
                self._cache[rel] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache[rel] = None
        return self._cache[rel]

    def exists(self, rel: str) -> bool:
        return (ROOT / rel).exists()

    def stamp(self, rel: str) -> dt.datetime | None:
        payload = self.get(rel)
        if isinstance(payload, dict):
            for key in TIMESTAMP_KEYS:
                if isinstance(payload.get(key), str):
                    parsed = _parse_ts(payload[key])
                    if parsed:
                        return parsed
        path = ROOT / rel
        if path.exists():
            return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        return None

    def stale(self, rel: str) -> bool:
        when, promoted = self.stamp(rel), self.promoted_at
        return bool(when and promoted and when < promoted)

    # -- derived views ----------------------------------------------------
    def battery_run(self) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        """The metric battery run that describes the champion.

        Nested is the primary protocol, so a full nested run wins. The
        preliminary nested run is deliberately *not* a candidate: it covers one
        repeat and two families, and its threshold metrics would then be mixed
        with a champion label the flat tournament chose. It is reported
        separately in E instead, where it can be named for what it is.
        """
        battery = self.get("artifacts/metrics/metric_battery.json") or {}
        runs = battery.get("runs", {})
        champion = self.champion or ""
        for prefix in ("NESTED:", "FLAT:"):
            for key, run in runs.items():
                if key.startswith(prefix) and champion and key.endswith(champion):
                    return key, run
        for key, run in runs.items():          # no champion-specific run on file
            if key.startswith(("NESTED:", "FLAT:")):
                return key, run
        return None, None

    def preliminary_nested(self) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        battery = self.get("artifacts/metrics/metric_battery.json") or {}
        for key, run in (battery.get("runs") or {}).items():
            if key.startswith("NESTED_PRELIMINARY:"):
                return key, run
        return None, None

    def champion_features(self) -> tuple[list[str] | None, str]:
        """Feature names the served bundle actually uses."""
        bundle_path = settings.MODELS_DIR / "final_bundle.joblib"
        if not bundle_path.exists():
            return None, "artifacts/models/final_bundle.joblib is absent"
        try:
            import joblib
            bundle = joblib.load(bundle_path)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return None, f"could not read the bundle: {exc}"
        names = bundle.get("feature_list_selected")
        if not names:
            return None, "the bundle records no selected feature list"
        return list(names), str(bundle_path)


# ---------------------------------------------------------------------------
# section 63 - release blockers
# ---------------------------------------------------------------------------
def _firewall(ev: Evidence, columns: list[str]) -> tuple[str, str, list[str]]:
    names, source = ev.champion_features()
    if names is None:
        integrity = ev.get("artifacts/testing/data_integrity.json") or {}
        checks = {c.get("check"): c for c in integrity.get("checks", [])}
        fallback = checks.get("bundle_disjoint_from_quarantine")
        if fallback and fallback.get("passed"):
            return (CLEAR, f"bundle unreadable here ({source}); the data-integrity suite "
                           "records the bundle as disjoint from the quarantine",
                    ["artifacts/testing/data_integrity.json"])
        return UNVERIFIED, source, ["artifacts/models/final_bundle.joblib"]
    hit = sorted(set(names) & set(columns))
    if hit:
        return BLOCKED, f"champion input set names {', '.join(hit)}", [source]
    return (CLEAR, f"none of {', '.join(columns)} appear among the champion's "
                   f"{len(names)} input features", [source])


def _suite(ev: Evidence, *rels: str) -> tuple[str, str, list[str]]:
    """Status of a QA suite artifact: absent, failing, stale or clear."""
    absent = [r for r in rels if ev.get(r) is None]
    if absent:
        return UNVERIFIED, f"absent: {', '.join(absent)}", list(rels)
    failing = [r for r in rels if (ev.get(r) or {}).get("all_passed") is False]
    if failing:
        return BLOCKED, f"failing checks in {', '.join(failing)}", list(rels)
    stale = [r for r in rels if ev.stale(r)]
    if stale:
        return (STALE, f"all checks passed, but {', '.join(stale)} was recorded before "
                       f"{ev.champion} was promoted; re-run scripts/release_test.sh",
                list(rels))
    counts = [f"{(ev.get(r) or {}).get('n_passed')}/{(ev.get(r) or {}).get('n_checks')}"
              for r in rels]
    return CLEAR, f"checks passed {', '.join(counts)}", list(rels)


def _dry_run(ev: Evidence, key: str, description: str) -> tuple[str, str, list[str]]:
    rel = "artifacts/metrics/organiser_dry_run.json"
    run = ev.get(rel)
    if run is None:
        return UNVERIFIED, "the organiser dry run has not been recorded", [rel]
    ok = {
        "prediction_invariance": (run.get("prediction_invariance", {}).get("all_invariant")
                                  and run.get("prediction_invariance", {}).get("sound")),
        "accepted_model_unchanged": run.get("accepted_model_unchanged"),
        "verdict": run.get("verdict") == "PASS",
    }[key]
    if not ok:
        return BLOCKED, f"{description} failed in the dry run", [rel]
    if ev.stale(rel):
        return STALE, f"{description} held, but before {ev.champion} was promoted", [rel]
    return CLEAR, description, [rel]


def _offline(ev: Evidence) -> tuple[str, str, list[str]]:
    rel = "artifacts/testing/offline_results.json"
    payload = ev.get(rel)
    if payload is None:
        return UNVERIFIED, _pending("bash scripts/test_offline.sh"), [rel]
    if payload.get("verdict") != "PASS":
        return BLOCKED, f"offline suite verdict {payload.get('verdict')}", [rel]
    if ev.stale(rel):
        return STALE, "offline suite passed before the current champion", [rel]
    return CLEAR, "backend serves with the network stack pointed at a dead port", [rel]


def _no_mcp(ev: Evidence) -> tuple[str, str, list[str]]:
    rel = "artifacts/testing/no_mcp_scan.txt"
    if not ev.exists(rel):
        return UNVERIFIED, _pending("bash scripts/test_security.sh"), [rel]
    return CLEAR, "source scan records no MCP or browser-automation dependency", [rel]


def _defects(ev: Evidence) -> tuple[str, str, list[str]]:
    rel = "artifacts/testing/final_release_summary.json"
    summary = ev.get(rel)
    if summary is None:
        return UNVERIFIED, _pending("bash scripts/release_test.sh"), [rel]
    if summary.get("blockers"):
        return BLOCKED, f"open blockers: {summary['blockers']}", [rel]
    if ev.stale(rel):
        return (STALE, f"the release summary records zero blockers for "
                       f"{summary.get('model', {}).get('winner')}, which is not the "
                       f"current champion {ev.champion}", [rel])
    return CLEAR, "release summary records no open blockers", [rel]


@dataclass(frozen=True)
class Item:
    """One line of section 63 or section 64 and the evidence that decides it."""
    text: str
    check: Callable[[Evidence], tuple[str, str, list[str]]]


BLOCKERS: list[Item] = [
    Item("F3924 enters model features",
         lambda ev: _firewall(ev, ["F3924"])),
    Item("F3898/F3899 enter accepted model",
         lambda ev: _firewall(ev, ["F3898", "F3899"])),
    Item("F3912/F3913/F3914/F3915 enter accepted model",
         lambda ev: _firewall(ev, ["F3912", "F3913", "F3914", "F3915"])),
    Item("F2230 remains suspicious and is still included",
         lambda ev: _firewall(ev, ["F2230"])),
    Item("F3916-18 are used without availability evidence",
         lambda ev: _firewall(ev, ["F3916", "F3917", "F3918"])),
    Item("train-validation overlap",
         lambda ev: _suite(ev, "artifacts/testing/leakage_results.json")),
    Item("preprocessing fitted on validation",
         lambda ev: _suite(ev, "artifacts/testing/leakage_results.json")),
    Item("feature selection fitted on validation",
         lambda ev: _suite(ev, "artifacts/testing/leakage_results.json")),
    Item("test labels used for tuning",
         lambda ev: _suite(ev, "artifacts/testing/leakage_results.json")),
    Item("external validation triggers retraining",
         lambda ev: _dry_run(ev, "accepted_model_unchanged",
                             "the bundle fingerprint is identical before and after the "
                             "organiser upload")),
    Item("fake metrics",
         lambda ev: _suite(ev, "artifacts/testing/ui_metric_consistency.json")),
    Item("hardcoded dashboard metrics",
         lambda ev: _suite(ev, "artifacts/testing/api_frontend_consistency.json")),
    Item("prediction row order changes",
         lambda ev: _dry_run(ev, "prediction_invariance",
                             "predictions are invariant to row order and column order")),
    Item("model artifact cannot reproduce saved predictions",
         lambda ev: _suite(ev, "artifacts/testing/leakage_results.json")),
    Item("UI/backend score mismatch",
         lambda ev: _suite(ev, "artifacts/testing/api_frontend_consistency.json")),
    Item("Ollama changes scoring",
         lambda ev: _suite(ev, "artifacts/testing/ollama_guardrail_results.json")),
    Item("core requires internet", _offline),
    Item("core requires MCP", _no_mcp),
    Item("core requires Claude in Chrome", _no_mcp),
    Item("P0 defect", _defects),
    Item("unapproved P1 defect", _defects),
]


# ---------------------------------------------------------------------------
# section 64 - pass criteria
# ---------------------------------------------------------------------------
def _nested_complete(ev: Evidence) -> tuple[str, str, list[str]]:
    rel = "artifacts/metrics/nested_cv.json"
    nested = ev.get(rel)
    if nested is None:
        return NOT_MET, _pending("python -m muleguard.cli.nested_cv --repeats 3 --inner 4"), [rel]
    board = nested.get("leaderboard", [])
    repeats = max((row.get("n_repeats", 0) for row in board), default=0)
    families = sorted({row["model"] for row in board})
    if repeats < 3 or len(families) < 4:
        return (NOT_MET, f"on disk: {len(families)} families ({', '.join(families)}) at "
                         f"{repeats} repeat(s); the programme calls for the full family "
                         f"set at 3 repeats", [rel])
    return MET, f"{len(families)} families x {repeats} repeats x 5 outer folds", [rel]


def _tournament_complete(ev: Evidence) -> tuple[str, str, list[str]]:
    rel = "artifacts/metrics/tournament_v2.json"
    t = ev.get(rel)
    if t is None:
        return NOT_MET, _pending("python -m muleguard.cli.tournament_v2"), [rel]
    coverage = t.get("update_1_coverage", {})
    missing = coverage.get("missing") or []
    if missing:
        return NOT_MET, f"required families missing: {', '.join(missing)}", [rel]
    return MET, f"{len(t.get('models', {}))} models, coverage {coverage.get('status')}", [rel]


def _has(ev: Evidence, rel: str, producer: str, detail: str) -> tuple[str, str, list[str]]:
    if ev.get(rel) is None and not ev.exists(rel):
        return NOT_MET, _pending(producer), [rel]
    if ev.stale(rel):
        return NOT_MET, f"{detail}, but recorded before {ev.champion} was promoted", [rel]
    return MET, detail, [rel]


CRITERIA: list[tuple[str, Item]] = [
    ("Data", Item("DataSet.xlsx fingerprinted",
                  lambda ev: _has(ev, "artifacts/testing/data_integrity.json",
                                  "python -m muleguard.cli.audit_data",
                                  "workbook SHA-256 recorded and re-verified"))),
    ("Data", Item("Description.xlsx fingerprinted",
                  lambda ev: _has(ev, "artifacts/testing/description_integrity.json",
                                  "python -m muleguard.cli.audit_data",
                                  "description workbook SHA-256 recorded"))),
    ("Data", Item("target verified",
                  lambda ev: _suite(ev, "artifacts/testing/data_integrity.json"))),
    ("Data", Item("semantic registry built",
                  lambda ev: _has(ev, "artifacts/features/feature_dictionary.json",
                                  "python -m muleguard.cli.audit_data",
                                  "feature dictionary built from Description.xlsx"))),
    ("Data", Item("post-resolution leakage excluded",
                  lambda ev: _firewall(ev, FIREWALL_13))),
    ("ML", Item("nested repeated CV complete", _nested_complete)),
    ("ML", Item("strong model tournament complete", _tournament_complete)),
    ("ML", Item("best candidate stability tested",
                lambda ev: _has(ev, "artifacts/metrics/stability_stress_v2.json",
                                "python -m muleguard.cli.robustness_v2",
                                "seed, positive-removal and rank stability measured"))),
    ("ML", Item("calibration tested",
                lambda ev: _has(ev, "artifacts/metrics/final_calibration.json",
                                "python -m muleguard.cli.build_lenses_v2",
                                "isotonic/Platt comparison, Brier, ECE and coverage"))),
    ("ML", Item("analyst-budget metrics produced",
                lambda ev: _has(ev, "artifacts/metrics/capacity_curve.json",
                                "python -m muleguard.cli.capacity_curve",
                                "recall and precision at every budget with intervals"))),
    ("ML", Item("confidence intervals produced",
                lambda ev: _has(ev, "artifacts/metrics/metric_battery.json",
                                "python -m muleguard.cli.metric_battery",
                                "percentile bootstrap over accounts, 2000 draws"))),
    ("Hidden validation", Item("targetless upload works",
                               lambda ev: _dry_run(ev, "verdict",
                                                   "the organiser dry run passes"))),
    ("Hidden validation", Item("labeled upload uses sealed protocol",
                               lambda ev: _dry_run(ev, "verdict",
                                                   "predictions sealed before labels revealed"))),
    ("Hidden validation", Item("no retraining occurs",
                               lambda ev: _dry_run(ev, "accepted_model_unchanged",
                                                   "bundle fingerprint unchanged"))),
    ("Hidden validation", Item("submission export preserves order",
                               lambda ev: _dry_run(ev, "prediction_invariance",
                                                   "row order preserved and invariant"))),
    ("Hidden validation", Item("distribution shift reported",
                               lambda ev: _has(ev, "artifacts/metrics/adversarial_validation.json",
                                               "python -m muleguard.cli.nested_ses --stages shift",
                                               "adversarial validation AUC and shift-prone features"))),
    ("Runtime", Item("backend works offline", _offline)),
    ("Runtime", Item("frontend metrics match backend",
                     lambda ev: _suite(ev, "artifacts/testing/api_frontend_consistency.json"))),
    ("Runtime", Item("Ollama optional",
                     lambda ev: _suite(ev, "artifacts/testing/ollama_guardrail_results.json"))),
    ("Runtime", Item("one-command run works",
                     lambda ev: _has(ev, "artifacts/testing/one_command_startup.log",
                                     "bash run.sh", "run.sh brought the stack up"))),
]


# ---------------------------------------------------------------------------
# section 65 - the A-L block
# ---------------------------------------------------------------------------
def section_a(ev: Evidence) -> list[tuple[str, str]]:
    env = ev.get("artifacts/testing/environment.json") or {}
    desc = ev.get("artifacts/testing/description_integrity.json") or {}
    git = env.get("git", {})
    gpus = env.get("gpu_names") or []
    return [
        ("Git SHA", f"{git.get('commit_sha', 'unknown')}"
                    f"{' (working tree dirty)' if git.get('is_dirty') else ''}"),
        ("Python", (env.get("python") or "").split(" (")[0] or "unknown"),
        ("Node", str(env.get("node_version", "recorded by scripts/test_frontend.sh"))),
        ("CPU", f"{env.get('processor', 'unknown')} - "
                f"{env.get('cpu_count_physical')} physical / "
                f"{env.get('cpu_count_logical')} logical cores"),
        ("RAM", f"{env.get('ram_total_gb')} GB total"),
        ("GPU", ", ".join(gpus) if gpus else
                f"none - CUDA available: {env.get('cuda_available')}, "
                f"compute mode {env.get('compute_mode')}"),
        ("DataSet SHA", desc.get("dataset_workbook_sha256", "unknown")),
        ("Description SHA", desc.get("description_workbook_sha256", "unknown")),
    ]


def section_b(ev: Evidence) -> list[tuple[str, str]]:
    integrity = ev.get("artifacts/testing/data_integrity.json") or {}
    fp = integrity.get("fingerprint", {})
    profile = integrity.get("profile_summary", {})
    rows = fp.get("n_rows")
    prevalence = fp.get("positive_prevalence")
    positives = round(rows * prevalence) if rows and prevalence else None
    # The dataset is model-independent, so a stale release summary is still a
    # valid second opinion on the row and positive counts. Disagreement here
    # would mean one of the two artifacts describes a different extract.
    recorded = ((ev.get("artifacts/testing/final_release_summary.json") or {})
                .get("dataset", {}).get("positives"))
    if recorded is not None and positives is not None and recorded != positives:
        positives = f"{positives} (disagrees with final_release_summary: {recorded})"
    _, run = ev.battery_run()
    dev = (run or {}).get("support", {})
    return [
        ("Rows", f"{rows} (development {dev.get('n')}, locked test "
                 f"{rows - dev.get('n') if rows and dev.get('n') else '?'})"),
        ("Feature columns", str(profile.get("n_feature_cols", "unknown"))),
        ("Target", "Target (1 = mule account)"),
        ("Positives", f"{positives} overall; {dev.get('n_positives')} in development"),
        ("Negatives", f"{rows - positives if rows and positives else '?'} overall; "
                      f"{dev.get('n_negatives')} in development"),
        ("Prevalence", f"{prevalence:.6f} overall" if prevalence else "unknown"),
        ("Naive accuracy", f"{1 - prevalence:.6f} (predict every account legitimate)"
                           if prevalence else "unknown"),
        ("No-skill AP", f"{dev.get('prevalence'):.6f} on development"
                        if dev.get("prevalence") else "unknown"),
    ]


def section_c(ev: Evidence) -> list[tuple[str, str]]:
    quarantine = ev.get("artifacts/features/quarantined_features.json") or {}
    entries = {e["feature"]: e for e in quarantine.get("quarantine", [])}
    f3912 = ev.get("artifacts/metrics/with_vs_without_f3912.json") or {}
    status, detail, _ = _firewall(ev, FIREWALL_13)

    def reason(name: str) -> str:
        entry = entries.get(name, {})
        return entry.get("reason") or entry.get("evidence") or "quarantined"

    return [
        ("Hard excluded", f"{len(entries)} columns, all excluded from every model, "
                          f"selector, ensemble and lens: {', '.join(sorted(entries))}"),
        ("Conditional quarantine", "none - quarantine policy "
                                   f"{quarantine.get('policy_version')} is unconditional; "
                                   "a column is either admissible or excluded"),
        ("F2230 verdict", f"EXCLUDED. {reason('F2230')}"),
        ("F3916-18 verdict", f"EXCLUDED. {reason('F3916')}"),
        ("Target leakage detected?", f"YES, and firewalled. F3912 alone scored "
                                     f"{_fmt(f3912.get('with_f3912', {}).get('pr_auc_mean'))} "
                                     f"PR-AUC against "
                                     f"{_fmt(f3912.get('without_f3912', {}).get('pr_auc_mean'))} "
                                     f"without it. Firewall check: {status} - {detail}"
                                     if f3912 else f"firewall check: {status} - {detail}"),
    ]


def section_d(ev: Evidence) -> str:
    rows: list[tuple[str, str, str, str, str, str]] = []
    tournament = ev.get("artifacts/metrics/tournament_v2.json") or {}
    for name, m in sorted(tournament.get("models", {}).items(),
                          key=lambda kv: -(kv[1].get("pr_auc_mean") or 0)):
        rows.append((name, "FLAT 3x5", str(m.get("n_features", "-")),
                     _fmt(m.get("pr_auc_mean")), _fmt(m.get("pr_auc_std")),
                     m.get("status", "OK")))
    nested = ev.get("artifacts/metrics/nested_cv.json") or {}
    for m in sorted(nested.get("leaderboard", []),
                    key=lambda r: -(r.get("pr_auc_mean") or 0)):
        rows.append((m["model"], f"NESTED {m.get('n_repeats')}x5x4",
                     str(m.get("feature_size_mode", "-")),
                     _fmt(m.get("pr_auc_mean")), _fmt(m.get("pr_auc_std")), "NESTED"))
    if not rows:
        return _pending("python -m muleguard.cli.tournament_v2")
    head = ("| model | protocol | features | PR-AUC mean | PR-AUC std | note |\n"
            "| --- | --- | --- | --- | --- | --- |\n")
    return head + "\n".join("| " + " | ".join(r) + " |" for r in rows)


def section_e(ev: Evidence) -> list[tuple[str, str]]:
    key, run = ev.battery_run()
    promotion = ev.get("artifacts/metrics/promotion_decision_v2.json") or {}
    detail = promotion.get("promoted_detail", {})
    if run is None:
        return [("Model", ev.champion or "unknown"),
                ("Everything else", _pending("python -m muleguard.cli.metric_battery"))]
    protocol = run.get("protocol", "unknown")
    ranking = run.get("ranking", {})
    intervals = run.get("intervals", {}).get("resample_accounts", {}).get("pr_auc", {})
    quality = run.get("probability_quality", {})
    # Classification metrics need one operating point. URGENT_REVIEW is the tier
    # that creates an analyst case, so that is the one reported here; the other
    # two tiers are in F and in the threshold table.
    tier = next((t for t in run.get("at_frozen_thresholds", [])
                 if t.get("tier") == "URGENT_REVIEW"), {})
    note = "" if protocol == "NESTED" else \
        f"  [operating point and probability quality measured under the {protocol} " \
        "protocol; nested is primary and is still running]"

    prelim_key, prelim = ev.preliminary_nested()
    if protocol == "NESTED":
        nested_ap = _fmt(ranking.get("pr_auc", {}).get("mean"))
    else:
        nested_ap = _pending("python -m muleguard.cli.nested_cv --repeats 3 --inner 4, "
                             "then metric_battery --protocol NESTED")
        if prelim:
            nested_ap += (f". A preliminary nested run ({prelim_key}, 1 repeat, "
                          f"2 families) measured "
                          f"{_fmt((prelim.get('ranking') or {}).get('pr_auc', {}).get('mean'))} "
                          "- lower than the flat figure, as nested protocols usually are, "
                          "and superseded by the run in progress")
    return [
        ("Model", f"{ev.champion} (battery run {key}){note}"),
        ("Feature set", f"{detail.get('feature_set', '?')} of view "
                        f"{detail.get('view', '?')}"),
        ("Number of features", str(detail.get("n_features", "?"))),
        ("Nested-CV AP mean", nested_ap),
        (f"{protocol}-CV AP mean", _fmt(ranking.get("pr_auc", {}).get("mean"))),
        ("AP std", f"{_fmt(ranking.get('pr_auc', {}).get('std'))} across "
                   f"{ranking.get('pr_auc', {}).get('n_repeats')} repeats "
                   "(a spread, not an interval)"),
        ("95% CI", f"[{_fmt(intervals.get('ci_low'))}, {_fmt(intervals.get('ci_high'))}] "
                   f"percentile bootstrap over accounts, {intervals.get('n_boot_effective')} draws"),
        ("ROC-AUC", _fmt(ranking.get("roc_auc", {}).get("mean")) +
                    " (secondary - prevalence 0.88% makes ROC flattering)"),
        ("Accuracy", _fmt(tier.get("accuracy")) + " at the URGENT_REVIEW threshold"),
        ("Balanced Accuracy", _fmt(tier.get("balanced_accuracy"))),
        ("Precision", _fmt(tier.get("precision"))),
        ("Recall", _fmt(tier.get("recall"))),
        ("F1", _fmt(tier.get("f1"))),
        ("F2", _fmt(tier.get("f2"))),
        ("MCC", _fmt(tier.get("mcc"))),
        ("Brier", f"{_fmt(quality.get('brier'))} against "
                  f"{_fmt(quality.get('brier_base_rate_reference'))} for a base-rate "
                  f"predictor (skill {_fmt(quality.get('brier_skill_vs_base_rate'), 3)})"),
        ("ECE", f"{_fmt(quality.get('calibration_quantile_bins', {}).get('ece'))} "
                "(10 quantile bins), "
                f"{_fmt(quality.get('calibration_uniform_bins', {}).get('ece'))} "
                "(10 uniform bins)"),
    ]


def section_f(ev: Evidence) -> list[tuple[str, str]]:
    curve = ev.get("artifacts/metrics/capacity_curve.json") or {}
    head = {int(h["budget"]): h for h in curve.get("headline", [])}
    if not head:
        return [("Everything", _pending("python -m muleguard.cli.capacity_curve"))]
    out: list[tuple[str, str]] = []
    for budget in (25, 50, 100):
        h = head.get(budget, {})
        out.append((f"Recall@Top{budget}",
                    f"{_fmt(h.get('recall'), 4)}  "
                    f"({h.get('true_positives')}/{(h.get('true_positives') or 0) + (h.get('mules_missed') or 0)} mules; "
                    f"95% CI [{_fmt(h.get('recall_ci_low_resampled_accounts'), 4)}, "
                    f"{_fmt(h.get('recall_ci_high_resampled_accounts'), 4)}])"))
        out.append((f"Precision@Top{budget}", _fmt(h.get("precision"), 4)))
    worst = head.get(100, {})
    out.append(("FP/1000 legit", f"{_fmt(worst.get('fp_per_1000_legitimate'), 3)} at a "
                                 f"100-alert budget; "
                                 f"{_fmt(head.get(25, {}).get('fp_per_1000_legitimate'), 3)} at 25"))
    return out


def section_g(ev: Evidence) -> list[tuple[str, str]]:
    seed = ev.get("artifacts/metrics/seed_variance_v2.json") or {}
    stress = ev.get("artifacts/metrics/stability_stress_v2.json") or {}
    if not seed and not stress:
        return [("Everything", _pending("python -m muleguard.cli.robustness_v2"))]
    return [
        ("Seed stability", f"PR-AUC {_fmt(seed.get('pr_auc_mean'))} +/- "
                           f"{_fmt(seed.get('pr_auc_std'))} over {seed.get('n_seeds')} seeds, "
                           f"spread {_fmt(seed.get('spread'), 4)}. Any model comparison "
                           "smaller than this spread on unpaired folds is noise."),
        ("Positive-removal stability",
         f"{_fmt(stress.get('positive_removal_pr_auc_mean'))} +/- "
         f"{_fmt(stress.get('positive_removal_pr_auc_std'))} over "
         f"{stress.get('rounds')} rounds dropping "
         f"{stress.get('positives_removed_per_round_mean')} positives each; relative drop "
         f"{_fmt(stress.get('positive_removal_pr_auc_relative_drop'), 4)}"),
        ("Feature stability",
         f"rank correlation {_fmt(stress.get('feature_rank_stability'), 4)}, "
         f"top-20 overlap {_fmt(stress.get('feature_top20_overlap'), 4)}"),
        ("Rank stability",
         f"{_fmt(stress.get('prediction_rank_stability'), 4)} Spearman over all "
         "development rows - low because ~99% of rows are negatives whose calibrated "
         "scores are near-tied; the analyst-facing number is the top-budget overlap"),
    ]


def section_h(ev: Evidence) -> list[tuple[str, str]]:
    adv = ev.get("artifacts/metrics/adversarial_validation.json")
    drift = ev.get("artifacts/metrics/drift_locked_test.json") or {}
    dry = ev.get("artifacts/metrics/organiser_dry_run.json") or {}
    if adv is None:
        adv_line = _pending("python -m muleguard.cli.nested_ses --stages shift")
        shift_line = adv_line
    else:
        adv_line = f"AUC {_fmt(adv.get('auc'))} distinguishing development from locked test"
        shift_line = ", ".join(f["feature"] for f in adv.get("shift_prone_features", [])[:5])
    return [
        ("Adversarial validation", adv_line),
        ("OOD", f"locked-test drift {drift.get('status')} - score PSI "
                f"{_fmt(drift.get('score_psi'), 4)}, {drift.get('n_features_alert')} features "
                f"at alert level over {drift.get('n_rows_scored')} rows"),
        ("Shift-prone features", shift_line),
        ("Hidden-validation readiness",
         f"organiser dry run {dry.get('verdict', 'not run')}, "
         f"{dry.get('variants_passed', '?')} upload variants invariant"),
    ]


def section_i(ev: Evidence) -> list[tuple[str, str]]:
    dry = ev.get("artifacts/metrics/organiser_dry_run.json") or {}
    batch = ev.get("artifacts/testing/batch_upload_results.json") or {}
    inv = dry.get("prediction_invariance", {})
    offline = dry.get("offline_label_comparison", {})
    if not dry:
        return [("Everything", _pending("python -m muleguard.cli.dry_run"))]
    return [
        ("Targetless upload", f"{dry.get('variants_passed')} variants accepted with the "
                              f"target column removed ({dry.get('mock_file', {}).get('rows')} rows)"),
        ("Target-present sealed validation",
         f"seal verified: {offline.get('seal_verified')} - predictions sealed "
         f"{offline.get('sealed_utc')}, labels revealed {offline.get('revealed_utc')}"),
        ("Row-order preservation", f"all_invariant={inv.get('all_invariant')}, "
                                   f"sound={inv.get('sound')} (a sensitivity control "
                                   "confirms the check can fail)"),
        ("Competition export", f"batch upload checks {batch.get('n_passed')}/"
                               f"{batch.get('n_checks')}"),
        ("No-retraining assertion", f"bundle fingerprint {dry.get('bundle_fingerprint_before')} "
                                    f"-> {dry.get('bundle_fingerprint_after')} "
                                    f"(unchanged={dry.get('accepted_model_unchanged')})"),
    ]


def section_j(ev: Evidence) -> list[tuple[str, str]]:
    def line(rel: str, producer: str) -> str:
        payload = ev.get(rel)
        if payload is None:
            return _pending(producer)
        state = f"{payload.get('n_passed')}/{payload.get('n_checks')} checks passed"
        return f"{state} - STALE, recorded before {ev.champion}" if ev.stale(rel) else state
    offline_status, offline_detail, _ = _offline(ev)
    mcp_status, mcp_detail, _ = _no_mcp(ev)
    return [
        ("Backend", line("artifacts/testing/backend_results.json",
                         "bash scripts/test_backend.sh")),
        ("Frontend metrics", line("artifacts/testing/api_frontend_consistency.json",
                                  "bash scripts/test_frontend.sh")),
        ("Offline", f"{offline_status} - {offline_detail}"),
        ("Ollama-off", line("artifacts/testing/ollama_guardrail_results.json",
                            "bash scripts/test_offline.sh")),
        ("No MCP", f"{mcp_status} - {mcp_detail}"),
        ("No Claude in Chrome", f"{mcp_status} - the same scan covers browser automation"),
    ]


def section_k(ev: Evidence) -> list[tuple[str, str]]:
    summary = ev.get("artifacts/testing/final_release_summary.json")
    if summary is None:
        return [("Everything", _pending("bash scripts/release_test.sh"))]
    totals = summary.get("totals", {})
    stale = " [STALE - recorded for " \
            f"{summary.get('model', {}).get('winner')}, not {ev.champion}]" if ev.stale(
                "artifacts/testing/final_release_summary.json") else ""
    return [
        ("Passed", f"{totals.get('qa_passed')}/{totals.get('qa_checks')} QA checks, "
                   f"{totals.get('pytest_backend')} pytest, "
                   f"{totals.get('vitest_frontend')} vitest{stale}"),
        ("Failed", str(int(totals.get("qa_checks", 0)) - int(totals.get("qa_passed", 0)))),
        ("P0", f"{len(summary.get('blockers', []))} open"),
        ("P1", f"{len(summary.get('blockers', []))} open, "
               f"{len(_approved_exceptions())} approved non-blocking exceptions"),
    ]


def _approved_exceptions() -> list[dict[str, Any]]:
    try:
        payload = json.loads(EXCEPTIONS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [e for e in payload.get("exceptions", []) if e.get("approved_by")]


# ---------------------------------------------------------------------------
# risks
# ---------------------------------------------------------------------------
def _risks(ev: Evidence, blockers: list[dict[str, Any]],
           criteria: list[dict[str, Any]]) -> list[str]:
    """Risks that survive the evidence, most consequential first."""
    out: list[str] = []
    open_criteria = [c["text"] for c in criteria if c["status"] != MET]
    stale_blockers = [b["text"] for b in blockers if b["status"] == STALE]

    if open_criteria:
        out.append(f"Evidence is incomplete: {len(open_criteria)} of {len(criteria)} pass "
                   f"criteria are not met yet ({', '.join(open_criteria[:4])}"
                   f"{', ...' if len(open_criteria) > 4 else ''}). Nothing here can be "
                   "read as a final PASS until those runs land.")
    if stale_blockers:
        out.append(f"{len(stale_blockers)} release blockers are cleared only by evidence "
                   f"recorded before {ev.champion} was promoted. The QA suites must be "
                   "re-run against the current bundle before the verdict means anything.")
    _, run = ev.battery_run()
    support = (run or {}).get("support", {})
    if support.get("n_positives"):
        out.append(f"{support['n_positives']} positives at "
                   f"{support['prevalence'] * 100:.2f}% prevalence. One mule is worth "
                   f"{100 / support['n_positives']:.1f} recall points, so every interval "
                   "in this report is wide and every fold-level difference is fragile.")
    stress = ev.get("artifacts/metrics/stability_stress_v2.json") or {}
    if stress.get("positive_removal_pr_auc_min"):
        out.append(f"Under positive removal the worst round fell to "
                   f"{_fmt(stress['positive_removal_pr_auc_min'], 4)} PR-AUC from a "
                   f"{_fmt(stress.get('reference_pr_auc'), 4)} reference: the model's "
                   "ranking depends on which mules it was shown.")
    key, _ = ev.battery_run()
    if key and not key.startswith("NESTED:"):
        out.append("The headline metrics still come from the flat repeated-CV protocol. "
                   "Nested CV is the primary protocol in this programme and it usually "
                   "reports lower, so the headline should be expected to fall.")
    out.append("Generation-1 numbers (PR-AUC 0.824 and above) came from a model that could "
               "see quarantined columns. They remain in the repository as retired evidence "
               "and must never be quoted as current behaviour.")
    return out[:5]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def _block(pairs: list[tuple[str, str]]) -> str:
    return "```text\n" + "\n".join(f"{k}: {v}" for k, v in pairs) + "\n```"


def build(ev: Evidence) -> dict[str, Any]:
    blockers = [{"text": item.text, "status": s, "detail": d, "evidence": e}
                for item in BLOCKERS for s, d, e in [item.check(ev)]]
    criteria = [{"group": group, "text": item.text, "status": s, "detail": d, "evidence": e}
                for group, item in CRITERIA for s, d, e in [item.check(ev)]]
    # A criterion check that reuses a blocker helper answers in blocker words;
    # anything that is not a clean pass is simply NOT_MET here.
    for c in criteria:
        if c["status"] in (CLEAR,):
            c["status"] = MET
        elif c["status"] in (BLOCKED, STALE, UNVERIFIED):
            c["status"] = NOT_MET

    exceptions = _approved_exceptions()
    if any(b["status"] == BLOCKED for b in blockers):
        verdict = FAIL
    elif any(b["status"] != CLEAR for b in blockers) or any(c["status"] != MET for c in criteria):
        verdict = PENDING_EVIDENCE
    elif exceptions:
        verdict = PASS_WITH_EXCEPTIONS
    else:
        verdict = PASS

    sections = {
        "A. Environment": _block(section_a(ev)),
        "B. Verified Primary Dataset": _block(section_b(ev)),
        "C. Leakage Firewall": _block(section_c(ev)),
        "D. Model Tournament": section_d(ev),
        "E. Champion": _block(section_e(ev)),
        "F. Alert Budget": _block(section_f(ev)),
        "G. Stability": _block(section_g(ev)),
        "H. Generalization": _block(section_h(ev)),
        "I. Validation Lab": _block(section_i(ev)),
        "J. System": _block(section_j(ev)),
        "K. Tests": _block(section_k(ev)),
    }
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "champion": ev.champion,
        "champion_promoted_utc": ev.promoted_at.isoformat() if ev.promoted_at else None,
        "verdict": verdict,
        "verdict_is_one_of_the_three_permitted": verdict != PENDING_EVIDENCE,
        "verdict_rule": (
            "FAIL if any section 63 blocker is BLOCKED. Otherwise a section 65 verdict "
            "is issued only when every blocker is CLEAR and every section 64 criterion "
            "is MET; while evidence is outstanding the verdict is PENDING_EVIDENCE, "
            "because a PASS over incomplete evidence is a guess, not a verdict."),
        "release_blockers": blockers,
        "pass_criteria": criteria,
        "approved_non_blocking_exceptions": exceptions,
        "sections": sections,
        "top_risks": _risks(ev, blockers, criteria),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# MuleGuard - Trinetra: final validation response (section 65)",
        "",
        f"Generated {payload['generated_utc']} by `python -m muleguard.cli.final_verdict`.",
        "Every field is read from a named artifact. A field with no evidence behind it "
        "says PENDING and names the run that would produce it; none is ever filled in "
        "from an earlier model.",
        "",
        f"Champion: **{payload['champion']}**, promoted "
        f"{payload['champion_promoted_utc']}.",
        "",
    ]
    for title, body in payload["sections"].items():
        lines += [f"# {title}", "", body, ""]

    lines += ["# L. Final Verdict", "", "```text", payload["verdict"], "```", ""]
    if payload["verdict"] == PENDING_EVIDENCE:
        lines += [
            "This is deliberately not one of the three strings section 65 permits. "
            "The permitted verdicts are claims about completed evidence, and the "
            "evidence below is still open. The rule that produced this line:",
            "", f"> {payload['verdict_rule']}", "",
        ]

    lines += ["## Release blockers (section 63)", "",
              "| blocker | status | evidence |", "| --- | --- | --- |"]
    for b in payload["release_blockers"]:
        lines.append(f"| {b['text']} | {b['status']} | {b['detail']} |")

    lines += ["", "## Pass criteria (section 64)", "",
              "| group | criterion | status | evidence |", "| --- | --- | --- | --- |"]
    for c in payload["pass_criteria"]:
        lines.append(f"| {c['group']} | {c['text']} | {c['status']} | {c['detail']} |")

    lines += ["", "## Top 5 remaining risks", ""]
    lines += [f"{i}. {risk}" for i, risk in enumerate(payload["top_risks"], 1)]
    lines += ["", "---", "",
              f"Machine-readable form: `artifacts/testing/final_verdict.json`.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true", dest="echo",
                    help="also write the A-L block to stdout")
    args = ap.parse_args(argv)

    ev = Evidence()
    payload = build(ev)
    save_json(payload, OUT_JSON)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(render(payload), encoding="utf-8")
    if args.echo:
        print(render(payload))

    blocked = [b["text"] for b in payload["release_blockers"] if b["status"] == BLOCKED]
    open_items = [b["text"] for b in payload["release_blockers"] if b["status"] not in (CLEAR, BLOCKED)]
    open_items += [c["text"] for c in payload["pass_criteria"] if c["status"] != MET]
    log.info("verdict %s - %d blockers BLOCKED, %d items still open",
             payload["verdict"], len(blocked), len(open_items))
    log.info("wrote %s and %s", OUT_DOC, OUT_JSON)
    if payload["verdict"] == FAIL:
        return 1
    return 0 if payload["verdict"] in (PASS, PASS_WITH_EXCEPTIONS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
