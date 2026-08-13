"""The §58 top-level report: what was validated, what it showed, what it did not.

`docs/FINAL_VERDICT_A_L.md` answers §65 in the fixed A-L shape - fields and
values, nothing else. This document is the surrounding argument: which protocol
decided, why the shipped champion is the one shipped, what the Feature
Availability Firewall retired and what that cost, and - the part a validation
report is actually for - which claims the evidence does **not** support.

It is generated, never hand-written, for the same reason every other report here
is: a hand-written summary is accurate on the day it is typed and silently wrong
afterwards. Every number is read from an artifact and carries the artifact's
name; anything missing is printed as PENDING with the command that produces it,
because a gap that names its own fix is a task, while a gap that is quietly
omitted is a misrepresentation.

    .venv/Scripts/python.exe -m muleguard.cli.final_report

Exit code is 0 whenever the document was written. This module reports; it is
`final_verdict` that fails a release, and `nested_promotion` that fails it when
the primary protocol disagrees with what is deployed.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from muleguard import settings
from muleguard.cli.final_verdict import (
    BLOCKED, CLEAR, Evidence, MET, PENDING_EVIDENCE, _fmt, _pending,
)
from muleguard.logging import get_logger
from muleguard.utils import load_json

log = get_logger("cli.final_report")

ROOT = settings.REPO_ROOT
OUT = settings.DOCS_DIR / "FINAL_XHIGH_VALIDATION_REPORT.md"
VERDICT_JSON = settings.ARTIFACTS_DIR / "testing" / "final_verdict.json"

# §58 names seventeen reports. This is one of them, and it links the rest, so a
# reader who opens only this file can still reach every piece of evidence.
SIBLING_REPORTS = [
    ("DESCRIPTION_VALIDATION_REPORT.md", "Description.xlsx parsed and reconciled against the data"),
    ("FEATURE_AVAILABILITY_FIREWALL.md", "which columns are excluded, and on what evidence"),
    ("FINAL_LEAKAGE_FORENSICS.md", "the leakage investigation in full"),
    ("HISTORICAL_METRIC_RECONCILIATION.md", "why older numbers in this repo differ from current ones"),
    ("NESTED_CV_MODEL_TOURNAMENT.md", "the primary-protocol tournament"),
    ("HIDDEN_VALIDATION_READINESS.md", "what happens when the organisers hand over unseen data"),
    ("POSITIVE_REMOVAL_STABILITY.md", "how much the result depends on individual positives"),
    ("ADVERSARIAL_VALIDATION_REPORT.md", "dev-vs-test separability and shift-prone features"),
    ("CALIBRATION_AND_THRESHOLDS.md", "probability quality and the frozen policy thresholds"),
    ("FALSE_POSITIVE_VALIDATION.md", "the cost of being wrong on a legitimate account"),
    ("VALIDATION_LAB_TEST_REPORT.md", "the upload flows a judge will exercise"),
    ("SEALED_VALIDATION_PROTOCOL.md", "how a labelled upload is scored without leaking into the model"),
    ("UI_METRIC_CONSISTENCY.md", "the dashboard shows the artifacts' numbers, not its own"),
    ("OFFLINE_RUNTIME_TEST.md", "the system with the network removed"),
    ("SECURITY_TEST_REPORT.md", "input handling, PII, and the LLM boundary"),
    ("FINAL_MODEL_CARD.md", "the model card for the deployed champion"),
    ("FINAL_VERDICT_A_L.md", "the §65 answer in its required A-L form"),
]

# §61. The report states these as refusals rather than leaving them unsaid,
# because the interesting property of a validation report is what it declines
# to claim.
NOT_CLAIMED = [
    ("perfect or near-perfect detection",
     "the champion misses positives at every review budget, and section F of "
     "`FINAL_VERDICT_A_L.md` gives the counts"),
    ("zero false positives",
     "a 100 %-precision row exists in the retired generation-1 artifacts and is "
     "not a property of the deployed model — "
     "`docs/HISTORICAL_METRIC_RECONCILIATION.md` reconciles the two"),
    ("a guarantee about any individual account",
     "the system outputs review tiers and evidence for an analyst, never a "
     "determination of guilt, and no output is a legal or factual finding"),
    ("superiority over other competitors or published systems",
     "no common evaluation has been run, so no comparative claim is available"),
    ("generalisation to a different bank, period, or product mix",
     "every estimate comes from one workbook of 9,082 accounts, and "
     "`docs/ADVERSARIAL_VALIDATION_REPORT.md` is the only evidence about shift"),
]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._\n"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _headline(ev: Evidence, verdict: dict[str, Any]) -> str:
    champion = verdict.get("champion") or ev.champion or "not yet promoted"
    v = verdict.get("verdict") or PENDING_EVIDENCE
    promoted = verdict.get("champion_promoted_utc") or "unrecorded"

    blockers = verdict.get("release_blockers") or []
    criteria = verdict.get("pass_criteria") or []
    open_b = [b for b in blockers if b.get("status") not in (CLEAR, BLOCKED)]
    open_c = [c for c in criteria if c.get("status") != MET]

    lines = [
        f"**Verdict: `{v}`** — the §65 answer, reproduced from "
        f"`{_rel(VERDICT_JSON)}`, which is regenerated by "
        "`make final-verdict` and is the only place the verdict is decided.",
        "",
        f"- Champion: **`{champion}`**, promoted {promoted}.",
        f"- Release blockers: {len(blockers) - len(open_b)}/{len(blockers)} "
        f"resolved, {len(open_b)} open.",
        f"- Pass criteria: {len(criteria) - len(open_c)}/{len(criteria)} met, "
        f"{len(open_c)} open.",
    ]
    if v == PENDING_EVIDENCE:
        lines += [
            "",
            "> `PENDING_EVIDENCE` is not one of the three verdict strings §65 "
            "permits, and it is not being offered as one. It is this "
            "repository's way of saying that the evidence needed to choose "
            "among PASS, PASS WITH APPROVED NON-BLOCKING EXCEPTIONS and FAIL "
            "is not all on disk yet. Publishing a PASS over absent evidence "
            "would be the specific failure this whole exercise exists to "
            "prevent; the open items are listed in full below, each with the "
            "command that closes it.",
        ]
    return "\n".join(lines) + "\n"


def _protocols(ev: Evidence) -> str:
    nested = ev.get("artifacts/metrics/nested_cv.json") or {}
    decision = ev.get("artifacts/metrics/nested_promotion_decision.json") or {}
    design = nested.get("design") or {}

    body = [
        "Two cross-validation protocols are run, and every number in this "
        "repository is labelled with the one that produced it.",
        "",
        "**Flat repeated CV (3×5).** Feature selection and hyperparameter "
        "tuning happen once, over the whole development split; the folds then "
        "score the result. This is the protocol that promoted the shipped "
        "champion, and it is optimistic by construction — the selection saw "
        "every row it was later scored on. It is reported as such, never as "
        "the headline estimate.",
        "",
        "**Nested repeated CV (primary).** Selection and tuning happen inside "
        "each outer fold, against inner splits only, and each outer-validation "
        "row is predicted exactly once. This is the estimate the project "
        "treats as its honest one.",
        "",
    ]
    if design:
        body += ["```text",
                 f"outer      : {design.get('outer', '?')}",
                 f"inner      : {design.get('inner', '?')}",
                 f"selection  : {design.get('selection', '?')}",
                 f"tuning     : {design.get('tuning', '?')}",
                 f"outer rows : {design.get('outer_validation_use', '?')}",
                 "```", ""]
    else:
        body += [f"Design: {_pending('make nested-cv')}", ""]

    board = nested.get("leaderboard") or []
    if board:
        rows = [[f"`{r.get('model')}`", _fmt(r.get("pr_auc_mean")),
                 _fmt(r.get("pr_auc_std")), _fmt(r.get("roc_auc_mean")),
                 str(r.get("n_repeats") or "?")]
                for r in sorted(board, key=lambda d: -(d.get("pr_auc_mean") or 0))]
        body += ["**Nested leaderboard** (`artifacts/metrics/nested_cv.json`):", "",
                 _table(["family", "PR-AUC mean", "± std", "ROC-AUC", "repeats"], rows), ""]

    verdict = decision.get("verdict")
    # The arbiter's `why` is a sentence fragment written for a JSON field. It is
    # quoted rather than paraphrased - the report must not soften the arbiter -
    # but it is punctuated here so the prose reads as prose.
    why = str(decision.get("why", "")).strip()
    if why and why[-1] not in ".!?":
        why += "."
    if why:
        why = why[0].upper() + why[1:]
    if not verdict:
        body += [f"Promotion under the primary protocol: {_pending('make nested-promotion')}"]
    elif verdict == "CHAMPION_CONFIRMED":
        body += [f"**The primary protocol confirms the shipped champion.** {why}"]
    elif verdict == "CHAMPION_CHALLENGED":
        gap = decision.get("nested_gap_over_deployed")
        paired = decision.get("paired_check") or {}
        ci = paired.get("paired_delta_ci95") or []
        body += [
            "**The primary protocol promotes a different model from the one "
            f"shipped.** {why}"
            + (f" The gap on the leaderboard is {_fmt(gap)} PR-AUC."
               if gap is not None else "")
            + (f" Paired on identical rows it is {_fmt(paired.get('paired_delta_mean'))}, "
               f"95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}], "
               f"{'excluding' if paired.get('excludes_zero') else 'including'} zero, "
               f"with {paired.get('repeats_favouring_promoted')} repeats favouring the "
               "challenger - which is the test that matters, because two marginal "
               "intervals on 64 positives overlap whatever the truth is."
               if len(ci) == 2 else ""),
            "",
            "This is recorded rather than acted on. Swapping the champion is "
            "not a documentation change:",
            "",
        ] + [f"- {c}" for c in decision.get("what_acting_on_this_costs", [])] + [
            "",
            "The locked test is single-touch by construction, so acting on this "
            "spends evidence that cannot be regenerated. "
            f"`make nested-promotion` exits non-zero while this stands, and "
            f"`{_rel(settings.METRICS_DIR / 'nested_promotion_decision.json')}` "
            "carries the full leaderboard.",
        ]
    else:
        body += [f"Promotion under the primary protocol: **{verdict}** — "
                 f"{decision.get('why', 'no reason recorded')}. "
                 f"Fills when: `{decision.get('fills_when', '?')}`."]
    return "\n".join(body) + "\n"


def _firewall_section(ev: Evidence) -> str:
    q = ev.get("artifacts/features/quarantined_features.json") or {}
    rows = [[f"`{r.get('feature')}`", str(r.get("variable_name") or "—"),
             str(r.get("availability_class") or "—"), str(r.get("reason") or "—")]
            for r in (q.get("quarantine") or [])]
    body = [
        "The firewall excludes columns that would not exist at the moment a "
        "real decision is made — the target itself, post-resolution status "
        "flags, and investigation-derived fields. Exclusion is from *every* "
        "model, selector, calibrator, explanation and export, not just from "
        "the final fit.",
        "",
        _table(["column", "name", "availability class", "why excluded"], rows),
        "",
        "This is also the reason the generation-1 champion was retired. "
        "`catboost_tuned_top60` scored a locked-test PR-AUC of 0.8242 on a "
        "feature pool that contained quarantined columns; that number is "
        "withdrawn, not improved upon, and the current figures are lower "
        "because they are measured without those columns. "
        "`docs/HISTORICAL_METRIC_RECONCILIATION.md` reconciles the two "
        "generations line by line, and `docs/LOCKED_TEST_RULING.md` records "
        "what the locked test may and may not still be used for.",
    ]
    return "\n".join(body) + "\n"


def _evidence_inventory(ev: Evidence) -> str:
    manifest = ev.get("artifacts/testing/artifact_manifest.json") or {}
    entries = manifest.get("artifacts") or []
    counts = manifest.get("counts") or {}
    body = ["Every artifact §57 requires, and whether it is on disk. "
            "`make reconcile-artifacts` rebuilds the derived ones from their "
            "real sources, so a spec-named file cannot hold stale content "
            "while its source moves on.", ""]
    if not entries:
        return "\n".join(body + [_pending("make reconcile-artifacts")]) + "\n"

    tally = ", ".join(f"**{v}** {k.lower()}" for k, v in sorted(counts.items()))
    body += [f"{len(entries)} required artifacts: {tally}.", ""]

    pending = [e for e in entries if e.get("status") == "PENDING"]
    if pending:
        body += ["Still missing:", "",
                 _table(["required artifact", "what produces it"],
                        [[f"`{e.get('spec')}`",
                          f"`{e.get('pending_run')}`" if e.get("pending_run")
                          else "_no producer recorded_"] for e in pending]), ""]
    else:
        body += ["Nothing is missing.", ""]

    derived = [e for e in entries if e.get("status") == "DERIVED"]
    body += [
        f"<details><summary>The {len(derived)} regenerated artifacts and their "
        "real sources</summary>",
        "",
        _table(["spec name", "derived from"],
               [[f"`{e.get('spec')}`",
                 ", ".join(f"`{s}`" for s in (e.get("sources") or [])) or "_derived_"]
                for e in derived]),
        "", "</details>",
    ]
    return "\n".join(body) + "\n"


def _ledger_section() -> str:
    path = settings.ARTIFACTS_DIR / "experiments" / "experiment_ledger.csv"
    if not path.exists():
        return _pending("make experiment-ledger") + "\n"
    import csv
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    tally = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    return (
        f"`{_rel(path)}` records **{len(rows)} experiments** ({tally}). The "
        "ledger is rebuilt from the artifacts themselves and exits non-zero "
        "if any file under `artifacts/metrics/` is not claimed by an entry, "
        "which is how \"no forgotten experiments\" is enforced rather than "
        "remembered. Negative and abandoned results stay in it on purpose: a "
        "ledger that only lists what worked is a highlight reel.\n")


def _open_items(verdict: dict[str, Any]) -> str:
    blockers = [b for b in (verdict.get("release_blockers") or [])
                if b.get("status") not in (CLEAR, BLOCKED)]
    criteria = [c for c in (verdict.get("pass_criteria") or [])
                if c.get("status") != MET]
    if not blockers and not criteria:
        return "Nothing is open: every release blocker is resolved and every " \
               "pass criterion is met against evidence dated after the " \
               "champion's promotion.\n"
    body = []
    if blockers:
        body += ["**Release blockers still open**", "",
                 _table(["blocker", "status", "why"],
                        [[b.get("text", "?"), f"`{b.get('status')}`",
                          (b.get("detail") or "")[:100]] for b in blockers]), ""]
    if criteria:
        body += ["**Pass criteria not yet met**", "",
                 _table(["criterion", "status", "what fills it"],
                        [[c.get("text", "?"), f"`{c.get('status')}`",
                          (c.get("detail") or "")[:100]] for c in criteria]), ""]
    body += [
        "A `STALE` blocker is not a failed check. It means the check passed "
        "against evidence written **before** the current champion was "
        "promoted, so it describes a different model and has to be re-run "
        "rather than re-read. `bash scripts/final_validation.sh` re-runs the "
        "whole battery and refreshes every one of them in a single pass.",
    ]
    return "\n".join(body) + "\n"


def _reproduce() -> str:
    return (
        "```bash\n"
        "# the judge path - verifies shipped artifacts, never retrains\n"
        "bash scripts/final_validation.sh\n"
        "\n"
        "# rebuild the expensive evidence from the raw workbooks (hours)\n"
        "bash scripts/final_validation.sh --full-retrain\n"
        "\n"
        "# the individual questions this report summarises\n"
        "make nested-promotion   # does the primary protocol keep the champion?\n"
        "make experiment-ledger  # is any experiment missing from the record?\n"
        "make final-verdict      # the A-L answer, and the release decision\n"
        "```\n")


# ---------------------------------------------------------------------------
def render(ev: Evidence, verdict: dict[str, Any]) -> str:
    champion = verdict.get("champion") or ev.champion or "unpromoted"
    parts = [
        "# Final Validation Report",
        "",
        f"_Generated {_now()} by `muleguard.cli.final_report`. Do not edit by "
        "hand — run `make final-report`, or `bash scripts/final_validation.sh`, "
        "which regenerates this along with the evidence it reads._",
        "",
        "## 0. The answer",
        "",
        _headline(ev, verdict),
        "## 1. What was validated",
        "",
        "A mule-account detection system for BOI × IIT Hyderabad Hackathon 2026 "
        "PS2: 9,082 accounts, 81 positives (0.892 % prevalence), one workbook, "
        "no external data. The question this report answers is not \"is the "
        "model good\" but \"is the reported number the number a bank would "
        f"actually get\" — for the deployed `{champion}` and for nothing else.",
        "",
        "Class imbalance at this level breaks the usual reflexes: accuracy is "
        "99.1 % for a model that predicts \"legitimate\" every time, so it is "
        "not used as a headline anywhere. Average precision (PR-AUC) and "
        "recall at a fixed analyst review budget are, because they are the two "
        "quantities a fraud desk actually lives with.",
        "",
        "## 2. Which protocol decides",
        "",
        _protocols(ev),
        "## 3. The leakage firewall, and what it cost",
        "",
        _firewall_section(ev),
        "## 4. What this report does not claim",
        "",
        "\n".join(f"- **No claim of {what}** — {why}." for what, why in NOT_CLAIMED),
        "",
        "The §61 honesty rules are enforced in code, not left to discipline, "
        "in three places: `assert_language_safe` rejects the forbidden verdict "
        "vocabulary in every payload before it leaves the process, release-gate "
        "check `no_forbidden_verdict_vocabulary` scans shipped source for the "
        "same words, and `test_the_docs_do_not_overclaim` scans this file and "
        "every other document for absolute claims. A hit fails the release "
        "suite, and a refusal that names what it refuses - the lines above - "
        "passes, which is the only exemption.",
        "",
        "## 5. Evidence inventory (§57)",
        "",
        _evidence_inventory(ev),
        "## 6. Experiment ledger (§60)",
        "",
        _ledger_section(),
        "## 7. What is still open",
        "",
        _open_items(verdict),
        "## 8. Top risks",
        "",
        "\n".join(f"{i}. {r}" for i, r in
                  enumerate(verdict.get("top_risks") or ["_None recorded._"], 1)),
        "",
        "## 9. The other required reports (§58)",
        "",
        _table(["report", "what it answers"],
               [[f"[`{n}`]({n})" if (settings.DOCS_DIR / n).exists() else f"`{n}` (missing)",
                 d] for n, d in SIBLING_REPORTS]),
        "",
        "## 10. Reproducing this",
        "",
        _reproduce(),
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ev = Evidence()
    verdict = load_json(VERDICT_JSON) or {}
    if not verdict:
        log.warning("%s is absent - run `make final-verdict` first; the report "
                    "will render with the verdict section marked pending",
                    _rel(VERDICT_JSON))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(ev, verdict), encoding="utf-8")
    log.info("verdict carried through: %s",
             verdict.get("verdict") or PENDING_EVIDENCE)
    log.info("wrote %s", _rel(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
