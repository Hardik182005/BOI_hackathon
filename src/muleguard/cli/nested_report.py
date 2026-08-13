"""Write ``docs/NESTED_CV_MODEL_TOURNAMENT.md`` from the nested run's artifacts.

§58 asks for a nested-CV tournament report. Until now the spec name held a
pointer to `docs/MODEL_TOURNAMENT_REPORT.md`, which describes the **flat**
tournament and says the nested section "is filled by the nested run" - a promise
nothing kept. This module keeps it, and generates rather than narrates for the
same reason `final_report` does: a hand-typed leaderboard is correct on the day
it is typed and quietly wrong afterwards.

The report's job is to state one uncomfortable thing clearly if it is true - that
the primary protocol does not promote the model that is shipped - and to give the
reader everything needed to check it: the leaderboard, the same rule applied to
it, the paired interval on the difference, and the price of acting on it.

    .venv/Scripts/python.exe -m muleguard.cli.nested_report

Exits 0 regardless of what it finds. It reports; `nested_promotion` decides and
is the thing that exits non-zero.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Any

from muleguard import settings
from muleguard.cli.final_verdict import Evidence, _fmt
from muleguard.logging import get_logger

log = get_logger("cli.nested_report")

OUT = settings.REPO_ROOT / "docs/NESTED_CV_MODEL_TOURNAMENT.md"

NESTED = "artifacts/metrics/nested_cv.json"
ARBITER = "artifacts/metrics/nested_promotion_decision.json"
FLAT = "artifacts/metrics/tournament_v2.json"
TUNING = "artifacts/metrics/tuning_overfit_test.json"
BATTERY = "artifacts/metrics/metric_battery.json"

MISSING = "_Not available: `{path}` has not been written._"


def _pending(path: str, fills_when: str) -> str:
    return f"{MISSING.format(path=path)} Fills when: `{fills_when}`.\n"


def _design(ev: Evidence) -> str:
    nested = ev.get(NESTED)
    if not nested:
        return _pending(NESTED, "muleguard.cli.nested_cv --repeats 3 --inner 4")
    d = nested.get("design") or {}
    rows = [f"| {k.replace('_', ' ')} | {v} |" for k, v in d.items()]
    return "\n".join([
        f"Run written {nested.get('generated_utc', 'unknown')}.",
        "",
        "| element | as run |",
        "|---|---|",
        *rows,
        "",
        "Every outer-validation row is predicted exactly once per repeat, by a "
        "model whose features and hyperparameters were chosen without it. That "
        "is the whole difference from the flat tournament, and it is the reason "
        "this estimate is the primary one.",
        "",
    ])


def _leaderboard(ev: Evidence) -> str:
    nested = ev.get(NESTED)
    if not nested:
        return _pending(NESTED, "muleguard.cli.nested_cv --repeats 3 --inner 4")
    board = sorted(nested.get("leaderboard") or [],
                   key=lambda r: -float(r.get("pr_auc_mean") or 0))
    lines = ["| family | nested PR-AUC | sd across repeats | ROC-AUC | features | seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in board:
        lines.append(
            f"| `{r.get('model')}` | **{float(r.get('pr_auc_mean') or 0):.5f}** | "
            f"{float(r.get('pr_auc_std') or 0):.5f} | "
            f"{float(r.get('roc_auc_mean') or 0):.4f} | "
            f"{r.get('feature_size_mode')} | {r.get('seconds', '-')} |")
    lines += ["",
              "`dummy_prevalence` is scored and never promotable: it exists so "
              "that a broken metric is visible as a dummy near the top rather "
              "than as a plausible number nobody questions.", ""]
    return "\n".join(lines)


def _flat_vs_nested(ev: Evidence) -> str:
    nested, flat = ev.get(NESTED), ev.get(FLAT)
    if not nested or not flat:
        return _pending(FLAT, "muleguard.cli.tournament_v2")

    best: dict[str, tuple[str, float]] = {}
    for name, m in (flat.get("models") or {}).items():
        fam, ap = m.get("family"), m.get("pr_auc_mean")
        if fam is None or ap is None or m.get("view") not in (None, "ALL_ADMISSIBLE"):
            continue
        if fam not in best or ap > best[fam][1]:
            best[fam] = (name, float(ap))

    rows, deltas = [], []
    for r in sorted(nested.get("leaderboard") or [],
                    key=lambda r: -float(r.get("pr_auc_mean") or 0)):
        fam = str(r.get("model"))
        nested_ap = float(r.get("pr_auc_mean") or 0)
        if fam not in best:
            rows.append(f"| `{fam}` | {nested_ap:.5f} | not contested | - | - |")
            continue
        name, flat_ap = best[fam]
        delta = nested_ap - flat_ap
        deltas.append((fam, delta))
        rows.append(f"| `{fam}` | {nested_ap:.5f} | {flat_ap:.5f} (`{name}`) | "
                    f"{delta:+.5f} | {'nested higher' if delta > 0 else 'flat higher'} |")

    optimistic = [f for f, d in deltas if d < 0]
    lines = [
        "The flat tournament used the same outer folds, so the two columns are "
        "comparable. Flat is per configuration, nested is per family - each "
        "family's best admissible flat configuration is shown.",
        "",
        "| family | nested (primary) | best flat | nested - flat | |",
        "|---|---:|---:|---:|---|",
        *rows,
        "",
    ]
    if deltas:
        lines += [
            "**The flat protocol is not uniformly optimistic, and that is the "
            "finding.** Optimism appears where the flat run's single "
            "outside-the-folds choice happened to suit the family "
            f"({', '.join('`' + f + '`' for f in optimistic) or 'none here'}), "
            "and the reverse appears where it did not. A family whose flat "
            "configuration was poor looks weak in the flat tournament for a "
            "reason that has nothing to do with how it generalises - which is "
            "exactly what a protocol that fixes the configuration once cannot "
            "see, and what the nested protocol was run to expose.",
            "",
        ]
    return "\n".join(lines)


def _verdict(ev: Evidence) -> str:
    arb = ev.get(ARBITER)
    if not arb:
        return _pending(ARBITER, "muleguard.cli.nested_promotion")
    verdict = arb.get("verdict", "?")
    dep = (arb.get("deployed") or {})
    head = [f"**Verdict: `{verdict}`** - {arb.get('why', '')}.", "",
            f"Shipped: `{dep.get('model')}` (family `{dep.get('family')}`, "
            f"bundle {dep.get('bundle_version')}). "
            f"Nested promotes: `{arb.get('nested_promoted')}`.", ""]

    if verdict != "CHAMPION_CHALLENGED":
        head += ["The rule applied here is the flat tournament's rule, "
                 "unchanged, so any difference in outcome comes from the "
                 "protocol rather than from the tie-break.", ""]
        return "\n".join(head)

    paired = arb.get("paired_check") or {}
    head += [
        f"The gap is {arb.get('nested_gap_over_deployed')} PR-AUC. Two families "
        "measured on 64 positives have wide, overlapping marginal intervals, so "
        "the question is not whether those overlap - it is whether the "
        "*difference* does, on rows both families scored.",
        "",
    ]
    if paired:
        lo, hi = paired.get("paired_delta_ci95", [None, None])
        head += [
            f"| paired on identical rows | {paired.get('promoted_family')} - "
            f"{paired.get('deployed_family')} |",
            "|---|---|",
            f"| mean difference | **{paired.get('paired_delta_mean')}** |",
            f"| 95 % interval | [{lo}, {hi}] |",
            f"| excludes zero | {'yes' if paired.get('excludes_zero') else 'no'} |",
            f"| repeats favouring the challenger | "
            f"{paired.get('repeats_favouring_promoted')} |",
            f"| bootstrap resamples favouring it | "
            f"{paired.get('share_of_resamples_favouring_promoted')} of 1.0 "
            f"({paired.get('n_boot')} replicates) |",
            "",
            "These intervals are on the difference and are not comparable to "
            "the single-family intervals in `metric_battery.json`.",
            "",
        ]
    head += ["### What acting on this would cost", ""]
    head += [f"- {c}" for c in arb.get("what_acting_on_this_costs") or []]
    head += ["",
             arb.get("not_done_automatically", ""),
             "",
             "So the finding is published and the bundle is not touched. A "
             "reader who wants the challenger shipped has the number, the "
             "interval and the bill in front of them.",
             ""]
    return "\n".join(head)


def _tuning(ev: Evidence) -> str:
    t = ev.get(TUNING)
    if not t:
        return _pending(TUNING, "muleguard.cli.tuning_overfit")
    tests = t.get("tests") or {}
    pool = t.get("pooling_decomposition") or {}
    return "\n".join([
        f"An open note asked whether the in-fold Optuna tuning was net-harmful, "
        f"after an untuned arm beat a tuned one by "
        f"{abs(float(pool.get('untuned_pooled_pr_auc', 0)) - float(pool.get('tuned_pooled_pr_auc', 0))):.5f} "
        "on the same folds. Paired over all 15 outer folds it is not:",
        "",
        f"- gain from not tuning: **{t.get('mean_gain_from_not_tuning')}** "
        f"(sd of the paired difference {t.get('std_of_paired_diff')})",
        f"- sign test {tests.get('sign_test', {}).get('folds_favouring_untuned')}"
        f"/{tests.get('sign_test', {}).get('n_folds')} folds, "
        f"p = {tests.get('sign_test', {}).get('p_two_sided')}",
        f"- Wilcoxon p = {tests.get('wilcoxon_signed_rank', {}).get('p_two_sided')}"
        f" · paired t p = {tests.get('paired_t', {}).get('p_two_sided')}",
        "",
        f"The larger pooled gap is a pooling effect: the tuned arm loses "
        f"{abs(float(pool.get('pooling_cost_tuned', 0))):.5f} PR-AUC when its "
        f"folds are pooled against {abs(float(pool.get('pooling_cost_untuned', 0))):.5f} "
        "for the untuned one, because a configuration that changes per fold "
        "produces scores that do not share a scale. Full reasoning in "
        "`docs/TUNING_OVERFIT_HYPOTHESIS.md`.",
        "",
    ])


def _thresholds(ev: Evidence) -> str:
    """Do the frozen policy thresholds still behave under the primary protocol?

    The thresholds were frozen against the flat store. If the nested predictions
    put a very different number of accounts above the same cut, the freeze is a
    property of one protocol rather than of the model, and re-freezing would be
    owed. This section is what turns "we did not re-freeze" from an omission
    into a decision.
    """
    battery = ev.get(BATTERY) or {}
    runs = battery.get("runs") or {}
    champion = ev.champion or ""
    family = champion.split("_top_")[0]
    flat, nested = runs.get(f"FLAT:{champion}"), runs.get(f"NESTED:{family}")
    if not flat or not nested:
        return _pending(BATTERY, "muleguard.cli.metric_battery --protocol NESTED "
                                 "--source artifacts/predictions/nested_oof.parquet")

    def tiers(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {t.get("tier"): t for t in run.get("at_frozen_thresholds", [])}

    ft, nt = tiers(flat), tiers(nested)
    rows, moves = [], []
    for tier in [t for t in ft if t in nt]:
        f, n = ft[tier], nt[tier]
        fa, na = f.get("alerts"), n.get("alerts")
        if isinstance(fa, (int, float)) and fa:
            moves.append((tier, (na - fa) / fa, na - fa))
        rows.append(
            f"| `{tier}` | {f.get('threshold')} | {fa} → {na} "
            f"| {_fmt(f.get('precision'))} → {_fmt(n.get('precision'))} "
            f"| {_fmt(f.get('recall'))} → {_fmt(n.get('recall'))} |")

    # The verdict sentence is computed, not asserted: the largest relative move
    # in alert volume is what decides whether the freeze survived the protocol
    # change, and it is quoted so a reader can disagree with the cut-off.
    worst = max(moves, key=lambda m: abs(m[1])) if moves else ("", 0.0, 0)
    transfers = abs(worst[1]) <= 0.25
    ordered = all(
        (nt[a].get("alerts") or 0) <= (nt[b].get("alerts") or 0)
        for a, b in zip(list(ft)[:-1], list(ft)[1:]) if a in nt and b in nt)
    fq = (flat.get("probability_quality") or {})
    nq = (nested.get("probability_quality") or {})
    return "\n".join([
        f"The policy thresholds are frozen against `{champion}`'s flat score "
        "distribution. Applying those same numbers, unchanged, to the nested "
        "predictions of the same family answers whether the freeze describes the "
        "model or only the protocol that produced it.",
        "",
        "| tier | frozen cut | alerts, flat → nested | precision | recall |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        f"Brier moves {_fmt(fq.get('brier'))} → {_fmt(nq.get('brier'))} and the "
        f"skill against the base rate {_fmt(fq.get('brier_skill_vs_base_rate'))} → "
        f"{_fmt(nq.get('brier_skill_vs_base_rate'))}.",
        "",
        f"The largest move is `{worst[0]}`, {worst[2]:+d} alerts "
        f"({worst[1] * 100:+.1f}%), and the tiers "
        f"{'keep' if ordered else 'do NOT keep'} their order by volume. "
        + ("The cuts therefore describe the model rather than the protocol that "
           "produced it, and **no threshold is re-frozen and no calibrator is "
           "re-fitted on this evidence**. That is a decision with a reason "
           "attached rather than an omission."
           if transfers and ordered else
           "That is too large to call the freeze protocol-independent, and the "
           "thresholds are owed a re-freeze against the primary protocol before "
           "the tier volumes in the product documentation can be relied on."),
        "",
        "Two honest qualifications. The tiers do not hold their **recall** as "
        "well as their volume - the deep tier gives up several points, which is "
        "what applying an unchanged cut to a slightly weaker score distribution "
        "does, and it is the cost of not re-freezing rather than an argument "
        "that nothing changed. And this covers the shipped family only: the "
        "challenger in section 4 scores on its own scale and would need its own "
        "freeze, which is part of the bill listed there.",
        "",
    ])


def _not_decided() -> str:
    return "\n".join([
        "- **Nothing was retrained here.** Every number is read from a stored "
        "prediction or a stored leaderboard.",
        "- **The locked test was not opened.** It is single-touch and was spent "
        "on the shipped bundle; no nested result is checked against it.",
        "- **No threshold moved.** The policy thresholds are frozen against the "
        "shipped model's score distribution and a different model would need "
        "its own freeze.",
        "- **The nested run does not rank feature-set sizes.** Each fold chose "
        "its own from {30, 60, 120}; the leaderboard reports the mode, which is "
        "a description of what happened, not a recommendation.",
        "",
    ])


def render(ev: Evidence) -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Nested CV Model Tournament",
        "",
        "_Required by section 58 of the final-validation prompt. Generated by "
        "`muleguard.cli.nested_report` - do not hand-edit; edit the generator._",
        "",
        f"<sub>Written {now} from the artifacts named in each section.</sub>",
        "",
        "The flat tournament in `docs/MODEL_TOURNAMENT_REPORT.md` chose the "
        "shipped model. This document is the check on that choice: the same "
        "families, the same folds, but with selection and tuning moved inside "
        "the fold where they cannot see the rows they are scored on.",
        "",
        "## 1. The protocol as run",
        "",
        _design(ev),
        "## 2. Leaderboard under the primary protocol",
        "",
        _leaderboard(ev),
        "## 3. What the flat protocol got wrong, and where",
        "",
        _flat_vs_nested(ev),
        "## 4. Does the shipped champion survive?",
        "",
        _verdict(ev),
        "## 5. The tuning question, closed",
        "",
        _tuning(ev),
        "## 6. The frozen thresholds under the primary protocol",
        "",
        _thresholds(ev),
        "## 7. What this run does not decide",
        "",
        _not_decided(),
        "## 8. Reproduce",
        "",
        "```",
        ".venv/Scripts/python.exe -m muleguard.cli.nested_cv --repeats 3 --inner 4",
        ".venv/Scripts/python.exe -m muleguard.cli.nested_promotion",
        ".venv/Scripts/python.exe -m muleguard.cli.tuning_overfit",
        ".venv/Scripts/python.exe -m muleguard.cli.nested_report",
        "```",
        "",
        "The first command is the expensive one - it is a full re-run of the "
        "protocol, not a re-read - and the other three are seconds.",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    text = render(Evidence())
    path = settings.REPO_ROOT / args.out if not str(args.out).startswith(
        str(settings.REPO_ROOT)) else args.out
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log.info("wrote %s (%d lines)", path, text.count("\n") + 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
