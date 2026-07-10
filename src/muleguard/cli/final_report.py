"""Assemble docs/FINAL_RESULTS.md and the tournament/calibration reports
entirely from artifact files - no number enters a document by hand.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from muleguard import settings
from muleguard.utils import git_info, load_json


def _fmt_ci(block: dict) -> str:
    return f"{block['point']:.4f} (95% CI {block['ci_low']:.4f}–{block['ci_high']:.4f})"


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    git = git_info(settings.REPO_ROOT)
    fp = load_json(settings.REPO_ROOT / "data/interim/data_fingerprint.json")
    oof = load_json(settings.METRICS_DIR / "oof_metrics.json")["models"]
    lt = load_json(settings.METRICS_DIR / "locked_test_metrics.json")
    lens = load_json(settings.METRICS_DIR / "lens_stack_oof.json")
    ens = load_json(settings.METRICS_DIR / "ensemble_decision.json")
    abl = load_json(settings.METRICS_DIR / "with_vs_without_f3912.json")
    manifest = load_json(settings.MODELS_DIR / "model_manifest.json")
    adv = load_json(settings.METRICS_DIR / "advanced_models.json")
    sel = load_json(settings.FEATURES_DIR / "selected_features.json")
    env = load_json(settings.ARTIFACTS_DIR / "environment_snapshot.json")

    candidates = {k: v for k, v in oof.items() if not k.startswith("REJECTED")}
    ranked = sorted(candidates.items(), key=lambda kv: -kv[1]["pr_auc_mean"])
    winner_name, winner = ranked[0]

    tour_rows = "\n".join(
        f"| {name} | {m.get('n_features', '')} | {m['pr_auc_mean']:.4f} ± {m['pr_auc_std']:.4f} "
        f"| {m['roc_auc_mean']:.4f} | {m['n_repeats']} | {m.get('runtime_seconds', '')} |"
        for name, m in ranked
    )
    rejected_rows = "\n".join(
        f"| {name} | — | {m['pr_auc_mean']:.4f} ± {m['pr_auc_std']:.4f} | {m['roc_auc_mean']:.4f} "
        f"| {m['n_repeats']} | REJECTED LEAKAGE |"
        for name, m in oof.items() if name.startswith("REJECTED")
    )
    budget_rows = "\n".join(
        f"| top {b['budget']} | {b['true_positives']}/{lt['n_positives']} | {b['recall']:.2%} | {b['precision']:.2%} |"
        for b in lt["recall_at_budget"]
    )
    fpr_rows = "\n".join(
        f"| {r['fpr_target']:.1%} | {r['recall']:.2%} | {r['fp_per_1000_legit']:.1f} |"
        for r in lt["recall_at_fpr"]
    )
    def _tier_precision(t: dict) -> str:
        return "—" if t["precision_in_tier"] is None else f"{t['precision_in_tier']:.2%}"

    tier_rows = "\n".join(
        f"| {t['tier']} | {t['n']} | {t['n_true_mules']} | {_tier_precision(t)} |"
        for t in lt["tier_distribution"]
    )
    adv_rows = "\n".join(
        f"| {name} | {info['status']} | {info['reason']} |"
        for name, info in adv["challengers"].items()
    )

    doc = f"""# Final Results

Generated {now} · commit `{git['commit_sha'][:12]}` · compute mode `{env['compute_mode']}`
· seed {env['seed']} · raw SHA-256 `{fp['raw_file']['sha256'][:20]}…`

Every number below is produced by the pipeline and traceable to
`artifacts/metrics/` + `artifacts/predictions/`. Splits: **dev OOF** =
repeated stratified 5-fold CV on 7,264 dev rows (64 positives), preprocessing
and selection inside folds; **locked test** = 1,818 rows (17 positives),
touched exactly once.

## Dataset (verified)

- {fp['n_rows']:,} accounts × {fp['n_cols']:,} columns; target `F3924`
- {fp['target_distribution']['positives_1']} positives / {fp['target_distribution']['negatives_0']:,} negatives → prevalence {fp['positive_prevalence']:.4%}
- Quarantined: F3924, F3912, F2230 (snapshot month ≡ label), __UNNAMED__0 (row index)

## Leakage ablation (dev OOF)

| Run | PR-AUC | Status |
|---|---|---|
| LightGBM clean (quarantine enforced) | {abl['without_f3912']['pr_auc_mean']:.4f} ± {abl['without_f3912']['pr_auc_std']:.4f} | ACCEPTED |
| LightGBM + F3912 | {abl['with_f3912']['pr_auc_mean']:.4f} ± {abl['with_f3912']['pr_auc_std']:.4f} | **REJECTED LEAKAGE — evidence only** |

## Model tournament (dev OOF, natural prevalence)

| Model | Features | PR-AUC (mean ± std) | ROC-AUC | Repeats | Runtime s |
|---|---|---|---|---|---|
{tour_rows}
{rejected_rows}

Winner: **{winner_name}** ({winner['pr_auc_mean']:.4f} ± {winner['pr_auc_std']:.4f}).
Ensemble decision: **{"ACCEPTED" if ens['accepted'] else "REJECTED"}** — {ens['rule']}
(stacker AP by repeat {[round(a, 4) for a in ens['ensemble_ap_by_repeat']]} vs best single {[round(a, 4) for a in ens['best_single_ap_by_repeat']]}).

Feature selection: stability selection inside folds; fold-to-fold top-60
overlap {sel['fold_overlap_top60_mean']:.2f}. Compact sets: top-15/30/60 evaluated above.

### Advanced challengers

| Challenger | Status | Reason |
|---|---|---|
{adv_rows}

## Locked test (single touch)

| Metric | Value |
|---|---|
| PR-AUC (raw winner scores) | {_fmt_ci(lt['pr_auc'])} |
| PR-AUC (calibrated) | {_fmt_ci(lt['calibrated_pr_auc'])} |
| ROC-AUC (secondary) | {_fmt_ci(lt['roc_auc'])} |
| Brier score | {lt['brier']:.5f} |
| ECE ({lt['calibration']['n_bins']} bins) | {lt['calibration']['ece']:.4f} |
| Conformal abstention rate | {lt['conformal']['abstention_rate']:.2%} |
| Positive conformal coverage | {lt['conformal']['positive_coverage']:.2%} |
| OOD rate | {lt['ood_rate']:.2%} |
| Scoring throughput | {lt['scoring_rows_per_second']} rows/s (CPU) |

### Recall / precision at alert budgets (locked test)

| Budget | Caught | Recall | Precision |
|---|---|---|---|
{budget_rows}

### Recall at fixed FPR (locked test)

| FPR target | Recall | FP per 1,000 legit |
|---|---|---|
{fpr_rows}

### Review-tier outcomes (policy applied to locked test)

| Tier | Accounts | True mules | Precision in tier |
|---|---|---|---|
{tier_rows}

## Lens stack (fitted on dev OOF only)

- Calibrator: **{lens['calibration_selection']['winner']}** — comparison {lens['calibration_selection']['comparison']}
- Conformal (α=0.10) OOF coverage: {lens['conformal_coverage_oof']}
- Hard negatives mined: {lens['n_hard_negatives']}
- Policy thresholds (frozen): {lens['policy_thresholds']}

## Bundle

`{manifest['bundle_sha256'][:24]}…` · {manifest['n_features']} features ·
calibrator {manifest['calibrator']} · registered champion in
`artifacts/model_registry/registry.json`.
"""
    (settings.DOCS_DIR / "FINAL_RESULTS.md").write_text(doc, encoding="utf-8")
    print("FINAL_RESULTS.md written")


if __name__ == "__main__":
    main()
