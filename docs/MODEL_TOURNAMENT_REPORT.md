# Model Tournament Report

> **This document describes the RETIRED, pre-firewall tournament.** Its
> leaderboard was computed on a pool that still contained the post-resolution
> columns `F3898`, `F3899`, `F3913`, `F3914`, `F3915`, and its winner
> `catboost_tuned_top60` selected four of the inadmissible columns. Its
> headline **0.8077 is retired**.
>
> The current tournament is `src/muleguard/cli/tournament_v2.py` →
> `artifacts/metrics/tournament_v2.json` /
> `artifacts/metrics/model_comparison_v2.csv`, narrated in
> `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md`. Promoted model:
> **`xgboost_top_120`, OOF PR-AUC 0.7690 ± 0.0266** over 3 repeats.
>
> The protocol description below still applies, with these changes: 3 repeats
> rather than 5; every matrix is built through
> `features/frame.build_model_frame()` so the firewall cannot be bypassed;
> and five availability views (A–E) are contested alongside the full admitted
> pool.

Measured leaderboard of the retired run:
`artifacts/metrics/model_comparison.csv`, full per-repeat detail in
`artifacts/metrics/oof_metrics.json`, plot
`artifacts/plots/model_comparison.png`. Headline numbers are also assembled
into `docs/FINAL_RESULTS.md`. Nothing in this document is a number typed by
hand.

## Protocol

- **Folds:** one immutable repeated stratified group-aware 5-fold design
  (5 repeats, seeds 42–46) shared by every contestant; assignments stored in
  `data/splits/cv_folds.parquet`.
- **In-fold everything:** constants/duplicate removal, imputation (linear
  models), feature selection, early stopping (carved from the training fold),
  hyperparameters (tuned on a single scouting repeat), calibration.
- **Metric:** out-of-fold Average Precision at natural prevalence, mean ± std
  across repeats, with per-repeat values preserved. Bootstrap CIs per repeat.
- **No locked-test access:** asserted in the harness and enforced by tests.

## Contestants

| Family | Config |
|---|---|
| Dummy (prevalence) | floor reference |
| Logistic L2 (class-weighted) | median-impute + standardise in-fold |
| LightGBM baseline | class-weighted, early stopping, full clean matrix |
| LightGBM tuned | Optuna (40 trials, TPE seed 42) on top-60; evaluated on full/60/30/15 |
| XGBoost tuned | Optuna (30 trials) on top-60 |
| CatBoost tuned | Optuna (25 trials) on top-60 |
| REJECTED: LightGBM + F3912 | leakage evidence only — red bar, never a candidate |

### Advanced challengers (Mode A guard)

TabPFN / TabICL / AutoGluon ran behind a feasibility guard;
statuses + exact skip reasons in `artifacts/metrics/advanced_models.json`
(CPU-16GB environment: TabICL requires GPU-class resources; AutoGluon lacks
Python 3.13 support at build time; each is a documented roadmap experiment,
not a silent omission).

## Key measured findings

1. **Compact features beat the full matrix.** The full 3,900-column cleaned
   matrix drowns 64 training positives in noise; the stability-selected
   compact sets substantially outperform it (see leaderboard - e.g. tuned
   top-60 vs full-matrix baseline). This is the graded deliverable of the
   problem statement: *identify the most relevant features*.
2. **Leakage inflates everything.** Re-admitting F3912 lifts OOF PR-AUC to
   ~0.94 — rejected as evidence, kept out of every model.
3. **Ensemble decision is evidence-gated.** A regularised logistic stacker
   over LGBM/XGB/CatBoost OOF predictions is accepted only if it beats the
   best single model on ≥ n−1 repeats (`ensemble_decision.json` records the
   verdict either way; model correlations included).

All three findings survive the leakage-free re-run, with different numbers:
compact still beats the full matrix (top-120 0.7690 vs full pool 0.5874);
leakage still inflates (the retired 0.8077 is the evidence); and the ensemble
is still evidence-gated (`ensemble_v2.json`, decision `SINGLE_MODEL_KEPT`,
with the addendum's literal criteria and our stricter per-repeat criterion
both recorded).

## Selection of the shipped model

The winner is the highest mean OOF PR-AUC among non-rejected candidates that
is also stable across repeats (std reported, per-repeat values inspected).
It is retrained on the full dev split with its tuned parameters and frozen
into `artifacts/models/final_bundle.joblib` (SHA-256 in the manifest);
the runner-up families ride along as agreement models for the policy engine.

Under the re-run this rule promoted `xgboost_top_120`, formalised by the
addendum UPDATE 13 generalization score applied inside a 0.01 PR-AUC band of
the leader (`artifacts/metrics/promotion_decision_v2.json`; the band contained
only the leader, so `tie_break_applied` is `false`).
