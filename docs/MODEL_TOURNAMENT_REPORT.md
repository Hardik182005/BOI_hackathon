# Model Tournament Report

Measured leaderboard (single source of truth):
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

## Selection of the shipped model

The winner is the highest mean OOF PR-AUC among non-rejected candidates that
is also stable across repeats (std reported, per-repeat values inspected).
It is retrained on the full dev split with its tuned parameters and frozen
into `artifacts/models/final_bundle.joblib` (SHA-256 in the manifest);
XGBoost and CatBoost ride along as agreement models for the policy engine.
