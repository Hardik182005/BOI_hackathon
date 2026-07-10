# Feature Selection Report

Machine-readable outputs: `artifacts/features/selected_features.json`,
`artifacts/features/selection_frequency.csv`,
plot `artifacts/plots/feature_stability.png`.

## Pipeline (all inside training folds)

1. **In-fold hygiene** (`FoldPreprocessor`, fitted per training fold):
   constants (incl. all-missing) removed; exact duplicate columns removed via
   missing-aware content hashes — the fold's validation split never influences
   any removal. On dev folds this typically removes ~360 constants and ~950+
   duplicate columns of the 3,921 candidates.
2. **Stability selection** (per training fold of repeat 0): 40 stratified
   subsamples at 70% rate; two independent selectors per subsample —
   (a) LightGBM gain importance (150 shallow trees, class-weighted),
   (b) L1 logistic regression (liblinear, C=0.1, class-weighted) on a
   univariate-screened (top-500 |r|), median-imputed, standardised view.
   A feature is "selected" in a subsample if it enters the top-60 of either
   selector with positive importance.
3. **Frequency aggregation** across folds → `selection_frequency.csv`;
   compact sets frozen: top-15, top-30, top-60, freq ≥ 0.6 (16 features),
   freq ≥ 0.8 (9 features).

## Measured stability

- Mean fold-to-fold overlap of top-60 sets: **0.54** — honest number: with 51
  positives per training fold, some churn in the tail is expected; the head
  of the ranking (freq ≥ 0.8, 9 features) is highly stable across folds.
- Compact-set ablation (top-15/30/60 vs full clean matrix) is measured in the
  tournament and reported in `docs/FINAL_RESULTS.md` / `MODEL_TOURNAMENT_REPORT.md`.

## Anti-leakage guarantees

- Quarantined columns (F3924, F3912, F2230, index) are excluded **before**
  any selector runs.
- Selection statistics never touch the locked test (enforced by an assertion
  in the harness plus `tests/model/test_leakage_guards.py`).
- The dev-frozen production list (top-60) is used for the final bundle; the
  compact metrics quoted for it come from folds where selection was re-run
  in-fold, so the quoted numbers do not inherit selection bias.

## Reporting integrity

Selected features are anonymised (`Fxxxx`); the report and UI show selection
frequency, SHAP importance, missingness and fold stability — never invented
business meanings. The 18 bank-hint features from Topic.pdf are present in
the data and their overlap with the selected set is visible in the frequency
table (hints were never used to force selection).
