# Feature Selection Report

Machine-readable outputs for the **current** run:
`artifacts/features/selected_features_v2.json`,
`artifacts/features/selection_frequency_v2.csv`. The pre-firewall outputs
(`selected_features.json`, `selection_frequency.csv`, plot
`artifacts/plots/feature_stability.png`) are retained as the retired record.

**What changed.** Selection now runs over the firewall-admitted pool only
(3,925 candidates for `ALL_ADMISSIBLE`, of which 369 were ever selected), per
availability view, with the ranking refitted independently inside every
training fold. The production list is the **top-120** set used by
`xgboost_top_120`, verified to contain 0 quarantined columns. The retired
top-60 list contained `F3898`, `F3913` and `F3914` in its most stable head;
overlap between it and the leakage-free top-60 is 12 of 60.

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

- Mean fold-to-fold overlap of top-60 sets in the retired run: **0.54** —
  honest number: with 51 positives per training fold, some churn in the tail
  is expected. The stable head reported for that run (freq ≥ 0.8, 9 features)
  is retired: three of those nine were post-resolution columns, so what was
  measured as stability was the stability of the leak.
- Compact-set ablation is now reported in
  `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md` §3 and
  `docs/UPGRADE_GAP_ANALYSIS.md` §3.1: top-120 0.7690 ± 0.0266 > top-60
  0.7408 ± 0.0498 > top-30 0.6897 ± 0.0122 > full pool 0.5874 ± 0.0276.
- Positive-removal stability of the promoted model's feature ranking:
  Spearman 0.7654 between rounds, top-20 overlap 0.70
  (`artifacts/metrics/stability_stress_v2.json`).

## Anti-leakage guarantees

- Quarantined columns are excluded **before** any selector runs. The list is
  no longer four entries: it is the 9 hard-quarantined columns (F3924, F3912,
  F3913, F3914, F3915, F3898, F3899, F2230, `__UNNAMED__0`), 3 conditionally
  quarantined risk-level flags (F3916/F3917/F3918) and the fairness exclusion
  F3892, per `configs/feature_availability.yaml`.
- Selection statistics never touch the locked test (enforced by an assertion
  in the harness plus `tests/model/test_leakage_guards.py`).
- The dev-frozen production list (now top-120) is used for the final bundle;
  the compact metrics quoted for it come from folds where selection was re-run
  in-fold, so the quoted numbers do not inherit selection bias.

## Reporting integrity

Selected features are anonymised (`Fxxxx`); the report and UI show selection
frequency, SHAP importance, missingness and fold stability — never invented
business meanings. The 18 bank-hint features from Topic.pdf are present in
the data and their overlap with the selected set is visible in the frequency
table (hints were never used to force selection).
