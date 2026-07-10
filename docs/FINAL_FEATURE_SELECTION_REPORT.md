# Final Feature Selection Report

Authoritative artifacts: `artifacts/features/final_selected_features.json`,
`artifacts/features/final_selection_frequency.csv` (copies of the live
`selected_features.json` / `selection_frequency.csv` with identical content),
plot `artifacts/plots/final_feature_stability.png`. Method detail:
`docs/FEATURE_SELECTION_REPORT.md`.

## Subset ablation (measured, 5-repeat OOF, tuned models)

| Subset | PR-AUC (mean ± std) | Note |
|---|---|---|
| top-60 (production) | **0.8077 ± 0.0450** (CatBoost) / 0.7717 ± 0.0528 (LGBM) | best |
| full cleaned matrix (~2,900 cols after in-fold hygiene) | 0.7637 ± 0.0223 (LGBM) | compact beats full |
| top-30 | 0.7562 ± 0.0255 | −0.02 vs top-60 (LGBM) |
| top-15 | 0.6440 ± 0.0420 | signal loss visible |
| top-100 | not run — bracketed by top-60 (better) and full (worse); skip recorded, not hidden |

Latency scales with width: top-60 inference is ~14× faster per fold than the
full matrix (52 s vs 710 s per full CV pass).

## Verified selection hygiene

- Constant/quasi-constant/duplicate handling and the stability selector
  (LightGBM gain + L1 logistic over 40 stratified subsamples) run **inside
  training folds**; the headline compact numbers come from fold-local
  selection, so they carry no selection bias.
- The frozen production list (top-60) is versioned in the bundle +
  `final_selected_features.json`; missing selected features at inference are
  a hard `SCHEMA_ERROR`.
- Fold-to-fold top-60 overlap 0.54 (honest churn in the tail with 51
  positives/fold); the freq ≥ 0.8 head (9 features: F3898, F3908, F3914,
  F3913, F1863, F158, F2074, F3886, F3640) is highly stable.
- Every top feature was cross-checked against the leakage audit: the
  strongest selected feature has single-feature CV PR-AUC ≤ 0.06 — the model
  works through feature *combinations*, not one suspicious column.
- Anonymous features keep their `Fxxxx` names everywhere; the only verified
  semantic in the top set is F3886 (account/product type).
- The 18 bank-hint features (Topic.pdf) are all present in the data; none
  entered the top-60 on merit — hints were context, never a selection input.
