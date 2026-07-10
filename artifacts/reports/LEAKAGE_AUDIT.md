# Leakage Audit

Generated 2026-07-10T18:56:17.931141+00:00 by `muleguard.cli.audit_data`.

## Method

For **every numeric feature**: pairwise-complete correlation with `F3924`,
**cross-validated** single-feature PR-AUC (5-fold, direction fixed on train folds),
quantile-binned mutual information (missing as own bin), exact label-reconstruction
check on two-valued columns, near-perfect separation and identifier-likeness checks.
Audit statistics are computed on the full file **for safety screening only** — they are
used to *exclude* features, never to select them (selection happens inside CV folds).

Flag thresholds: |corr| ≥ 0.95, single-feature CV PR-AUC ≥ 0.9.

## Result: 3 feature(s) flagged

| feature | target_corr | single_feature_cv_pr_auc | label_reconstruction_rate | flag_high_corr | flag_high_single_ap | flag_label_copy | flag_perfect_separation | flag_identifier |
|---|---|---|---|---|---|---|---|---|
| __UNNAMED__0 | 0.1628 | 1.0 | 0.0 | False | True | False | True | True |
| F3912 | 0.9691 | 0.9433 | 0.9875 | True | True | True | False | False |
| F2230 | -0.0404 | 0.5945 | 1.0 | False | False | True | False | False |


## Quarantine (mandatory + flagged)

| feature | reason | disposition |
|---|---|---|
| F3924 | target variable (F3924) | EXCLUDED_FROM_ALL_TRAINING |
| F3912 | target leak: measured |corr|=0.97, single-feature CV PR-AUC=0.94, balanced label reconstruction=0.99 | EXCLUDED_FROM_ALL_TRAINING |
| F2230 | snapshot-month label artifact: ALL 9,001 negatives are the 2025-10 snapshot, ALL 81 positives are Sep/Nov/Dec snapshots - the month deterministically reconstructs the target (balanced reconstruction 1.0). Also invalidates any out-of-time split on this column. | EXCLUDED_FROM_ALL_TRAINING |
| __UNNAMED__0 | row-index / identifier column | EXCLUDED_FROM_ALL_TRAINING |


Policy: quarantined features are excluded from every model, ensemble, selector, calibration and explanation; ablation runs that include them are labelled REJECTED LEAKAGE evidence only

The with/without-F3912 ablation (evidence that the quarantine matters) is produced by
the baseline phase: `artifacts/metrics/with_vs_without_f3912.json` and
`artifacts/plots/leakage_ablation.png` — the F3912 run is labelled **REJECTED LEAKAGE**.
