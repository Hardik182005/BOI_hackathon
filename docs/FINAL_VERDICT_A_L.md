# MuleGuard - Trinetra: final validation response (section 65)

Generated 2026-08-13T02:47:39+00:00 by `python -m muleguard.cli.final_verdict`.
Every field is read from a named artifact. A field with no evidence behind it says PENDING and names the run that would produce it; none is ever filled in from an earlier model.

Champion: **xgboost_top_120**, promoted 2026-08-12T07:39:08.059521+00:00.

# A. Environment

```text
Git SHA: 1f222e1708cea177236e275bee408bcd51a40f26 (working tree dirty)
Python: 3.13.2
Node: recorded by scripts/test_frontend.sh
CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel - 4 physical / 8 logical cores
RAM: 16.93 GB total
GPU: none - CUDA available: False, compute mode cpu_16gb
DataSet SHA: 7d1be90fe23b57460184f2f7566d572cc3a40fc6d17d436fe33219e80eabc204
Description SHA: 7d30652b72d4b79b3feeb3e14ebe67177988b3155912ea33f284f140e8774b20
```

# B. Verified Primary Dataset

```text
Rows: 9082 (development 7264, locked test 1818)
Feature columns: 3924
Target: Target (1 = mule account)
Positives: 81 overall; 64 in development
Negatives: 9001 overall; 7200 in development
Prevalence: 0.008919 overall
Naive accuracy: 0.991081 (predict every account legitimate)
No-skill AP: 0.008811 on development
```

# C. Leakage Firewall

```text
Hard excluded: 13 columns, all excluded from every model, selector, ensemble and lens: F2230, F3892, F3898, F3899, F3912, F3913, F3914, F3915, F3916, F3917, F3918, F3924, __UNNAMED__0
Conditional quarantine: none - quarantine policy 2.0 is unconditional; a column is either admissible or excluded
F2230 verdict: EXCLUDED. MNTH snapshot month - deterministically reconstructs the label in this extract
F3916-18 verdict: EXCLUDED. L3_FLG - customer risk level, time-of-availability undetermined
Target leakage detected?: YES, and firewalled. F3912 alone scored 0.94190 PR-AUC against 0.61416 without it. Firewall check: CLEAR - none of F2230, F3892, F3898, F3899, F3912, F3913, F3914, F3915, F3916, F3917, F3918, F3924, __UNNAMED__0 appear among the champion's 120 input features
```

# D. Model Tournament

| model | protocol | features | PR-AUC mean | PR-AUC std | note |
| --- | --- | --- | --- | --- | --- |
| tabpfn_top_60 | FLAT 3x5 | 60 | 0.91100 | 0.00436 | OK |
| xgboost_top_120 | FLAT 3x5 | 120 | 0.76904 | 0.02663 | OK |
| xgboost_top_60 | FLAT 3x5 | 60 | 0.74082 | 0.04981 | OK |
| lightgbm_top_60 | FLAT 3x5 | 60 | 0.69918 | 0.02733 | OK |
| catboost_top_60 | FLAT 3x5 | 60 | 0.69630 | 0.04168 | OK |
| xgboost_top_30 | FLAT 3x5 | 30 | 0.68974 | 0.01224 | OK |
| lightgbm_top_120 | FLAT 3x5 | 120 | 0.68621 | 0.04045 | OK |
| lightgbm_top_30 | FLAT 3x5 | 30 | 0.66281 | 0.01609 | OK |
| lightgbm_viewA_top_60 | FLAT 3x5 | 60 | 0.65844 | 0.02341 | OK |
| catboost_top_120 | FLAT 3x5 | 120 | 0.64670 | 0.06993 | OK |
| lightgbm_viewB_top_60 | FLAT 3x5 | 60 | 0.64543 | 0.04847 | OK |
| lightgbm_top_250 | FLAT 3x5 | 250 | 0.64419 | 0.02999 | OK |
| lightgbm_full_pool | FLAT 3x5 | 3925 | 0.58743 | 0.02758 | OK |
| catboost_top_30 | FLAT 3x5 | 30 | 0.56136 | 0.11250 | OK |
| lightgbm_top_15 | FLAT 3x5 | 15 | 0.33566 | 0.13990 | OK |
| lightgbm_viewE_top_60 | FLAT 3x5 | 60 | 0.30611 | 0.08419 | OK |
| lightgbm_freq_ge_0_50 | FLAT 3x5 | 13 | 0.19166 | 0.02470 | OK |
| elasticnet_top30 | FLAT 3x5 | 30 | 0.08954 | 0.00449 | OK |
| logistic_top30 | FLAT 3x5 | 30 | 0.08816 | 0.00454 | OK |
| lightgbm_viewC_top_15 | FLAT 3x5 | 15 | 0.02949 | 0.01101 | OK |
| lightgbm_viewD_top_15 | FLAT 3x5 | 15 | 0.02059 | 0.00205 | OK |
| dummy_prevalence | FLAT 3x5 | 30 | 0.00871 | 0.00000 | OK |
| xgboost | NESTED 1x5x4 | 120 | 0.66792 | 0.00000 | NESTED |
| dummy_prevalence | NESTED 1x5x4 | 120 | 0.00931 | 0.00000 | NESTED |

# E. Champion

```text
Model: xgboost_top_120 (battery run FLAT:xgboost_top_120)  [operating point and probability quality measured under the FLAT protocol; nested is primary and is still running]
Feature set: top_120 of view ALL_ADMISSIBLE
Number of features: 120
Nested-CV AP mean: PENDING - produced by: python -m muleguard.cli.nested_cv --repeats 3 --inner 4, then metric_battery --protocol NESTED. A preliminary nested run (NESTED_PRELIMINARY:xgboost, 1 repeat, 2 families) measured 0.66792 - lower than the flat figure, as nested protocols usually are, and superseded by the run in progress
FLAT-CV AP mean: 0.76904
AP std: 0.02663 across 3 repeats (a spread, not an interval)
95% CI: [0.67555, 0.85245] percentile bootstrap over accounts, 2000 draws
ROC-AUC: 0.95771 (secondary - prevalence 0.88% makes ROC flattering)
Accuracy: 0.99202 at the URGENT_REVIEW threshold
Balanced Accuracy: 0.91080
Precision: 0.53000
Recall: 0.82812
F1: 0.64634
F2: 0.74438
MCC: 0.65893
Brier: 0.00313 against 0.00873 for a base-rate predictor (skill 0.642)
ECE: 0.00096 (10 quantile bins), 0.00149 (10 uniform bins)
```

# F. Alert Budget

```text
Recall@Top25: 0.3906  (25/64 mules; 95% CI [0.3086, 0.5000])
Precision@Top25: 1.0000
Recall@Top50: 0.6875  (44/64 mules; 95% CI [0.5844, 0.7869])
Precision@Top50: 0.8800
Recall@Top100: 0.8281  (53/64 mules; 95% CI [0.7241, 0.9118])
Precision@Top100: 0.5300
FP/1000 legit: 6.528 at a 100-alert budget; 0.000 at 25
```

# G. Stability

```text
Seed stability: PR-AUC 0.76294 +/- 0.03222 over 5 seeds, spread 0.0905. Any model comparison smaller than this spread on unpaired folds is noise.
Positive-removal stability: 0.72398 +/- 0.03545 over 15 rounds dropping 30.0 positives each; relative drop 0.0337
Feature stability: rank correlation 0.7696, top-20 overlap 0.6805
Rank stability: 0.3694 Spearman over all development rows - low because ~99% of rows are negatives whose calibrated scores are near-tied; the analyst-facing number is the top-budget overlap
```

# H. Generalization

```text
Adversarial validation: PENDING - produced by: python -m muleguard.cli.nested_ses --stages shift
OOD: locked-test drift STABLE - score PSI 0.0000, 0 features at alert level over 1818 rows
Shift-prone features: PENDING - produced by: python -m muleguard.cli.nested_ses --stages shift
Hidden-validation readiness: organiser dry run PASS, 8/8 upload variants invariant
```

# I. Validation Lab

```text
Targetless upload: 8/8 variants accepted with the target column removed (1818 rows)
Target-present sealed validation: seal verified: True - predictions sealed 2026-08-12T10:59:04.490653+00:00, labels revealed 2026-08-12T10:59:38.459657+00:00
Row-order preservation: all_invariant=True, sound=True (a sensitivity control confirms the check can fail)
Competition export: batch upload checks 4/4
No-retraining assertion: bundle fingerprint afd0dc1d8fc02eb9 -> afd0dc1d8fc02eb9 (unchanged=True)
```

# J. System

```text
Backend: 15/15 checks passed - STALE, recorded before xgboost_top_120
Frontend metrics: 6/6 checks passed - STALE, recorded before xgboost_top_120
Offline: UNVERIFIED - PENDING - produced by: bash scripts/test_offline.sh
Ollama-off: 16/16 checks passed - STALE, recorded before xgboost_top_120
No MCP: CLEAR - source scan records no MCP or browser-automation dependency
No Claude in Chrome: CLEAR - the same scan covers browser automation
```

# K. Tests

```text
Passed: 89/89 QA checks, 93 pytest, 3 vitest [STALE - recorded for catboost_tuned_top60, not xgboost_top_120]
Failed: 0
P0: 0 open
P1: 0 open, 0 approved non-blocking exceptions
```

# L. Final Verdict

```text
PENDING_EVIDENCE
```

This is deliberately not one of the three strings section 65 permits. The permitted verdicts are claims about completed evidence, and the evidence below is still open. The rule that produced this line:

> FAIL if any section 63 blocker is BLOCKED. Otherwise a section 65 verdict is issued only when every blocker is CLEAR and every section 64 criterion is MET; while evidence is outstanding the verdict is PENDING_EVIDENCE, because a PASS over incomplete evidence is a guess, not a verdict.

## Release blockers (section 63)

| blocker | status | evidence |
| --- | --- | --- |
| F3924 enters model features | CLEAR | none of F3924 appear among the champion's 120 input features |
| F3898/F3899 enter accepted model | CLEAR | none of F3898, F3899 appear among the champion's 120 input features |
| F3912/F3913/F3914/F3915 enter accepted model | CLEAR | none of F3912, F3913, F3914, F3915 appear among the champion's 120 input features |
| F2230 remains suspicious and is still included | CLEAR | none of F2230 appear among the champion's 120 input features |
| F3916-18 are used without availability evidence | CLEAR | none of F3916, F3917, F3918 appear among the champion's 120 input features |
| train-validation overlap | STALE | all checks passed, but artifacts/testing/leakage_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| preprocessing fitted on validation | STALE | all checks passed, but artifacts/testing/leakage_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| feature selection fitted on validation | STALE | all checks passed, but artifacts/testing/leakage_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| test labels used for tuning | STALE | all checks passed, but artifacts/testing/leakage_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| external validation triggers retraining | CLEAR | the bundle fingerprint is identical before and after the organiser upload |
| fake metrics | STALE | all checks passed, but artifacts/testing/ui_metric_consistency.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| hardcoded dashboard metrics | STALE | all checks passed, but artifacts/testing/api_frontend_consistency.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| prediction row order changes | CLEAR | predictions are invariant to row order and column order |
| model artifact cannot reproduce saved predictions | STALE | all checks passed, but artifacts/testing/leakage_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| UI/backend score mismatch | STALE | all checks passed, but artifacts/testing/api_frontend_consistency.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| Ollama changes scoring | STALE | all checks passed, but artifacts/testing/ollama_guardrail_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| core requires internet | UNVERIFIED | PENDING - produced by: bash scripts/test_offline.sh |
| core requires MCP | CLEAR | source scan records no MCP or browser-automation dependency |
| core requires Claude in Chrome | CLEAR | source scan records no MCP or browser-automation dependency |
| P0 defect | STALE | the release summary records zero blockers for catboost_tuned_top60, which is not the current champion xgboost_top_120 |
| unapproved P1 defect | STALE | the release summary records zero blockers for catboost_tuned_top60, which is not the current champion xgboost_top_120 |

## Pass criteria (section 64)

| group | criterion | status | evidence |
| --- | --- | --- | --- |
| Data | DataSet.xlsx fingerprinted | NOT_MET | workbook SHA-256 recorded and re-verified, but recorded before xgboost_top_120 was promoted |
| Data | Description.xlsx fingerprinted | MET | description workbook SHA-256 recorded |
| Data | target verified | NOT_MET | all checks passed, but artifacts/testing/data_integrity.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| Data | semantic registry built | MET | feature dictionary built from Description.xlsx |
| Data | post-resolution leakage excluded | MET | none of F2230, F3892, F3898, F3899, F3912, F3913, F3914, F3915, F3916, F3917, F3918, F3924, __UNNAMED__0 appear among the champion's 120 input features |
| ML | nested repeated CV complete | NOT_MET | on disk: 2 families (dummy_prevalence, xgboost) at 1 repeat(s); the programme calls for the full family set at 3 repeats |
| ML | strong model tournament complete | MET | 22 models, coverage COMPLETE |
| ML | best candidate stability tested | MET | seed, positive-removal and rank stability measured |
| ML | calibration tested | MET | isotonic/Platt comparison, Brier, ECE and coverage |
| ML | analyst-budget metrics produced | MET | recall and precision at every budget with intervals |
| ML | confidence intervals produced | MET | percentile bootstrap over accounts, 2000 draws |
| Hidden validation | targetless upload works | MET | the organiser dry run passes |
| Hidden validation | labeled upload uses sealed protocol | MET | predictions sealed before labels revealed |
| Hidden validation | no retraining occurs | MET | bundle fingerprint unchanged |
| Hidden validation | submission export preserves order | MET | row order preserved and invariant |
| Hidden validation | distribution shift reported | NOT_MET | PENDING - produced by: python -m muleguard.cli.nested_ses --stages shift |
| Runtime | backend works offline | NOT_MET | PENDING - produced by: bash scripts/test_offline.sh |
| Runtime | frontend metrics match backend | NOT_MET | all checks passed, but artifacts/testing/api_frontend_consistency.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| Runtime | Ollama optional | NOT_MET | all checks passed, but artifacts/testing/ollama_guardrail_results.json was recorded before xgboost_top_120 was promoted; re-run scripts/release_test.sh |
| Runtime | one-command run works | NOT_MET | run.sh brought the stack up, but recorded before xgboost_top_120 was promoted |

## Top 5 remaining risks

1. Evidence is incomplete: 8 of 20 pass criteria are not met yet (DataSet.xlsx fingerprinted, target verified, nested repeated CV complete, distribution shift reported, ...). Nothing here can be read as a final PASS until those runs land.
2. 11 release blockers are cleared only by evidence recorded before xgboost_top_120 was promoted. The QA suites must be re-run against the current bundle before the verdict means anything.
3. 64 positives at 0.88% prevalence. One mule is worth 1.6 recall points, so every interval in this report is wide and every fold-level difference is fragile.
4. Under positive removal the worst round fell to 0.6413 PR-AUC from a 0.7493 reference: the model's ranking depends on which mules it was shown.
5. The headline metrics still come from the flat repeated-CV protocol. Nested CV is the primary protocol in this programme and it usually reports lower, so the headline should be expected to fall.

---

Machine-readable form: `artifacts/testing/final_verdict.json`.
