# MuleGuard - Trinetra: final validation response (section 65)

Generated 2026-08-23T11:20:24+00:00 by `python -m muleguard.cli.final_verdict`.
Every field is read from a named artifact. A field with no evidence behind it says PENDING and names the run that would produce it; none is ever filled in from an earlier model.

Champion: **xgboost_top_120**, promoted 2026-08-13T02:53:19.184309+00:00.

# A. Environment

```text
Git SHA: f280c6fc576b61bbe4edb4c2981ff5fa253e831f (working tree dirty)
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
| catboost | NESTED 3x5x4 | 120 | 0.80653 | 0.00845 | NESTED |
| histgb | NESTED 3x5x4 | 120 | 0.76735 | 0.02949 | NESTED |
| xgboost | NESTED 3x5x4 | 120 | 0.75393 | 0.00740 | NESTED |
| lightgbm | NESTED 3x5x4 | 120 | 0.70046 | 0.02362 | NESTED |
| extratrees | NESTED 3x5x4 | 120 | 0.54839 | 0.06962 | NESTED |
| logistic_l1l2 | NESTED 3x5x4 | 120 | 0.16386 | 0.00308 | NESTED |
| dummy_prevalence | NESTED 3x5x4 | 120 | 0.00910 | 0.00022 | NESTED |

**Arbiter (CHAMPION_CHALLENGED):** under the primary nested protocol the promotion rule selects `catboost`, not the shipped `xgboost_top_120`. Scored on identical rows the gap is 0.05279 PR-AUC, 95% CI [0.02374, 0.08602], 3/3 repeats favouring the challenger. The swap was **not** taken. This file records the finding. Swapping the champion is a decision with a locked-test cost attached and is not taken by a report. Section E therefore describes an artefact that places third of 6 under the protocol this project calls primary.

# E. Champion

```text
Model: xgboost_top_120 (battery run FLAT:xgboost_top_120)  [operating point and probability quality measured under the FLAT protocol, which is the run that scored this exact bundle; the primary nested figure for family 'xgboost' is the Nested-CV row below]
Feature set: top_120 of view ALL_ADMISSIBLE
Number of features: 120
Nested-CV AP mean: 0.75393 +/- 0.00740 for family 'xgboost' over 3 repeats (NESTED:xgboost) - the nested protocol selects its own feature-set size per fold, so this is the family's figure, not this bundle's
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
Positive-removal stability (NESTED, primary): 0.76780 against a 0.78161 reference, relative drop 0.0177. Paired over 15 outer folds the loss is -0.01381, 95% CI [-0.02130, -0.00631], worse in 14 of 15 (sign p 0.00098). The drop is small and real, not small and deniable.
Feature stability (NESTED vs FLAT): rank correlation 0.3944 nested against 0.7696 flat. The gap is structural, not noise: the flat run fits feature selection once over pooled development data, so dropping training positives cannot disturb a choice already made, while the nested run re-selects inside every outer fold and the choice moves. Read 0.3944 as the honest figure - when the mules this model learned from change, so does most of what it cites.
Seed stability: PR-AUC 0.76294 +/- 0.03222 over 5 seeds, spread 0.0905. Any model comparison smaller than this spread on unpaired folds is noise.
Positive-removal stability (FLAT): 0.72398 +/- 0.03545 over 15 rounds dropping 30.0 positives each; relative drop 0.0337. Worst round 0.6413
Feature stability (FLAT): rank correlation 0.7696, top-20 overlap 0.6805 - see the nested figure above, which is the one to quote
Rank stability (FLAT): 0.3694 Spearman over all development rows - low because ~99% of rows are negatives whose calibrated scores are near-tied; the analyst-facing number is the top-budget overlap
```

# H. Generalization

```text
Adversarial validation: PENDING - produced by: python -m muleguard.cli.nested_ses --stages shift
OOD: locked-test drift STABLE - score PSI 0.0000, 0 features at alert level over 1818 rows
Shift-prone features: PENDING - produced by: python -m muleguard.cli.nested_ses --stages shift
Hidden-validation readiness: organiser dry run PASS, 11/11 upload variants invariant
```

# I. Validation Lab

```text
Targetless upload: 11/11 variants accepted with the target column removed (1818 rows)
Target-present sealed validation: seal verified: True - predictions sealed 2026-08-14T08:38:38.092004+00:00, labels revealed 2026-08-14T08:42:03.018116+00:00
Row-order preservation: all_invariant=True, sound=True (a sensitivity control confirms the check can fail)
Competition export: batch upload checks 4/4
No-retraining assertion: bundle fingerprint afd0dc1d8fc02eb9 -> afd0dc1d8fc02eb9 (unchanged=True)
```

# J. System

```text
Backend: 15/15 checks passed
Frontend metrics: 6/6 checks passed
Offline: CLEAR - backend serves with the network stack pointed at a dead port
Ollama-off: 16/16 checks passed
No MCP: CLEAR - source scan records no MCP or browser-automation dependency
No Claude in Chrome: CLEAR - the same scan covers browser automation
```

# K. Tests

```text
Passed: 90/90 QA checks, 93 pytest, 3 vitest
Failed: 0
P0: 0 open
P1: 0 open, 0 approved non-blocking exceptions
```

# L. Final Verdict

```text
PENDING_EVIDENCE
```

Scope: the evidence is incomplete, so nothing is certified either way yet.

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
| train-validation overlap | CLEAR | checks passed 6/6 |
| preprocessing fitted on validation | CLEAR | checks passed 6/6 |
| feature selection fitted on validation | CLEAR | checks passed 6/6 |
| test labels used for tuning | CLEAR | checks passed 6/6 |
| external validation triggers retraining | CLEAR | the bundle fingerprint is identical before and after the organiser upload |
| fake metrics | CLEAR | checks passed 6/6 |
| hardcoded dashboard metrics | CLEAR | checks passed 6/6 |
| prediction row order changes | CLEAR | predictions are invariant to row order and column order |
| model artifact cannot reproduce saved predictions | CLEAR | checks passed 6/6 |
| UI/backend score mismatch | CLEAR | checks passed 6/6 |
| Ollama changes scoring | CLEAR | checks passed 16/16 |
| core requires internet | CLEAR | backend serves with the network stack pointed at a dead port |
| core requires MCP | CLEAR | source scan records no MCP or browser-automation dependency |
| core requires Claude in Chrome | CLEAR | source scan records no MCP or browser-automation dependency |
| P0 defect | CLEAR | release summary records no open blockers |
| unapproved P1 defect | CLEAR | release summary records no open blockers |

## Pass criteria (section 64)

| group | criterion | status | evidence |
| --- | --- | --- | --- |
| Data | DataSet.xlsx fingerprinted | MET | workbook SHA-256 recorded and re-verified |
| Data | Description.xlsx fingerprinted | MET | description workbook SHA-256 recorded |
| Data | target verified | MET | checks passed 11/11 |
| Data | semantic registry built | NOT_MET | feature dictionary built from Description.xlsx, but recorded before xgboost_top_120 was promoted |
| Data | post-resolution leakage excluded | MET | none of F2230, F3892, F3898, F3899, F3912, F3913, F3914, F3915, F3916, F3917, F3918, F3924, __UNNAMED__0 appear among the champion's 120 input features |
| ML | nested repeated CV complete | MET | 7 families x 3 repeats x 5 outer folds |
| ML | strong model tournament complete | MET | 22 models, coverage COMPLETE |
| ML | best candidate stability tested | NOT_MET | seed, positive-removal and rank stability measured, but recorded before xgboost_top_120 was promoted |
| ML | calibration tested | MET | isotonic/Platt comparison, Brier, ECE and coverage |
| ML | analyst-budget metrics produced | MET | recall and precision at every budget with intervals |
| ML | confidence intervals produced | MET | percentile bootstrap over accounts, 2000 draws |
| Hidden validation | targetless upload works | MET | the organiser dry run passes |
| Hidden validation | labeled upload uses sealed protocol | MET | predictions sealed before labels revealed |
| Hidden validation | no retraining occurs | MET | bundle fingerprint unchanged |
| Hidden validation | submission export preserves order | MET | row order preserved and invariant |
| Hidden validation | distribution shift reported | NOT_MET | PENDING - produced by: python -m muleguard.cli.nested_ses --stages shift |
| Runtime | backend works offline | MET | backend serves with the network stack pointed at a dead port |
| Runtime | frontend metrics match backend | MET | checks passed 6/6 |
| Runtime | Ollama optional | MET | checks passed 16/16 |
| Runtime | one-command run works | MET | run.sh brought the stack up |

## Top 5 remaining risks

1. Evidence is incomplete: 3 of 20 pass criteria are not met yet (semantic registry built, best candidate stability tested, distribution shift reported). Nothing here can be read as a final PASS until those runs land.
2. 64 positives at 0.88% prevalence. One mule is worth 1.6 recall points, so every interval in this report is wide and every fold-level difference is fragile.
3. Under positive removal the worst round fell to 0.6413 PR-AUC from a 0.7493 reference: the model's ranking depends on which mules it was shown.
4. The headline metrics come from the flat protocol, which is not the primary one. Under nested CV the shipped family scores 0.75393 against the flat 0.76904 (-0.01511). On the same rows `catboost` beats it by 0.05279 PR-AUC with a 95% interval excluding zero, and the champion was left in place rather than swapped. Read the headline as what the shipped artefact does, not as the best estimate available.
5. Generation-1 numbers (PR-AUC 0.824 and above) came from a model that could see quarantined columns. They remain in the repository as retired evidence and must never be quoted as current behaviour.

---

Machine-readable form: `artifacts/testing/final_verdict.json`.
