# Final Results

Generated 2026-07-10T21:43:41.405791+00:00 · commit `e831082acffb` · compute mode `cpu_16gb`
· seed 42 · raw SHA-256 `7d1be90fe23b57460184…`

Every number below is produced by the pipeline and traceable to
`artifacts/metrics/` + `artifacts/predictions/`. Splits: **dev OOF** =
repeated stratified 5-fold CV on 7,264 dev rows (64 positives), preprocessing
and selection inside folds; **locked test** = 1,818 rows (17 positives),
touched exactly once.

## Dataset (verified)

- 9,082 accounts × 3,925 columns; target `F3924`
- 81 positives / 9,001 negatives → prevalence 0.8919%
- Quarantined: F3924, F3912, F2230 (snapshot month ≡ label), __UNNAMED__0 (row index)

## Leakage ablation (dev OOF)

| Run | PR-AUC | Status |
|---|---|---|
| LightGBM clean (quarantine enforced) | 0.6142 ± 0.0474 | ACCEPTED |
| LightGBM + F3912 | 0.9419 ± 0.0119 | **REJECTED LEAKAGE — evidence only** |

## Model tournament (dev OOF, natural prevalence)

| Model | Features | PR-AUC (mean ± std) | ROC-AUC | Repeats | Runtime s |
|---|---|---|---|---|---|
| tabpfn_top60 |  | 0.8970 ± 0.0000 | 0.9890 | 1 | 3144.3 |
| catboost_tuned_top60 | 60 | 0.8077 ± 0.0450 | 0.9648 | 5 | 142.2 |
| lightgbm_tuned_top60 | 60 | 0.7717 ± 0.0528 | 0.9614 | 5 | 51.5 |
| lightgbm_tuned_full | full_clean | 0.7637 ± 0.0223 | 0.9602 | 5 | 710.2 |
| lightgbm_tuned_top30 | 30 | 0.7562 ± 0.0255 | 0.9647 | 5 | 51.9 |
| xgboost_tuned_top60 | 60 | 0.7560 ± 0.0658 | 0.9483 | 5 | 57.3 |
| lightgbm_tuned_top15 | 15 | 0.6440 ± 0.0420 | 0.9515 | 5 | 39.4 |
| lightgbm_baseline |  | 0.6142 ± 0.0474 | 0.9359 | 5 | 736.0 |
| logistic_l2 |  | 0.2328 ± 0.0120 | 0.9202 | 5 | 70.1 |
| dummy_prevalence |  | 0.0087 ± 0.0000 | 0.4938 | 5 | 59.6 |
| REJECTED_leakage_lgbm_with_F3912 | — | 0.9419 ± 0.0119 | 0.9933 | 2 | REJECTED LEAKAGE |

Winner: **tabpfn_top60** (0.8970 ± 0.0000).
Ensemble decision: **REJECTED** — stacker accepted only if it beats the best single model on >= n-1 repeats
(stacker AP by repeat [0.8443, 0.838, 0.8666, 0.8076, 0.8416] vs best single [0.7681, 0.8384, 0.869, 0.7462, 0.817]).

Feature selection: stability selection inside folds; fold-to-fold top-60
overlap 0.54. Compact sets: top-15/30/60 evaluated above.

### Advanced challengers

| Challenger | Status | Reason |
|---|---|---|
| tabpfn | RAN | completed on top-60 features, 1 repeat (cached OOF result); challenger only - winner eligibility requires 5-repeat evidence, and CPU runtime (3144.3s for one repeat) is operationally prohibitive at ~26 min/fold |
| tabicl | SKIPPED | package not installed; TabICL(v2) requires GPU-class resources for its in-context regime - Mode A (CPU 16GB) excludes it. Documented as a Mode B/C experiment. |
| autogluon | SKIPPED | AutoGluon does not support Python 3.13 at build time (requires <=3.12); benchmark documented as a roadmap item on a compatible environment |

## Locked test (single touch)

| Metric | Value |
|---|---|
| PR-AUC — production scorer (winner, calibrated) | 0.8242 (95% CI 0.6536–0.9584) |
| PR-AUC — LightGBM agreement model (secondary) | 0.8608 (95% CI 0.7031–0.9819) |
| ROC-AUC (secondary) | 0.9940 (95% CI 0.9877–0.9990) |
| Brier score | 0.00258 |
| ECE (10 bins) | 0.0027 |
| Conformal abstention rate | 0.55% |
| Positive conformal coverage | 76.47% |
| OOD rate | 0.11% |
| Scoring throughput | 687.7 rows/s (CPU) |

### Recall / precision at alert budgets (locked test)

| Budget | Caught | Recall | Precision |
|---|---|---|---|
| top 18 | 13/17 | 76.47% | 72.22% |
| top 25 | 13/17 | 76.47% | 52.00% |
| top 50 | 14/17 | 82.35% | 28.00% |
| top 100 | 17/17 | 100.00% | 17.00% |

### Recall at fixed FPR (locked test)

| FPR target | Recall | FP per 1,000 legit |
|---|---|---|
| 0.5% | 76.47% | 5.6 |
| 1.0% | 76.47% | 10.5 |

### Review-tier outcomes (policy applied to locked test)

| Tier | Accounts | True mules | Precision in tier |
|---|---|---|---|
| CRITICAL_REVIEW | 12 | 12 | 100.00% |
| URGENT_REVIEW | 6 | 1 | 16.67% |
| STANDARD_REVIEW | 26 | 0 | 0.00% |
| OOD_REVIEW | 2 | 0 | 0.00% |
| MONITOR | 1772 | 4 | 0.23% |

## Lens stack (fitted on dev OOF only)

- Calibrator: **platt** — comparison {'platt': {'brier': 0.0023493878976461576, 'ece': 0.0005267187771415963}, 'isotonic': {'brier': 0.0024545057969935172, 'ece': 0.0011328873237070357}}
- Conformal (α=0.10) OOF coverage: {'target_coverage': 0.9, 'positive_coverage': 0.9375, 'negative_coverage': 0.9709722222222222, 'abstention_rate': 0.07034691629955947, 'share_high': 0.036894273127753306, 'share_low': 0.8927588105726872}
- Hard negatives mined: 144
- Policy thresholds (frozen): {'critical_risk': 0.9638304837100166, 'urgent_risk': 0.048153197236171764, 'standard_risk': 0.017246147513913988, 'anomaly_escalation_pct': 99.0, 'policy_version': '1.0'}

## Bundle

`04fafaee25ae82c7a5a2e6ec…` · 60 features ·
calibrator platt · registered champion in
`artifacts/model_registry/registry.json`.
