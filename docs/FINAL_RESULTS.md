# Final Results

> **SUPERSEDED — RETIRED, LEAKAGE-INFLATED RUN. Do not quote any model result
> from this document as a current result.**
>
> Every model number below was produced before the Feature Availability
> Firewall re-quarantined the post-resolution columns `F3898`, `F3899`,
> `F3912`, `F3913`, `F3914`, `F3915`, the target `F3924`, the raw index
> column, and (conditionally) `F3916`/`F3917`/`F3918`. The accepted model of
> this run, `catboost_tuned_top60`, selected `F3898`, `F3913`, `F3914` and
> `F3916` — all inadmissible — so its **0.8077 is retired**.
>
> Current leakage-free results:
> **`xgboost_top_120`, OOF PR-AUC 0.7690 ± 0.0266** —
> see `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md`,
> `artifacts/metrics/tournament_v2.json`,
> `artifacts/metrics/promotion_decision_v2.json`.
>
> This file is retained unedited below the banner as the reproducible record of
> how the leaked result was produced, and of the dataset facts (§ Dataset) that
> did not change.

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

## Model tournament (dev OOF, natural prevalence) — RETIRED

**Every row in this table was computed on the pre-firewall feature pool and is
retired.** `catboost_tuned_top60` at 0.8077 selected three post-resolution
columns plus one of undetermined availability; `tabpfn_top60` at 0.8970 was a
single-repeat run on the same leaked top-60 set and is retired with it. The
leakage-free replacement leaderboard is in
`docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md` §3.

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

Winner of this retired run: **tabpfn_top60** (0.8970 ± 0.0000) as raw leader,
with `catboost_tuned_top60` (0.8077 ± 0.0450) taken into the bundle under the
5-repeat eligibility rule. **Both figures are retired.** The current promoted
model is `xgboost_top_120` at 0.7690 ± 0.0266
(`artifacts/metrics/promotion_decision_v2.json`).

In the re-run tournament `tabpfn_top_60` **did** complete, once
`ignore_pretraining_limits=True` was set: status `OK`, OOF PR-AUC
**0.9110 ± 0.0044** over three repeats. That figure is *not* retired and is not
in doubt — it was earned on firewall-admitted columns only, and the control that
settles it is that `xgboost_top_60` sees the identical 60 columns over the
identical folds and scores 0.7408. It is nonetheless **not promoted**: a single
`predict_proba` call costs 438 s and the model has no attribution path, so it
can neither answer an analyst nor produce a §17 ProofGraph. See
`UPGRADE_GAP_ANALYSIS.md` §3.1.1 and
`artifacts/metrics/challenger_review_v2.json`.
Ensemble decision: **REJECTED** — stacker accepted only if it beats the best single model on >= n-1 repeats
(stacker AP by repeat [0.8443, 0.838, 0.8666, 0.8076, 0.8416] vs best single [0.7681, 0.8384, 0.869, 0.7462, 0.817]).

Feature selection: stability selection inside folds; fold-to-fold top-60
overlap 0.54. Compact sets: top-15/30/60 evaluated above.

### Advanced challengers

| Challenger | Status | Reason |
|---|---|---|
| tabpfn | RAN — challenger, verified, not promoted | Completed 3 repeats on the firewall-admitted top-60 set at OOF PR-AUC 0.9110 ± 0.0044, satisfying UPDATE 1's three-independent-fold-seeds gate. Blocked on serving, not on evidence: 438 s per single-row `predict_proba` and no attribution path, so no §17 ProofGraph. See `UPGRADE_GAP_ANALYSIS.md` §3.1.1. |
| tabicl | SKIPPED | package not installed; TabICL(v2) requires GPU-class resources for its in-context regime - Mode A (CPU 16GB) excludes it. Documented as a Mode B/C experiment. |
| autogluon | SKIPPED | AutoGluon does not support Python 3.13 at build time (requires <=3.12); benchmark documented as a roadmap item on a compatible environment |

## Locked test (single touch) — RETIRED BUNDLE

**These figures belong to the retired `catboost_tuned_top60` bundle
(`04fafaee25ae82c7…`) and must not be attributed to the current champion.**
`artifacts/metrics/locked_test_touch_log.json` records all three touches
against that bundle. The locked test has **not** been re-evaluated under
`xgboost_top_120`, and no locked-test number is published for it.

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

## Lens stack (fitted on dev OOF only) — RETIRED

Superseded by `artifacts/metrics/lens_stack_oof_v2.json`, whose `supersedes`
field names this run as "pre-firewall, leaky feature set". Current values:
Platt calibrator, Brier 0.003128, ECE 0.001489; frozen policy thresholds
critical 0.93385 / urgent 0.09774 / standard 0.01318.

- Calibrator: **platt** — comparison {'platt': {'brier': 0.0023493878976461576, 'ece': 0.0005267187771415963}, 'isotonic': {'brier': 0.0024545057969935172, 'ece': 0.0011328873237070357}}
- Conformal (α=0.10) OOF coverage: {'target_coverage': 0.9, 'positive_coverage': 0.9375, 'negative_coverage': 0.9709722222222222, 'abstention_rate': 0.07034691629955947, 'share_high': 0.036894273127753306, 'share_low': 0.8927588105726872}
- Hard negatives mined: 144
- Policy thresholds (frozen): {'critical_risk': 0.9638304837100166, 'urgent_risk': 0.048153197236171764, 'standard_risk': 0.017246147513913988, 'anomaly_escalation_pct': 99.0, 'policy_version': '1.0'}

## Bundle — RETIRED

`04fafaee25ae82c7a5a2e6ec…` · 60 features · calibrator platt ·
`status: "retired"` in `artifacts/model_registry/registry.json`.

Current champion bundle: `d12914de5abee99a…` · `xgboost_top_120` · 120
firewall-admitted features · calibrator platt ·
`leakage_status: FIREWALL_ADMITTED_ONLY` · `ensemble_decision:
SINGLE_MODEL_KEPT` (`artifacts/models/model_manifest.json`).
