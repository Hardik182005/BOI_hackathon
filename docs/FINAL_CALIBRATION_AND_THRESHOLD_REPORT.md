# Final Calibration and Threshold Report

Authoritative artifact for the **current** model:
`artifacts/metrics/lens_stack_oof_v2.json` (its `supersedes` field names
`lens_stack_oof.json` as "pre-firewall, leaky feature set"), plus
`artifacts/models/model_manifest.json` for the frozen thresholds. Retired-run
artifacts: `artifacts/metrics/final_threshold_table.csv`,
`final_locked_test_metrics.json`, `lens_stack_oof.json`,
`artifacts/model_registry/policy_snapshot.json`. Method detail:
`docs/CALIBRATION_AND_THRESHOLD_REPORT.md`.

## Calibration — current model (`xgboost_top_120`, dev OOF)

- Platt vs isotonic compared with cross-fitted OOF probabilities; isotonic
  needed ≥2% relative wins on BOTH Brier and ECE (small-positives guard) and
  did not clear it → **Platt selected**, refit on all dev OOF, frozen in the
  bundle. Measured: Platt Brier 0.003128 / ECE 0.001489 versus isotonic
  0.003159 / 0.001698.
- Frozen policy thresholds: critical 0.93385, urgent 0.09774, standard
  0.01318, anomaly escalation at the 99th percentile (policy version 1.0).
  These replace the retired bundle's 0.96383 / 0.04815 / 0.01725; the score
  distribution changed when the leaked columns were removed, so the thresholds
  had to be re-derived rather than carried over.
- Conformal (α=0.10) OOF coverage: positive 0.9375, negative 0.9474,
  abstention 4.69%.

## Calibration — RETIRED run (locked test, retired bundle)

- Locked test (single touch): **Brier 0.00258, ECE 0.0027**. **These belong to
  the retired `catboost_tuned_top60` bundle.** The locked test has not been
  re-touched under the current champion
  (`artifacts/metrics/locked_test_touch_log.json`), so no locked-test
  calibration figure exists for `xgboost_top_120` and none is asserted here.

## Uncertainty layers

- **Mondrian conformal** (α=0.10, class-conditional, finite-sample corrected)
  on crossfit-calibrated OOF: locked-test abstention 0.55%, positive coverage
  76.5% (13/17 mules in HIGH_RISK or UNCERTAIN sets — the conservative
  small-n quantile is stated, not hidden).
- **Model disagreement** (LGBM/XGB/CatBoost spread) attached to every score.
- **OOD detection**: locked-test OOD rate 0.11%; synthetic extremes route to
  OOD_REVIEW (e2e scenario D).
- **Anomaly challenger**: IsolationForest percentile ≥99 escalates MONITOR.

## Tier outcomes on the locked test (RETIRED bundle, policy applied, thresholds dev-frozen)

**Retired.** The table below was produced by the retired `catboost_tuned_top60`
bundle under its own thresholds. It is not a description of current behaviour,
and the "100.0%" precision figure in it must not be quoted as a result of the
current system.

| Tier | Accounts | True mules | FP | Precision in tier | Recall contribution |
|---|---|---|---|---|---|
| CRITICAL_REVIEW | 12 | 12 | 0 | 100.0% | 70.6% of all mules |
| URGENT_REVIEW | 6 | 1 | 5 | 16.7% | +5.9% |
| STANDARD_REVIEW | 26 | 0 | 26 | 0% | high-recall screen band |
| OOD_REVIEW | 2 | 0 | 2 | — | routed for input reasons |
| MONITOR | 1,772 | 4 | — | 0.23% miss rate in monitor pool | monitored, never "safe" |

Forbidden outputs (`GUILTY`, `CRIMINAL`, `CERTIFIED_CLEAN`,
`PERMANENTLY_SAFE`, automatic `FREEZE`) do not exist as states anywhere in
code — enforced by unit tests and the policy engine's closed tier set.

Edge cases (missing features, unseen category, extreme values, disagreement)
are exercised in `tests/e2e/test_robustness.py` and the QA harness; each
routes to SCHEMA_ERROR or a review tier, never to a confident pass.
