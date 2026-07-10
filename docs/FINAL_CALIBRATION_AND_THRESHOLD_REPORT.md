# Final Calibration and Threshold Report

Authoritative artifacts: `artifacts/metrics/final_threshold_table.csv`,
`final_locked_test_metrics.json`, `lens_stack_oof.json`,
`artifacts/model_registry/policy_snapshot.json`. Method detail:
`docs/CALIBRATION_AND_THRESHOLD_REPORT.md`.

## Calibration (selected on dev OOF, measured on locked test)

- Platt vs isotonic compared with 5-fold cross-fitted OOF probabilities;
  isotonic needed ≥2% relative wins on BOTH Brier and ECE (small-positives
  guard) and did not clear it → **Platt selected**, refit on all dev OOF,
  frozen in the bundle.
- Locked test (single touch): **Brier 0.00258, ECE 0.0027** — the calibrated
  probability driving tiers is trustworthy, not just a ranking.

## Uncertainty layers

- **Mondrian conformal** (α=0.10, class-conditional, finite-sample corrected)
  on crossfit-calibrated OOF: locked-test abstention 0.55%, positive coverage
  76.5% (13/17 mules in HIGH_RISK or UNCERTAIN sets — the conservative
  small-n quantile is stated, not hidden).
- **Model disagreement** (LGBM/XGB/CatBoost spread) attached to every score.
- **OOD detection**: locked-test OOD rate 0.11%; synthetic extremes route to
  OOD_REVIEW (e2e scenario D).
- **Anomaly challenger**: IsolationForest percentile ≥99 escalates MONITOR.

## Tier outcomes on the locked test (policy applied, thresholds dev-frozen)

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
