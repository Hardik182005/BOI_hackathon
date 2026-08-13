# Calibration and Threshold Report

Describes the deployed champion **`xgboost_top_120`, bundle v2.0.0**.

Measured values:

| what | authoritative source | value |
| --- | --- | --- |
| calibrator selection + conformal coverage | `artifacts/metrics/lens_stack_oof_v2.json` | Platt wins: Brier **0.003128** vs isotonic 0.003159, ECE **0.001489** vs 0.001698 (dev OOF, 3×5) |
| frozen tier thresholds | `artifacts/models/final_bundle.joblib` → `policy_thresholds` | critical 0.93385, urgent 0.09774, standard 0.01318 |
| locked-test Brier / ECE | **not asserted** | the locked test is single-touch and was spent on the retired run; no calibration metric is claimed for the champion on it |

Three unsuffixed files under `artifacts/` describe the **retired**,
pre-firewall `catboost_tuned_top60` bundle and must not be read as current:
`locked_test_metrics.json` (PR-AUC 0.82423, Brier 0.00258), `threshold_table.csv`
and `final_threshold_table.csv` (the 12-alert / 100 %-precision CRITICAL row),
and `artifacts/model_registry/policy_snapshot.json` (critical 0.96383). The
snapshot file in particular disagrees with the bundle; **the bundle is what
serves traffic**, and the snapshot has not been regenerated since the firewall.
`policy_version` still reads `"1.0"` in both, which under-labels a threshold set
that was re-frozen for generation 2 — recorded as a known defect in
`docs/METRIC_BATTERY.md`, not silently corrected here.

The retirement itself is documented in `docs/HISTORICAL_METRIC_RECONCILIATION.md`
and `docs/LOCKED_TEST_RULING.md`; the generation-2 counterpart of this document
is `docs/FINAL_CALIBRATION_AND_THRESHOLD_REPORT.md`.

## Why calibration matters here

Raw tree scores are rankings, not probabilities. Review tiers, alert budgets
and any future cost analysis need a true probability, so the score driving
decisions is a calibrated one — and its quality is measured (Brier, ECE,
reliability curve), not assumed.

## Calibrator selection (dev OOF only)

- Candidates: Platt (sigmoid on logit) vs isotonic regression.
- Method: 5-fold **cross-fitted** calibrated probabilities over the
  repeat-averaged OOF scores of the winning model — every point is calibrated
  by a model that never saw it; compared on Brier + ECE.
- Small-positives guard: with 64 dev positives, isotonic overfits easily, so
  isotonic must beat Platt by ≥2% relative on **both** metrics to win;
  otherwise Platt is selected. The measured comparison and the winner are
  recorded in the lens artifact and the bundle manifest.
- The final calibrator is refit on all dev OOF scores and frozen into the
  bundle.

## Conformal abstention

Mondrian (class-conditional) split conformal, α = 0.10, finite-sample
(n+1)-corrected quantiles per class, fitted on the crossfit-calibrated OOF
probabilities. Outputs HIGH_RISK_SET / LOW_RISK_SET / UNCERTAIN_SET; empirical
OOF coverage and locked-test abstention are in the artifacts. With 64
positive calibration points the positive-class quantile is conservative —
by design: under-confidence routes to humans, never to silent passes.

## Tier thresholds (frozen from dev OOF calibrated distribution)

| Tier | Derivation |
|---|---|
| CRITICAL_REVIEW | top-25 dev-OOF calibrated scores (analyst daily critical capacity) |
| URGENT_REVIEW | top-100 dev-OOF calibrated scores |
| STANDARD_REVIEW | threshold achieving 90% recall on dev OOF positives (capped to nest under URGENT) |
| OOD_REVIEW | any OOD trip, regardless of score |
| MONITOR | below STANDARD, escalatable by conformal uncertainty or anomaly percentile ≥ 99 |

The locked test never influenced any threshold; its per-tier precision is the
*first* out-of-sample read of the policy and is reported as such. Hidden
thresholds are not exposed in user-facing screens (dashboard shows tiers and
scores, not cutoffs; exact values live in the registry snapshot for auditors).
