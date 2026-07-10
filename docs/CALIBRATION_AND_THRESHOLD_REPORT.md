# Calibration and Threshold Report

Measured values: `artifacts/metrics/lens_stack_oof.json` (selection +
coverage), `artifacts/metrics/locked_test_metrics.json` (final Brier/ECE),
`artifacts/metrics/threshold_table.csv` (confusion at tier thresholds),
`artifacts/model_registry/policy_snapshot.json` (frozen thresholds).

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
