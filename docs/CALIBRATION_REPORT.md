# Calibration Report

Master prompt §14–15. How a model score becomes a probability an analyst can act
on, and where the thresholds came from.

Authoritative artefacts: `artifacts/metrics/lens_stack_oof_v2.json` (current
champion) and `artifacts/models/model_manifest.json` (frozen thresholds).
Method detail: `docs/CALIBRATION_AND_THRESHOLD_REPORT.md`. Threshold table:
`artifacts/metrics/thresholds.csv`.

---

## 1. Why calibrate at all

A raw gradient-boosting score is a ranking, not a probability. Two things depend
on it being a real probability:

1. **Proportionate action.** "This account is 93 % likely to be a mule" and "this
   account is in the top 25" support different decisions. Only the first lets a
   reviewer weigh effort against expected value.
2. **Threshold stability.** Tiers defined on an uncalibrated score move whenever
   the score distribution shifts. Tiers defined on a calibrated probability mean
   the same thing across model versions.

A miscalibrated score is also a false-positive problem: an analyst who learns
that "0.9" usually means "probably nothing" stops reading the number.

---

## 2. Platt vs isotonic — decided by a rule fixed in advance

Both were fitted on cross-fitted OOF probabilities and compared under a rule
written **before** the comparison:

> Isotonic must beat Platt by ≥ 2 % relative on **both** Brier and ECE. With few
> positives, prefer the simpler calibrator.

| Calibrator | Brier | ECE |
|---|---:|---:|
| **Platt (selected)** | **0.003128** | **0.001489** |
| isotonic | 0.003159 | 0.001698 |

Isotonic lost on both, so the rule never even had to arbitrate — but the rule
existing in advance is the point. Isotonic regression fitted on **64 positives**
is a step function with very few steps; it can look better in-sample and
generalise badly, and with a rule written afterwards it would have been easy to
justify either choice.

Platt was refit on all dev OOF predictions and frozen into the bundle.

---

## 3. Frozen policy thresholds

`policy_version 1.0`:

| Tier | Threshold |
|---|---:|
| critical | 0.93385 |
| urgent | 0.09774 |
| standard | 0.01318 |
| anomaly escalation | 99th percentile of the Isolation Forest score |

**These replace the retired bundle's 0.96383 / 0.04815 / 0.01725.** When the
leaked columns were removed the score distribution changed shape, so the
thresholds had to be re-derived rather than carried across. Carrying them over
would have silently changed what each tier meant.

Every threshold was derived on the **development split only**. The locked test
was never used for calibration or threshold tuning — enforced by the touch log
and by a release-gate check.

---

## 4. Operating points, measured

Development OOF (`lens_stack_oof_v2.json`):

| Alert budget | Recall | Precision | True positives |
|---:|---:|---:|---:|
| top 25 | 0.391 | **1.000** | 25 |
| top 50 | 0.688 | 0.880 | 44 |
| top 73 | 0.766 | 0.671 | 49 |
| top 100 | 0.828 | 0.530 | 53 |

| FPR target | Recall | FP per 1,000 legitimate |
|---:|---:|---:|
| 0.5 % | 0.797 | 5.0 |
| 1.0 % | 0.844 | 10.0 |

The precision column is published so that an alert budget can be chosen against a
real trade-off. At 25 alerts the team wastes no reviews; at 100 it finds 15 more
mules and opens 47 legitimate customers' accounts. That is a business decision,
not a modelling one, and the table is what makes it a decision rather than a
default.

---

## 5. Uncertainty layers

Calibration answers "how likely?". These answer "how sure are we of that?".

### Mondrian conformal prediction (α = 0.10, class-conditional)

| | |
|---|---:|
| positive coverage | 0.9375 |
| negative coverage | 0.9474 |
| **abstention rate** | **0.0469** |
| share high | 0.0603 |
| share low | 0.8928 |

The system **declines to commit on 4.7 % of cases**. An abstention is not a
failure — it is a decision to route to a human with the ambiguity stated, and it
appears in the ProofGraph as a `doubt:conformal` node.

### Model disagreement

LightGBM / XGBoost / CatBoost spread is attached to every score. Under UPDATE 6
disagreement is treated as **uncertainty**, never as a reason to raise risk — it
widens the review band instead.

### Out-of-distribution detection

Inputs unlike the training distribution route to `OOD_REVIEW` rather than
receiving a confident score. Release-gate check `ood_routes_to_review` PASS.

### Anomaly challenger

An Isolation Forest that never saw the labels escalates `MONITOR` cases at
percentile ≥ 99, and — more usefully — contributes to the **defence** side when it
finds a high-scoring account unremarkable.

---

## 6. What the tiers are not

Forbidden states — `GUILTY`, `CRIMINAL`, `CERTIFIED_CLEAN`, `PERMANENTLY_SAFE`,
automatic `FREEZE` — **do not exist anywhere in the code**. The tier set is
closed, and a release-gate check scans shipped source for the vocabulary.

The `MONITOR` tier is explicitly *monitored, never safe*. No threshold in this
system produces a clean bill of health.

Edge cases — missing features, unseen categories, extreme values, model
disagreement — route to `SCHEMA_ERROR` or to a review tier. **None of them can
produce a confident pass.** Exercised in `tests/e2e/test_robustness.py`.

---

## 7. A retired table, kept visible

The locked-test tier outcomes below were produced by the **retired**
`catboost_tuned_top60` bundle under its own thresholds. The locked test has not
been re-touched under the current champion, so **no locked-test calibration
figure exists for `xgboost_top_120` and none is asserted**.

| Tier | Accounts | True mules | FP | Precision | Recall contribution |
|---|---:|---:|---:|---:|---|
| CRITICAL_REVIEW | 12 | 12 | 0 | 100.0 % | 70.6 % of all mules |
| URGENT_REVIEW | 6 | 1 | 5 | 16.7 % | +5.9 % |
| STANDARD_REVIEW | 26 | 0 | 26 | 0 % | high-recall screen band |
| OOD_REVIEW | 2 | 0 | 2 | — | routed for input reasons |
| MONITOR | 1,772 | 4 | — | 0.23 % miss rate | monitored, never "safe" |

It is retained because it shows the tier separation the design is aiming for. The
**100 % figure must not be quoted as a result of the current system** — that is
why the caveat is on the table rather than in a footnote.

Retired-run calibration, for completeness: Brier 0.00258, ECE 0.0027 on the
locked test.
