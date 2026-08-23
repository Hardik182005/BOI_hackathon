# Pre-change baseline — what the USP upgrade is forbidden to change

Artifact: `artifacts/upgrade_baseline/usp_prechange_baseline.json`
Produced by: `.venv/Scripts/python.exe -m muleguard.cli.usp_baseline --save`
Verified by: `.venv/Scripts/python.exe -m muleguard.cli.usp_baseline --check`
Taken at commit `33ac429d422b297db7b0a4b0bf184544828e64c1` (branch `main`), before
any Cohort Radar or Account-Control file existed.

Two USP layers are being added — the **Trinetra Mule-Farm Cohort Radar** and the
**Account-Control Ambiguity Guardrail**. Both are strictly post-model: they read
a score that has already been produced and never feed anything back. That is a
design claim, and a design claim is worth exactly as much as the measurement
that checks it. This document is the measurement taken beforehand.

## Why a frozen baseline rather than a README quote

A metric copied out of a report proves only that the report is self-consistent
with itself. Every accuracy figure below is **recomputed from the saved
out-of-fold prediction store**, so a stale artifact describing a model that has
since changed fails the run instead of being echoed back. The one number this
recomputation is checked against is the registry's own headline, and it agrees:

| | value |
|---|---|
| PR-AUC recomputed from `final_model_oof_predictions.parquet` | **0.7690395605299595** |
| PR-AUC recorded in `artifacts/model_registry/registry.json` | 0.76904 |
| PR-AUC recorded inside `final_bundle.joblib` | 0.76904 |
| verdict | **agrees** (tolerance 1e-4; the registry stores five decimals) |

`Recall@100 = 0.78125` likewise reproduces the 0.7813 printed on the model card,
from the predictions rather than from the card.

## The champion, exactly as it stands

| property | value |
|---|---|
| model | `xgboost_top_120` (family `xgboost`) |
| bundle | `artifacts/models/final_bundle.joblib` |
| bundle sha256 | `d12914de5abee99aa18b80e3d41ae2da800dbf9355b53f3a9c0e06806921efc5` |
| data fingerprint | `7d1be90fe23b57460184f2f7566d572cc3a40fc6d17d436fe33219e80eabc204` |
| view | `ALL_ADMISSIBLE` |
| seed | 42 |
| selected features | 120 (order-sensitive hash `daed5969411c3316`) |
| meta-features required | `MG_ALERT_CONVERGENCE`, `MG_PASSTHROUGH_7D`, `MG_RAIL_FRAGMENTATION` |
| calibrator | Platt, `a = 1.1342429437437733`, `b = -0.8585338390023346` |
| ensemble accepted | no — single model kept |

The calibrator is captured by its **fitted constants**, not by its class name.
"The calibrator is unchanged" then means something about behaviour rather than
something about imports.

### Policy thresholds

| threshold | value |
|---|---|
| `critical_risk` | 0.9338487190474695 |
| `urgent_risk` | 0.09773832436704985 |
| `standard_risk` | 0.013182620557866929 |
| `anomaly_escalation_pct` | 99.0 |
| `policy_version` | 1.0 |

## Accuracy, recomputed (3 repeats, out-of-fold, development partition only)

Each repeat is evaluated separately and the results averaged — the protocol the
registry itself uses. Pooling the three repeats into one vector would mix three
predictions of the same account and quietly inflate the sample.

Probability metrics are taken **after** applying the frozen Platt calibrator to
the raw out-of-fold scores, which is the production path exactly: raw, then
calibrate, then threshold. Thresholding raw scores at a calibrated-scale cut-off
would describe a system nobody runs. Ranking metrics are unaffected either way —
Platt is monotone — so PR-AUC, ROC-AUC and the top-K budgets are identical under
both readings.

The operating point is the policy's own `standard_risk` line, 0.013182620557866929,
not the textbook 0.5.

| metric | value |
|---|---|
| PR-AUC | 0.7690395605299595 |
| ROC-AUC | 0.957709056712963 |
| accuracy | 0.961958516886931 |
| balanced accuracy | 0.9137037037037037 |
| precision | 0.1942277472995442 |
| recall | 0.8645833333333334 |
| F1 | 0.31206452281537606 |
| F2 | 0.4964991648579049 |
| MCC | 0.39421701922209573 |
| Brier (calibrated) | 0.003458115345471382 |
| ECE (calibrated, 10 uniform bins) | 0.0024080675000670393 |
| false positives per 1000 legitimate | 37.175925925925924 |
| Recall@25 / Precision@25 | 0.390625 / 1.0 |
| Recall@50 / Precision@50 | 0.6458333333333334 / 0.8266666666666667 |
| Recall@100 / Precision@100 | 0.78125 / 0.5 |
| n / positives / prevalence | 7264 / 64 / 0.00881057268722467 |

A repeat-averaged variant (one score per account, averaged across repeats, then
calibrated) is recorded alongside in the JSON. It is reported **as well as**, not
instead of, the per-repeat average, because the two answer different questions:
per-repeat estimates a single model run, repeat-averaged describes the
ensemble-of-repeats behind the published calibration report.

## Canonical probe rows

400 development accounts, chosen by seed 42 and sorted by row index, scored
through the **live scoring path** — the same `score_rows` the API calls. For each
one the baseline stores the raw per-family probabilities, the calibrated risk,
the risk tier, the decision, the auto-action, the conformal status, the OOD
status, model agreement, and whether the merchant safeguard fired. All at full
float precision; rounding for display would hide exactly the drift the
comparison exists to catch.

Probe set hash: `2690340d4432468d`. The locked test is **not** among them. A
regression check does not get to consume the locked test.

## Leakage quarantine, read live

Not copied into this document from a prompt — read from the firewall config at
snapshot time, which is what the Cohort Radar will also have to obey.

| class | features |
|---|---|
| hard quarantine | `F2230`, `F3898`, `F3899`, `F3912`, `F3913`, `F3914`, `F3915`, `F3924`, `__UNNAMED__0` |
| conditional quarantine | `F3916`, `F3917`, `F3918` |
| fairness-excluded | `F3892` |
| contextual-only (never a model input) | `F3890`, `F3891`, `F3894` |

Policy version 2.0; quarantine hash `877858d2c8cbe6d9`; target column `F3924`.

## Frozen artifact hashes

| artifact | sha256 (first 12) |
|---|---|
| `predictions/final_model_oof_predictions.parquet` | `69c73c8fa03a` |
| `predictions/final_locked_test_predictions.parquet` | `5587931aba06` |
| `metrics/locked_test_metrics.json` | `607e6079746c` |
| `metrics/final_locked_test_metrics.json` | `607e6079746c` |
| `metrics/final_oof_metrics.json` | `7ae54feea5c9` |
| `metrics/final_calibration.json` | `c41c937230f6` |
| `metrics/final_accuracy_table.csv` | `4cdae65b1caa` |
| `metrics/global_shap_importance.json` | `671a3309af41` |
| `models/model_manifest.json` | `cb1e97d59259` |
| `features/quarantined_features.json` | `50211ca78fff` |

Inputs: `DataSet.xlsx` `7d1be90fe23b`, `Description.xlsx` `7d30652b72d4`.

## The pass condition

`--check` recomputes every item above and diffs. It fails on **any** of:

* a change to the bundle hash, model id, feature list hash, calibrator
  constants, thresholds, or quarantine hash;
* a calibrated or raw probability moving by more than **1e-12** on any probe row;
* a single risk-tier or policy-action mismatch on any probe row;
* any classification metric that is not bit-identical.

Nothing here rounds before comparing, and the tolerance exists only so that a
platform-level float difference is *reported* rather than crashing the run. The
honest expectation is exactly `0.0`. A non-zero difference is not to be
explained away as small — it is to be traced to its cause.

Run immediately after being written, against the unchanged repo, `--check`
reported max absolute probability difference `0.000e+00`, zero tier mismatches,
and a difference of `+0.000e+00` on every metric. The comparator round-trips, so
a later non-zero result means the code changed, not the measurement.

## What the baseline deliberately does not do

* It does not evaluate the locked test. The locked-test artifacts are **hashed**,
  never re-scored — repeatedly touching a held-out set is how it stops being one.
* It does not retrain anything, and `--save` refuses to overwrite an existing
  baseline without `--force`. A baseline that can be silently regenerated after a
  change is not a baseline; it is a description of the change agreeing with
  itself.
