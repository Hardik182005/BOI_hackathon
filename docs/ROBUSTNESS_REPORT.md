# Robustness Report

Executable suite: `tests/e2e/test_robustness.py` (runs against the frozen
bundle on dev rows only; pass/fail status is part of the release gate's
`tests_pass` check).

## Perturbations exercised and required behaviour

| Perturbation | Required behaviour | Test |
|---|---|---|
| Duplicate identical requests | bit-identical scores and tiers (determinism) | `test_duplicate_requests_identical` |
| Column order shuffled | identical scores (schema is name-based, not positional) | `test_column_order_invariance` |
| Extreme values (1e9) on 15 selected features | routed to `OOD_REVIEW`, never a confident normal label | `test_extreme_values_route_to_ood` |
| Half of selected features nulled | no crash; scores stay in [0,1]; tier remains a known value (missingness feeds the OOD/missingness signal) | `test_added_missingness_does_not_crash...` |
| ±1% noise on top features | median calibrated-risk shift < 0.15 (local stability near thresholds) | `test_small_perturbations_keep_scores_stable` |
| Omitted selected feature | hard `422 SCHEMA_ERROR` — silent zero-fill is forbidden | `test_missing_selected_feature_raises_schema_error` (model suite) + API test |
| Target column smuggled into request | rejected with 422 | API integration test |
| Corrupted/oversized uploads | Pydantic size limits + batch cap 500 | API schemas |
| Malformed LLM output (planted) | validator rejects; deterministic fallback | `test_llm_guardrails` (13 cases) + demo scene |
| Ollama down | scoring unchanged; reports fall back | integration tests run without Ollama |
| Latency | < 1 s/row single-threaded CPU bound asserted; measured throughput in locked-test metrics | `test_scoring_latency_batch` |

## Rare-positive stress test (measured)

Source: `artifacts/metrics/stability_stress_v2.json` — model
`xgboost_top_120`, 3 rounds, 12.5% of *training* positives removed per round
(mean 30 removed). The evaluation fold is identical in every round, so all
spread is attributable to label scarcity and nothing else.

| Measure | Value |
|---|---:|
| Reference PR-AUC | 0.74927 |
| PR-AUC after positive removal | 0.7383 ± 0.0190 (min 0.71217) |
| Relative drop | 1.46% |
| Recall@25 / @50 / @100 | 0.3906 / 0.6458 / 0.7500 |
| Recall std @25 / @50 / @100 | 0.0000 / 0.0147 / 0.0128 |
| Feature-importance rank stability (Spearman) | 0.7654 |
| Top-20 feature overlap between rounds | 0.70 |
| All-row prediction rank stability (Spearman) | 0.1656 |
| Top-K set overlap (Jaccard) @25 / @50 / @100 | 0.57 / 0.6588 / 0.43 |

Two honest readings, both recorded in the artifact:

- The all-row Spearman of 0.1656 is reported **unchanged** because it is the
  figure the published robustness thresholds were defined on. Its caveat: about
  99% of those rows are negatives whose calibrated scores sit in a narrow band
  near zero, so their relative order moves freely between rounds without any
  account changing review status. A low all-row figure is weak evidence on its
  own.
- The top-budget Jaccard overlaps measure the part of the ranking an analyst
  actually works through. The artifact states explicitly that they are a
  diagnostic reported **alongside** the badge, not an input to it: the badge
  thresholds were fixed before these experiments ran and have not been
  redefined to suit the result.

The feature-rank figures answer a separate question — whether the model keeps
citing the same evidence when the mules it learned from change. A detector
whose stated reasons rewrite themselves under label churn cannot be explained
to an analyst; at Spearman 0.7654 and 0.70 top-20 overlap, this one mostly does
not.

## Business look-alike scenario

`muleguard.cli.demo` selects a real legitimate dev account with the highest
screener score where the hard-negative verifier disagrees — the policy caps
fast-tracking and routes to review. Stored with full evidence in
`artifacts/evidence/demo_scenarios.json`.

## Honest boundaries

- Adversarial *training-time* poisoning is out of scope (the bank controls
  the feature pipeline); inference-time gaming is mitigated by hidden
  thresholds, relational features, anomaly/OOD challengers and drift alarms.
- No new labelled data was invented for robustness testing; all perturbations
  are synthetic transformations of real dev rows and are labelled as such.
