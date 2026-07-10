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
