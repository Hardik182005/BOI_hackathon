# Final Release Gate

Generated 2026-07-10T23:08:26.650937+00:00 · commit `c8e0d253acf8`

## Verdict: **PASS** (14/14 checks passed)

| Check | Result | Detail |
|---|---|---|
| no_target_or_f3912_leakage | PASS | 60 bundle features disjoint from 4 quarantined |
| no_split_overlap | PASS | test=1818 dev=7264 overlap=0 |
| locked_test_single_touch | PASS | touches=3 forced=0 |
| metrics_trace_to_predictions | PASS | 11 models verified |
| model_load_and_deterministic | PASS | 10-row rescore identical |
| llm_cannot_alter_score | PASS | rejected with 2 reasons |
| no_auto_freeze_paths | PASS | policy emits no auto actions; freeze needs approver in API |
| ood_routes_to_review | PASS | tier=OOD_REVIEW status=OUT_OF_DISTRIBUTION |
| scoring_survives_ollama_outage | PASS | fallback ok; planted hallucination rejected for 8 reasons |
| raw_data_unmodified | PASS | SHA-256 verified |
| no_secrets_committed | PASS | .env not tracked |
| tests_pass | PASS | -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html |
| probabilities_bounded | PASS | OOF + locked test bounded |
| artifacts_complete | PASS | 19 artifacts present |