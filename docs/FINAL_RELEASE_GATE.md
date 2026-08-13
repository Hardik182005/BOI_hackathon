# Final Release Gate

Generated 2026-08-13T17:55:46.609002+00:00 · commit `f280c6fc576b`

## Verdict: **FAIL** (22/23 checks passed)

| Check | Result | Detail |
|---|---|---|
| no_target_or_f3912_leakage | PASS | 120 bundle features disjoint from 4 quarantined |
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
| tests_pass | **FAIL** | 1 failed, 407 passed in 256.77s (0:04:16) |
| probabilities_bounded | PASS | OOF + locked test bounded |
| artifacts_complete | PASS | 19 artifacts present |
| addendum_artifacts_complete | PASS | 10 addendum artifacts present |
| robustness_grade_not_hand_picked | PASS | grade=LOW limited_by=['prediction_rank_stability'] |
| label_audit_changed_nothing | PASS | 1 rows flagged, all HUMAN_REVIEW_ONLY |
| merchant_verifier_cannot_lower_risk | PASS | risk held at 0.91/INVESTIGATE; confidence -> MEDIUM; 1 adjustments all carry their trigger values |
| graph_never_fabricates_edges | PASS | default UNAVAILABLE; contract forbids derived edges |
| shield_reports_no_leaked_feature | PASS | 54 STABLE / 66 WATCH / 0 SHIFT_PRONE / 0 LEAKAGE |
| no_forbidden_verdict_vocabulary | PASS | 5 forbidden verdict words absent from shipped source |
| attack_surface_covered | PASS | 49 security tests collected covering sqli, xss, path_traversal, csv_injection |
| organiser_dry_run_passed | PASS | 8/8 variants, invariance sound=True, model unchanged=True, locked-test PR-AUC 0.7262714933700882 |

## Blockers

- tests_pass: 1 failed, 407 passed in 256.77s (0:04:16)