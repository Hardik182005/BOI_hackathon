# Final Release Test Report

Generated 2026-08-13T17:55:48.846717+00:00 · commit `f280c6fc576b` · **Verdict: FAIL**

## Environment

| Item | Value |
|---|---|
| OS | Windows-11-10.0.22631-SP0 |
| Python / Node | 3.13.2 / v24.16.0 |
| Compute mode | cpu_16gb (16.93 GB RAM, CUDA=False) |
| Dataset SHA-256 | `7d1be90fe23b57460184f2f7566d572c…` |
| Model bundle SHA-256 | `d12914de5abee99aa18b80e3d41ae2da…` |

## Dataset (verified)

9,082 rows × 3,925 cols · 81 positives (0.8919%) ·
quarantined: F3924 (target), F3912 (leak), F2230 (month≡label), __UNNAMED__0 (index) ·
120 selected features in production.

## Model

| Item | Value |
|---|---|
| Best single (5-repeat OOF) | **catboost_tuned_top60** — PR-AUC 0.8077 ± 0.0450 |
| Locked test (production scorer) | PR-AUC 0.8242 (95% CI 0.6536–0.9584) |
| Calibration | Brier 0.00258, ECE 0.0027 |
| Ensemble | rejected by pre-registered ≥4/5-repeats rule |
| Challengers | tabpfn:RAN, tabicl:SKIPPED, autogluon:SKIPPED (TabPFN 1-repeat OOF 0.8969585278109471) |
| Throughput | 687.7 rows/s CPU |

### Recall / precision at analyst budgets (locked test)

| Budget | Recall | Precision |
|---|---|---|
| top 18 | 76.5% | 72.2% |
| top 25 | 76.5% | 52.0% |
| top 50 | 82.4% | 28.0% |
| top 100 | 100.0% | 17.0% |

## QA suites (live evidence in artifacts/testing/)

| Suite | Status | Checks |
|---|---|---|
| backend_test_results | PASS | 15/15 |
| data_integrity_results | PASS | 11/11 |
| leakage_test_results | PASS | 6/6 |
| ollama_guardrail_results | PASS | 16/16 |
| performance_results | PASS | 7/7 |
| e2e_results | PASS | 10/10 |
| api_frontend_consistency | PASS | 6/6 |
| security_results | PASS | 8/8 |
| frontend_test_results | PASS | 7/7 |
| batch_upload_results | PASS | 4/4 |

Plus: backend pytest **93 passed**, frontend vitest **3 passed**, one-command
startup log `artifacts/testing/one_command_startup.log`.

## ML release gate

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
| merchant_verifier_cannot_lower_risk | PASS | risk held at 0.91/INVESTIGATE; confidence -> MEDIUM; 1 adjustments all carry their trigger |
| graph_never_fabricates_edges | PASS | default UNAVAILABLE; contract forbids derived edges |
| shield_reports_no_leaked_feature | PASS | 54 STABLE / 66 WATCH / 0 SHIFT_PRONE / 0 LEAKAGE |
| no_forbidden_verdict_vocabulary | PASS | 5 forbidden verdict words absent from shipped source |
| attack_surface_covered | PASS | 49 security tests collected covering sqli, xss, path_traversal, csv_injection |
| organiser_dry_run_passed | PASS | 8/8 variants, invariance sound=True, model unchanged=True, locked-test PR-AUC 0.7262714933 |

## Defects

- P0: none open
- P1: none open
- P2 (approved, non-blocking): TabPFN/TabICL/AutoGluon documented skips on
  this hardware/Python; frontend has no dedicated batch-upload page (API
  endpoint + CSV download implemented; UI page is a roadmap item).

## Verdict: **FAIL**
Blockers: ML release gate: FAIL
