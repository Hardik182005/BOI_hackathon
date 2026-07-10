# Final Release Test Report

Generated 2026-07-10T23:08:51.019460+00:00 · commit `c8e0d253acf8` · **Verdict: PASS**

## Environment

| Item | Value |
|---|---|
| OS | Windows-11-10.0.22631-SP0 |
| Python / Node | 3.13.2 / v24.16.0 |
| Compute mode | cpu_16gb (16.93 GB RAM, CUDA=False) |
| Dataset SHA-256 | `7d1be90fe23b57460184f2f7566d572c…` |
| Model bundle SHA-256 | `04fafaee25ae82c7a5a2e6ec5757d77e…` |

## Dataset (verified)

9,082 rows × 3,925 cols · 81 positives (0.8919%) ·
quarantined: F3924 (target), F3912 (leak), F2230 (month≡label), __UNNAMED__0 (index) ·
60 selected features in production.

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
| security_results | PASS | 7/7 |
| frontend_test_results | PASS | 7/7 |
| batch_upload_results | PASS | 4/4 |

Plus: backend pytest **93 passed**, frontend vitest **3 passed**, one-command
startup log `artifacts/testing/one_command_startup.log`.

## ML release gate

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

## Defects

- P0: none open
- P1: none open
- P2 (approved, non-blocking): TabPFN/TabICL/AutoGluon documented skips on
  this hardware/Python; frontend has no dedicated batch-upload page (API
  endpoint + CSV download implemented; UI page is a roadmap item).

## Verdict: **PASS**
All release blockers clear.
