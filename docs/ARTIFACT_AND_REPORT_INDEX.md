# Artifact and report index

Sections 57 and 58 fix a list of file names. This project had already written most of that evidence under its own names. Rather than rename the evidence - which would break every document that cites it - each required name is produced from the real source and says so in its own `__provenance__` block.

Regenerate with `make reconcile-artifacts`. Nothing here is written by hand, so the table cannot claim a file that is not there.

## Section 57 - artifacts

| required path | status | source | note |
|---|---|---|---|
| `artifacts/testing/environment.json` | DERIVED | `artifacts/environment_snapshot.json` |  |
| `artifacts/testing/data_integrity.json` | DERIVED | `artifacts/testing/data_integrity_results.json` |  |
| `artifacts/testing/description_integrity.json` | DERIVED | `artifacts/features/feature_dictionary.json` |  |
| `artifacts/testing/leakage_results.json` | DERIVED | `artifacts/testing/leakage_test_results.json` |  |
| `artifacts/testing/nested_cv_results.json` | DERIVED | `artifacts/metrics/nested_cv.json` |  |
| `artifacts/testing/positive_removal_results.json` | DERIVED | `artifacts/metrics/nested_positive_removal.json`, `artifacts/metrics/stability_stress_v2.json` | **FLAT_FALLBACK** - the nested arm has not run; flat-CV stand-in, not comparable to nested numbers |
| `artifacts/testing/adversarial_validation.json` | PENDING | `artifacts/metrics/nested_shift_shield.json` | waiting on `muleguard.cli.nested_ses --stages shift` |
| `artifacts/testing/calibration_results.json` | DERIVED | `artifacts/metrics/lens_stack_oof_v2.json` |  |
| `artifacts/testing/threshold_results.json` | DERIVED | `artifacts/metrics/lens_stack_oof_v2.json` |  |
| `artifacts/testing/backend_results.json` | DERIVED | `artifacts/testing/backend_test_results.json` |  |
| `artifacts/testing/ui_metric_consistency.json` | DERIVED | `artifacts/testing/api_frontend_consistency.json` |  |
| `artifacts/testing/offline_results.json` | PENDING | - | waiting on `bash scripts/test_offline.sh` |
| `artifacts/testing/security_results.json` | PRESENT | - | written directly by the security suite |
| `artifacts/metrics/model_comparison.csv` | PRESENT | - | generation-1 tournament table; model_comparison_v2.csv is the post-firewall one and the ledger is the full record |
| `artifacts/metrics/final_nested_cv.json` | DERIVED | `artifacts/metrics/nested_cv.json` |  |
| `artifacts/metrics/final_budget_metrics.csv` | DERIVED | `artifacts/metrics/capacity_curve.json` |  |
| `artifacts/metrics/final_calibration.json` | DERIVED | `artifacts/metrics/lens_stack_oof_v2.json` |  |
| `artifacts/metrics/feature_subset_ablation.csv` | DERIVED | `artifacts/metrics/nested_feature_family_arms.json`, `artifacts/metrics/family_dropout_v2.json` | **FLAT_FALLBACK** - the nested arm has not run; flat-CV stand-in, not comparable to nested numbers |
| `artifacts/metrics/alert_context_ablation.csv` | DERIVED | `artifacts/metrics/alert_context_ablation_v2.json` |  |
| `artifacts/metrics/leakage_ablation.csv` | PRESENT | - | written by the leakage ablation run |
| `artifacts/predictions/all_outer_oof_predictions.parquet` | DERIVED | `artifacts/predictions/nested_oof.parquet` |  |
| `artifacts/predictions/final_model_oof_predictions.parquet` | DERIVED | `artifacts/predictions/oof_v2.parquet` | champion dev OOF; re-copied whenever the champion changes |
| `artifacts/features/feature_dictionary.json` | PRESENT | - |  |
| `artifacts/features/selected_features.json` | PRESENT | - |  |
| `artifacts/features/selection_frequency.csv` | PRESENT | - |  |
| `artifacts/features/quarantined_features.json` | PRESENT | - |  |

`PRESENT` - the pipeline already writes this exact path. `DERIVED` - regenerated here from the named source. `PENDING` - the run that produces it has not finished. `MISSING` - neither the file nor its source exists.

## Section 58 - reports

| required name | authoritative file | covers |
|---|---|---|
| `docs/FINAL_XHIGH_VALIDATION_REPORT.md` | _not written yet_ | written last, from the completed nested evidence |
| `docs/DESCRIPTION_VALIDATION_REPORT.md` | `docs/FEATURE_DICTIONARY_REPORT.md` (pointer written) | Description.xlsx parsing, coverage and the availability ruling per field |
| `docs/FEATURE_AVAILABILITY_FIREWALL.md` | `docs/FEATURE_AVAILABILITY_AUDIT.md` (pointer written) | the firewall itself: what was quarantined and on what evidence |
| `docs/FINAL_LEAKAGE_FORENSICS.md` | `docs/FINAL_DATA_AND_LEAKAGE_AUDIT.md` (pointer written) | post-outcome field forensics, including the F3912 probe |
| `docs/HISTORICAL_METRIC_RECONCILIATION.md` | itself | already under the spec name |
| `docs/NESTED_CV_MODEL_TOURNAMENT.md` | `docs/MODEL_TOURNAMENT_REPORT.md` (pointer written) | the tournament; the nested results section is filled by the nested run |
| `docs/HIDDEN_VALIDATION_READINESS.md` | `docs/HIDDEN_VALIDATION_STRATEGY.md` (pointer written) | what happens when the organiser's file arrives |
| `docs/POSITIVE_REMOVAL_STABILITY.md` | `docs/ROBUSTNESS_REPORT.md` (pointer written) | positive-removal rounds and the grade they feed |
| `docs/ADVERSARIAL_VALIDATION_REPORT.md` | `docs/NESTED_STABILITY_ENSEMBLE_SHIFT.md` (pointer written) | section 23 of the pre-registered nested programme |
| `docs/CALIBRATION_AND_THRESHOLDS.md` | `docs/FINAL_CALIBRATION_AND_THRESHOLD_REPORT.md` (pointer written) | calibration choice, threshold freeze and the policy version |
| `docs/FALSE_POSITIVE_VALIDATION.md` | `docs/FALSE_POSITIVE_CONTROL_REPORT.md` (pointer written) | false-positive control, including the merchant verifier |
| `docs/VALIDATION_LAB_TEST_REPORT.md` | `docs/VALIDATION_LAB_REPORT.md` (pointer written) | the judge-facing validation lab |
| `docs/SEALED_VALIDATION_PROTOCOL.md` | `docs/LOCKED_TEST_RULING.md` (pointer written) | the seal, the single touch, and why the locked test is reference only |
| `docs/UI_METRIC_CONSISTENCY.md` | `docs/FINAL_FRONTEND_UI_REPORT.md` (pointer written) | every number shown in the UI traced to the artifact it came from |
| `docs/OFFLINE_RUNTIME_TEST.md` | `docs/NO_MCP_NO_BROWSER_AGENT_COMPLIANCE.md` (pointer written) | offline behaviour with the narrator disabled and no network |
| `docs/SECURITY_TEST_REPORT.md` | `docs/FINAL_SECURITY_REPORT.md` (pointer written) | security suite results |
| `docs/FINAL_MODEL_CARD.md` | `docs/MODEL_CARD.md` (pointer written) | the model card |

A pointer file carries no numbers of its own. It names the report that does, so a reader looking for the required name finds the real thing one hop away rather than a second copy that can go stale.
