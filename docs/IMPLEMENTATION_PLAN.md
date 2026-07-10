# Implementation Plan

Sequenced exactly per the master prompt; ML truth before product, product before polish. Status is updated as phases complete.

| Phase | Maps to | Deliverables | Exit criteria |
|---|---|---|---|
| 0. Repo & environment audit | Week 1 | `CURRENT_STATE_AUDIT.md`, `environment_snapshot.json`, initial commit | Environment + commit SHA recorded; existing tests (none) recorded |
| 1. Immutable data handling | Week 1 | read-only raw copy + SHA-256, one-time XLSX→Parquet, `schema.json`, `data_fingerprint.json`, independent-engine validation | Row/col counts, header equality, target distribution, spot checks all pass; raw untouched |
| 2. Data audit + leakage firewall | Week 1 | per-feature profile JSON/CSV, `DATA_AUDIT_REPORT.md`, `LEAKAGE_AUDIT.md`, `configs/leakage_quarantine.yaml`, F3912 ablation evidence | Every feature has dtype/missing/constant/duplicate/suspicion status; quarantine list justified with measured evidence |
| 3. Immutable splits | Week 1 | `locked_test_indices.parquet`, `cv_folds.parquet` (5×5), duplicate-group scan | Splits reproducible from seed; ≥12 positives in locked test; no duplicate-group straddling |
| 4. Leakage-free baselines | Week 1 | dummy / logistic / LightGBM OOF metrics, recall@budget, bootstrap CIs, first honest report | Baseline reruns identically; metrics trace to saved prediction files |
| 5. Feature selection + tournament | Week 2 | in-fold filtering, stability selection, compact sets (15/30/60), tuned LGBM/XGB/CatBoost, TabPFN/TabICL feasibility, OOF store, ensemble decision | Winner chosen on repeated OOF evidence; compact set documented |
| 6. Trinetra lenses | Week 3 | calibration (Platt vs isotonic), Mondrian conformal abstention, hard-negative verifier, IsolationForest + OOD, deterministic policy engine | Uncertain/OOD route to review; thresholds from dev data only |
| 7. Locked test (single touch) + explainability | Week 3/4 | final dev-trained models score locked test once; SHAP reason codes, cohort percentiles, counterfactual sensitivity, evidence packets, all required plots | Locked-test metrics with CIs; every plot states split + metric |
| 8. API + audit trail + guarded LLM | Week 3 | FastAPI endpoints, SQLite append-only audit, model registry, Ollama client + hallucination validator + deterministic fallback | Scoring identical with Ollama up/down; LLM cannot alter scores |
| 9. Test suites | Week 3/4 | unit/model/integration/security tests | All green; leakage guards enforced by tests |
| 10. Dashboard | Week 4 | React/TS 7-page analyst dashboard reading only real artifacts | No blank/overlapping screens; loading/empty/error states |
| 11. Hardening + release | Week 4 | robustness suite, drift module, Docker, 21 docs, model card, judge Q&A, demo script, release gate verdict | Release gate PASS or documented FAIL with open risks |

## Timeboxing note (Mode A, CPU-only)

Optuna: 40 (LGBM) / 30 (XGB) / 25 (CatBoost) trials, single 5-fold during scouting, 5×5 repeated CV for finalists only. AutoGluon: skipped — no Python 3.13 support at build time (documented in tournament report). TabPFN/TabICL: attempted only behind an explicit memory/runtime guard with graceful skip.
