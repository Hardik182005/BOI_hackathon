# MuleGuard · Trinetra — Final Full Testing, Accuracy, UI, Backend and One-Command Release Prompt

You are now acting as a Principal QA Engineer, ML Validation Lead, Backend Reliability Engineer, Frontend Test Engineer, MLOps Engineer, Security Reviewer, Responsible-AI Reviewer, and Release Manager for the MuleGuard · Trinetra project.

Your job is to perform a complete, strict, evidence-based validation of the entire project and fix every issue found before declaring it ready.

This is not a planning-only task. Inspect the repository, run the system, execute tests, fix defects, rerun tests, generate evidence, and produce a final PASS/FAIL release report.

Do not use MCP.
Do not use Claude in Chrome.
Do not use browser-control agents.
Do not use external browser automation services.
Do not require any Claude extension.
Do not require any LLM to start or run the main application.
Do not require Ollama for risk scoring.
Do not use an LLM inside the training, scoring, calibration, thresholding, or final decision pipeline.

The entire core system must work locally from the terminal and backend services, with one simple command.

---

# 1. Authoritative product requirements

The system is MuleGuard, powered by the Trinetra detection engine, for suspicious mule-account classification.

The official tabular target is:

- Input features: F1 through F3923
- Target: F3924
- 1 = suspicious/mule-labelled account
- 0 = legitimate/not currently labelled mule

Critical project rules:

- F3924 must never enter the feature matrix.
- F3912 is a suspected leakage feature and must remain quarantined from the accepted model.
- Index or unnamed ID columns must not enter training.
- The model must optimise for rare-event detection.
- Accuracy is not the primary metric.
- PR-AUC / Average Precision is the primary metric.
- Every metric must come from actual saved predictions.
- No fake dashboard numbers.
- No hardcoded model metrics.
- No test-set tuning.
- No automatic account freezing.
- No claim of criminal guilt.
- Uncertain and out-of-distribution cases must go to human review.
- The system must continue working when Ollama is stopped.
- The frontend must be a clean white-background, black-text interface.

---

# 2. Immediate first actions

Before changing code:

1. Print:
   - current git branch
   - git commit SHA
   - operating system
   - Python version
   - Node version
   - package manager versions
   - available RAM
   - CPU
   - GPU/CUDA availability
   - free disk space

2. Inspect:
   - repository structure
   - backend
   - frontend
   - training scripts
   - model artifacts
   - dataset paths
   - configuration files
   - Docker files
   - shell scripts
   - existing tests
   - logs
   - current README instructions

3. Run all currently available tests before making fixes.

4. Save the initial state in:
   - `docs/FINAL_TEST_INITIAL_AUDIT.md`
   - `artifacts/testing/initial_test_results.json`

5. Do not delete working code.
6. Do not rewrite large modules unless necessary.
7. Fix root causes, not only symptoms.

---

# 3. No-MCP and no-Claude-in-Chrome validation

Search the complete repository for:

- MCP configuration
- MCP servers
- Claude browser integrations
- Claude-in-Chrome instructions
- Playwright dependencies used only for Claude browser control
- Puppeteer dependencies used only for agent browsing
- browser-agent startup requirements
- external AI-agent services

Create:

- `artifacts/testing/no_mcp_scan.txt`
- `docs/NO_MCP_NO_BROWSER_AGENT_COMPLIANCE.md`

Requirements:

- The application must not depend on MCP.
- The application must not depend on Claude in Chrome.
- The application must not depend on an agent controlling Chrome.
- Normal frontend E2E testing may use local Playwright only if it is installed inside the repository and invoked from terminal commands; it must not require Claude browser tooling.
- If Playwright is unnecessary, prefer component and API tests without it.
- Remove or disable any accidental MCP startup dependency.
- Remove any documentation that says users must install Claude in Chrome.

Release blocker:

- FAIL if the product cannot run without MCP or Claude browser tools.

---

# 4. One-command startup requirement

The entire local project must start using one command.

Preferred command:

```bash
./run.sh
```

or, if the repository already uses Make:

```bash
make run
```

Choose one canonical command and document it clearly.

The command must:

1. Check required local dependencies.
2. Create missing local directories.
3. Validate environment variables.
4. Use safe defaults for local demo.
5. Start the backend.
6. Start the frontend.
7. Load the currently approved model artifact.
8. Run database migrations if required.
9. Wait for health checks.
10. Print exact local URLs.
11. Print model version.
12. Print whether Ollama is available or disabled.
13. Keep logs readable.
14. Stop all child processes cleanly on Ctrl+C.
15. Return a non-zero exit code if startup fails.

The one-command start must not:

- retrain the model every time
- require MCP
- require Claude in Chrome
- require an external LLM API
- require manual opening of multiple terminals
- silently swallow backend failures
- show a ready message before health checks pass

Required URLs:

- Frontend: `http://localhost:<frontend-port>`
- Backend: `http://localhost:<backend-port>`
- API docs: `http://localhost:<backend-port>/docs`
- Health: `http://localhost:<backend-port>/health/ready`

Create:

- `run.sh` or canonical equivalent
- `scripts/stop.sh` if useful
- `docs/ONE_COMMAND_RUN_GUIDE.md`
- `artifacts/testing/one_command_startup.log`

Test from a clean shell.

Release blocker:

- FAIL if more than one manual command is required for normal local startup.

---

# 5. Backend independence and reliability

The backend is the source of truth.

The backend must run and score accounts even when:

- frontend is stopped
- Ollama is stopped
- internet is disconnected
- browser is closed
- no external API key exists

Test:

1. Start backend alone.
2. Call health endpoint.
3. Call model metadata endpoint.
4. Score one valid row.
5. Score a batch.
6. Request case details.
7. Generate deterministic explanation without Ollama.
8. Stop Ollama and repeat.
9. Disconnect frontend and repeat.
10. Simulate invalid input and verify safe error handling.

Required endpoints should include or be equivalent to:

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/model`
- `POST /v1/score`
- `POST /v1/score/batch`
- `GET /v1/cases`
- `GET /v1/cases/{case_id}`
- `POST /v1/cases/{case_id}/decision`
- `POST /v1/cases/{case_id}/feedback`
- `GET /v1/metrics/summary`
- `GET /v1/drift/status`

Backend requirements:

- deterministic predictions for identical inputs
- model loaded once at startup
- clear schema validation
- request IDs
- structured logs
- safe timeouts
- no secrets in logs
- graceful model-load failure
- correct HTTP status codes
- no test labels required at inference
- no silent zero-filling of missing required selected features
- OOD or schema-error routing for invalid feature sets
- no LLM dependency for scoring

Create:

- `artifacts/testing/backend_test_results.json`
- `docs/BACKEND_RELIABILITY_REPORT.md`

---

# 6. Dataset integrity and leakage testing

Locate the official dataset.

Verify and record:

- file hash
- file size
- rows
- columns
- target distribution
- missing target count
- duplicate rows
- duplicate columns
- constant columns
- quasi-constant columns
- categorical columns
- numeric columns
- date-like columns
- index-like columns
- selected features
- quarantined features

Mandatory assertions:

- F3924 is target only.
- F3924 is absent from every model input.
- F3912 is absent from every accepted model input.
- unnamed/index columns are absent.
- locked test rows do not overlap development rows.
- validation folds do not overlap.
- preprocessing is fitted only on training folds.
- calibration is fitted only on development data.
- thresholds are selected only on development data.
- final locked test is evaluated only once in the release pipeline.

Create tests that fail automatically if any quarantined feature enters training.

Create:

- `artifacts/testing/data_integrity_results.json`
- `artifacts/testing/leakage_test_results.json`
- `docs/FINAL_DATA_AND_LEAKAGE_AUDIT.md`

Release blocker:

- FAIL if leakage is detected.
- FAIL if F3912 enters the accepted model.
- FAIL if F3924 enters the feature matrix.
- FAIL if any test-set tuning is found.

---

# 7. Accuracy and model-quality testing

The goal is not to claim “best accuracy.” The goal is to find and verify the strongest leakage-free model supported by the dataset.

Run a fair model tournament using identical splits and metrics.

Required candidates:

- Dummy prevalence baseline
- Class-weighted logistic regression
- LightGBM
- CatBoost
- XGBoost

Advanced challengers, only if supported safely:

- TabICLv2 on selected features
- TabPFN on selected features
- AutoGluon benchmark
- Isolation Forest as anomaly challenger, not primary classifier
- Hard-negative verifier
- Calibrated stacked ensemble

Do not use an Ollama or Hugging Face text LLM as the primary F3924 classifier.

Evaluation protocol:

- locked stratified test split
- repeated stratified cross-validation on development data
- all preprocessing inside folds
- feature selection inside folds
- class weights
- no default SMOTE
- SMOTE only as a controlled ablation inside training folds
- save all out-of-fold predictions
- save all locked-test predictions

Primary metric:

- PR-AUC / Average Precision

Also calculate:

- recall
- precision
- F1
- ROC-AUC as secondary only
- recall at top 25
- recall at top 50
- recall at top 100
- precision at top 25
- precision at top 50
- precision at top 100
- recall at top 1%
- false positives per 1,000 legitimate accounts
- precision in highest-risk tier
- Brier score
- Expected Calibration Error
- calibration curve
- confusion matrix for every operational threshold
- 95% bootstrap confidence intervals
- inference latency
- model size
- peak memory
- training time

Selection rules:

1. Do not choose a model from one lucky split.
2. Compare repeated-fold mean, standard deviation, and confidence intervals.
3. Prefer a slightly lower PR-AUC model if it is materially better calibrated, more stable, more explainable, and operationally safer.
4. Accept an ensemble only if it consistently improves over the best single model.
5. Reject any model whose performance depends on F3912.
6. Never claim state-of-the-art without evidence.
7. Never say “100% accurate.”
8. Report the exact best achieved leakage-free result.

Create:

- `artifacts/metrics/final_model_comparison.csv`
- `artifacts/metrics/final_oof_metrics.json`
- `artifacts/metrics/final_locked_test_metrics.json`
- `artifacts/predictions/final_oof_predictions.parquet`
- `artifacts/predictions/final_locked_test_predictions.parquet`
- `artifacts/plots/final_precision_recall_curve.png`
- `artifacts/plots/final_calibration_curve.png`
- `artifacts/plots/final_recall_at_budget.png`
- `artifacts/plots/final_confusion_matrices.png`
- `artifacts/plots/final_feature_stability.png`
- `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md`

---

# 8. Feature-selection testing

Test:

- constant-feature removal
- quasi-constant removal
- duplicate-column detection
- correlation clustering
- stable feature selection
- top 15 features
- top 30 features
- top 60 features
- top 100 features
- full cleaned matrix

For each subset report:

- PR-AUC
- recall at alert budget
- calibration
- variance across folds
- inference latency
- explanation stability

Verify:

- feature selection is fitted only within training folds
- selected-feature list is versioned
- selected features are present at inference
- anonymous features are not assigned invented meanings
- importance ranks are reasonably stable

Create:

- `artifacts/features/final_selected_features.json`
- `artifacts/features/final_selection_frequency.csv`
- `docs/FINAL_FEATURE_SELECTION_REPORT.md`

---

# 9. Calibration, uncertainty and decision-tier testing

Test:

- Platt/sigmoid calibration
- isotonic calibration
- selected calibration method
- conformal or abstention layer
- model disagreement
- OOD detection
- anomaly disagreement

Required decision outcomes:

- `CRITICAL_REVIEW`
- `URGENT_REVIEW`
- `STANDARD_REVIEW`
- `OOD_REVIEW`
- `MONITOR`

Never return:

- `GUILTY`
- `CRIMINAL`
- `CERTIFIED_CLEAN`
- `PERMANENTLY_SAFE`
- automatic `FREEZE`

Thresholds must be derived from development data.

For each tier report:

- number of accounts
- number of true positives
- number of false positives
- precision
- recall contribution
- coverage
- uncertainty
- recommended human action

Test edge cases:

- highly suspicious account
- legitimate business-like high-volume account
- low-risk account
- model disagreement
- OOD input
- missing selected features
- unseen category
- extreme valid values

Create:

- `artifacts/metrics/final_threshold_table.csv`
- `docs/FINAL_CALIBRATION_AND_THRESHOLD_REPORT.md`

---

# 10. Hallucination and Ollama guardrail testing

Ollama is optional and may only generate a readable explanation from already verified structured results.

The core score and risk tier must be created before any Ollama call.

Test with Ollama:

1. Available and valid.
2. Unavailable.
3. Timeout.
4. Invalid JSON.
5. Changed risk score.
6. Changed risk tier.
7. Invented feature.
8. Invented transaction amount.
9. Invented date or customer fact.
10. Criminal-guilt claim.
11. Unsupported bank action.
12. Missing required limitation.
13. Excessively long output.
14. Prompt-injection text inside a feature value.
15. Repeated retries or service instability.

The validator must reject output if:

- score differs from backend score
- tier differs
- feature not in allowed input appears
- unsupported amount/date/person appears
- guilt is claimed
- unsupported action appears
- JSON schema fails
- required disclaimer is missing

Required fallback:

- deterministic template
- same score
- same tier
- same reason list
- clear limitations
- no LLM dependency

Test the system with Ollama completely stopped.

Create:

- `artifacts/testing/ollama_guardrail_results.json`
- `docs/FINAL_HALLUCINATION_GUARDRAIL_REPORT.md`

Release blocker:

- FAIL if Ollama can alter score, tier, action, or evidence.
- FAIL if the system breaks when Ollama is unavailable.

---

# 11. Frontend visual requirements

The frontend must use:

- white background
- black primary text
- neutral grey borders
- minimal accent colours
- no dark theme
- no gradients
- no excessive shadows
- no neon colours
- no decorative animations that reduce clarity
- no crowded dashboard
- no overlapping elements
- no unreadable charts
- no fake statistics

Recommended visual system:

- Page background: white
- Cards: white
- Main text: near-black
- Secondary text: dark grey
- Borders: light grey
- Critical: restrained red
- Urgent/high: amber
- Positive/healthy status: restrained green
- Information: restrained blue
- Risk colours must never override text readability

Use black text on white backgrounds everywhere except small status chips where accessibility contrast is verified.

Do not use green for negative outcomes.
Do not use red for normal/healthy outcomes.

---

# 12. Frontend functional testing

Test every screen:

1. Executive Overview
2. Alert Queue
3. Case Detail
4. Model Performance
5. Feature Intelligence
6. Drift and Monitoring
7. Model Card
8. Settings/configuration if present

Validate:

- no blank screen after login/start
- no console errors
- no unhandled promise rejection
- no broken API calls
- no overlap at common resolutions
- no clipped tables
- no off-screen buttons
- no inaccessible modal
- no invisible text
- charts resize correctly
- tooltips readable
- loading state
- empty state
- error state
- backend unavailable state
- Ollama unavailable state
- batch upload progress
- report download
- analyst decision workflow
- filter and sort
- pagination
- responsive desktop layout

Test at:

- 1366×768
- 1440×900
- 1920×1080
- 1280×720

Use terminal-invoked local frontend tests.
Do not use Claude in Chrome.

Create:

- `artifacts/testing/frontend_test_results.json`
- `docs/FINAL_FRONTEND_UI_REPORT.md`

Release blocker:

- FAIL if any main page is blank.
- FAIL if major charts overlap.
- FAIL if model metrics shown in UI do not match backend artifacts.
- FAIL if the UI labels an account holder as criminal.

---

# 13. API-to-frontend consistency testing

For sampled cases:

1. Call backend directly.
2. Save response.
3. Render frontend through the local test runner.
4. Compare:
   - account reference
   - risk score
   - tier
   - model version
   - reasons
   - uncertainty
   - OOD status
   - limitations

The frontend must not:

- round in a misleading way
- change the tier
- invent a reason
- invent a metric
- display stale cached output after model change

Create:

- `artifacts/testing/api_frontend_consistency.json`

Release blocker:

- FAIL if frontend and backend disagree.

---

# 14. Batch-upload testing

Test official-format CSV/XLSX batch scoring.

Cases:

- valid full file
- valid small file
- missing target column for inference
- target column present but ignored safely
- missing required selected feature
- duplicate column
- wrong datatype
- corrupted file
- oversized file
- formula-injection content
- extra unknown columns
- reordered columns

Requirements:

- safe parsing
- clear errors
- no application crash
- no target leakage
- output downloadable as CSV
- output contains model version
- output contains risk score, tier, uncertainty, and review status
- output does not expose hidden threshold logic
- no formulas executed

Create:

- `artifacts/testing/batch_upload_results.json`

---

# 15. Performance and load testing

Measure:

- cold backend startup time
- model load time
- single-row latency
- p50 latency
- p95 latency
- p99 latency
- batch throughput
- peak RAM
- CPU utilisation
- frontend first render
- API error rate under load

Test at safe local levels:

- 1 concurrent request
- 5 concurrent requests
- 10 concurrent requests
- 25 concurrent requests if hardware permits
- batch sizes 10, 100, 1,000, and full dataset if safe

Requirements:

- no model reload per request
- no unbounded memory growth
- no crash
- useful timeout errors
- batch scoring progress
- safe worker count
- one-command runner chooses sane defaults for 16 GB RAM

Create:

- `artifacts/testing/performance_results.json`
- `docs/FINAL_PERFORMANCE_REPORT.md`

---

# 16. Offline and dependency testing

Test in offline mode:

- no internet
- no Ollama
- no external API keys
- no MCP
- no browser agent

The following must still work:

- backend start
- frontend start
- model load
- scoring
- batch scoring
- deterministic explanations
- dashboard
- evidence export
- audit logging

Optional advanced models may require pre-downloaded weights, but the approved released model must already be packaged locally or clearly downloaded during setup—not silently fetched at runtime.

Release blocker:

- FAIL if core operation requires internet.

---

# 17. Security testing

Test:

- secrets in repository
- secrets in logs
- unsafe pickle/joblib loads
- path traversal
- arbitrary file overwrite
- formula injection in CSV
- oversized upload
- invalid MIME type
- malformed JSON
- SQL injection
- XSS in analyst notes
- CORS misconfiguration
- dependency vulnerabilities
- model artifact checksum
- unauthorised high-impact action
- audit-log tampering
- PII exposure

Requirements:

- high-impact actions require explicit analyst confirmation
- audit trail includes actor, time, model version, reason
- no automatic freeze
- no customer data sent externally by default

Create:

- `artifacts/testing/security_results.json`
- `docs/FINAL_SECURITY_REPORT.md`

---

# 18. End-to-end scenarios

Execute these end-to-end:

## Scenario A: High-risk mule-like account

Expected:

- high calibrated risk
- critical or urgent review
- clear technical reasons
- human review required
- no automatic freeze

## Scenario B: Legitimate business look-alike

Expected:

- risk may initially be elevated
- hard-negative/context layer reduces confidence or abstains
- manual review
- no criminal label
- no automatic freeze

## Scenario C: Low-current-risk account

Expected:

- `MONITOR`
- “not currently flagged”
- never “permanently safe”

## Scenario D: OOD account

Expected:

- `OOD_REVIEW`
- no confident normal result

## Scenario E: Ollama offline

Expected:

- same score and tier
- deterministic explanation
- no broken page

## Scenario F: Ollama hallucination

Expected:

- invalid explanation rejected
- deterministic fallback used
- audit event recorded

## Scenario G: Invalid batch upload

Expected:

- safe rejection
- clear error
- no backend crash

## Scenario H: Model restart

Expected:

- same model version
- same deterministic predictions

## Scenario I: Analyst decision

Expected:

- explicit action
- actor and timestamp stored
- no direct automatic bank action

## Scenario J: Drift warning

Expected:

- monitoring alert
- no automatic model promotion

Create:

- `artifacts/testing/e2e_results.json`
- `docs/FINAL_E2E_REPORT.md`

---

# 19. Clean-install test

Test from a clean checkout or clean temporary copy.

Steps:

1. Copy repository without caches.
2. Follow README exactly.
3. Run setup.
4. Run tests.
5. Run one-command startup.
6. Score a sample.
7. Open frontend.
8. Stop application.
9. Restart.
10. Verify persistence and model consistency.

No hidden local machine dependency is allowed.

Create:

- `artifacts/testing/clean_install_log.txt`

Release blocker:

- FAIL if a new developer cannot run the project from documented instructions.

---

# 20. Final commands

Create canonical commands:

```bash
./run.sh
./scripts/test_all.sh
./scripts/test_ml.sh
./scripts/test_backend.sh
./scripts/test_frontend.sh
./scripts/test_e2e.sh
./scripts/test_security.sh
./scripts/test_offline.sh
./scripts/stop.sh
```

Or equivalent Make targets:

```bash
make run
make test
make test-ml
make test-backend
make test-frontend
make test-e2e
make test-security
make test-offline
make stop
```

One canonical command must run the full release test suite:

```bash
./scripts/release_test.sh
```

It must:

- execute all required tests
- aggregate results
- return non-zero on failure
- generate final reports
- not mark PASS when blockers remain

---

# 21. Final release report

Create:

- `docs/FINAL_RELEASE_TEST_REPORT.md`
- `artifacts/testing/final_release_summary.json`

The report must include:

## Environment

- commit SHA
- OS
- Python/Node versions
- compute mode
- RAM/GPU
- dataset hash
- model artifact hash

## Dataset

- verified rows/columns
- target distribution
- quarantined features
- selected feature count

## Model

- candidates tested
- best single model
- ensemble decision
- OOF PR-AUC with CI
- locked-test PR-AUC with CI
- recall at top 25/50/100
- precision at top 25/50/100
- Brier score
- ECE
- inference latency
- model size

## System

- one-command startup result
- backend independent result
- offline result
- Ollama-off result
- frontend pages tested
- API/frontend consistency
- load test
- security test

## Defects

- P0 defects
- P1 defects
- P2 defects
- fixed defects
- open approved exceptions

## Final verdict

Only one:

- `PASS`
- `PASS WITH APPROVED NON-BLOCKING EXCEPTIONS`
- `FAIL`

---

# 22. Mandatory release blockers

Return `FAIL` if any are true:

- F3924 enters features
- F3912 enters accepted model
- index/ID leakage
- split overlap
- test-set tuning
- fake/hardcoded metrics
- model artifacts missing
- predictions not reproducible
- backend requires frontend
- scoring requires Ollama
- scoring requires internet
- MCP required
- Claude in Chrome required
- one-command startup broken
- frontend blank screen
- major UI overlap
- frontend/backend score mismatch
- Ollama changes score or tier
- LLM invents evidence without rejection
- automatic freeze possible
- criminal-guilt wording appears
- OOD receives confident safe result
- raw dataset modified
- secrets committed
- P0 defect remains
- unapproved P1 defect remains

---

# 23. Required final response after execution

After completing the work, respond with:

1. What was tested.
2. What was fixed.
3. Exact one-command startup command.
4. Exact full-test command.
5. Verified dataset shape and class balance.
6. Confirmed leakage exclusions.
7. Best leakage-free model achieved.
8. OOF PR-AUC and confidence interval.
9. Locked-test PR-AUC and confidence interval.
10. Recall and precision at alert budgets.
11. Calibration results.
12. Frontend status.
13. Backend status.
14. Offline/Ollama-off status.
15. Number of tests passed/failed.
16. Remaining risks.
17. Final release verdict.

Do not say “best accuracy” without the actual evidence.
Do not say “100%.”
Do not claim completion when blockers remain.

---

# 24. Start now

Execute in this order:

1. Repository and environment audit.
2. Existing-test baseline.
3. No-MCP/no-Claude-browser scan.
4. One-command startup implementation and validation.
5. Dataset and leakage verification.
6. Full model tournament.
7. Feature-selection and calibration tests.
8. Backend reliability tests.
9. Ollama hallucination/guardrail tests.
10. Frontend white-background/black-text tests.
11. API/frontend consistency.
12. Batch upload tests.
13. Performance and offline tests.
14. Security tests.
15. End-to-end scenarios.
16. Clean-install test.
17. Fix every P0/P1 issue.
18. Rerun the complete suite.
19. Generate final evidence.
20. Return the release verdict.
