# Model Card — MuleGuard · Trinetra

Machine-readable manifest: `artifacts/models/model_manifest.json`.
Measured performance with CIs: `docs/FINAL_RESULTS.md`,
`artifacts/metrics/locked_test_metrics.json`.

## Purpose

Rank bank accounts by behavioural similarity to labelled mule accounts so
human analysts review the highest-risk cases first. The output is a
**calibrated behavioural risk score with quantified uncertainty**, mapped by a
deterministic policy to review tiers (CRITICAL/URGENT/STANDARD/OOD/MONITOR).
It is not a fraud verdict, not proof of intent, and never triggers automatic
punitive action.

## Intended users & decisions

Bank fraud analysts and their supervisors. Supported decision: *which
accounts to review first, with what evidence*. Unsupported decisions:
freezing, prosecution, customer communication — these require human process.

## Training data

Provided portal dataset (PS2): 9,082 accounts × 3,923 features, 81 labelled
mules (0.8919%). Development 7,264 rows / locked test 1,818 rows (17
positives), stratified, frozen before training, touched once. Raw file
SHA-256 recorded in `data/interim/data_fingerprint.json`.

## Architecture

Compact stability-selected feature set → class-weighted LightGBM (winner) with
XGBoost/CatBoost agreement models → Platt/isotonic calibration (OOF-selected)
→ Mondrian conformal abstention → hard-negative verifier (Lens 2) →
IsolationForest + OOD challengers (Lens 3) → deterministic policy engine →
human analyst. Optional local LLM narrates verified facts under a
hallucination validator with deterministic fallback.

## Exclusions (leakage firewall — measured evidence)

- `F3924` (target), `F3912` (|corr| .97, single-feature CV PR-AUC .94),
  `F2230` (snapshot month ≡ label by dataset construction),
  `__UNNAMED__0` (row index; file sorted by label).
- In-fold: constants and exact duplicate columns removed on train statistics.

## Metrics

Primary PR-AUC with bootstrap 95% CIs; recall/precision at alert budgets
(25/50/100/1%); recall at fixed FPR; Brier/ECE; per-tier precision; runtime.
Accuracy is deliberately not reported (0.89% prevalence makes it meaningless).

## Ethical & safety constraints

No automatic freezes; review-tier language only; "low risk" =
*not currently flagged, monitoring continues*; uncertain/OOD inputs go to
humans; anonymous features never receive invented semantics; behavioural
score never described as criminal intent; append-only audit of every score,
action and report; local-only inference (no customer data leaves the machine).

## Known limitations

Flat snapshot (no sequences, no counterparty graph → no ring detection
claims); behaviour ≠ intent; sleeper mules invisible pre-activation; 81
positives → wide CIs (reported); label collection confounded with snapshot
month → no valid out-of-time test on this file (documented, not faked);
PU learning disabled after label-trust audit.

## Version & governance

Bundle version, SHA-256, git commit, calibrator choice, policy snapshot and
data fingerprint are in the manifest and the model registry. Promotion and
rollback are recorded human decisions.
