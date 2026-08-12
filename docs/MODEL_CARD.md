# Model Card — MuleGuard · Trinetra

Machine-readable manifest: `artifacts/models/model_manifest.json`.
Measured performance: `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md`,
`artifacts/metrics/tournament_v2.json`,
`artifacts/metrics/promotion_decision_v2.json`,
`artifacts/metrics/lens_stack_oof_v2.json`.

**Current model: `xgboost_top_120` — OOF PR-AUC 0.7690 ± 0.0266** (3 repeats,
120 firewall-admitted features), ROC-AUC 0.9577, Recall@100 0.7813.
It supersedes `catboost_tuned_top60`, whose 0.8077 is **retired** as
leakage-inflated: that set contained `F3898`, `F3913`, `F3914` (post-resolution)
and `F3916` (undetermined availability). `docs/FINAL_RESULTS.md` and
`artifacts/metrics/locked_test_metrics.json` describe the retired bundle.

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

Feature Availability Firewall (admissibility by time-of-availability) →
compact stability-selected feature set (top-120) → class-weighted **XGBoost**
(`xgboost_top_120`, promoted; no ensemble — `SINGLE_MODEL_KEPT`) with
LightGBM/CatBoost agreement models → Platt/isotonic calibration (OOF-selected)
→ Mondrian conformal abstention → hard-negative verifier (Lens 2) →
IsolationForest + OOD challengers (Lens 3) → deterministic policy engine →
human analyst. Optional local LLM narrates verified facts under a
hallucination validator with deterministic fallback.

## Exclusions (leakage firewall — measured evidence)

- **Hard quarantine (9)** — `F3924` (target); `F3912` (|corr| .97,
  single-feature CV PR-AUC .94), `F3913`, `F3914`, `F3915` (the four mutually
  exclusive resolution outcomes — removing only one leaves the family
  reconstructible); `F3898`, `F3899` (resolve-day durations, written after the
  decision); `F2230` (snapshot month ≡ label by dataset construction);
  `__UNNAMED__0` (row index; file sorted by label).
- **Conditional quarantine (3)** — `F3916`, `F3917`, `F3918` (customer risk
  level flags), held out until the bank confirms they are written independently
  of alert resolution; evaluated only as a labelled ablation.
- **Fairness exclusion (1)** — `F3892` (gender).
- Policy of record: `configs/feature_availability.yaml` (v2.0), enforced by
  `src/muleguard/features/firewall.py`; the promoted 120-feature set contains
  **0 quarantined columns**.
- In-fold: constants and exact duplicate columns removed on train statistics.

## Metrics

Primary PR-AUC with bootstrap 95% CIs; recall/precision at alert budgets
(25/50/100/1%); recall at fixed FPR; Brier/ECE; per-tier precision; runtime.
Accuracy is deliberately not reported (0.89% prevalence makes it meaningless).

Generalization evidence (`artifacts/metrics/tournament_v2.json`): an ablation
restricted to the bank's pre-existing risk/prior variables scores 0.0295, and
one restricted to alert-context metadata scores 0.0206, against a prevalence
floor of 0.0087 and a full-model 0.7690. The model's performance comes from
behavioural aggregates, not from pre-existing risk flags or alert metadata.

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
