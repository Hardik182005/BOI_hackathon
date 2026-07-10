# Drift Monitoring Report

Mechanism report; the live status is `GET /v1/drift/status` and the latest
measured snapshot is `artifacts/metrics/drift_locked_test.json`.

## What is monitored

| Signal | Method |
|---|---|
| Input feature drift | PSI per kept feature against a frozen dev baseline (quantile bins, missing-aware) |
| Score drift | PSI of the calibrated-risk distribution vs dev OOF |
| Missingness drift | per-feature missing-rate shift vs baseline |
| Schema drift | scoring requests validate the selected-feature contract; missing features are a hard `SCHEMA_ERROR`, and category novelty feeds the OOD detector |
| OOD rate | share of scored rows routed to `OOD_REVIEW` |
| Calibration drift | Brier/ECE recomputed when verified labels arrive via analyst feedback |
| Alert-volume drift | tier counts per scored batch (dashboard) |
| LLM health | narrator failure and hallucination-rejection counts stored with each generated report |

Bands: PSI < 0.10 stable, 0.10–0.25 moderate, > 0.25 alert — stated
conventions, not discovered results. Sequential detectors (ADWIN/DDM) apply
when a real scoring stream exists; on a single static file we run batch PSI
(honest scope note).

## Baseline

Frozen at locked-test evaluation from the development rows only:
per-feature quantile bins + reference samples + missing rates + the dev
calibrated-score distribution (`artifacts/metrics/drift_baseline.joblib`).

## Champion / challenger governance

1. Challenger trained on approved data with the same immutable folds.
2. Compared on OOF PR-AUC, recall@budget, Brier/ECE.
3. Promotion is a recorded human decision in the model registry — never
   automatic. 4. Rollback = reload the previous hashed bundle (kept in
   `artifacts/model_registry/`). 5. Analyst verdicts (CONFIRMED_MULE /
   FALSE_POSITIVE) accumulate in `analyst_feedback` as fresh labels for
   retraining.
