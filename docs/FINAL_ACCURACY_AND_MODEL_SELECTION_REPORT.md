# Final Accuracy and Model Selection Report

All numbers are pipeline-generated; authoritative machine-readable copies:
`artifacts/metrics/final_model_comparison.csv`, `final_oof_metrics.json`,
`final_locked_test_metrics.json`, predictions in `artifacts/predictions/final_*.parquet`.
Nothing here is typed by hand — this document narrates those artifacts
(assembled alongside `docs/FINAL_RESULTS.md`, which carries the full tables).

## Protocol

- Locked stratified test (1,818 rows / 17 positives) frozen before training,
  evaluated once; development = repeated stratified group-aware 5-fold CV
  (5 repeats, 7,264 rows / 64 positives); every preprocessing/selection/
  calibration step inside training folds; class weights, no SMOTE; natural
  prevalence preserved everywhere.
- Primary metric PR-AUC; ROC-AUC secondary; recall/precision at top-25/50/
  100/1%; FP per 1,000 legit; Brier/ECE; bootstrap 95% CIs (n=2000);
  latency/throughput/model size recorded.

## Tournament outcome (dev OOF, 5 repeats)

| Model | PR-AUC | Note |
|---|---|---|
| CatBoost tuned, top-60 | **0.8077 ± 0.0450** | winner |
| LightGBM tuned, top-60 | 0.7717 ± 0.0528 | agreement model |
| LightGBM tuned, full clean matrix | 0.7637 ± 0.0223 | compact set beats full matrix |
| LightGBM tuned, top-30 | 0.7562 ± 0.0255 | |
| XGBoost tuned, top-60 | 0.7560 ± 0.0658 | agreement model |
| LightGBM tuned, top-15 | 0.6440 ± 0.0420 | |
| LightGBM baseline (untuned, full) | 0.6142 ± 0.0474 | |
| Logistic L2 | 0.2328 ± 0.0120 | |
| Dummy prevalence | 0.0087 | floor |
| *LightGBM + F3912* | *0.9419 ± 0.0119* | **REJECTED LEAKAGE — evidence only** |

Challengers: **TabPFN v3** ran on top-60 (1 repeat): OOF PR-AUC **0.8970**,
but runtime 3,144 s per repeat (~26 min/fold CPU) and single-repeat evidence —
recorded as a promising GPU-era challenger, ineligible for the bundle slot
under the pre-registered 5-repeat rule and operationally prohibitive on this
hardware. TabICL and AutoGluon: documented skips
(`artifacts/metrics/advanced_models.json`).

Ensemble: logistic stacker won 3/5 repeats vs CatBoost — **rejected** by the
pre-registered ≥4/5 rule (`ensemble_decision.json`).

## Selection rationale

CatBoost top-60 wins on repeated-fold mean while remaining stable (std in
family range), fast (142 s per full 5×5 evaluation; 688 rows/s inference in
the bundle), calibratable (Platt selected on OOF; locked-test Brier 0.00258,
ECE 0.0027) and explainable (TreeSHAP). No candidate depends on F3912/F2230.

## Locked test — single touch (production scorer)

- PR-AUC **0.8242** (95% CI 0.6536–0.9584); ROC-AUC 0.9908 (secondary)
- Recall@25 = 88.2% (15/17) at 60% precision; recall@100 = 94.1%
- CRITICAL tier: 12 accounts, 12 true mules — 100% precision in tier
- Conformal abstention 0.55%; positive coverage 76.5%; OOD rate 0.11%
- The CI is wide because 17 positives exist — reported, not hidden.

## Honesty notes

No "state-of-the-art" or "100%" claims. The 0.94 leakage number is displayed
only as rejected evidence. The strongest *defensible* number is the one above,
with its confidence interval and its single-touch provenance.
