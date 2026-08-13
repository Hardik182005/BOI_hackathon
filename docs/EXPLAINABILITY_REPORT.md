# Explainability Report

Mechanisms and language rules; per-case artifacts are produced live
(evidence packets) and globally in `artifacts/plots/`.

## Local explanations (per scored account)

- **Exact TreeSHAP** on the deployed champion — no sampling approximation. The
  explainer follows the champion rather than assuming a family: `xgboost` goes
  through `booster.predict(..., pred_contribs=True)`, `lightgbm` through
  `booster_.predict(..., pred_contrib=True)`, `catboost` through
  `get_feature_importance(type="ShapValues")` (`explain/reason_codes.py:63-75`).
  The shipped bundle is **`xgboost_top_120` (v2.0.0)**, so the XGBoost path is
  the one a judge exercises. An earlier revision of this document said
  "the winning LightGBM"; that named the generation-1 champion and is corrected
  here — see `docs/HISTORICAL_METRIC_RECONCILIATION.md`.
- Each of the top-5 drivers reports: feature id, verified semantic name (only
  for the 7 registry columns), raw value, legitimate-cohort median, cohort
  percentile, direction, SHAP contribution, plus model version and the data
  fingerprint via the bundle manifest.
- Language rule for anonymised features (enforced in narrator + validator):
  comparative-numerical only — *"F1702 is at the 99.2nd percentile of the
  legitimate cohort and increases the model score."* Never semantics, never
  intent.

## Global explanations

- Stability-selection frequency table
  (`artifacts/features/selection_frequency_v2.csv`) and plot (fold-to-fold
  overlap recorded in `selected_features_v2.json`). The unsuffixed
  `selection_frequency.csv` / `selected_features.json` are the **generation-1**
  top-60 selection, run before the Feature Availability Firewall; they are kept
  for provenance and are not what the shipped model explains.
- Global mean |SHAP| ranking over the dev OOF folds
  (`artifacts/metrics/global_shap_importance.json`, `make global-shap`).
- Global model comparison and leakage-ablation plots.
- Compact-vs-full feature ablation in the tournament metrics: the current ladder
  is top-15/30/60/**120**/250 versus the full clean matrix, and the production
  set is top-120 (`artifacts/metrics/model_comparison_v2.csv`).
- Partial-dependence/ALE deliberately not shipped for anonymised features:
  a PD curve on an unnamed column invites over-interpretation (documented
  choice).

## Counterfactual sensitivity ("model sensitivity examples")

- Only top INCREASES_RISK, non-registry (behavioural, mutable) features are
  moved — to the legitimate-cohort median; the row is rescored by the actual
  model; the score delta and threshold crossing are reported.
- Explicitly labelled *"model sensitivity example (not a causal statement)"*.
- Demographic/registry fields (gender, occupation, region, segment) are never
  proposed for change.

## Evidence packets

JSON + printable HTML (`explain/evidence_packet.py`): masked reference,
decision tier, risk + uncertainty (agreement, conformal, OOD, anomaly),
verified reasons, narrative (source-labelled: `ollama` validated or
`deterministic`), analyst action history, model version, explicit limitations.
CSV exports pass a formula-injection sanitiser.
