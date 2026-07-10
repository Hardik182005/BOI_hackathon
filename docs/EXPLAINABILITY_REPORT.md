# Explainability Report

Mechanisms and language rules; per-case artifacts are produced live
(evidence packets) and globally in `artifacts/plots/`.

## Local explanations (per scored account)

- **Exact TreeSHAP** (`pred_contrib`) on the winning LightGBM — no sampling
  approximation.
- Each of the top-5 drivers reports: feature id, verified semantic name (only
  for the 7 registry columns), raw value, legitimate-cohort median, cohort
  percentile, direction, SHAP contribution, plus model version and the data
  fingerprint via the bundle manifest.
- Language rule for anonymised features (enforced in narrator + validator):
  comparative-numerical only — *"F1702 is at the 99.2nd percentile of the
  legitimate cohort and increases the model score."* Never semantics, never
  intent.

## Global explanations

- Stability-selection frequency table (`artifacts/features/selection_frequency.csv`)
  and plot (fold-to-fold overlap recorded in `selected_features.json`).
- Global model comparison and leakage-ablation plots.
- Compact-vs-full feature ablation in the tournament metrics (top-15/30/60 vs
  full clean matrix).
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
