# Trinetra Lens Report

Mechanism report; measured lens numbers live in
`artifacts/metrics/lens_stack_oof.json` and `docs/FINAL_RESULTS.md`.

## Lens 1 — Detect the behaviour

- Final base models (LightGBM winner + XGBoost + CatBoost on the compact
  stability-selected feature set) trained on the development split.
- Outputs per account: raw scores per model, calibrated risk, model agreement
  (1 − max pairwise spread), conformal set, SHAP top drivers vs the
  legitimate-cohort percentiles.
- The screening posture is high-recall: the STANDARD_REVIEW threshold is
  derived from a 90% dev-OOF recall target.

## Lens 2 — Protect the look-alike (false positives)

1. **Hard-negative verifier** — mined from OOF predictions only: the top ~2%
   highest-scored *legitimate* accounts form the hard-negative band; a second
   LightGBM learns positives vs (hard negatives + 3× background). At scoring
   time, when the verifier disputes a CRITICAL case and conformal evidence is
   not conclusive, policy caps it to URGENT_REVIEW — review, not fast-track.
2. **Calibration** — Platt vs isotonic compared with cross-fitted OOF
   probabilities on Brier + ECE; isotonic must win both by ≥2% relative to be
   chosen (small-positives overfit guard). Selection recorded in the bundle.
3. **Mondrian conformal abstention** — class-conditional split conformal on
   crossfit-calibrated OOF probabilities (α = 0.10, finite-sample corrected).
   Outputs HIGH_RISK_SET / LOW_RISK_SET / UNCERTAIN_SET; UNCERTAIN escalates
   MONITOR cases into STANDARD_REVIEW. The system is allowed to say
   "I don't know" — and does.

## Lens 3 — Recover the missed (false negatives)

1. **Isolation Forest challenger** fit on the legitimate dev cohort;
   scores converted to dev-referenced percentiles. Percentile ≥ 99 escalates
   an otherwise-MONITOR account to review (second opinion, never an override).
2. **OOD detector** — missingness-profile z-score, widened range violations,
   kNN distance in robust-scaled space vs the dev 99.9% quantile. Any trip →
   OOD_REVIEW: the model's score is explicitly not trusted for such inputs.
3. **Positive-unlabelled learning: disabled by design.** The label-trust audit
   found the negatives are a single-month snapshot and undiscovered mules are
   plausible inside them, but the same audit found time≡label confounding —
   a PU model here would mostly relearn the snapshot artifact. Documented as
   a roadmap item pending cleaner labels. (Honest skip, per audit.)
4. **Never certify clean** — the benign state everywhere (policy, API, UI,
   narratives) is `NOT_CURRENTLY_FLAGGED` / "monitoring continues".

## Action layer

Deterministic policy engine (`action/policy.py`), thresholds frozen from dev
OOF distributions into the registry policy snapshot. Tier precedence:
OOD → risk bands → safety escalations (conformal uncertainty, anomaly
disagreement) → look-alike protection. Every high-impact recommendation
requires analyst identity + reason + second approver and lands in the
append-only audit log. No model or LLM can execute anything.
