# Assumptions and Limits

Every item is labelled: **[MEASURED]** re-verified from raw data by this pipeline · **[ASSUMPTION]** design assumption · **[LIMIT]** honest scope boundary.

## Data semantics

1. **[MEASURED → see `data/interim/data_fingerprint.json`]** Shape, class balance, missingness, constants, duplicates are recomputed from `DataSet.xlsx`; the submission PDF numbers are treated as unverified claims until reproduced.
2. **[ASSUMPTION]** `F3924` is the binary target with `1 = suspicious/mule`, per Topic.pdf ("feature 3924 is the target variable"). Label reliability of `0` is unknown — negatives may contain undiscovered mules (motivates Lens 3, "never certify clean").
3. **[ASSUMPTION]** All `Fxxxx` columns except a small interpretable set are anonymised behavioural aggregates. **No business semantics are invented for them anywhere in code, UI, or reports.** Explanations reference values vs cohort percentiles only.
4. **[LIMIT]** The file is a flat per-account snapshot: no transaction sequences, no counterparty edges. Therefore no sequence models, no GNN, no claim of network detection. Graph detection is a production extension once counterparty data exists (e.g. via DPIP), not part of this model.
5. **[LIMIT]** Behaviour is not intent. The model scores behavioural similarity to labelled mules; it cannot distinguish wilful mule vs hacked victim vs coerced victim. Cause is decided by human analysts with enrichment data.
6. **[LIMIT]** Dormant/sleeper mules with no behavioural footprint are not claimed detectable before activation.

## Methodology

7. **[ASSUMPTION]** Leakage audit statistics (per-feature correlation/single-feature CV PR-AUC) are computed on the full dataset **for the audit only**; this is safety-conservative (used to *exclude* features, never to select them). Predictive feature *selection* happens strictly inside CV training folds.
8. **[ASSUMPTION]** `F3912` is quarantined by default based on measured leak evidence; final adjudication would belong to the data owner (bank). The with/without ablation is preserved as evidence and clearly marked "rejected leakage".
9. **[ASSUMPTION]** Locked test = 20% stratified, seed 42, frozen to parquet before any model training; touched exactly once at the end. If duplicate feature-rows span dev/test, whole duplicate groups are kept on one side (group-aware split).
10. **[ASSUMPTION]** With ~81 positives total, per-fold calibration data is tiny; simpler calibrators (Platt) are preferred over isotonic when isotonic shows overfit (selected by OOF Brier/ECE, never on locked test).
11. **[LIMIT]** Confidence intervals are bootstrap-based and wide by nature at 0.89% prevalence; they are reported, not hidden.
12. **[MEASURED → LIMIT]** Out-of-time stress testing is **invalid on this file**: the audit found `F2230` (snapshot month) perfectly separates the classes — all 9,001 negatives are the 2025-10 snapshot, all 81 positives are Sep/Nov/Dec snapshots. Time ≡ label, so any "train on earlier months, test on later" split would be testing the leak, not the model. Documented in `artifacts/metrics/temporal_stress_metrics.json` as NOT_VALID with evidence; `F2230` is quarantined.

## Product / safety

13. The system never freezes accounts; outputs are review tiers (`CRITICAL_REVIEW`, `URGENT_REVIEW`, `STANDARD_REVIEW`, `OOD_REVIEW`, `MONITOR`). "Freeze candidate" appears only as a recommendation requiring an authorised analyst.
14. Ollama (local LLM) is an optional narrator of verified structured facts. It can never compute/alter a risk score, tier, threshold, or action; its output is schema-validated and rejected on any hallucination, with a deterministic fallback. Scoring works with Ollama stopped.
15. Low-risk accounts are "not currently flagged; continue monitoring" — never "safe/clean".
16. Demo enrichment (analyst names, case notes) is synthetic and labelled as such; account references are masked synthetic IDs, not real customers.

## Engineering environment

17. **[ASSUMPTION]** Compute Mode A (16 GB CPU laptop): Optuna budgets capped (40/30/25 trials), repeated CV 5×5 for finalists, TabPFN/TabICL only behind a feasibility guard, AutoGluon skipped (no Python 3.13 support at build time) — each skip documented in the tournament report.
18. All installs and artifacts on E: drive (user constraint); C: has ~37 GB free and is not used for new packages.
