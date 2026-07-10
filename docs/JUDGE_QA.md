# Judge Q&A

Answers reflect what is actually implemented; measured numbers live in
`docs/FINAL_RESULTS.md` and `artifacts/metrics/`.

**Why not use accuracy?**
At 0.89% prevalence, predicting "legitimate" for everyone scores ~99.1%
accuracy and catches zero mules. We report PR-AUC (Average Precision),
recall/precision at analyst alert budgets, recall at fixed FPR, Brier/ECE,
and bootstrap CIs — the honest family for rare-event detection.

**Why remove F3912?**
Measured on the raw file: |correlation| 0.969 with the target, single-feature
*cross-validated* PR-AUC 0.943, balanced label reconstruction 0.988. Adding it
inflates LightGBM from 0.614 to 0.942 OOF PR-AUC — the signature of a
post-outcome field. A model shipped with it would collapse in production the
moment it scores accounts whose investigations haven't happened yet.

**Did you find any leakage beyond F3912?**
Yes — our audit found a leak the design document had missed: **F2230, the
snapshot month, separates the classes perfectly** (all 9,001 negatives are the
Oct-2025 snapshot; all 81 positives are Sep/Nov/Dec). The file is even
physically sorted by label, so the row index is a perfect predictor too. Both
are quarantined; this is why our honest headline is lower than naive analyses
of this dataset.

**Why not use SMOTE?**
Synthetic oversampling distorts the base rate and corrupts calibrated
probabilities that our review-tier thresholds depend on. We change the loss
(class weights) and the metric, not the data. Natural prevalence is preserved
in every validation and test split.

**Why not use an LLM as the classifier?**
The task is tabular classification with 3,900 anonymised numeric features.
Gradient-boosted trees are the state of the art for this regime (Grinsztajn
et al. 2022); an LLM adds cost, latency, nondeterminism and hallucination risk
with no accuracy evidence. Our LLM is confined to narrating verified facts and
is machine-validated against altering them.

**Why are LightGBM/CatBoost/XGBoost strong here?**
Native missing-value handling (27.6% of cells are missing), robustness to
irrelevant/duplicate features, class weighting, monotone training on CPU,
deterministic seeds, and exact TreeSHAP explanations.

**What did TabICL/TabPFN add?**
On this hardware (CPU-only, 16 GB) they were run/skipped per a documented
feasibility guard — see `artifacts/metrics/advanced_models.json`. They are
challengers, not defaults; anything without repeated OOF improvement is not
shipped.

**How do you prevent false positives?**
Lens 2: a hard-negative verifier trained on the exact legitimate accounts the
screener confuses for mules; probability calibration (Platt/isotonic selected
by Brier/ECE on OOF); Mondrian conformal abstention that routes uncertain
cases to humans; and policy protection that keeps verifier-disputed cases in
review rather than fast-tracking them.

**How do you catch missed mules?**
Lens 3: an Isolation Forest challenger (fit on the legitimate cohort) plus an
OOD detector; disagreement with the supervised score escalates to review. And
structurally: no account is ever labelled "safe" — the negative state is
"not currently flagged; monitoring continues".

**Can the system identify intent?**
No, and it never claims to. Behaviour is not intent: a hacked victim and a
wilful mule can look identical in features. Cause is decided by human analysts
with KYC/registry/device enrichment; the UI and reports say "behavioural risk".

**Can it identify sleeper accounts?**
Not before they act — dormant mules have no behavioural footprint in this
snapshot. We state this openly; onboarding-time signals (account age, tenure
bucket) provide partial early risk context only.

**Why no GNN?**
The file is a flat per-account matrix: no sender→receiver edges exist. A graph
model without edges would be theatre. Counterparty-graph detection is the
documented roadmap once DPIP-style signals provide real edges.

**What happens when data drifts?**
PSI/JS monitoring of features, scores and missingness against a frozen dev
baseline; ADWIN-style batch checks; champion/challenger retraining where
promotion is a recorded human decision with rollback. Analyst verdicts feed
back as fresh labels.

**How are thresholds selected?**
From the dev OOF calibrated-risk distribution only: CRITICAL/URGENT sized to
analyst daily capacities (25/100), STANDARD from a 90% recall target. Saved to
a frozen policy snapshot; the locked test never influenced them.

**Can the model automatically freeze an account?**
No code path exists for that. The policy engine emits review tiers; a freeze
*recommendation* requires an analyst id, a reason and a second approver, all
recorded in an append-only audit log.

**What happens when Ollama hallucinates?**
A validator rejects output that changes scores/tiers, invents features or
amounts, asserts guilt, or steps outside the action allowlist — then a
deterministic template takes over. We demo this live with a planted
hallucination.

**What happens when Ollama is offline?**
Nothing user-visible: scoring is LLM-free by construction and reports fall
back to deterministic narratives (integration-tested).

**How does the bank audit a decision?**
Every score, case action, feedback and report is in SQLite with UTC
timestamps, actor identity, model version and correlation id;
`audit_events` is append-only (SQLite triggers block UPDATE/DELETE). Evidence
packets bundle the score, reasons, uncertainty, actions and limitations.

**What is implemented versus roadmap?**
Implemented: everything in `docs/FINAL_RESULTS.md` and the release gate.
Roadmap (stated, not claimed): counterparty graph layer, federated/DPIP
integration, full RBAC/SSO, PostgreSQL deployment, TabICL/TabPFN on GPU.
