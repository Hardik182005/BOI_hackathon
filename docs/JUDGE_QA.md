# Judge Q&A

Answers reflect what is actually implemented; measured numbers live in
`docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md` and
`artifacts/metrics/*_v2.json`. (`docs/FINAL_RESULTS.md` is the retired
pre-firewall run and is labelled as such.)

**What is the headline number?**
`xgboost_top_120`: **OOF PR-AUC 0.7690 ± 0.0266** over 3 CV repeats on 120
firewall-admitted features; ROC-AUC 0.9577; Recall@100 0.7813. Against a
0.89% prevalence that is roughly an 86× lift. Source:
`artifacts/metrics/tournament_v2.json`,
`artifacts/metrics/promotion_decision_v2.json`.

**An earlier version of this project reported 0.8077. What happened?**
That model (`catboost_tuned_top60`) selected `F3898` (how long the
investigation took), `F3913` and `F3914` (the alert's closing status) and
`F3916` (a risk flag of undetermined timing). None of those exist for an
account that has not yet been investigated, which is the only kind of account
the system is asked about. The number was inflated by them and is **retired**.
We found it ourselves, published the retirement in the model registry's
`supersedes` block, and replaced it with a lower number that measures the
actual task.

**How do you know the new number isn't leakage too?**
Two ablations, in `artifacts/metrics/tournament_v2.json`. A model restricted
to the bank's own pre-existing risk variables scores **0.0295**. A model
restricted to alert-context metadata scores **0.0206**. The prevalence floor is
0.0087. If the detector were re-reading a pre-existing verdict or the shape of
the alert, one of those two would score well; neither does, while behavioural
features reach 0.7690.

**Why not use accuracy?**
At 0.89% prevalence, predicting "legitimate" for everyone scores ~99.1%
accuracy and catches zero mules. We report PR-AUC (Average Precision),
recall/precision at analyst alert budgets, recall at fixed FPR, Brier/ECE,
and bootstrap CIs — the honest family for rare-event detection.

**Why remove F3912?**
Measured on the raw file: |correlation| 0.969 with the target, single-feature
*cross-validated* PR-AUC 0.943, balanced label reconstruction 0.988. Adding it
inflated LightGBM from 0.614 to 0.942 OOF PR-AUC in the pre-firewall run — the
signature of a post-outcome field. A model shipped with it would collapse in
production the moment it scores accounts whose investigations haven't happened
yet. (Both figures are from that retired run; the point they illustrate is
unchanged.) Removing `F3912` alone was not enough: it is one of a four-column
near-one-hot family, and `1 − max(F3913, F3914, F3915)` reconstructs it on
86.95% of rows. All four go, plus the two resolve-day columns —
`docs/UPGRADE_GAP_ANALYSIS.md` §1.1.

**Did you find any leakage beyond F3912?**
Yes — our audit found a leak the design document had missed: **F2230, the
snapshot month, separates the classes perfectly** (all 9,001 negatives are the
Oct-2025 snapshot; all 81 positives are Sep/Nov/Dec). The file is even
physically sorted by label, so the row index is a perfect predictor too. Both
are quarantined; this is why our honest headline is lower than naive analyses
of this dataset.

**Why not use SMOTE?**
We measured it rather than asserting it: six resampling arms across the same 15
nested outer folds (`artifacts/metrics/smote_ablation.json`, verdict
**`KEEP_BASELINE`**). The honest reading is not "SMOTE hurt" — every SMOTE ratio
came out *slightly positive* (+0.005 to +0.012 AP) and simply failed the
pre-registered sign test. What decides it is the control arm: plain random
duplication buys +0.005 on its own, so synthesising new points beyond
re-weighting is worth roughly +0.007 against a fold spread of ±0.10. That is
not a reason to manufacture minority examples by interpolating between 81 real
mules in 3,924 dimensions. Random undersampling is the one unambiguous result:
−0.134, p = 0.0001. So we change the loss (class weights) and the metric, not
the data, and natural prevalence is preserved in every validation and test split.

One caveat we should state before a judge finds it: this ablation scored
**ranking only**. The calibration argument — that synthetic oversampling
distorts the base rate and corrupts the probabilities our review tiers depend on
— is the textbook reason and it is why the thresholds are built the way they
are, but this run did not measure Brier or ECE per arm, so we are not claiming
it as our own finding. `smote_0.10` also clears Wilcoxon (p = 0.010) while
failing the sign test (p = 0.12); the sign test was named before the run, so it
stands rather than being swapped for the test that reads better.

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

## Cohort Radar and the Account-Control Guardrail

**You show a cohort of similar accounts. Isn't that a mule ring?**
No, and we will not call it one. Behavioural similarity has innocent
generators - the same salary cycle, the same merchant category, the same
product, the same month-end. What we can say is that these accounts behave
unusually similarly. Who owns them, who operates them and whether they are
connected is not in this dataset. The panel says so in a mandatory disclaimer,
and the forbidden phrasings ("criminal network", "same handler", "connected
mule ring", "controlled by the same person") are enumerated in
`artifacts/models/cohort_radar_manifest.json` and asserted absent by tests.

**Does cohort membership raise the risk score?**
No. Zero effect, and it is measured rather than asserted: 400 accounts
re-scored end-to-end before and after the feature exists give
**max |delta probability| = 0.000e+00** against a `1e-12` tolerance, zero tier
changes, zero policy-action changes, and all 20 published metrics identical.
See `docs/USP_ACCURACY_REGRESSION.md`; reproduce with
`python -m muleguard.cli.usp_baseline --check`, where the exit code is the
verdict.

**Why not blend them? `0.8 * model + 0.2 * cohort` would score better.**
Three reasons. It destroys the calibration - our probabilities are Platt-
calibrated at Brier 0.0031 and ECE 0.0015, and a blend produces a number that
is no longer a probability of anything while still being displayed as a
percentage. It is circular - neighbours are found using the model's own input
features, so blending re-counts the same evidence and calls the second count
corroboration. And it escalates by association - a legitimate high-volume
merchant that resembles a suspicious account would gain risk *because of who
it resembles*, which is the one mechanism a fraud system must never have.

**Does the Cohort Radar work?**
For a known mule account, 6.8 of its top 10 behavioural neighbours are also
mule accounts - a **77.5x lift** over the 0.88% base rate, with Hit@10 = 0.94.
For a legitimate account the same lookup returns 0.22 positives in 10.
Evaluated outer-fold-safe on the model's own nested-CV fold map, with the
similarity transform refit per fold on training rows only and labels read only
after the neighbours are fixed. Source:
`artifacts/metrics/cohort_radar_retrieval.json`. The qualifier matters:
**this is a retrieval-quality diagnostic, not classifier accuracy**, and it
does not prove common ownership.

**Could a judge's uploaded labels leak into the cohort results?**
No, and we test it adversarially rather than structurally. Setting `F3924 = 1`
on the query row leaves the ranking unchanged. Setting all 11 named forbidden
columns to an extreme value leaves the ranking unchanged. Label-shaped keys
added to an uploaded row are ignored entirely. The frozen weights and scaling
statistics are byte-identical after an adversarial query. 11/11 checks in
`artifacts/testing/cohort_radar_leakage.json`, release-blocking.

**Do the cohorts just group people by age, occupation or locality?**
Measured, not assumed. Cohort neighbours agree on **occupation 0.998x** as
often as two random accounts - literally at chance. Area, age-decade and gender
land between 1.09x and 1.12x, which is a behavioural correlation and not a
matching effect, because none of those fields is in the fingerprint at all.
Behavioural features carry **94.04%** of the similarity weight; the only
profile feature is product name at 0.05%. Full method:
`docs/COHORT_RADAR_FAIRNESS_AUDIT.md`.

**Why does the system refuse to say whether someone is a mule?**
Because it cannot know. Three different questions: how unusual the behaviour
is (our classifier answers this), who operated the account (device, SIM,
credential and KYC history would - we have none), and whether the holder knew
(an interview would). We report `ASSESSED`, `NOT_AVAILABLE` and `UNKNOWN`
respectively. A large share of Indian mule accounts belong to students and gig
workers who were tricked or paid; the activity is real and the culpability is a
separate finding. The system will never emit "witting mule", "criminal",
"victim" or "handler", and will never automatically FREEZE, FILE_STR,
DECLARE_MULE or CERTIFY_CLEAN. `automatic_actions_permitted` is `[]` on every
card at every risk level.

**Where is the transaction graph?**
There isn't one, and that is a data fact rather than a design choice. A
transaction graph needs sender/receiver pairs; this dataset is account-level
aggregates. We have two graphs: the **ProofGraph** (one account, edges mean
"this evidence raised this decision") and the **Cohort Radar** (many accounts,
edges mean "these two behave similarly", labelled `BEHAVIORALLY_SIMILAR_TO`).
No directional arrows are drawn anywhere, because an arrow implies money
movement we cannot evidence.

**Is any of this a neural network, a vector DB or a GNN?**
No. The retrieval is a weighted Gower distance over 120 columns computed with
NumPy - no FAISS, no embedding model, no Neo4j, no GNN. That is sufficient at
this portfolio size and, more importantly, inspectable: we can show an
investigator *which features* drove a match, which no learned embedding would
allow.

---

**What is implemented versus roadmap?**
Implemented: everything in `docs/FINAL_RESULTS.md` and the release gate.
Roadmap (stated, not claimed): counterparty graph layer, federated/DPIP
integration, full RBAC/SSO, PostgreSQL deployment, TabICL/TabPFN on GPU.
