# Demo Script (7–9 minutes)

Preparation: API on :8001, dashboard on :5173, `muleguard.cli.demo --via-api`
already run (populates the queue with the five scenario cases), Ollama
running with any preferred model (optional — Scene 6 kills it live).

Scenes 7 and 8 are the post-model USP layers. If time is short, cut Scene 5 or
the hallucination sub-step of Scene 6 — not Scene 8: the refusal to infer intent
is the part of this system that most needs saying out loud.

All numbers spoken must be read from the screen (they come from
`artifacts/metrics/`), never from memory.

---

## Scene 1 — The problem (Executive Overview page)  ~45 s

- Point at the prevalence figure: **81 mules in 9,082 accounts (0.89%)**.
- "A model that flags nobody is 99.1% accurate and catches zero mules. That's
  why you will not see an accuracy tile anywhere in this product — the
  headline is PR-AUC and recall at your analysts' real daily budget."
- Target: `F3924` per the problem statement.

## Scene 2 — Leakage integrity (Model Performance page, leakage panel)  ~60 s

- Show the red **REJECTED LEAKAGE** bar: with F3912 the model scored 0.94
  PR-AUC in the pre-firewall run; clean it scored 0.61.
- "Our audit didn't stop at the known suspect. It found that the **snapshot
  month column separates the classes perfectly** — every legitimate account is
  the October snapshot, every mule is Sep/Nov/Dec. The file is even sorted by
  label. All of it is quarantined automatically, with the evidence saved."
- "And it didn't stop there either. A later pass found that our own accepted
  model was reading three columns that only exist after an analyst has already
  closed the case, plus one whose timing nobody could confirm. That model
  reported 0.8077. We retired it. The model we ship scores **0.7690**, and
  it is the first number we have that measures the actual task."
- "The lower number is the honest one — it's the one that survives production."

## Scene 3 — Trinetra Lens 1: detect (Alert Queue → top case)  ~50 s

- Open the highest-risk case (`DEMO-HIGH_RISK_MULE`).
- Show calibrated risk, model agreement, and the SHAP driver table with
  cohort percentiles: "F-features compared against the legitimate cohort —
  numeric evidence, no invented meanings."

## Scene 4 — Trinetra Lens 2: spare the look-alike  ~50 s

- Open `DEMO-BUSINESS_LOOKALIKE`.
- "The screener finds this legitimate account suspicious — but the
  hard-negative verifier, trained on exactly the accounts that fool the
  screener, disagrees. Policy answer: routed to human review, capped below
  fast-track. Nobody gets auto-punished for looking like a mule."

## Scene 5 — Trinetra Lens 3: never certify the unseen  ~40 s

- Open `DEMO-MODEL_DISAGREEMENT`: supervised score low, anomaly percentile
  top-1% → escalated to STANDARD_REVIEW.
- Open `DEMO-OOD_SYNTHETIC` (labelled synthetic): extreme values → OOD_REVIEW,
  "the model refuses to pretend it understands data it has never seen."
- "And the calm case says *not currently flagged — monitoring continues*.
  Nothing is ever certified clean."

## Scene 6 — Explainability + guarded local GenAI (Case Detail)  ~90 s

1. Click **Generate (local LLM)** — readable narrative appears, labelled with
   its source model; risk numbers identical to the score panel.
2. Stop Ollama (`taskkill` / `ollama stop` on the host). Generate again —
   deterministic narrative appears instantly; scoring unaffected.
3. Show the planted-hallucination rejection from the demo evidence
   (`demo_scenarios.json`): invented ₹ amount, guilt claim, tier change —
   all rejected with reasons; fallback used.

## Scene 7 — Cohort Radar: is this account one, or one of forty?  ~60 s

On the same high-risk case, open the **Behaviourally similar accounts** panel.

1. Ten neighbours appear, each with a similarity band and **the features that
   actually drove the match** — say one out loud from the screen (they are
   amount, velocity and rail features, never demographics).
2. Read the panel disclaimer aloud, verbatim. It is the point of the scene:

   > "Behavioural similarity is not proof of a shared owner, handler, or
   > criminal network."

3. "There are no arrows on this panel. A transaction graph needs sender and
   receiver pairs; this dataset is account-level aggregates and doesn't have
   them. So we don't draw one, and we don't imply one."
4. Point at the neighbours' own risk tiers — **unchanged**. "Being in this
   cohort did not raise anyone's score. Not by a little. The probabilities are
   identical to twelve decimal places before and after this panel exists —
   `docs/USP_ACCURACY_REGRESSION.md`, and the measured difference is zero."

If a judge asks *"does it work?"*: for a known mule account, 6.8 of the top 10
behavioural neighbours are also mule accounts — a **77× lift** over the 0.88 %
base rate. For a legitimate account the same lookup returns 0.22 in 10. Read it
from `artifacts/metrics/cohort_radar_retrieval.json`. Say the qualifier too:
**this is a retrieval diagnostic, not classifier accuracy.**

## Scene 8 — What we refuse to conclude (Account-Control card)  ~45 s

Scroll to the **Account control and intent** card under the ProofGraph.

1. Three rows: behavioural mule risk `ASSESSED` · account-control evidence
   `NOT_AVAILABLE` · intent attribution `UNKNOWN`.
2. "Our model can tell you this account behaves like a mule account. It cannot
   tell you who was operating it, or whether the person whose name is on it
   knew. In India a large share of mule accounts belong to students and gig
   workers who were tricked or paid. The activity is real; the culpability is a
   separate finding, and an investigator makes it — not a gradient-boosted tree."
3. Show the seven-item verification checklist — device history, SIM change,
   credential resets, KYC changes, counterparties, customer interview, raw
   trail — all marked `NOT_IN_THIS_DATASET`. "This is what to go and get. We are
   not pretending we already have it."
4. "This card connects to the decision by `REQUIRES_HUMAN_VERIFICATION`, with
   weight zero. Never by `RAISED_BY` — an edge pointing the other way would make
   a limitation look like evidence."

## Scene 9 — Bank impact (Executive Overview)  ~45 s

- Recall at top-25/50/100 alert budgets, read from the **out-of-fold**
  numbers in `artifacts/metrics/lens_stack_oof_v2.json` (0.391 / 0.688 /
  0.828). Say out loud that these are cross-validated, not locked-test,
  figures.
- Precision inside the CRITICAL tier, read from the screen.
- "One thing we will not show you is a locked-test score for this model. The
  locked test was spent on the model we retired. We are not going to touch it
  again to make a nicer slide — the honest position is that we have
  cross-validated evidence and no held-out evidence, and the touch log says
  so."
- "Every number on screen traces to an artifact file with a bootstrap CI, and
  every decision an analyst takes lands in an append-only audit log. No
  recovered-money claims — that would require a live simulation we haven't
  run."

---

Contingencies:
- API down → `make serve-api`; dashboard shows explicit error states (no fake data).
- Ollama already off → Scene 6 step 2 becomes step 1.
- Judge asks for code: show `configs/feature_availability.yaml` (the current
  policy — 9 hard-quarantined columns, 3 conditional, 1 fairness exclusion),
  `src/muleguard/features/firewall.py`, and
  `tests/model/test_leakage_guards.py`. `configs/leakage_quarantine.yaml` is
  the earlier four-entry list and is retained only as history.
