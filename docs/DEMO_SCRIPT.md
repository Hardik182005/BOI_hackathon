# Demo Script (5–7 minutes)

Preparation: API on :8000, dashboard on :5173, `muleguard.cli.demo --via-api`
already run (populates the queue with the five scenario cases), Ollama
running with any preferred model (optional — Scene 6 kills it live).

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

- Show the red **REJECTED LEAKAGE** bar: with F3912 the model scores 0.94
  PR-AUC; clean it scores 0.61.
- "Our audit didn't stop at the known suspect. It found that the **snapshot
  month column separates the classes perfectly** — every legitimate account is
  the October snapshot, every mule is Sep/Nov/Dec. The file is even sorted by
  label. All of it is quarantined automatically, with the evidence saved."
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

## Scene 7 — Bank impact (Executive Overview)  ~45 s

- Recall at top-25/50/100 alert budgets from the locked test.
- Precision inside the CRITICAL tier.
- "Every number on screen traces to an artifact file with a bootstrap CI, the
  locked test was touched exactly once, and every decision an analyst takes
  lands in an append-only audit log. No recovered-money claims — that would
  require a live simulation we haven't run."

---

Contingencies:
- API down → `make serve-api`; dashboard shows explicit error states (no fake data).
- Ollama already off → Scene 6 step 2 becomes step 1.
- Judge asks for code: show `configs/leakage_quarantine.yaml` and
  `tests/model/test_leakage_guards.py`.
