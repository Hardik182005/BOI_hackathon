# Account-Control Ambiguity Guardrail

> **Status:** shipped · post-model guardrail · zero effect on any score
> **Code:** [`src/muleguard/usp/control_attribution.py`](../src/muleguard/usp/control_attribution.py)
> **Tests:** [`tests/unit/test_control_attribution.py`](../tests/unit/test_control_attribution.py) · [`tests/regression/test_proofgraph_unchanged.py`](../tests/regression/test_proofgraph_unchanged.py)
> **Spec:** sections 20–23 of the Cohort Radar / Zero-Regression USP brief

---

## 1. The problem this exists to prevent

A behavioural classifier trained on account-level aggregates can rank how unusual
an account's activity is. It cannot tell you who was sitting at the keyboard, or
whether the person whose name is on the account knew what was happening.

Those are three different questions, and most mule-detection demos blur them. A
model outputs `0.94`, a slide says "mule account", and a *behavioural percentile*
has quietly become an *accusation about a person*. Under Indian mule-account
typologies this is not a pedantic distinction: a large share of mule accounts
belong to students, gig workers and rural customers who rented, sold, or were
tricked out of their credentials. The activity is real. The culpability is a
separate finding, established by an investigator — not by a gradient-boosted tree.

MuleGuard keeps the three apart structurally:

| Question | Who can answer it | What MuleGuard reports |
|---|---|---|
| **Behavioural mule risk** — how unusual is this account's aggregate activity? | the frozen champion classifier | `ASSESSED`, with probability, tier and band |
| **Account-control evidence** — who actually operated the account? | device / SIM / credential / KYC history | `NOT_AVAILABLE` |
| **Intent attribution** — did the holder knowingly participate? | a customer interview and an investigation | `UNKNOWN` |

The dataset behind this project contains the first kind of evidence and none of
the second or third. So the system says so — in the UI, in the API response and
in the ProofGraph — rather than letting silence be read as agreement.

---

## 2. What the guardrail emits

`control_attribution(risk_probability=..., risk_tier=...)` returns a deterministic
card: three concept blocks, a limitation statement, and a verification checklist.
Nothing in it is learned, scored, or ranked.

### 2.1 Limitation statement (fixed string, emitted verbatim)

> The supplied account-level feature dataset supports behavioural-risk assessment
> but does not establish who controlled the account or whether the account holder
> knowingly participated.

### 2.2 The verification checklist (section 21)

Seven items, fixed order, no scoring. Each is marked `NOT_IN_THIS_DATASET` unless
a deployment explicitly declares the feed available via `sources_available`.
Nothing is assumed available by default — assuming would produce a card claiming
evidence exists when the lookup behind it would return nothing.

| # | Check | Why it bears on control |
|---|---|---|
| 1 | Recent device / login ownership history | Shows whether the account was operated from the customer's own device |
| 2 | SIM / mobile-number change | A recent change can indicate loss of control of the second factor |
| 3 | Credential / password-reset history | Resets close to the activity window bear on who held access |
| 4 | KYC / contact-detail changes | Redirected contact details separate the holder from the operator |
| 5 | Beneficiary / counterparty relationships | Establishes whether counterparties are known to the customer |
| 6 | Customer confirmation / interview | The only source that speaks to awareness and intent |
| 7 | Raw transaction trail | Aggregates cannot show the individual movements behind them |

The checklist is framed as **recommended enrichment** — what an investigator
should obtain *before* any high-impact action. It is not a to-do list the system
pretends to have completed.

### 2.3 The risk band

`band` is **derived from the tier the policy already set**, never re-thresholded:

| Tier | Band |
|---|---|
| `CRITICAL_REVIEW`, `URGENT_REVIEW` | `HIGH` |
| `STANDARD_REVIEW` | `MODERATE` |
| `OOD_REVIEW` | `NOT_COMPARABLE` |
| otherwise | `LOW` |

Inventing a second set of cut-offs here would create a number that could disagree
with the decision the system actually made.

---

## 3. Language the system will never emit (section 20)

These categories describe a *person*, which aggregate behaviour cannot establish:

```
witting mule · unwitting mule · coerced mule · criminal · victim
handler · money launderer · accomplice
```

They live in `control_attribution.NEVER_INFERRED` **so the test suite can assert
none of them is ever emitted**, and are deliberately absent from every payload —
naming them in the output would be the same mistake in quotation marks.
[`tests/unit/test_control_attribution.py`](../tests/unit/test_control_attribution.py) walks the full serialised card and the
ProofGraph node and fails on any occurrence.

---

## 4. Actions the system will never take automatically (section 23)

```
FREEZE · FILE_STR · DECLARE_MULE · DECLARE_CRIMINAL · CERTIFY_CLEAN
```

`automatic_actions_permitted` is `[]` on every card, for every account, at every
risk level. There is no configuration that turns it into a non-empty list. The
highest-severity output the system produces is a **review priority with an
evidence pack attached**.

---

## 5. How it attaches to the ProofGraph (section 22)

This is the part that is easy to get subtly wrong, so it is worth being explicit.

```
   evidence  ──RAISED_BY──▶ ┌──────────────────────────┐
   patterns  ──RAISED_BY──▶ │        DECISION          │
                            │  (risk, tier, action)    │
                            └────────────┬─────────────┘
                                         │
                          REQUIRES_HUMAN_VERIFICATION   weight = 0.0
                                         │
                                         ▼
                            ┌──────────────────────────┐
                            │   CONTROL_ATTRIBUTION    │
                            │   modifies_risk = false  │
                            └──────────────────────────┘
```

- The edge runs **decision → control_attribution**, never the reverse.
- The relation is **`REQUIRES_HUMAN_VERIFICATION`**, never `RAISED_BY`.
- The edge weight is **exactly `0.0`**.
- The node is never an edge *source*, so nothing downstream can consume it as
  evidence.

The reason for the direction: an edge pointing the other way would make a
limitation look like evidence. `RAISED_BY` means *"this fact pushed the score"*.
A statement about what we **cannot** establish did not push anything. It is not
evidence for a conclusion — it is a **condition on acting upon one**.

`tests/regression/test_proofgraph_unchanged.py` asserts all four properties, and
additionally rebuilds the courtroom verdict from the core nodes alone and checks
it is identical — i.e. the Model Courtroom never saw this node.

---

## 6. Zero-regression guarantee

The guardrail is **post-model** in the strict sense used throughout this repo:

| Cannot touch | Verified by |
|---|---|
| `risk_probability` | `test_score_invariance_after_usp.py` — max abs diff ≤ `1e-12` |
| `risk_tier` | tier mismatch count `== 0`, row by row |
| `model_votes` | per-model-family raw score walk |
| calibrator | calibrator + threshold hash equality |
| policy thresholds | policy action mismatch count `== 0` |

`affects_model_output` is `False` on every card — and that field is not a claim,
it is checked. The card is produced *after* scoring, *from* the score, and is
excluded from every code path that feeds a metric.

---

## 7. What a judge sees

In the Trinetra UI the card renders beneath the ProofGraph as three stacked rows —
`ASSESSED`, `NOT_AVAILABLE`, `UNKNOWN` — with the limitation statement below and
the seven checklist items collapsed by default.

The intended reaction is the honest one: *this system knows the difference between
"this account behaves like a mule account" and "this person is a mule", and it
refuses to cross that line even when crossing it would look more impressive.*

---

## 8. Related

- [`docs/COHORT_RADAR.md`](COHORT_RADAR.md) — the behavioural-similarity USP, which carries the same restraint about network language
- [`docs/COHORT_RADAR_FAIRNESS_AUDIT.md`](COHORT_RADAR_FAIRNESS_AUDIT.md) — evidence that cohorts group by behaviour, not by people
- [`docs/ASSUMPTIONS_AND_LIMITS.md`](ASSUMPTIONS_AND_LIMITS.md) — dataset-level limitations
- [`docs/FINAL_MODEL_CARD.md`](FINAL_MODEL_CARD.md) — intended use and out-of-scope use
- [`docs/PROOFGRAPH_DESIGN.md`](PROOFGRAPH_DESIGN.md) — the evidence graph this node attaches to
