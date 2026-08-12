# False-Positive Control Report

Master prompt §16 and §22, addendum UPDATE 9.

The winning principle names four things, and this is the third:
*better hidden-validation generalization + auditable proof + **false-positive
protection** + excellent UX.*

---

## 1. Why this is a first-class concern, not a footnote

Prevalence is **0.89 %**. At any usable alert budget, most of what an analyst
opens will be a legitimate customer.

The arithmetic is unforgiving. On the development OOF at a top-100 budget the
champion catches 53 of 64 mules — and hands the team **47 legitimate accounts to
investigate**. Each of those is a real customer whose account gets scrutinised,
whose transactions get questioned, and whose bank now has a file on them.

Two consequences follow, and both are operational rather than statistical:

1. **A queue that is mostly wrong trains reviewers to stop reading.** Alert
   fatigue is not a soft problem; it is the mechanism by which a good model
   becomes a rubber stamp.
2. **A false positive has a victim.** It is not a symmetric error against a
   missed mule — it is a different kind of harm, to a different person.

So false-positive control is built into five separate layers, and none of them
works by lowering a score.

---

## 2. Layer 1 — never accuse

The system produces **review recommendations**, never findings.

`FORBIDDEN_TERMS` are rejected mechanically by `assert_language_safe`, which
walks every serialised payload before it leaves the process: `guilty`,
`criminal`, `fraudster`, `permanently_safe`, `certified_clean`, `auto_freeze`,
`confirmed mule`, `proven mule`. A payload containing any of them raises
`UnsafeLanguage` and is never served.

The five permitted verdicts all describe **work for a person to do**:
`REVIEW_RECOMMENDED`, `ENHANCED_REVIEW_RECOMMENDED`, `MONITOR_ONLY`,
`INSUFFICIENT_EVIDENCE_TO_ESCALATE`, `NO_ACTION_INDICATED`.

There is **no automatic freezing anywhere in this system**. High-impact actions
require human approval. A false positive therefore costs a review, not a frozen
account.

---

## 3. Layer 2 — the ProofGraph defence panel

Covered fully in `PROOFGRAPH_DESIGN.md`; its role here is the important one.

Every alert ships with a defence column of equal visual and structural weight:
negative-SHAP features, plus six measured structural doubts (model disagreement,
non-decisive conformal set, out-of-distribution input, unremarkable anomaly
percentile, missing deciding inputs, verifier declining to confirm).

**This is the false-positive control that acts at the moment of decision.** A
reviewer looking at a weak case can see that it is weak, from the same screen, in
the same format, without leaving the alert. A reviewer who can only see
incriminating evidence has no mechanism for dismissing a case except intuition.

The courtroom encodes it: a contested case — any uncertainty node present, or
model families in disagreement — **cannot** receive
`ENHANCED_REVIEW_RECOMMENDED`, no matter how lopsided the evidence balance.
Observed live on `CASE-18A744455E`: risk 1.000, evidence balance 0.9156, one
uncertainty node, verdict downgraded to `REVIEW_RECOMMENDED`.

---

## 4. Layer 3 — the Merchant Legitimacy Verifier (UPDATE 9)

### The competitor pattern we refused

A common approach is to multiply a merchant's risk score by a hand-picked
constant — `score × 0.70`. UPDATE 9 rules it out, and the reasons are worth
stating because they generalise:

- **It is unmeasurable.** There is no experiment that validates 0.70, and no way
  to tell whether it helped.
- **It silently destroys calibration.** A calibrated probability multiplied by an
  arbitrary constant is no longer a probability, and every downstream threshold
  is now wrong in an undocumented way.
- **It hides the adjustment.** Nobody reviewing an alert can see that it happened.

### What we built instead

A **learned** model, trained on firewall-admitted merchant and business-context
features (`artifacts/metrics/merchant_verifier_v2.json`):

| | |
|---|---:|
| View | `E_profile_merchant` |
| Features | 1,331 (from 1,376 in the view) |
| Dev rows / positives | 7,264 / 64 |
| OOF PR-AUC | 0.63604 |
| OOF ROC-AUC | 0.97461 |
| Lift over prevalence | 72.19× |

### The exclusion that makes it evidence

`TOTAL_ALL_RAILS` — 45 columns of aggregate debit/credit volume across every
payment rail — was **dropped from this model even though it would raise its
PR-AUC**:

> Exculpatory evidence is only worth anything if it is independent of the
> accusation.

Total-rail volume is the primary *mule* signal, not evidence of a legitimate
business. A verifier that leaned on it would be re-deriving the risk score and
then presenting the result as an independent second opinion, which is worse than
having no second opinion at all.

### Does the band carry real information?

| Band | Accounts | Mules | Prevalence |
|---|---:|---:|---:|
| `STRONG_BUSINESS_EVIDENCE` | 1,452 | **0** | **0.0 %** |
| `SOME_BUSINESS_EVIDENCE` | 2,180 | — | — |
| `LITTLE_BUSINESS_EVIDENCE` | 3,632 | — | — |
| book overall | 9,082 | 81 | 0.881 % |

Measured on **out-of-fold predictions**. Zero mules among 1,452 accounts with
strong business evidence, against a book prevalence of 0.881 %. Verdict:
`USEFUL_AS_EXCULPATORY_CONTEXT`.

Top evidence drivers are recognisably business-like: standing instructions,
fees-and-charges patterns, loan servicing, POS activity, GST.

### What it may and may not do

| May | May **never** |
|---|---|
| lower the confidence attached to an **automatic escalation** | modify the calibrated risk, the model score, or any threshold |
| appear as exculpatory evidence in the ProofGraph defence panel | remove an account from a review queue |
| | act as a merchant whitelist |
| | apply a fixed multiplier to any score |

It influences **auto-escalation confidence**, not the probability. Every
adjustment is auditable: the band, the drivers, and the fact that it fired all
appear in the case payload. Zero mules in the strong band is a *measured
regularity*, not a guarantee — which is exactly why the band informs confidence
and never removes an account from review.

---

## 5. Layer 4 — hard-negative mining and conformal abstention

**144 hard negatives** were mined from the development OOF: legitimate accounts
the model scored highly. They are the look-alikes — the accounts whose behaviour
genuinely resembles a mule's — and they are what the verifier is trained to
separate from true positives.

**Mondrian conformal prediction** at 90 % target coverage
(`lens_stack_oof_v2.json`):

| | |
|---|---:|
| positive coverage | 0.9375 |
| negative coverage | 0.9474 |
| abstention rate | **0.0469** |
| share high | 0.0603 |
| share low | 0.8928 |

The system **abstains on 4.7 %** of cases rather than issuing a confident wrong
answer. An abstention is not a failure to decide — it is a decision to route to a
human with the ambiguity stated, and it appears in the ProofGraph as a
`doubt:conformal` node.

An **Isolation Forest** that never saw the labels runs alongside. When it finds
an account unremarkable (percentile < 80) that fact enters the defence side —
independent corroboration that a high supervised score may be an artefact.

---

## 6. Layer 5 — calibration and tiered thresholds

A probability that means what it says is itself a false-positive control: it lets
a reviewer make a proportionate decision instead of reacting to a rank.

Platt calibration was selected over isotonic under a rule fixed in advance —
*isotonic must beat Platt by ≥ 2 % relative on **both** Brier and ECE; with few
positives, prefer the simpler calibrator*:

| Calibrator | Brier | ECE |
|---|---:|---:|
| **platt** (selected) | **0.003128** | **0.001489** |
| isotonic | 0.003159 | 0.001698 |

Frozen policy thresholds (`policy_version 1.0`): critical 0.93385, urgent
0.09774, standard 0.01318, anomaly escalation at the 99th percentile.

Tiering means a marginal case does not enter the same queue as a strong one.
Locked-test tier outcomes under the retired bundle showed 100 % precision in
`CRITICAL_REVIEW` against 0.23 % in `MONITOR` — the separation the tiers exist to
produce.

### Operating points, measured

Development OOF (`lens_stack_oof_v2.json`):

| Budget | Recall | Precision | TPs |
|---|---:|---:|---:|
| top 25 | 0.391 | **1.000** | 25 |
| top 50 | 0.688 | 0.880 | 44 |
| top 73 | 0.766 | 0.671 | 49 |
| top 100 | 0.828 | 0.530 | 53 |

| FPR target | Recall | FP per 1,000 legitimate |
|---|---:|---:|
| 0.5 % | 0.797 | 5.0 |
| 1.0 % | 0.844 | 10.0 |

**25 alerts, 25 mules, zero false positives** is the operating point a team can
actually work. The precision curve is published so the budget can be chosen
against a real trade-off rather than a slogan.

Locked test: PR-AUC 0.7263, **lift 77.7×** over prevalence, Recall@100 0.824.

---

## 7. What we refused to do

| Refused | Why |
|---|---|
| `score × 0.70` for merchants | UPDATE 9 — unmeasurable, decalibrating, invisible |
| any hard-coded whitelist | one compromised whitelisted account and the control is a blind spot |
| raising risk when models disagree | UPDATE 6 — disagreement is uncertainty; it widens the review band and never raises a score |
| auto-adapting thresholds | UPDATE 7 — thresholds that move on their own cannot be audited after an incident |
| automatic freezing | no code path exists; human approval is required for high-impact actions |
| suppressing alerts on a low compatibility score | UPDATE 3 — the score warns, it never modifies a prediction |

---

## 8. Label-noise audit (UPDATE 5)

`artifacts/metrics/label_noise_audit_v2.json` identifies high-scoring negatives
by consensus across model families — candidates for *"labelled legitimate,
behaves like a mule"*.

Its guarantees are as important as its findings:

> **Do not automatically flip labels. Do not delete difficult mule examples.**

Neither happens. The audit is published as evidence for a human to consider, and
some of those rows are almost certainly genuine false negatives in the ground
truth rather than model errors — which is precisely why an automated system must
not act on them.

Full detail: `docs/LABEL_NOISE_AUDIT.md`.

---

## 9. Summary

| Layer | Mechanism | Evidence |
|---|---|---|
| 1 | no accusatory language, no automatic action | `assert_language_safe`, 5 review-only verdicts |
| 2 | ProofGraph defence panel, contested cases downgraded | live case: balance 0.92 → `REVIEW_RECOMMENDED` |
| 3 | learned Merchant Legitimacy Verifier | 0 mules in 1,452 strong-evidence accounts vs 0.881 % book |
| 4 | hard negatives + conformal abstention + unsupervised anomaly | 144 mined; 4.7 % abstention at 90 % coverage |
| 5 | calibration + tiered thresholds | Brier 0.0031, ECE 0.0015; 25 alerts → 25 mules, 0 FP |

None of these layers lowers a risk score. Every one of them gives a human a
better reason to close a case.
