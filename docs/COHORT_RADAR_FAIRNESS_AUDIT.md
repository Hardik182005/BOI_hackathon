# Cohort Radar — Fairness and Sensitive-Attribute Audit

> **Verdict:** PASS
> **Artifact:** [`artifacts/testing/cohort_radar_fairness.json`](../artifacts/testing/cohort_radar_fairness.json)
> **Reproduce:** `python -m muleguard.cli.cohort_evidence --skip-retrieval`
> **Code:** [`src/muleguard/usp/cohort_audit.py`](../src/muleguard/usp/cohort_audit.py) · `fairness_report()`
> **Spec:** section 32 of the Cohort Radar / Zero-Regression USP brief
> **Radar version:** 1.0 · 120 fingerprint features · weight sum exactly `1.000000000000`

---

## 1. The failure mode this audit is looking for

The Cohort Radar answers one question: *which other accounts behave like this
one?* The honest version of that feature groups accounts by **what they did**.
The dishonest version groups them by **who they are** — and then presents the
result as a behavioural finding.

That failure has a specific shape in Indian retail banking data, and it is worth
naming plainly before showing the numbers:

- **"Students like students."** Age and occupation correlate with transaction
  style. A similarity function that leans on them will hand an investigator a
  cohort of twenty 21-year-olds and call it a behavioural pattern. The
  investigator then sees a "mule cohort" that is really an age bracket.
- **"Same locality = mule cohort."** Area category correlates with branch,
  product mix and cash habits. Leaning on it produces cohorts that track
  geography, which in this country tracks community.

Both would be *statistically defensible* — the features do carry signal — and
both would be indefensible in a review room. So the audit does not stop at
checking the feature list. **Excluding a protected attribute from the fingerprint
is necessary and not sufficient**: a behavioural feature can carry a demographic
signal, and a cohort panel that groups by occupation while claiming to group by
behaviour is the same failure with better paperwork.

This audit therefore measures the **outcome**, not the intent.

---

## 2. Test 1 — Are protected or profile attributes in the fingerprint at all?

The similarity fingerprint is built from the firewall-admitted feature set. Every
profile attribute is checked against the fitted transform's actual feature list.

| Attribute | Column | Firewall class | In fingerprint? |
|---|---|---|---|
| Gender | `F3892` | `fairness_excluded` | **No** |
| Area category | `F3890` | `contextual_only` | **No** |
| Customer occupation | `F3891` | `contextual_only` | **No** |
| Age in years | `F3894` | `contextual_only` | **No** |

```
"profile_fields_in_fingerprint": []
```

`F3892` (gender) is **hard-excluded by the fairness policy** and can never enter.
`F3890` / `F3891` / `F3894` are classed `contextual_only` — admissible as display
context for a human reviewer, never as a matching signal — and the Radar honours
that classification rather than re-deciding it locally. The classes are read from
the **live firewall config**, not hardcoded here, so a change to the policy
changes this audit rather than silently diverging from it.

**Result: PASS.** None of the four is present among the 120 fingerprint features.

---

## 3. Test 2 — Do behavioural features actually dominate the similarity?

Section 32 requires that behavioural features dominate matching. Absence of
demographics is not the same as dominance of behaviour, so the weight mass is
apportioned by the feature dictionary's `availability_class`:

| Availability class | Features | Total weight | Share |
|---|---:|---:|---:|
| `BEHAVIORAL` | 114 | 0.940425 | **94.04 %** |
| `ALERT_CONTEXT` | 2 | 0.038759 | 3.88 % |
| `MG_META` (engineered behavioural) | 3 | 0.020291 | 2.03 % |
| `PROFILE` | 1 | 0.000525 | **0.05 %** |
| **Total** | **120** | **1.000000** | **100.00 %** |

**96.07 %** of the similarity mass is behavioural once the three engineered
`MG_*` meta-features (pass-through velocity, rail fragmentation, alert
convergence — all derived from transaction behaviour) are counted with the
behavioural block.

The single `PROFILE` feature is `F3886` (*product name of the corresponding
account*) — the sole categorical in the fingerprint, carrying **0.0525 %** of the
weight. It is a product-type field, not a person-type field, and at that weight
it cannot move a ranking: two accounts sharing a product code gain less
similarity than a rounding difference on any of the top thirty behavioural
features.

### Top ten features by similarity weight

| Rank | Feature | Description | Class | Weight |
|---:|---|---|---|---:|
| 1 | `F1815` | Non-cash non-cheque **debit** amount, last 31D | BEHAVIORAL | 0.105505 |
| 2 | `F1813` | Non-cash non-cheque total amount, last 31D | BEHAVIORAL | 0.061955 |
| 3 | `F1057` | Max non-cash non-cheque total amount, last 14D | BEHAVIORAL | 0.061013 |
| 4 | `F1166` | Max non-cash non-cheque **credit** amount, last 31D | BEHAVIORAL | 0.057460 |
| 5 | `F3908` | Alert description flag | ALERT_CONTEXT | 0.037426 |
| 6 | `F1597` | Non-cash non-cheque total amount, last 7D | BEHAVIORAL | 0.036542 |
| 7 | `F3105` | Deviation of total vs average: UPI credit txns, 7D→14D | BEHAVIORAL | 0.025339 |
| 8 | `F3805` | Total transaction amount, last 14D | BEHAVIORAL | 0.024363 |
| 9 | `F2397` | Deviation of mobile-banking total txns, 14D→31D | BEHAVIORAL | 0.022122 |
| 10 | `F2029` | Average non-cash non-cheque total amount, last 14D | BEHAVIORAL | 0.018919 |

Top-10 weight share: **0.4506**. Every one of them is a money-movement quantity
or a deviation-from-own-baseline term. Nine of ten are `BEHAVIORAL`; none is a
demographic. This is what a mule-behaviour fingerprint is supposed to look like:
*amount, velocity, rail, and change-against-your-own-history.*

**Result: PASS.**

---

## 4. Test 3 — Empirically, do cohorts cluster by demography anyway?

This is the test that actually settles it, because it is the only one a
correlated-proxy feature cannot pass by accident.

**Method.** 300 seeded reference accounts (`seed=42`), their top-10 cohorts —
3,000 neighbour pairs — measured against **3,000 uniformly random reference
pairs** drawn from the same population. For each profile attribute, how often do
the two accounts in a pair agree? Age is bucketed by decade; the others are
compared exactly. A ratio near **1.0** means the attribute is not organising the
neighbourhoods.

| Attribute | Neighbour concordance | Random-pair concordance | Ratio to chance |
|---|---:|---:|---:|
| Customer occupation (`F3891`) | 0.2707 | 0.2713 | **0.998** |
| Age decade (`F3894`) | 0.1963 | 0.1807 | **1.087** |
| Area category (`F3890`) | 0.2897 | 0.2607 | **1.111** |
| Gender (`F3892`) | 0.4713 | 0.4210 | **1.120** |

Read the first row carefully, because it is the direct refutation of the
"students like students" failure: **two cohort neighbours agree on occupation
slightly less often than two accounts picked at random.** Occupation carries
literally no organising power in this similarity space — the ratio is 0.998.

The other three sit between 1.09× and 1.12× chance. Those are real but small
residual correlations, and this audit reports them rather than rounding them away:

- **They cannot be a matching effect**, because none of the three fields is in the
  fingerprint. There is no path by which the similarity function could read them.
- **They are a behavioural-correlation effect.** Accounts with similar
  transaction amounts, rails and velocities are somewhat more likely to share a
  demographic bucket, because real banking behaviour is not independent of who is
  doing the banking. A 1.11× lift on area category means roughly 29 pairs in 100
  share an area against 26 in 100 at chance.
- **For scale:** the Radar's *behavioural* retrieval lift over base mule
  prevalence is **77.47×** (see [`docs/COHORT_RADAR.md`](COHORT_RADAR.md) §
  retrieval evaluation). A 1.11× demographic echo against a 77× behavioural
  signal is not what the cohorts are made of.

Gender at 1.120× is the largest of the four and deserves a sentence of its own:
`F3892` is `fairness_excluded`, i.e. the strictest class the firewall has. It is
absent from the fingerprint, absent from the classifier's selected features, and
absent from every scoring path. The residual is a downstream correlation in the
underlying transaction behaviour, and the correct response to it is the one taken
here — measure it, publish it, and do not use the field.

**Result: PASS.** No attribute exceeds 1.12× chance; occupation is at chance.

---

## 5. What is *not* claimed

- This audit does **not** claim the underlying transaction data is free of
  demographic correlation. It plainly is not, and section 4 quantifies it.
- It does **not** claim the cohorts are demographically uniform in outcome — that
  would require the group-wise error-rate analysis in
  [`docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`](FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md),
  which covers the **classifier**. This document covers the **Radar**.
- It does **not** claim cohort membership means anything about a person. The
  Radar returns *behaviourally similar accounts* and says so in the panel
  disclaimer; see [`docs/ACCOUNT_CONTROL_AMBIGUITY.md`](ACCOUNT_CONTROL_AMBIGUITY.md)
  for why that distinction is enforced structurally rather than by wording.

---

## 6. Standing guarantees

| Guarantee | Enforced by |
|---|---|
| `fairness_excluded` fields can never enter the fingerprint | `cohort_audit.leakage_report()` check 4, release-blocking |
| `contextual_only` fields can never enter the fingerprint | firewall admission, shared with the classifier path |
| The firewall config is read live, never hardcoded | `firewall.config()` at audit time; quarantine hash compared |
| Cohort membership never changes a risk score | `test_score_invariance_after_usp.py`, ≤ 1e-12 |
| Cohort membership never auto-escalates a neighbour | `test_false_positive_safety.py`, `automatic_actions_permitted == []` |

**This audit is release-blocking.** `fairness_report()` emits `FAIL` if any
profile attribute appears in the fingerprint, and
`python -m muleguard.cli.cohort_evidence` exits non-zero on `FAIL`.

---

## 7. Related

- [`docs/COHORT_RADAR.md`](COHORT_RADAR.md) — how the similarity transform is built and evaluated
- [`docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`](FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md) — the classifier's fairness audit
- [`docs/FEATURE_AVAILABILITY_FIREWALL.md`](FEATURE_AVAILABILITY_FIREWALL.md) — the admission policy both paths share
- [`docs/ACCOUNT_CONTROL_AMBIGUITY.md`](ACCOUNT_CONTROL_AMBIGUITY.md) — why behaviour is never converted into an accusation
