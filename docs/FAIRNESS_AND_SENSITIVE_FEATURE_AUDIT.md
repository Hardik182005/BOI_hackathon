# Fairness and Sensitive-Feature Audit

Master prompt §24. Measured on the development out-of-fold predictions of the
served champion `xgboost_top_120` (7,264 rows, 64 positives), at the top-100
alert budget.

---

## 1. The four sensitive columns

`Description.xlsx` marks four columns sensitive. All four are `PROFILE` class.

| Column | Variable | Meaning | Kind | In the champion's 120? | Disposition |
|---|---|---|---|:--:|---|
| `F3892` | `GENDER` | customer gender | gender | **No** | **`EXCLUDED_BY_FAIRNESS_POLICY`** |
| `F3890` | `AREA_CATEGORY` | area category of the customer | geography | **No** | admissible as context, not selected |
| `F3891` | `CUST_OCCP` | occupation code | occupation | **No** | admissible as context, not selected |
| `F3894` | `AGE_IN_YRS` | customer age as of alert date | age | **No** | admissible as context, not selected |

**None of the four is a feature of the served model.** Only one `PROFILE` column
entered the champion's 120 at all: `F3886 PRODUCT_NAME`, the product the account
is held under.

### Gender is excluded by policy, not by accident

`F3892` is the only one hard-excluded. It is not quarantined for leakage — it is
excluded because it is a protected attribute with **no defensible causal link to
mule behaviour**, and the only thing it can contribute to a mule model is proxy
discrimination.

The firewall keeps this separate from the leakage quarantine:
`fairness.excluded_by_default` is its own config section with its own flag
(`include_gender`, default `False`, requiring a documented, approved
justification to flip). Conflating the two would mean a future leakage review
could quietly readmit a protected attribute as a side effect.

Age, occupation and area are handled differently and deliberately. They are
**not** hard-excluded — they are legitimate risk context in AML work, and a
blanket ban would be a fairness gesture rather than a fairness control. They are
available to the firewall as `PROFILE` columns; stability selection simply did
not choose any of them. That is the honest description, and it is stronger than a
ban: the model did not want them.

---

## 2. Measured disparity, at the top-100 budget

Excluding an attribute from the feature list does not guarantee the model is not
reproducing it through correlated behaviour. So the output was measured directly.

Overall selection rate: 100 / 7,264 = **1.38 %**. Overall Recall@100 = **0.828**.

### Selection rate (who gets flagged)

| Group | n | Selection rate | vs overall | Group prevalence |
|---|---:|---:|---:|---:|
| **Gender** | | | | |
| M | 4,003 | 1.90 % | 1.38× | 1.27 % |
| F | 1,140 | 1.40 % | 1.02× | 0.88 % |
| (not recorded) | 2,072 | 0.39 % | 0.28× | 0.14 % |
| **Area** | | | | |
| R (rural) | 1,595 | 2.32 % | **1.69×** | 1.38 % |
| SU (semi-urban) | 1,915 | 1.93 % | 1.40× | 0.89 % |
| U (urban) | 1,403 | 0.71 % | 0.52× | 0.64 % |
| M (metro) | 2,351 | 0.68 % | 0.49× | 0.68 % |
| **Occupation** | | | | |
| student | 944 | 3.18 % | **2.31×** | 1.59 % |
| agriculture | 883 | 2.72 % | **1.97×** | 1.47 % |
| salaried | 1,534 | 0.98 % | 0.71× | 0.85 % |
| self-employed | 3,151 | 0.89 % | 0.65× | 0.63 % |
| housewife | 540 | 0.56 % | 0.40× | 0.56 % |
| **Age** | | | | |
| 25–34 | 2,062 | 2.04 % | **1.48×** | 1.21 % |
| < 25 | 1,755 | 1.31 % | 0.95× | 0.97 % |
| 35–44 | 1,626 | 1.29 % | 0.94× | 0.68 % |
| 45–54 | 896 | 1.00 % | 0.73× | 0.67 % |
| 55–64 | 550 | 0.55 % | 0.40× | 0.55 % |
| 65+ | 375 | 0.53 % | 0.39× | 0.53 % |

**Selection rates are not equal across groups**, and we are not going to present
them as if they were.

The key column is the last one. In every case the group with the higher selection
rate also has the higher **actual mule prevalence** in the labelled data —
students 1.59 %, agriculture 1.47 %, rural 1.38 %, ages 25–34 1.21 %, against a
book average of 0.88 %. The model is tracking a real difference in base rates,
not an attribute it was never given.

That is an explanation, not an exoneration. Mule recruitment genuinely targets
students and rural account holders, so a detector that found no difference would
be failing. But base rates in a labelled extract are themselves partly a product
of where the bank looked, and this audit cannot separate those two causes. **The
disparity is published so that a human can make that judgement, and the numbers
are on the record.**

---

## 3. Equal opportunity: does the model catch mules equally well?

Selection rate is the wrong headline metric for a review queue. The question that
matters to a customer is: *if I am a mule, am I equally likely to be caught, and
if I am not, am I equally likely to be left alone?* Recall by group answers the
first.

Recall@100 by group (overall 0.828):

| Group | Positives | Recall@100 |
|---|---:|---:|
| **Gender** M | 51 | 0.902 |
| **Gender** F | 10 | 0.700 |
| **Area** R | 22 | 0.955 |
| **Area** SU | 17 | 0.882 |
| **Area** M | 16 | 0.688 |
| **Area** U | 9 | 0.667 |
| **Occupation** student | 15 | 1.000 |
| **Occupation** agriculture | 13 | 0.923 |
| **Occupation** self-employed | 20 | 0.900 |
| **Occupation** salaried | 13 | 0.538 |
| **Age** 25–34 | 25 | 0.920 |
| **Age** < 25 | 17 | 0.824 |
| **Age** 45–54 | 6 | 0.833 |
| **Age** 35–44 | 11 | 0.727 |

Recall ranges from 0.54 (salaried) to 1.00 (student).

**The statistical caveat is decisive and must be read before the numbers.** These
groups contain between 3 and 51 positives. The salaried group's 0.538 is 7 of 13
— a single case moves it by 0.077, and the 95 % interval on 7/13 spans roughly
0.29 to 0.79. **With 64 positives in total, no group-level fairness metric here is
statistically meaningful.** They are reported because publishing an
under-powered measurement with its power stated is more honest than omitting the
audit and claiming there was nothing to see.

Groups with fewer than 3 positives are not reported at all, and no group-level
metric was used to make any modelling decision.

---

## 4. What was done about it

Nothing automatic — deliberately.

| Not done | Why |
|---|---|
| per-group threshold adjustment | with 3–51 positives per group, a group-specific threshold would be fitted to noise, and it would mean two customers with identical behaviour receiving different treatment because of a protected attribute. That is a fairness harm, not a fix |
| reweighting to equalise selection rates | would suppress alerts in genuinely higher-prevalence segments — a fairness metric improved at the cost of undetected mules in the populations most targeted by recruitment |
| dropping age/occupation/area entirely | they are not in the model already; removing them from the *registry* would only hide the ability to audit |

What **is** done:

1. **Gender is excluded from the model outright**, by policy, with a flag that
   requires documented justification to change.
2. **The disparity is measured and published** — this document — rather than left
   for someone else to discover.
3. **Every alert is reviewed by a human.** There is no automatic freezing and no
   automatic adverse action anywhere in the system. A disparity in a *review
   queue* is materially different from a disparity in an *account closure*.
4. **Every alert carries a defence panel.** A reviewer sees the exculpatory
   evidence alongside the incriminating evidence, so a case supported only by
   weak signals can be closed on the same screen.
5. **No node in a ProofGraph can cite a protected attribute**, because no
   protected attribute is in the model — every node's `source` must name a real
   consumed column.

---

## 5. Proxy risk

Excluding gender does not prevent a model from reconstructing it from correlated
behaviour. We state the limit honestly: **we did not train a gender-prediction
model on the admitted feature set, so we cannot bound proxy leakage
quantitatively.**

What can be said:

- The champion's 120 features are overwhelmingly **transaction-rail aggregates**
  (`NON_CASH_CHQ` 46, `UPI` + `UPI_XFER` 26, `TOTAL_ALL_RAILS` 11) and more than
  half are deviation or ratio transforms measuring an account **against its own
  baseline**, not against a population. A within-account deviation is
  structurally less able to encode a demographic than a level would be.
- The single non-behavioural feature is `F3886 PRODUCT_NAME` — the banking
  product — which is a commercial rather than demographic attribute, though it is
  not independent of customer segment.
- `MG_PROFILE_MISMATCH` judges an account against its **own occupation and
  segment peer groups**, using bank-supplied deviation columns. This is peer
  comparison rather than demographic scoring, but it is the place where a
  demographic could most plausibly enter, and it is named here for that reason.
  It did not enter the champion's 120.

Recommended before production deployment: fit a classifier for each protected
attribute on the admitted feature set and report its AUC as a direct proxy-leakage
bound. That work is **not** in this submission and is listed as a gap rather than
implied to be done.

---

## 6. Language and dignity guardrails

Fairness is also about what the system says.

`assert_language_safe` walks every serialised payload and rejects `guilty`,
`criminal`, `fraudster`, `permanently_safe`, `certified_clean`, `auto_freeze`,
`confirmed mule`, `proven mule`. The ACCOUNT root node states it in the payload
itself:

> This is a behavioural ranking for human review, not a finding about the account
> holder.

The five permitted verdicts describe work to be done, never a characterisation of
a person. No customer identifier enters the counterfactual-twin index — twins
carry opaque references only. No raw PII is written to logs.

---

## 7. Summary

| Question | Answer |
|---|---|
| Is any protected attribute a model feature? | **No.** None of `F3890`/`F3891`/`F3892`/`F3894` is among the champion's 120 |
| Is gender available at all? | No — hard-excluded by fairness policy, flag defaults off |
| Are selection rates equal across groups? | **No.** Up to 2.31× for students, 1.69× rural, 1.48× ages 25–34 |
| Does the disparity track real base rates? | Yes in every case measured — but base rates in this extract are themselves partly an artefact of where the bank looked |
| Is recall equal across groups? | Ranges 0.54–1.00, and with 3–51 positives per group **none of it is statistically meaningful** |
| Were group metrics used to tune anything? | No |
| Is proxy leakage bounded? | **No** — measuring it is an open gap, stated as such |
| Can a disparity cause automatic adverse action? | No — human review is required for every action; nothing is automatic |
