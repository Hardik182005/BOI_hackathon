# Fairness Validation: Sensitive-Attribute Ablation

Master prompt §31. The companion document, `docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`
(§24), **measures** what the shipped model does to each group. This document
**tests** the position that produced it: does giving the model a protected or
demographic attribute improve detection enough to justify using it?

Produced by `python -m muleguard.cli.fairness_ablation` (`make fairness-ablation`).
Raw payload: `artifacts/metrics/fairness_ablation.json`. Runtime 2,269 s.

**The locked test set was not read.** Every number here comes from the
development split (7,264 rows, 64 positives) through repeated nested
cross-validation. The payload records `design.locked_test_read: false`.

---

## 1. The four sensitive columns and their disposition

`Description.xlsx` marks four columns sensitive. All four are `PROFILE` class.
None is a feature of the served champion `xgboost_top_120`.

| Column | Variable | Kind | Disposition | Governed by |
|---|---|---|---|---|
| `F3892` | `GENDER` | gender | **`EXCLUDED_BY_FAIRNESS_POLICY`** — never reaches the frame | `configs/feature_availability.yaml` → `fairness.excluded_by_default`, enforced at `features/firewall.py:143` |
| `F3890` | `AREA_CATEGORY` | geography | admissible; offered to selection, not chosen | `fairness.contextual_only` |
| `F3891` | `CUST_OCCP` | occupation | admissible; offered to selection, not chosen | `fairness.contextual_only` |
| `F3894` | `AGE_IN_YRS` | age | admissible; offered to selection, not chosen | `fairness.contextual_only` |

The distinction is deliberate and is the thing this run was built to test. Gender
is *banned*. Age, occupation and area are *available and were simply not
selected*. The first is a policy claim that needs justifying; the second is an
empirical claim that can be checked — and both are checked below.

---

## 2. Design

Eight arms, all scored on the **same 15 outer folds** (3 repeats × 5 folds,
4 inner folds for selection), the same `xgboost` at default configuration, the
same seed. The per-fold ranking is computed once inside outer-train and **shared
by every arm** — no arm re-ranks, so any two arms differ only in which columns
were taken from an identical ranking.

| Arm | What it does |
|---|---|
| `baseline` | the shipped pool: top 120 of the fold's ranking |
| `sensitive_excluded` | the three in-frame sensitive columns struck from the ranking before the top 120 is taken |
| `geography_forced` / `occupation_forced` / `age_forced` | one sensitive column forced into the pool |
| `gender_forced` | `F3892` injected (see §4) |
| `sensitive_forced` | all three in-frame columns forced |
| `all_four_forced` | all three plus injected gender |

**Forcing adds rather than swaps.** A forced column becomes feature 121 (or 122,
123, 124) rather than displacing the 120th-ranked feature. A swap would remove a
feature the selector wanted at the same moment it adds one it did not, and a null
result would then be unreadable — you could not tell whether the attribute failed
to help or whether the displaced feature was simply better.

Metric is average precision per fold. Comparison is `models.paired.paired_report`
against `baseline` on byte-identical folds, so the **per-fold difference** is the
unit of evidence, not the difference of two means.

### The decision rule, fixed before the run

> A sensitive attribute is re-admitted only if its arm's **mean paired difference
> is positive AND the sign test rejects at 0.05**. Anything else — including a
> positive mean the sign test cannot separate from noise — is `KEEP_EXCLUSION`.

This is recorded verbatim in the payload at `decision_rule.text`. It was fixed
before any arm was scored, which is the only reason the result below can be
called a test rather than a description.

---

## 3. Results

**Verdict: `KEEP_EXCLUSION`. Arms meeting the rule: none.**

| Arm | PR-AUC | ± | Paired diff | 95 % CI | sign p | Wilcoxon p | t p | up/down/tied | MDE80 |
|---|---:|---:|---:|---|---:|---:|---:|:--:|---:|
| `baseline` | 0.78161 | 0.10031 | — | — | — | — | — | — | — |
| `sensitive_excluded` | 0.78149 | 0.10011 | −0.00011 | [−0.00036, +0.00013] | 1.000 | 0.317 | 0.334 | 0/1/14 | 0.00032 |
| `geography_forced` | 0.77658 | 0.10468 | −0.00503 | [−0.01233, +0.00228] | **0.035** | 0.135 | 0.162 | 3/12/0 | 0.00954 |
| `occupation_forced` | 0.77712 | 0.10215 | −0.00449 | [−0.01080, +0.00183] | 0.118 | 0.121 | 0.150 | 4/11/0 | 0.00825 |
| `age_forced` | 0.78310 | 0.09805 | **+0.00150** | [−0.00622, +0.00921] | 0.424 | 0.397 | 0.684 | 9/5/1 | 0.01007 |
| `gender_forced` | 0.77679 | 0.10031 | −0.00481 | [−0.01123, +0.00160] | 0.118 | 0.064 | 0.130 | 4/11/0 | 0.00838 |
| `sensitive_forced` | 0.77608 | 0.09971 | −0.00553 | [−0.01182, +0.00076] | 0.302 | 0.083 | 0.080 | 5/10/0 | 0.00822 |
| `all_four_forced` | 0.78039 | 0.09496 | −0.00122 | [−0.00892, +0.00649] | 1.000 | 0.890 | 0.739 | 8/7/0 | 0.01006 |

Three tests are reported together for every arm because they can disagree, and a
disagreement is information. Read `geography_forced` carefully: its sign test
rejects at 0.035 while the other two do not — but **its mean is negative**. The
significant sign test there is evidence *against* geography, not for it. It fails
the decision rule on the first clause, not the second.

`age_forced` is the only arm with a positive mean (+0.00150). It gains on 9 folds,
loses on 5, ties on 1, and its confidence interval spans zero by a factor of four.
It fails the rule on the second clause.

### What the selector did on its own

Before any arm was scored, the run recorded where each sensitive column landed in
each fold's ranking:

| Column | Folds it entered the ranking at all | Folds inside the top 120 | Best rank seen |
|---|:--:|:--:|---:|
| `F3890` `AREA_CATEGORY` | 1 / 15 | 0 | 514 |
| `F3891` `CUST_OCCP` | 0 / 15 | 0 | — |
| `F3894` `AGE_IN_YRS` | 10 / 15 | **1** | 69 |

Out of roughly 508–530 columns that survive to the ranking in a given fold.
Occupation was never ranked at all; area was ranked once, at 514. Age is the only
one the selector takes seriously — it is ranked in two thirds of folds, and in
**one** fold it reached rank 69 and was therefore already in that fold's baseline
120.

That single fold is the same fold that shows up twice elsewhere in the results:
it is the one fold `sensitive_excluded` changes, and the one fold `age_forced`
leaves untouched (14 of 15 pools changed, arm sizes `120, 121`). The three
numbers are the same event seen from three directions, which is a useful check
that the arms are constructed the way the design says they are.

### What these results do and do not establish

**They do establish**, on this data and this design:

- The fairness exclusion is close to free. `sensitive_excluded` changed the
  feature pool in **1 of 15 folds** and cost 0.00011 AP. It is nearly a no-op
  because the selector almost never wanted the attributes in the first place; the
  policy is not paying for itself with lost detection.
- Gender does not help. `gender_forced` is **−0.00481**, i.e. the model is
  slightly *worse* with it. The policy and the measurement point the same way.
  This is the strongest available answer to the objection that the exclusion is
  a principled sacrifice of accuracy: on this evidence there is no sacrifice.
- No arm cleared the bar that was set before the run.

**They do not establish**:

- **That any of these attributes has no effect.** The MDE80 column is the point:
  this design could only reliably detect effects of 0.0003 (for
  `sensitive_excluded`, where 14 folds are byte-identical) up to about 0.010 AP
  for the forced arms. An attribute worth +0.005 AP would very likely have gone
  undetected here. "No effect was detected" is the claim; "there is no effect"
  is not.
- **That gender would not be selected if offered.** This run injects gender into
  a fixed pool; it does not re-run stability selection with gender admitted. See
  §4.
- **Anything about the locked test set**, which was not read.
- **Anything about proxy leakage.** A model that reconstructs gender from
  behaviour would show exactly this result — no gain from the explicit column,
  because the information is already there. Bounding proxy leakage remains the
  open gap recorded in §5 of the companion audit.

---

## 4. Gender: policy, flag, and what would flip it

`F3892` is removed by the firewall before the model frame exists. It is not
quarantined for leakage — it is excluded because it is a protected attribute with
no defensible causal link to mule behaviour.

**The flag.** `configs/feature_availability.yaml` → `fairness.excluded_by_default`
is its own config section, separate from `hard_quarantine`, and is honoured at
`features/firewall.py:143`:

```python
if col in cfg.fairness_excluded and not include_gender:
```

`include_gender` defaults to `False` through `firewall.admitted_features` and
`features.frame.build_model_frame`, whose docstring reads "admit F3892 — ONLY for
the fairness ablation." Keeping fairness exclusion separate from leakage
quarantine means a future leakage review cannot readmit a protected attribute as
a side effect.

**How gender was tested.** Because `F3892` never reaches the frame, it has no
rank, and rebuilding the ranking with it admitted would change *every* column's
rank — the arms would no longer be paired on identical pools. So the run calls
`build_model_frame(include_gender=True)`, **asserts that the resulting frame is
the baseline frame plus exactly one column** (any perturbation of the other
columns raises and aborts the run), lifts that single encoded column, and appends
it to each fold's matrix. `gender_forced` therefore differs from `baseline` in
exactly one column and nothing else. The assertion passed;
`gender_handling.isolation_asserted: true` in the payload.

The cost of that choice is stated in the payload as
`gender_handling.not_answered_by_this_run`: this run answers *"does a model with
gender beat one without?"*, not *"would selection have chosen gender?"* The first
is the question the policy turns on.

**What evidence would be required to flip the exclusion.** The config already
requires "a documented, approved justification." Concretely, that would mean at
minimum:

1. A `gender_forced`-style arm with a **positive** mean paired difference whose
   sign test rejects at 0.05 — the rule in §2, which this run failed by a clear
   margin and in the wrong direction.
2. A re-ranked arm confirming that stability selection chooses `F3892` when it is
   admitted, so the gain is not an artefact of forcing.
3. A slice audit showing the gain does not come from raising the selection rate
   of one gender group.
4. A named accountable approver and a recorded justification, because clause 1
   alone is an accuracy argument and the exclusion is not only an accuracy
   decision.

None of these conditions is met. The flag stays off.

---

## 5. Slice performance across arms

Group labels are read from the **raw** columns, not from model features, so
gender can be sliced on even though the model never sees it. Scores are the mean
of the three outer-fold OOF matrices; the alert budget is the top 100 of 7,264.

**Reporting rule, chosen before the numbers were seen:** a selection rate is
reported only when the group has **≥ 50 rows** (below that, one alert moves the
rate by more than 2 %), and a recall only when the group has **≥ 3 positives**
(matching the companion audit, so the two documents can be read together).
Suppressed groups still report their size — the fact that a group is too small
to measure is itself a finding and is not hidden.

**Suppressed here:** gender `O`, n = 49, 0 positives (below the row floor); age
`(not recorded)`, n = 3 (below the row floor); occupation `others` (138) and
`retired` (74) and age `65+` (372) report selection rates but **not** recall,
having 0, 0 and 2 positives respectively. One dev row carries an impossible
negative age; it is counted explicitly rather than silently absorbed into the
`< 25` bin.

The question this section answers is not "is the model fair" — §2 and §3 of the
companion audit answer that, and answer it with a published disparity. The
question here is **does forcing a sensitive attribute in change who gets
flagged.** Overall selection rate is 1.38 % in every arm (the budget is fixed);
overall Recall@100 is 0.797 for `baseline` and 0.813 for every other arm.

### Gender slice, by arm

| Arm | M (n=4,003) sel / rec | F (n=1,140) sel / rec | not recorded (n=2,072) sel / rec |
|---|---:|---:|---:|
| `baseline` | 1.80 % / 0.863 | 1.67 % / 0.700 | 0.39 % / 0.000 |
| `sensitive_excluded` | 1.82 % / 0.882 | 1.58 % / 0.700 | 0.39 % / 0.000 |
| `gender_forced` | 1.82 % / 0.882 | 1.67 % / 0.700 | 0.34 % / 0.000 |
| `all_four_forced` | 1.80 % / 0.882 | 1.67 % / 0.700 | 0.39 % / 0.000 |

Giving the model gender moves the female selection rate by nothing at all
(1.67 % → 1.67 %) and female recall by nothing (0.700 → 0.700, i.e. 7 of 10 in
both). The largest movement anywhere in the table is one alert.

### Geography and age slices, forced vs baseline

| Group | baseline sel | forced sel | baseline rec | forced rec |
|---|---:|---:|---:|---:|
| Area R (n=1,595) | 2.45 % | 2.32 % | 0.909 | 0.909 |
| Area SU (n=1,915) | 1.72 % | 1.83 % | 0.882 | 0.882 |
| Area M (n=2,351) | 0.81 % | 0.77 % | 0.688 | 0.688 |
| Area U (n=1,403) | 0.64 % | 0.71 % | 0.556 | 0.667 |
| Age 25–34 (n=2,062) | 2.04 % | 1.99 % | 0.840 | 0.840 |
| Age 35–44 (n=1,626) | 1.35 % | 1.42 % | 0.727 | 0.727 |
| Age 45–54 (n=896) | 0.78 % | 0.78 % | 0.833 | 1.000 |
| Age 65+ (n=372) | 0.54 % | 0.27 % | suppressed | suppressed |

**The decisive caveat, unchanged from the companion audit:** these groups hold
between 3 and 51 positives. Area U's recall moving 0.556 → 0.667 is **one case
out of nine**. Age 45–54's 0.833 → 1.000 is **one case out of six**. Nothing in
this table is statistically meaningful, and no group-level number here was used
to make any decision. It is published because an under-powered measurement with
its power stated is more honest than a silent omission.

What can be said at this resolution: forcing a demographic in did not produce a
visible redistribution of the alert budget across that demographic's own groups.
That is a weak claim and it is meant to be.

---

## 6. "Never the sole reason for escalation" — what the code actually enforces

The fairness config states the rule (`fairness.rules`: a demographic must never
be the sole driver). Here is what enforces it, and where the enforcement stops.

**Enforced in code:**

| Guarantee | Where |
|---|---|
| No demographic can reach the action decision. `decide()` takes a closed input surface — `calibrated_risk`, `conformal_set`, `ood_status`, `anomaly_percentile`, `model_agreement`, `verifier_flag`, `thresholds`. There is no path by which a feature value, demographic or otherwise, is an argument. | `src/muleguard/action/policy.py::decide` |
| No escalation is automatic. Every non-`MONITOR` tier returns `auto_action: None` and `decision: "HUMAN_REVIEW_REQUIRED"`. There is no automatic freeze or adverse action anywhere in the system. | `src/muleguard/action/policy.py::decide` |
| A reason code can only name a consumed column. Reasons are TreeSHAP over `feature_list_kept`, so no column outside the model's 120 can be cited. | `src/muleguard/models/scoring.py:301` → `explain/reason_codes.py::shap_reason_codes` |
| A demographic is never proposed as something the customer should change. `counterfactual_sensitivity` skips any feature carrying a `verified_semantic_name` — the set that includes `F3890`, `F3891`, `F3892`, `F3893`. | `src/muleguard/explain/reason_codes.py:149` |
| `F3892` cannot be in a shipped bundle. The release gate requires `feature_list_selected ∪ feature_list_kept` to be disjoint from the 13-entry quarantine manifest, which contains `F3892`. | `src/muleguard/cli/release_gate.py:41` check `no_target_or_f3912_leakage` |
| No serialised payload may name a quarantined feature. | `explain/error_atlas.py::assert_no_quarantined_feature`, unit-tested at `tests/unit/test_error_atlas.py:196` |
| No `F3892` in the final selection. | `tests/unit/test_evidence_bookkeeping.py:517::test_no_quarantined_feature_reaches_the_final_selection` |

**Not enforced in code — stated plainly:**

There is **no check of the form "if the top reason names a demographic, require a
corroborating non-demographic reason."** For gender the guarantee holds by
construction, because `F3892` is not in the model and cannot be named. But
`F3890`, `F3891` and `F3894` are *admissible*. They are absent from today's
champion by the selector's choice, not by a rule — and a future retrain that
selected one of them could produce a case where a demographic is the single
largest SHAP contributor, with nothing in the pipeline objecting.

Note on `firewall.assert_clean` (`features/firewall.py:183`). An earlier draft of
this document recorded that it checks `hard_quarantine` and `FORBIDDEN_CLASSES`
only — not `fairness_excluded` — and argued the omission was deliberate, because
enforcing the fairness exclusion there would stop
`build_model_frame(include_gender=True)` producing a matrix and this ablation
could never have run.

**That is no longer true, and the argument for it was wrong.** As of 2026-08-14
`assert_clean` checks all three config lists and refuses all 13 manifest columns,
gender included (`docs/FEATURE_AVAILABILITY_AUDIT.md` §8.1). The ablation was
then run to completion *under the tightened check* — which is the direct refutation
of the claim that the tightening would have prevented it. Nothing prevented it,
because `build_model_frame` skips the assertion when `extra_allowed` is set
(`features/frame.py:207`), which is precisely the labelled escape hatch a
deliberate ablation is supposed to use.

The distinction worth keeping is the audit one: the release gate
(`release_gate.py:41`) is still the *shipping* block, because it tests the frozen
bundle rather than a call site a wrong training script could bypass. `assert_clean`
is now genuinely the second line of defence its docstring always promised, rather
than a second line with four columns missing from it.

**What closing the gap would require:** a guard in
`explain/reason_codes.shap_reason_codes` (or in `models/scoring._score` at the
point reasons are attached) that detects a demographic in the top reason slot and
either demotes it or requires a corroborating behavioural reason, plus a
release-gate check asserting that guard is active. That work is **not** in this
submission and is recorded here as a gap rather than implied to be done.

---

## 7. Summary

| Question | Answer |
|---|---|
| Does gender improve the model? | **No.** −0.00481 AP paired, 4 folds up / 11 down. It is slightly worse with gender |
| Does any sensitive attribute clear the pre-fixed re-admission bar? | **No.** None. Verdict `KEEP_EXCLUSION` |
| Is the geography arm's significant sign test evidence *for* geography? | **No** — its mean is negative. It is evidence against |
| What does the exclusion policy cost? | 0.00011 AP, and it changed the feature pool in 1 of 15 folds |
| Was gender tested honestly? | Yes — injected as a single isolated column, with the isolation asserted at runtime; and the one question this cannot answer is recorded in the payload |
| Does "no effect detected" mean "no effect"? | **No.** MDE80 is 0.0003–0.010 AP; smaller true effects would have been missed |
| Is "never the sole reason" enforced in code? | **Partly.** Enforced by construction for gender; **not** enforced for age/occupation/area, which remain admissible. Gap named in §6 |
| Was the locked test set read? | **No** — `design.locked_test_read: false` |

Companion document: **`docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`** (§24) —
the measurement of what the shipped model does to each group, including the
published selection-rate disparity, the equal-opportunity table, the reasons no
per-group threshold adjustment was made, and the unbounded proxy-leakage gap.
That document reports the disparity; this one tests the exclusion. Neither
substitutes for the other.
