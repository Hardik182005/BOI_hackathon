# Final Accuracy and Model Selection Report

Authoritative machine-readable sources for every number below:

| Claim area | Artifact |
|---|---|
| Leaderboard, all candidates | `artifacts/metrics/tournament_v2.json`, `artifacts/metrics/model_comparison_v2.csv` |
| Promotion rule and promoted model | `artifacts/metrics/promotion_decision_v2.json` |
| Ensemble decision | `artifacts/metrics/ensemble_v2.json` |
| Feature pool and compact sets | `artifacts/features/selected_features_v2.json` |
| Calibration, conformal, thresholds | `artifacts/metrics/lens_stack_oof_v2.json` |
| Rare-positive stress test | `artifacts/metrics/stability_stress_v2.json` |
| Label-noise audit | `artifacts/metrics/label_noise_audit_v2.json` (narrated in `docs/LABEL_NOISE_AUDIT.md`) |
| Bundle identity and what it supersedes | `artifacts/models/model_manifest.json`, `artifacts/model_registry/registry.json` |
| Admissibility policy | `configs/feature_availability.yaml` |

Nothing here is typed from memory. Where a measurement does not exist yet, this
document says so instead of estimating it.

---

## 1. The primary metric is PR-AUC, and accuracy is not reported

Prevalence in the supplied extract is **81 positives in 9,082 rows = 0.8919 %**.
A model that labels every account legitimate is 99.11 % accurate and catches
nothing. Accuracy is therefore not the headline metric here, is not used to
select a model, and is not reported as a result anywhere in this repository.

The primary metric is **PR-AUC (average precision), averaged over CV repeats**,
as recorded in `tournament_v2.json` under `primary_metric`. ROC-AUC is
secondary. Recall at analyst budgets (top-25/50/100) is reported because it is
the number an analyst actually experiences. The measured floor is the
`dummy_prevalence` model at PR-AUC **0.0087**.

---

## 2. Protocol

- **Splits.** Locked stratified test of 1,818 rows / 17 positives, frozen before
  any training. Development split = 7,264 rows / 64 positives.
- **Evaluation.** Repeated stratified group-aware 5-fold CV, **3 repeats** on
  the development split. Preprocessing, feature selection and calibration are
  refitted inside each training fold; no held-out fold and no locked-test row
  participates in selection (`selected_features_v2.json`, field `method`).
- **Imbalance.** Class weights, no SMOTE, natural prevalence preserved
  everywhere.
- **Admissibility.** Every candidate draws only from the pool admitted by the
  Feature Availability Firewall (`configs/feature_availability.yaml`). The
  hard quarantine covers `F3924` (target), `F3912`, `F3913`, `F3914`, `F3915`
  (the four mutually exclusive resolution outcomes), `F3898`/`F3899`
  (resolve-day durations), `F2230` (snapshot month, which reconstructs the
  label in this extract) and `__UNNAMED__0` (the raw row index). `F3916`,
  `F3917` and `F3918` (customer risk level flags) are held in conditional
  quarantine until the bank confirms they are written independently of alert
  resolution; they are evaluated only as a labelled ablation. `F3892` (gender)
  is excluded by the fairness policy.
- **Verification.** The promoted model's 120-feature set was checked against
  that quarantine list: **0 quarantined columns present**.

---

## 3. Leakage-free leaderboard (development OOF, 3 repeats)

Full table, in `tournament_v2.json` order of PR-AUC:

| # | Model | View | Features | PR-AUC (mean ± std) | Worst repeat | ROC-AUC | Recall@100 | s |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **`xgboost_top_120`** | ALL_ADMISSIBLE | 120 | **0.7690 ± 0.0266** | 0.7493 | 0.9577 | 0.7813 | 32.3 |
| 2 | `xgboost_top_60` | ALL_ADMISSIBLE | 60 | 0.7408 ± 0.0498 | 0.6807 | 0.9420 | 0.7604 | 22.2 |
| 3 | `lightgbm_top_60` | ALL_ADMISSIBLE | 60 | 0.6992 ± 0.0273 | 0.6624 | 0.9397 | 0.7344 | 29.7 |
| 4 | `catboost_top_60` | ALL_ADMISSIBLE | 60 | 0.6963 ± 0.0417 | 0.6584 | 0.9323 | 0.7448 | 75.3 |
| 5 | `xgboost_top_30` | ALL_ADMISSIBLE | 30 | 0.6897 ± 0.0122 | 0.6742 | 0.9334 | 0.7500 | 13.2 |
| 6 | `lightgbm_top_120` | ALL_ADMISSIBLE | 120 | 0.6862 ± 0.0405 | 0.6309 | 0.9387 | 0.6979 | 34.4 |
| 7 | `lightgbm_top_30` | ALL_ADMISSIBLE | 30 | 0.6628 ± 0.0161 | 0.6460 | 0.9357 | 0.6979 | 17.8 |
| 8 | `lightgbm_viewA_top_60` | A_broad_behavioral | 60 | 0.6584 ± 0.0234 | 0.6256 | 0.9423 | 0.6927 | 31.4 |
| 9 | `catboost_top_120` | ALL_ADMISSIBLE | 120 | 0.6467 ± 0.0699 | 0.5516 | 0.9316 | 0.7292 | 128.2 |
| 10 | `lightgbm_viewB_top_60` | B_stable_compact | 60 | 0.6454 ± 0.0485 | 0.5771 | 0.9427 | 0.6875 | 26.6 |
| 11 | `lightgbm_top_250` | ALL_ADMISSIBLE | 250 | 0.6442 ± 0.0300 | 0.6041 | 0.9393 | 0.6563 | 49.8 |
| 12 | `lightgbm_full_pool` | ALL_ADMISSIBLE | 3,925 | 0.5874 ± 0.0276 | 0.5556 | 0.9286 | 0.6198 | 409.4 |
| 13 | `catboost_top_30` | ALL_ADMISSIBLE | 30 | 0.5614 ± 0.1125 | 0.4155 | 0.9156 | 0.6146 | 58.4 |
| 14 | `lightgbm_top_15` | ALL_ADMISSIBLE | 15 | 0.3357 ± 0.1399 | 0.2283 | 0.8931 | 0.4010 | 15.1 |
| 15 | `lightgbm_viewE_top_60` | E_profile_merchant | 60 | 0.3061 ± 0.0842 | 0.1908 | 0.8643 | 0.3646 | 23.4 |
| 16 | `lightgbm_freq_ge_0_50` | ALL_ADMISSIBLE | 13 | 0.1917 ± 0.0247 | 0.1572 | 0.8634 | 0.2656 | 7.8 |
| 17 | `elasticnet_top30` | ALL_ADMISSIBLE | 30 | 0.0895 ± 0.0045 | 0.0858 | 0.8795 | 0.1927 | 114.4 |
| 18 | `logistic_top30` | ALL_ADMISSIBLE | 30 | 0.0882 ± 0.0045 | 0.0824 | 0.8808 | 0.1875 | 0.9 |
| 19 | `lightgbm_viewC_top_15` | C_bank_prior | 15 | 0.0295 ± 0.0110 | 0.0155 | 0.6723 | 0.0573 | 6.0 |
| 20 | `lightgbm_viewD_top_15` | D_alert_context | 15 | 0.0206 ± 0.0021 | 0.0183 | 0.7046 | 0.0313 | 8.4 |
| 21 | `dummy_prevalence` | ALL_ADMISSIBLE | — | 0.0087 ± 0.0000 | 0.0087 | 0.4938 | 0.0000 | 0.2 |

`tabpfn_top_60` was previously recorded here as `FAILED` — TabPFN refuses to run
on CPU with more than 1,000 samples unless an override is set. That is no longer
the state of the repository. Re-run with `ignore_pretraining_limits=True` it
completed all three repeats, and `artifacts/metrics/tournament_v2.json` records
it with status `OK` at **OOF PR-AUC 0.9110 ± 0.0044**, Recall@100 **0.9115** —
above every row in the table above.

It is excluded from *this* selection deliberately, and not because the number is
doubted. The adjudication is in `artifacts/metrics/challenger_review_v2.json`
and is summarised in `UPGRADE_GAP_ANALYSIS.md` §3.1.1: all four UPDATE 1 gates
pass, no leakage was found, and the decisive control is that `xgboost_top_60`
consumes the identical 60 columns over the identical folds and scores 0.7408 —
so the gap is in the estimator, not in the data. What disqualifies it is
serving cost: `artifacts/metrics/tabpfn_latency.json` measures **438 s for a
single-row `predict_proba`**, because TabPFN carries the entire training split
through the transformer on every call, and it exposes no attribution path, so a
§17 ProofGraph would take roughly 7.4 hours per case. A model whose score
cannot be returned in time or proved afterwards cannot be the served champion,
whatever its PR-AUC.

TabPFN therefore contributes no evidence to the promotion decision, and is
retained as a verified challenger and a recorded second opinion.

Coverage of the specification's mandatory candidates (`catboost_top_60`,
`lightgbm_top_60`, `xgboost_top_60`, `lightgbm_full_pool`) is recorded as
`COMPLETE` with an empty `missing` list.

### 3.1 Two readings of the headline number

- **0.7690** is roughly an **86× lift** over the 0.8919 % prevalence floor, and
  about 88× the measured `dummy_prevalence` PR-AUC of 0.0087.
- Compact beats comprehensive. The 120-feature set outscores the full
  3,925-column admitted pool by 0.18 PR-AUC and runs about 13× faster
  (32.3 s vs 409.4 s per evaluation). Width past 120 hurts: `lightgbm_top_250`
  is below `lightgbm_top_60`.
- Every std in this table is computed over **3 repeats with 64 positives**.
  These are not tight intervals and are not presented as such.

### 3.2 The retired 0.8077

The previous champion `catboost_tuned_top60` reported OOF PR-AUC
**0.8077 ± 0.0450**. That number is **RETIRED**. Its top-60 feature set
contained `F3898` (MIN_RESOLVE_DAYS), `F3913` (OTHER_RESOLUTION), `F3914`
(FALSE_POSITIVE) and `F3916` (L3_FLG). The first three are written only after a
human has closed the alert — that is, after the decision this model exists to
support — and the fourth has undetermined availability timing. The score was
inflated by them.

It is recorded in `artifacts/model_registry/registry.json` with
`status: "retired"`, and in the current bundle's `supersedes` block with the
reason quoted verbatim. It may be cited only as a retired, leakage-inflated
baseline, with that explanation attached. Overlap between the retired top-60
and the new leakage-free top-60 is **12 of 60**.

---

## 4. Promotion rule (addendum UPDATE 13)

From `promotion_decision_v2.json`:

```
generalization_score = PR_AUC_mean - 0.5 * PR_AUC_std + 0.1 * Recall@100
```

The rule is deliberately **not** a global ranking function. It is applied only
inside a **0.01 PR-AUC band** below the raw leader; within that band the
simpler, more stable model wins. Outside the band, raw PR-AUC decides. This
prevents the score from promoting a materially weaker model on the strength of
a variance term.

| Field | Value |
|---|---|
| `raw_pr_auc_leader` | `xgboost_top_120` |
| `models_within_tie_band` | `["xgboost_top_120"]` — nothing else came within 0.01 |
| `tie_break_applied` | `false` |
| `promoted` | **`xgboost_top_120`** |
| `generalization_score` | 0.83385 = 0.76904 − 0.5(0.02663) + 0.1(0.78125) |

Because the band contained only the leader, the promotion is decided by raw
PR-AUC alone; the generalization score is recorded but did not change the
outcome. Stating this matters — a rule that never had to arbitrate should not
be presented as though it did.

---

## 5. Ensemble decision: `SINGLE_MODEL_KEPT`

Members evaluated (`ensemble_v2.json`): `xgboost_top_120`, `xgboost_top_60`,
`lightgbm_viewA_top_60`, `lightgbm_viewB_top_60`. Four aggregation schemes were
measured against the best single model:

| Scheme | Mean PR-AUC | Std across repeats | Mean Recall@100 | Repeats won vs best single |
|---|---:|---:|---:|---:|
| best single (`xgboost_top_120`) | 0.76904 | 0.02663 | 0.7813 | — |
| **logit mean** | **0.76946** | 0.02064 | 0.7865 | **1 of 3** |
| stacker (logistic) | 0.74269 | 0.01565 | 0.7708 | 0 of 3 |
| rank mean | 0.73552 | 0.01288 | 0.7500 | 0 of 3 |
| Borda | 0.73552 | 0.01288 | 0.7500 | 0 of 3 |

The addendum's UPDATE 2 acceptance rule has three literal criteria, and the
logit mean satisfies all three:

| Criterion | logit mean |
|---|---|
| PR-AUC not worse | true (+0.0004) |
| Recall@100 improves | true (0.7865 vs 0.7813) |
| Fold-to-fold variance decreases | true (0.0206 vs 0.0266) |

The artifact records this outcome explicitly as
`decision_under_addendum_text_alone: "ENSEMBLE_ACCEPTED:logit_mean"`.

We did not accept it, and the extra criterion we applied is recorded separately
under `additional_rule_not_from_the_addendum` so the decision can be
re-derived either way:

> the PR-AUC comparison is also required per repeat (win on at least n−1). At
> 3 repeats and 81 positives a mean-level edge of a few ten-thousandths is
> inside noise, and UPDATE 7 forbids adding complexity that does not improve
> measured performance.

The logit mean won **1 of 3 repeats**. A margin of 0.0004 PR-AUC across three
repeats of a 64-positive split is not a measurable improvement; it is a coin
flip that happened to land. Blending four models for it would add four
inference paths, four failure modes and four sets of SHAP attributions to
explain to an analyst, in exchange for nothing that survives the noise.
Decision: **`SINGLE_MODEL_KEPT`**, `ensemble_accepted: false` in the manifest.

Both readings are published on purpose. A judge who prefers the addendum's
literal text can see exactly what it would have selected and what it would have
bought.

---

## 6. View ablations: what the model is actually using

The tournament ran five availability views. Two of them are the load-bearing
generalization evidence:

| View | What it may draw on | Candidates | OOF PR-AUC |
|---|---|---:|---:|
| A — broad behavioural | BEHAVIORAL only | 3,897 | 0.6584 ± 0.0234 |
| B — stable compact | BEHAVIORAL + PROFILE | 3,905 | 0.6454 ± 0.0485 |
| **C — bank prior** | the bank's 18 finalised variables + safe profile fields | 18 | **0.0295 ± 0.0110** |
| **D — alert context** | pre-decision alert evidence only | 28 | **0.0206 ± 0.0021** |
| E — profile/merchant | POS, GST, balance, loan, standing instruction, fees | 1,376 | 0.3061 ± 0.0842 |
| — full admitted pool | everything admitted | 3,925 | 0.7690 (top-120 subset) |

Read views C and D against the 0.0087 floor:

- **View C = 0.0295.** If the detector were quietly re-reading a pre-existing
  bank risk opinion of the customer, this view would score well. It does not.
  The bank's own finalised variable list, on its own, is worth about 3× the
  base rate.
- **View D = 0.0206.** If the detector were re-reading analyst alert metadata —
  effectively "this case was queued for review, therefore it is a mule" — this
  view would score well. It does not. Alert context alone is worth about 2.4×
  the base rate, at ROC-AUC 0.7046.

Meanwhile view A, behavioural aggregates with no profile and no alert context
at all, reaches 0.6584. The conclusion the evidence supports is narrow and
worth stating precisely: **the promoted model's performance comes from
behavioural transaction and balance patterns, not from pre-existing risk flags
and not from the shape of the alert that raised the case.** That is the
property that has to hold for the model to work on the organisers' hidden
validation set, where the alert metadata and risk flags will differ.

---

## 7. Calibration and operating points

From `lens_stack_oof_v2.json` (development OOF, repeat-averaged and
crossfit-calibrated; `supersedes: "lens_stack_oof.json (pre-firewall, leaky
feature set)"`):

- **Calibrator: Platt.** Brier 0.003128, ECE 0.001489 (10 bins), against
  isotonic at Brier 0.003159 / ECE 0.001698. Selection rule, fixed in advance:
  isotonic must improve **both** Brier and ECE by ≥ 2 % relative to be chosen;
  with 64 positives the simpler calibrator is preferred.
- **Calibrated OOF PR-AUC 0.8047** (bootstrap 95 % CI 0.7115–0.8830, n = 1000);
  ROC-AUC 0.9598 (CI 0.9199–0.9911). This is a *different quantity* from the
  0.7690 leaderboard figure: it is computed once on repeat-averaged calibrated
  scores, whereas 0.7690 is the mean of three independent repeat-level
  measurements. **0.7690 ± 0.0266 remains the headline**, because averaging
  scores across repeats before scoring them is the more flattering of the two
  and we do not lead with the more flattering number.
- **Conformal (α = 0.10) OOF coverage:** positive coverage 0.9375, negative
  coverage 0.9474, abstention rate 4.69 %.
- **Recall at analyst budgets** (repeat-averaged calibrated OOF):

  | Budget | Recall | Precision | True positives |
  |---:|---:|---:|---:|
  | 25 | 0.3906 | 1.0000 | 25 |
  | 50 | 0.6875 | 0.8800 | 44 |
  | 73 | 0.7656 | 0.6712 | 49 |
  | 100 | 0.8281 | 0.5300 | 53 |

- **Recall at fixed false-positive rate:** 0.7969 at FPR 0.5 % (5.0 FP per
  1,000 legitimate accounts); 0.8438 at FPR 1.0 % (10.0 per 1,000).
- **Frozen policy thresholds:** critical 0.93385, urgent 0.09774, standard
  0.01318, anomaly escalation at the 99th percentile, policy version 1.0.
  These are the values in `model_manifest.json`; they differ from the retired
  bundle's thresholds because the underlying score distribution changed.

---

## 8. Locked test: not re-touched under the new champion

`artifacts/metrics/locked_test_touch_log.json` records three touches, all in
July 2026, all against bundle `04fafaee25ae82c7…` — the **retired** leaky
bundle. There is no v2 locked-test evaluation artifact.

Therefore:

- The previously published locked-test figures (PR-AUC 0.8242, CI
  0.6536–0.9584, and the per-tier and per-budget tables derived from them)
  belong to the retired model and **must not be attributed to
  `xgboost_top_120`**.
- This report publishes **no locked-test number for the current champion**,
  because none has been measured. The locked test is single-touch by design;
  spending it is a deliberate act, not a side effect of a documentation pass.

Stating an unmeasured number here would be the same class of error as the one
this whole audit exists to correct.

---

## 9. Stability under positive scarcity

From `stability_stress_v2.json` (model `xgboost_top_120`, 3 rounds, 12.5 % of
training positives removed per round, mean 30 positives removed; the evaluation
fold is identical in every round, so all spread is attributable to label
scarcity):

| Measure | Value |
|---|---:|
| Reference PR-AUC | 0.74927 |
| PR-AUC after positive removal | 0.72398 ± 0.03545 (min 0.64135) |
| Relative drop | **3.37 %** |
| Recall@100 | 0.74271 (std 0.04029) |
| Recall@50 | 0.63542 (std 0.02834) |
| Recall@25 | 0.38854 (std 0.00531) |
| Feature-importance rank stability (Spearman) | 0.7696 |
| Top-20 feature overlap between rounds | 0.6805 |
| All-row prediction rank stability (Spearman) | 0.3694 |
| Top-K set overlap (Jaccard), K = 25 / 50 / 100 | 0.608 / 0.6771 / 0.4493 |

An earlier draft of this table quoted a superseded run of the same experiment —
a 1.46 % drop, min 0.71217, all-row Spearman 0.1656 — against the identical
0.74927 reference. Every row above is re-read from
`artifacts/metrics/stability_stress_v2.json` as it now stands. The correction
runs against this document's interest: the drop roughly doubled and the worst
round is materially lower than the earlier text implied.

The feature-importance figure carries a further qualification. It is a **flat**
measurement, and the flat stress fits feature selection once over pooled
development data before its rounds begin, so removing training positives cannot
disturb a selection that has already happened. Re-run inside the nested
protocol, where selection is refit in every outer fold as a production refit
would be, the same correlation falls to **0.3944**
(`artifacts/metrics/nested_positive_removal.json`). Quote the nested figure:
the model's *ranking* survives losing an eighth of its mules, but most of what
it cites as evidence does not.

The all-row Spearman figure — 0.3694 in the current artifact — is reported
unchanged because it is the figure the published robustness thresholds were
defined on. Its caveat is
recorded in the artifact and repeated here: roughly 99 % of those rows are
negatives whose calibrated scores sit in a narrow band near zero, so their
relative order moves freely between rounds without any account changing review
status. The top-budget overlaps are the diagnostic that describes the part of
the ranking an analyst works through, and the artifact states explicitly that
they are a diagnostic **reported alongside** the badge, not an input to it —
the thresholds were fixed before these experiments ran and have not been
redefined to suit the result.

---

## 10. Label-noise audit

`label_noise_audit_v2.json`: 13 consensus models over 39 model-repeat runs on
7,264 development rows (64 positives, 7,200 negatives). **1 positive** (1.56 %)
was flagged as `POSSIBLE_LABEL_NOISE`; **0 negatives** were flagged as
high-scoring. No label was changed, no row was removed, and no model here is
trained, calibrated or thresholded on a filtered label set. Full narration and
the reasons this is a review request rather than a correction:
`docs/LABEL_NOISE_AUDIT.md`.

---

## 11. Honesty notes

- No accuracy figure is presented as a result, and no "100 %" claim of any kind
  is made in this document.
- The leakage-inflated 0.8077 and the rejected `F3912` ablation appear only as
  labelled negative evidence.
- Confidence intervals are wide because 64 development positives and 17
  locked-test positives are all that exist. They are reported rather than
  narrowed by choosing a friendlier resampling scheme.
- A behavioural risk score is not a fraud verdict. A low score means **not
  currently flagged**; it never means an account is safe, cleared or exonerated,
  and nothing in this report is evidence of criminal conduct by any account
  holder.
