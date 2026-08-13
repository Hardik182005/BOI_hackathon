# Stability, ensembling and distribution shift, re-run on nested folds

Final-validation sections 15-23, 53 and 54.

Sections 15-23 were answered once already, under a **flat** repeated-CV protocol
(`artifacts/metrics/ensemble_v2.json`, `stability_stress_v2.json`,
`seed_variance_v2.json`). Those numbers cannot be placed next to the nested
tournament, and several of them were computed with a fitted quantity - a blend
weight, a stacker, a noise flag - learned from the same out-of-fold vector they
were then scored against. This document re-runs them on the nested outer folds
so that every arm is paired against every other arm on byte-identical
partitions, and so that every fitted quantity is fitted inside the outer-training
partition only.

**Everything below the pre-registration line was fixed before the run.** The
decision rules live in `src/muleguard/cli/nested_ses.py`, were written before any
arm was scored, and are evaluated mechanically by that file - the code emits the
verdict, this document does not get to choose it afterwards.

---

## The headline, stated before the numbers

An outer-validation fold holds **1,453 rows and about 13 positives**. Average
precision computed on 13 positives is a coarse statistic: moving one positive
across the decision boundary moves it visibly. With 15 such folds, this design
can detect large effects and nothing else.

So the expected outcome of almost every comparison in this document is *not
significant*, and that is the honest result rather than a problem to be
engineered around. Where a null result is reported it is accompanied by the
minimum effect the design could have detected at 80 % power, so that "we found
nothing" is never mistaken for "there is nothing".

## Design

| | |
|---|---|
| protocol | nested repeated CV, paired on identical outer folds |
| outer | stratified 5-fold x 3 repeats = **15 outer folds** |
| inner | stratified 4-fold inside each outer-training partition |
| fold source | `nested.build_outer_folds(frame, n_repeats=3, n_inner=4)` - the same function the nested tournament calls |
| dev | 7,264 rows, 64 positives (0.881 %) |
| per fold | train 5,811 (+51), valid 1,453 (+13) |
| features | 120 columns, ranked inside the outer-training partition |
| hyperparameters | each family at its default configuration, identical across arms |
| locked test | **not read.** It is a historical holdout, for reference only |

Everything fitted - preprocessing, imputation, scaling, the feature ranking,
ensemble weights, stacker coefficients, calibration curves, label-noise flags -
is fitted on the outer-training partition and applied to the validation
partition. `fold.yva` appears exactly once per arm, in the scoring call.

That property is enforced by test, not by inspection. `tests/unit/
test_nested_experiments.py::test_every_combiner_ignores_validation_labels`
shuffles the validation labels and requires every arm's predictions to come back
bit-identical; `test_compute_bases_does_not_read_validation_labels` flips every
validation label and requires the same. An implementation that fitted anything
on the validation partition fails both immediately.

### Which yardstick, and which one is wrong here

The project's seed-noise floor of **0.0905 PR-AUC is an unpaired figure**: it is
the spread of whole-model scores across independent seeds and splits. Applying
it to a paired, fold-held-fixed comparison would be a category error - it would
declare every real effect in this document undetectable, because a paired
difference on fixed folds has a far smaller spread than an unpaired score
difference.

Every comparison here is therefore judged against **its own
`std_of_paired_diff`**, which is printed next to the mean in every artifact, and
every artifact carries a `yardstick` string saying so. Nothing in this document
is compared against 0.0905.

### Three tests, always all three

Each comparison reports an exact two-sided **sign test**, a **Wilcoxon
signed-rank** test, a **paired t** test, the mean paired difference, and a
t-based 95 % confidence interval. All three are reported even when they
disagree, and no rule anywhere selects the most flattering one.

They can disagree, and the disagreement carries information:

* the **sign test** sees direction only, so a few large wins against many small
  losses will not move it;
* the **Wilcoxon** test sees direction and the rank of the magnitudes, so a
  single outlier cannot carry it;
* the **paired t** sees magnitude, so a couple of large wins can carry it even
  when most folds got slightly worse.

`tests/unit/test_paired.py::test_disagreement_is_detected_not_hidden` pins this
down with a constructed case: two folds gain 0.40 and 0.35, thirteen lose 0.01.
The mean is positive and the t-test says p = 0.258 (not significant), while the
sign test says p = 0.007 **against** the arm. Reporting either alone would
misrepresent the experiment.

---

## Pre-registered decision rules

These were written before the run.

### Section 53 - ensembles

Accept an ensemble over the best single model only if **all three** hold:

* **C1** mean paired difference in AP > 0, **or** the 95 % CI of the mean
  contains 0 (a statistical tie);
* **C2** the standard deviation of the 15 fold APs decreases, **or** the mean
  per-fold Recall@10 improves;
* **C3** stress does not degrade materially: worst-fold difference >= -0.05
  **and** the mean difference over the 5 folds where the baseline scored lowest
  is >= 0.

C3 is a **proxy**. Section 53 asks for "external-like stress" and no external
set exists in this session, so the hardest folds inside this dataset stand in
for it. That substitution is a weakening of the rule and is labelled as such
wherever the verdict appears.

Otherwise: **prefer the strongest simple single model.**

The baseline is `best_single_inner` - in each fold, the family with the best
**inner** average precision. Choosing the baseline by its outer-validation score
would hand it a winner's curse and make every ensemble look better than it is.

### Section 18 - seed bagging

* **ADOPT** if the mean paired gain > 0 and at least 2 of the 3 tests give
  p < 0.05.
* **ADOPT_FOR_STABILITY_ONLY** if not ADOPT, the 95 % CI contains 0, and the
  across-fold AP spread decreases.
* **NO_CHANGE** otherwise - the single-seed model is 5x cheaper and a tie does
  not buy a 5x bill.
* **NOT_APPLICABLE** when the family has no stochastic component, so the five
  seeds fit the identical model and the arm cannot move. `histgb` is the case
  here: sklearn's `HistGradientBoosting` draws no random subsample, so
  `random_state` never reaches the fit. Reporting NO_CHANGE for such a family
  would read as evidence that seed averaging was tried and did not help. It was
  not tried - there was nothing to average, and a measured zero and an
  arithmetically impossible one are different claims.

Seeds are `fold_seed + {0, 7919, 15838, 23757, 31676}`, the offsets already used
in `seed_variance_v2.json`, so the nested and flat numbers differ by protocol
alone. Offset 0 is the single-seed arm: the model that would have shipped.

### Section 21 - positive removal

12 rounds per fold, 12.5 % of training positives removed per round (matching the
flat `stability_stress_v2.json`), **validation partition unchanged in every
round**, xgboost at its default configuration. Graded against the thresholds
already published in `src/muleguard/models/robustness.py` - relative AP drop,
AP spread, prediction rank stability - not against new thresholds invented here.

### Section 22 - label noise

A positive is flagged `POSSIBLE_LABEL_NOISE` when **every** family ranks it
below the median of all rows. Nothing is relabelled and nothing is deleted.

Down-weighting flagged positives is kept only if the **fold-local** arm - flags
derived from that fold's own inner-OOF predictions - beats the baseline with a
positive mean paired gain and at least 2 of 3 tests p < 0.05.

A second arm is run with flags derived from the **globally pooled** predictions.
That arm is **rejected evidence**: for a given fold, the pooled flags were
produced partly by models fitted on rows sitting in that fold's validation
partition. It is run only to measure how large that leak is, and it can never
justify a change no matter what it scores.

### Section 23 - adversarial validation

The outer split is random and stratified, so a classifier trying to separate
outer-train rows from outer-validation rows must score about 0.5. An AUC outside
**[0.45, 0.55]** is a stop-and-investigate: it would mean the folds are not
exchangeable, which would invalidate every paired comparison in this document.

Per-feature shift classes, applied to the columns each fold actually selected:

* **SHIFT_PRONE** - mean KS >= 0.10, or mean missing-rate shift >= 0.05, or
  unseen-category rate >= 0.02;
* **WATCH** - mean KS >= 0.05, or selected in fewer than half the folds;
* **STABLE** - otherwise.

### Sections 15, 16, 17 - feature pools

An arm replaces the full clean set only if its mean paired difference is
positive and at least 2 of 3 tests give p < 0.05. A pool is not kept because a
bank marked it, and the alert-context block is not accepted on a large jump
without a timing review.

### Section 54 - generalization report

The components - CV stability, seed stability, positive-removal stability,
feature stability, adversarial AUC, OOD rate, calibration stability - are
**reported separately**. The optional HIGH/MEDIUM/LOW label is a summary of the
already-published `ROBUSTNESS_THRESHOLDS`, with one addition (adversarial AUC
within [0.45, 0.55]). It is not a new score and it is never used to select a
model.

---

## Firewall verification

Every artifact in this programme carries a `firewall` block recording that the
13 hard-quarantined columns - `F2230`, `F3892`, `F3898`, `F3899`, `F3912`,
`F3913`, `F3914`, `F3915`, `F3916`, `F3917`, `F3918`, `F3924`, `__UNNAMED__0` -
are absent from the model frame **and** from every fold matrix. The check runs
before any stage and raises rather than warning. The verification is recorded
because the requirement is to verify, not to assume.

<!-- RESULTS BELOW THIS LINE -->

## Results

_Pending: the run is in progress. Nothing is written here until the artifacts
exist._

## What is not concluded

_Pending._
