# Open question: is the inner-fold Optuna tuning net-harmful?

**Status: HYPOTHESIS, not a finding.** The decisive number does not exist yet. This note records
the observation and the evidence while it is fresh so that it is resolved deliberately in the
nested-tournament write-up rather than quietly forgotten.

## The observation

Two runs of `HistGradientBoostingClassifier` on the **same 15 outer folds** (3 repeats x 5 folds,
`harness.dev_split(3)`, `n_inner=4`, `selector_top_k=200`):

| run | protocol | PR-AUC | source |
|---|---|---|---|
| nested tournament, Optuna-tuned, 15 trials/fold, feature size chosen on inner folds | nested | **0.76735 +/- 0.02949** | `logs/nested_cv_full.log`, 2026-08-12 20:34:25 |
| missingness ablation, WITHOUT arm, fixed hyperparameters, fixed `top_120` | nested | **0.79068 +/- 0.01179** | `logs/missingness_ablation.log`, 2026-08-12 21:40:56 |

The untuned run is **0.02333 higher** and less variable across repeats.

## Why this is not obviously explained away

Three explanations were checked and none of them accounts for it.

**It is not a different estimator.** `hgb_factory` in `src/muleguard/cli/nested_cv.py:141` and
`hgb_fit_predict` in `src/muleguard/cli/missingness_ablation.py:66` construct the same object with
the same arguments — `max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=20,
l2_regularization=1.0, early_stopping=False`, fitted with the same balanced `sample_weight` and
the same `random_state`. The untuned arm *is* the tuned arm's default configuration.

**It is not a search space that excludes the winner.** `hgb_space` covers `max_iter` 150-600 step
50, `max_depth` 2-8, `learning_rate` 0.01-0.15 log, `min_samples_leaf` 5-60, `l2_regularization`
1e-3 to 20 log. Every default value above lies inside that range. The tuner could have selected
the untuned configuration and did not.

**It is not a different split.** Both call `nested.build_outer_folds` with `n_repeats=3`,
`n_inner=4`, and the same `harness.dev_split(3)`, so the outer folds are byte-identical. That was
the reason for holding the folds fixed in the first place.

The tuned run also chooses its feature-set size from {30, 60, 120} on inner folds, where the
untuned arm is pinned to 120. That is a second degree of freedom, not a confound to be dismissed —
it may be the whole effect.

## The hypothesis

The inner-CV objective is too noisy to select on. Each outer-training partition holds roughly 51
positives; the inner 4-fold pools them into a single average-precision figure, and Optuna then
takes the **maximum over 15 trials** of that figure. Maximising a noisy statistic selects partly
for noise, and the selected configuration carries that noise out to the outer fold, where it does
not reproduce. Under that account, tuning does not merely fail to help — it actively costs
performance, and the cost is roughly the 0.023 seen here.

This is a well-understood failure mode at this sample size. It would also explain the untuned
arm's *lower* spread across repeats (0.01179 against 0.02949): a fixed configuration cannot
inherit variance from a selection step that does not happen.

## Why it is not a finding yet

The comparison above is between two means. It has not been paired, and pairing is exactly what
makes fold-held-fixed comparisons informative.

Note also that `artifacts/metrics/nested_cv.json` on disk at the time of writing still holds an
**older, underpowered run** (1 repeat, 2 trials, `xgboost 0.66792`). Its numbers must not be used
for this comparison.

## How to resolve it

The full 15-pair test is possible without re-running anything, because both halves are already
persisted:

* `artifacts/predictions/nested_oof.parquet` — written by `nested_cv.py:305` with one row per
  (model, repeat, row_index) carrying `target` and `score`;
* `artifacts/splits/nested_cv_assignments.parquet` — 108,960 rows of
  `(repeat, outer_fold, row_index, role, inner_fold)`, verified present.

Joining the two on `(repeat, row_index)` and filtering `role == "valid"` recovers per-outer-fold
average precision for the tuned run, which pairs directly against the ablation's 15 stored
`fold_ap` values.

1. When the nested tournament finishes, perform that join and run the paired comparison on all 15
   folds — the same three tests used in the missingness ablation (sign, Wilcoxon, paired t), and
   report all three rather than the most convenient.
2. If the effect survives pairing, the correct response is *not* to delete tuning and quote the
   higher number. It is to report both, state that the tuning budget is not affordable at this
   number of positives, and prefer the simpler estimator on the grounds of stability — which is a
   defensible engineering decision, where "the untuned number was bigger" alone is not.

## What must not be done with this

This note must not be used to select whichever configuration reports better. The 0.79068 came from
an arm of an ablation designed to answer a different question, and promoting it on that basis would
be exactly the selection-by-inspection this project has avoided elsewhere. Nothing is re-promoted
on the strength of this note.
