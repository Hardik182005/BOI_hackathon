# Missed-Mule Error Atlas

Generated 2026-08-12T12:43:45.366291+00:00 from `artifacts/metrics/error_atlas.json`. Do not edit by hand - regenerate with `python -m muleguard.cli.error_atlas`.

## What this is

For every account carrying a mule label in the development data that the served champion did **not** place inside the analyst review budget, this measures why and assigns one category.

It is a diagnostic instrument, not a patch mechanism. It never modifies a score, a threshold, a prediction, a feature set or a model; it hashes its inputs before and after to prove it. No language model is involved - every field is a measured quantity, a configuration value, or a fixed string written by hand in this repository. Everything is development out-of-fold; the locked test split was not read.

**The rule that matters:** a pattern found here leaves this system as a hypothesis for nested cross-validation. It does not become a rule, a feature or an override. Writing a special case for any individual account listed below would be a failure of this exercise.

## Scope

| | |
|---|---|
| Champion | `xgboost_top_120` |
| Analyst budget K | 100 (configs/thresholds.yaml tiers.urgent_review.daily_alert_capacity) |
| Development rows | 7264 |
| Labelled mules | 64 (prevalence 0.0088) |
| Surfaced within budget | 53 |
| **Missed** | **11** |
| Recall at budget | 0.8281 |

> 11 misses. Every count below is a count over 11 accounts. Nothing here is a rate that would survive a confidence interval, and no pattern seen in a handful of accounts should be treated as established. The categories describe these accounts; they do not estimate a population.

## Which number counts as a miss

Repeated cross-validation can be collapsed to a single ranking three ways, and they disagree. All three are reported; the Atlas uses `score_average` because `build_lenses_v2` averages raw scores to derive the thresholds the system serves, so that is the deployed operating point - not the most flattering one.

| Convention | Recall@K | Misses |
|---|---|---|
| score_average | 0.8281 | 11 |
| rank_average | 0.8125 | 12 |
| per_repeat_mean | 0.7812 | [16, 13, 13] |

## Category definitions

Six categories, one per miss. Rules are evaluated in the order below and the **first rule that fires wins**; every rule is still evaluated and recorded, so the artifact shows which explanations were available and which the order selected. The order runs from *the model could not have known* to *the model had every chance and still did not*.

Constants: `{'missingness_z': 4.0, 'range_violation_share': 0.2, 'missing_fraction': 0.5, 'threshold_miss_capacity_multiple': 2.0, 'neighbourhood_k': 10}`. Four of the six rules have no free parameter - their tests are definitional. `missingness_z` and `range_violation_share` are the served OOD lens's own published thresholds, reused rather than reinvented. `threshold_miss_capacity_multiple` is the one chosen number.

| # | Category | Exact test | Count |
|---|---|---|---|
| 1 | `MISSING_DATA` | `missingness_z >= 4.0 OR missing_fraction >= 0.5` | 0 |
| 2 | `OOD_PATTERN` | `knn_distance > knn_threshold (the lens's own dev-quantile gate) OR range_violation_share > 0.2` | 0 |
| 3 | `THRESHOLD_MISS` | `champion_rank <= 2.0 x budget_k` | 2 |
| 4 | `MODEL_DISAGREEMENT` | `n_peer_families_within_budget >= 1` | 5 |
| 5 | `LOOKALIKE_MULE` | `n_mule_labels_in_neighbourhood == 0 AND distance_to_nearest_legitimate_account < distance_to_nearest_known_mule` | 4 |
| 6 | `LOW_SIGNAL_MULE` | `residual - reached only when no rule above fired` | 0 |

**MISSING_DATA** - The champion could not read evidence that was not there. Both thresholds are the deployed OOD lens's own published values, so the Atlas does not invent a second definition of 'too much missing data'.

**OOD_PATTERN** - The evidence was present but unlike the development cohort, so the champion was extrapolating. Delegated verbatim to the OOD lens the system already serves.

**THRESHOLD_MISS** - The champion did rank the account near the top of the book; what excluded it was the size of the review queue, not the model's opinion. Placed above the estimator rules because 'the operating point cut it' is the more parsimonious explanation when both apply.

**MODEL_DISAGREEMENT** - At least one other model family, scored out-of-fold on the same folds and the same labels, would have placed this account inside the same budget. The signal was therefore recoverable from this data; this champion did not recover it.

**LOOKALIKE_MULE** - In the champion's own input space this account sits inside the legitimate book: no mule-labelled account is among its nearest neighbours, and a legitimate account is nearer than any mule-labelled one. Both tests are definitional - 'zero' and 'closer than' - so there is no threshold to tune.

**LOW_SIGNAL_MULE** - No measured quantity separated this account from the legitimate book: the evidence was present and in distribution, the champion did not rank it near the budget, no peer family surfaced it, and its neighbourhood is not exclusively legitimate. Recorded as an unattributed miss.

## Do these rules actually discriminate?

A rule that fires just as often on the mules the champion *did* surface explains nothing. Firing rates over the 53 labelled mules the champion DID surface within the budget:

| Rule | Fires on caught mules |
|---|---|
| `missing_data` | 0 / 53 (0.000) |
| `ood_pattern` | 0 / 53 (0.000) |
| `threshold_miss` | 53 / 53 (1.000) |
| `model_disagreement` | 52 / 53 (0.981) |
| `lookalike_mule` | 16 / 53 (0.302) |

## Per-miss findings

Distances are in the champion's robust-scaled input space; `d(mule)` and `d(legit)` are to the nearest mule-labelled and nearest legitimate development account respectively, the account itself excluded.

| Row | Champion rank | Category | d(mule) | d(legit) | Mules in 10-NN | Peer families in budget | Missing frac | Anomaly pct |
|---|---|---|---|---|---|---|---|---|
| 9010 | 423 | `MODEL_DISAGREEMENT` | 1701.45 | 937.46 | 0 | 1 | 0.158 | 52.8 |
| 9039 | 260 | `MODEL_DISAGREEMENT` | 7606.60 | 7309.28 | 0 | 1 | 0.133 | 83.1 |
| 9047 | 1949 | `LOOKALIKE_MULE` | 13.23 | 11.30 | 0 | 0 | 0.217 | 80.2 |
| 9053 | 115 | `THRESHOLD_MISS` | 7.18 | 6.98 | 2 | 1 | 0.100 | 63.1 |
| 9057 | 2131 | `LOOKALIKE_MULE` | 104959.62 | 321.41 | 0 | 0 | 0.142 | 48.1 |
| 9067 | 4804 | `LOOKALIKE_MULE` | 17.89 | 8.56 | 0 | 0 | 0.192 | 61.1 |
| 9069 | 201 | `MODEL_DISAGREEMENT` | 394.01 | 391.48 | 1 | 3 | 0.175 | 13.4 |
| 9072 | 298 | `MODEL_DISAGREEMENT` | 15562.20 | 445.85 | 0 | 1 | 0.117 | 69.2 |
| 9077 | 6125 | `LOOKALIKE_MULE` | 601.42 | 37.29 | 0 | 0 | 0.150 | 44.5 |
| 9078 | 788 | `MODEL_DISAGREEMENT` | 14.78 | 8.89 | 0 | 1 | 0.133 | 20.1 |
| 9081 | 198 | `THRESHOLD_MISS` | 15562.20 | 696.89 | 0 | 2 | 0.175 | 86.2 |

Full evidence per account - all four family scores, top attributions, missingness by feature family, merchant/context verdict, anomaly percentile, neighbour distances and every rule's verdict - is in `artifacts/metrics/error_atlas.json` under `misses`.

**Neighbour fields are feature-space proximity only.** This dataset has no edge table. "Closest known mule" means the nearest account in the champion's own 120-column input space after robust scaling. It is not a transfer, a shared counterparty or any relationship between the two accounts, and it must never be presented as one.

## Cross-family agreement

Same folds, same labels, same budget.

| Family | Own misses | Catches n of champion's 11 | Has misses champion catches |
|---|---|---|---|
| `catboost_top_120` | 15 | 1 | 5 |
| `lightgbm_top_120` | 15 | 2 | 6 |
| `tabpfn_top_60` | 5 | 7 | 1 |

4 of the champion's misses were outside the budget for **every** measured family.

## Observations on the served OOD lens

`OOD_PATTERN` delegates to the lens the system already serves rather than inventing a second definition, so what that lens does on this data decides what the category can mean. Zero misses landed in it, and the reason matters.

| | |
|---|---|
| k-NN gate | 38947879.66 (quantile 0.999 of the development k-NN distances, k=10 (configs/thresholds.yaml ood.knn_quantile)) |
| Largest development k-NN distance | 346942400.00 |
| Median development k-NN distance | 114.98 |
| Development rows above the gate | 8 |
| Rows the lens flags in total | 11 |
| ...of which carry a mule label | 0 |

The gate **does** fire - it is a development quantile, so a few rows sit above it by construction. The lens flags 11 development rows and none of them carries a mule label. So the OOD_PATTERN count of zero among the misses is not an artefact of an incapable gate - the gate does fire, just never on these accounts.

The range-violation limb is a different matter: its maximum over all development rows is 0.0000. Zero, and necessarily so: the lens's accepted ranges were widened from these same development rows, so no development row can violate them. This limb is a production-drift detector and carries no information in this in-sample analysis. Reported so that a zero count is not mistaken for evidence.

**The Atlas changed none of this.** Adjusting a served threshold is outside what this instrument is permitted to do. These are observations for whoever owns the lens.

## Attribution method

Method: **EXACT_TREESHAP**. Model-native feature importances were NOT used. Per-row exact TreeSHAP was affordable at this scale, so no substitution was made.

The tournament stored scores but not per-fold models, so each scoring model was rebuilt with the same folds, preprocessor and deterministic fold seed, then checked against its stored out-of-fold scores across the whole validation fold. Fidelity:

> EXACT - every refit reproduced its stored out-of-fold scores to 0.0 absolute, so these attributions come from the same model that produced the score being explained

Reproduction required `OMP_NUM_THREADS=4`. XGBoost's `hist` builder sums gradients in thread order, so at a different thread count the refits diverged from the stored scores by up to 0.16 absolute. That is a property of the estimator, not a bug found here, but it means attributions computed at the wrong thread count would explain a model that never scored these accounts.

## Hypotheses for nested-CV testing

> **None of these has been acted on. Nothing in this file has changed a score, a threshold, a feature set or a model, and none of these hypotheses may do so until it has beaten the existing champion through nested cross-validation. Hand-writing a rule for any individual account listed in this artifact would be a failure of the exercise, not a fix.**

### H1_recall_oriented_ensemble

- **Status:** `UNTESTED - NOT IMPLEMENTED, NOT WIRED, NO EFFECT ON ANY SCORE`
- **What triggered it:** tabpfn_top_60 catches 7 of the champion's 11 misses at K=100, and has 1 miss the champion catches.
- **Hypothesis:** A blend of xgboost_top_120 with tabpfn_top_60, selected on recall at the analyst budget rather than on PR-AUC, surfaces more labelled mules within the same queue than xgboost_top_120 alone.
- **How to test it:** Nested CV with recall@K as the selection objective, blend weights chosen in the inner loop only, scored in the outer loop. Must beat the champion on the outer folds, not on these misses.
- **Why it might fail:** challenger_review_v2 already found the rank blend does not beat the best single member on PR-AUC. Recall at a budget is a different objective, but the prior is not encouraging, and 7 overlapping accounts is far too few to select a blend weight on.

### H2_neighbourhood_label_density_feature

- **Status:** `UNTESTED - NOT IMPLEMENTED, NOT WIRED, NO EFFECT ON ANY SCORE`
- **What triggered it:** 4 misses sit in a neighbourhood with no mule-labelled account in it and a legitimate account nearer than any mule-labelled one; the same test fires on 16 of 53 caught mules, so it separates but far from cleanly.
- **Hypothesis:** A feature counting mule labels among an account's k nearest neighbours in the champion's input space adds signal the tree models are not extracting.
- **How to test it:** Nested CV. The neighbour graph and the label counts must be built inside each training fold only - building them on all development rows leaks the validation labels straight into the feature and will produce a spectacular and entirely false gain.
- **Why it might fail:** With this prevalence a fold-local neighbourhood contains almost no positives, so the feature may be near-constant zero and add nothing.

### H3_missingness_is_not_the_explanation

- **Status:** `OBSERVATION - NOT ACTED ON`
- **What triggered it:** Zero misses met either missingness test.
- **Hypothesis:** A negative result, recorded so it is not rediscovered: missing data does not explain these misses. Missingness-structure work is worth doing on its own merits but should not be justified by this Atlas.
- **How to test it:** Nothing to test here; this entry exists to close off a wrong lead.
- **Why it might fail:** Not applicable.

### H4_operating_point_sensitivity

- **Status:** `OBSERVATION - NOT ACTED ON`
- **What triggered it:** 2 miss(es) were ranked within twice the budget - the champion ranked them highly and the queue length excluded them.
- **Hypothesis:** For these accounts the binding constraint is review capacity, not model quality. Their recovery is a capacity question and no model change would be credited for it.
- **How to test it:** Not a modelling change. If capacity is ever revised, the gain must be attributed to capacity and not reported as a model improvement.
- **Why it might fail:** Capacity is an operational constraint set outside this system.

### H5_irreducible_on_current_features

- **Status:** `UNTESTED - NOT IMPLEMENTED, NOT WIRED, NO EFFECT ON ANY SCORE`
- **What triggered it:** 4 miss(es) were outside the budget for every measured family.
- **Hypothesis:** These accounts are not separable by any estimator on the current feature set. If they are to be recovered it will take new evidence, not a new estimator.
- **How to test it:** Nested CV on any genuinely new evidence source. Re-tuning existing models against these accounts would be fitting to 4 rows and is exactly the failure mode this Atlas is written to avoid.
- **Why it might fail:** There may be no further evidence available in this dataset.

## Limits

- 11 misses. These are descriptions of 11 accounts, not estimates of a population. No confidence interval would survive this sample size.

- The anomaly detector, the OOD lens, the hard-negative verifier and the merchant verifier in the served bundle were fitted on development data. Their readings on these accounts are in-sample context, not independent corroboration. The IsolationForest is the exception: it was fitted on rows not carrying a mule label, so these accounts were outside its fit.

- Quarantined columns cannot appear anywhere above: the writer raises rather than emits if one is present.

