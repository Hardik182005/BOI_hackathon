# Hidden-Validation Strategy

Master prompt §8–10 and §45. The organiser's hidden validation set is the only
score that decides this competition, and §45 is blunt about the ordering:

> **DO NOT SKIP STEPS 3–8 TO WORK ON THE GRAPH OR UI FIRST. The hidden
> validation score is the priority.**

This document states what we optimised for, what we deliberately refused to
optimise for, and what the evidence says about how the champion will behave on
data it has never seen.

---

## 1. The premise: the local score is not the goal

The training extract is 9,082 accounts × 3,925 columns with **81 positives**
(prevalence 0.891 %). At that scale, the difference between a model that
generalises and a model that has memorised 81 rows is invisible in a single
number. Every design decision below follows from that.

Concretely, three things can inflate a local number without moving the hidden
score at all: **target leakage**, **evaluation reuse**, and **fitting to
81 particular positives**. Each has its own countermeasure.

| Failure mode | Countermeasure | Document |
|---|---|---|
| Leakage — features that exist only because the label exists | Feature Availability Firewall; 13 columns hard-quarantined | `FEATURE_AVAILABILITY_AUDIT.md`, `LEAKAGE_AUDIT.md` |
| Evaluation reuse — tuning against a test set until it stops being a test set | locked holdout, touch-logged, never used for selection/calibration/thresholds | §3 below |
| Positive overfit — a model that needs *these* 81 mules | positive-removal stress test, 15 rounds | §5 below |

---

## 2. Track A — repeated stratified out-of-fold (the selection metric)

**5 folds × 3 repeats, stratified, at natural prevalence.** Every fold carries
roughly 4–5 positives in validation, which is small enough that a single fold's
PR-AUC is nearly meaningless; the repeats are what make the mean readable.

Rules that make the number honest:

- **Preprocessing, imputation, feature selection and calibration all happen
  inside the fold.** Any of them fitted outside would let information from the
  validation rows into the model that scores them.
- **PR-AUC is the primary metric.** At 0.89 % prevalence, ROC-AUC is dominated by
  the negatives and a model can look excellent at 0.96 ROC-AUC while being
  useless in the top-100. Accuracy is not used as a selection metric anywhere in
  this repository.
- **Recall@TopK is reported alongside**, because the deployed artefact is an
  analyst queue with a finite budget, not a probability.
- Never resampled: no SMOTE, no class rebalancing that would make the OOF number
  describe a portfolio nobody will ever see.

Champion result: **`xgboost_top_120`, OOF PR-AUC 0.76904 ± 0.02663**, ROC-AUC
0.95771, Recall@100 0.78125 (`artifacts/metrics/tournament_v2.json`).

### 2.1 What counts as a real difference

Before comparing candidates we measured how much the OOF number moves for
reasons that have nothing to do with the model. Five seeds × two repeats of the
**identical** champion configuration (`seed_variance_v2.json`):

| | |
|---|---:|
| PR-AUC mean | 0.76294 |
| PR-AUC std | 0.03222 |
| min / max | 0.71618 / 0.80668 |
| **spread** | **0.0905** |

The same model, changing nothing but the random seed, spans 0.09 PR-AUC. So
**differences below 0.0905 between candidate models are not differences.** This
figure is published and is what the tie-break rule is calibrated against; it is
the reason we do not report a leaderboard ordering to four decimal places and
call the top row a winner.

---

## 3. Track B — the locked holdout

**1,818 rows, 17 positives (prevalence 0.935 %)**, stratified, split before any
modelling, mask persisted at `artifacts/splits/` and asserted absent from
`cv_folds.parquet` by `run_oof`.

What the locked test is **never** used for:

- feature selection
- calibration fitting
- threshold tuning
- model selection or promotion
- early stopping

Every touch is appended to `artifacts/metrics/locked_test_touch_log.json` with a
timestamp, the bundle SHA, and the resulting PR-AUC. Three touches are on record
against the retired `04fafaee25ae82c7` bundle; the current champion's locked-test
figure was produced once, through the Validation Lab, during the §42 rehearsal.

**Champion on the locked test: PR-AUC 0.7263**, ROC-AUC 0.9665, lift 77.7×,
Recall@100 0.824 (`artifacts/metrics/organiser_dry_run.json`).

0.7263 against 0.7690 OOF is the ordinary direction and size of drop for 17
positives. It is reported as measured. We did not re-tune to close it — closing
it is exactly the act that would turn the holdout into a second training set.

Addendum UPDATE 1 puts this as a standing rule: **never tune again against the
already-viewed locked test.**

---

## 4. Track C — out-of-time: recorded as impossible, not fabricated

§10 asks for an out-of-time evaluation. On this extract it cannot be done
honestly, and we publish the reason rather than a number.

Measured from the raw file (`artifacts/metrics/temporal_stress_metrics.json`),
the snapshot-month column `F2230` deterministically separates the classes:

| Snapshot month | Accounts | Positives |
|---|---:|---:|
| 2025-09 | 48 | 48 |
| 2025-10 | 9,001 | 0 |
| 2025-11 | 23 | 23 |
| 2025-12 | 10 | 10 |

Balanced label reconstruction from `F2230` alone = **1.0**. Every negative sits
in one month; every positive sits in one of three others. A "train on earlier
months, test on later months" split would measure the label-collection process,
not generalisation — and would return a spectacular, meaningless score.

Disposition: `F2230` is quarantined (`configs/leakage_quarantine.yaml`), Track C
is recorded with status `NOT_VALID` and its evidence, and primary evaluation
rests on Tracks A and B. This is the single clearest example of the principle
that governs the whole strategy: **a published impossibility beats a fabricated
metric.**

---

## 5. Rare-positive stress: does the model need *these* 81 mules?

Addendum UPDATE 4. `artifacts/metrics/stability_stress_v2.json` — **15 rounds**,
each removing **12.5 %** of the training positives (mean 30 positives removed per
round) and refitting. The evaluation fold is identical in every round, so the
spread is attributable to label scarcity and nothing else.

| Statistic | Value |
|---|---:|
| reference PR-AUC | 0.74927 |
| positive-removal PR-AUC mean | 0.72398 |
| `positive_removal_pr_auc_std` | **0.03545** |
| PR-AUC min over rounds | 0.64135 |
| relative drop | **3.37 %** |
| `positive_removal_recall_std` @25/@50/@100 | 0.0053 / 0.0283 / 0.0403 |
| `prediction_rank_stability` (all rows) | **0.3694** |
| `top_budget_rank_stability` @25/@50/@100 | 0.608 / 0.677 / 0.449 |
| `feature_rank_stability` | 0.7696 |
| `feature_top20_overlap` | 0.6805 |

Removing an eighth of the mules costs **3.4 %** of PR-AUC. The model is not
resting on a handful of memorised positives.

`feature_rank_stability` answers a question the PR-AUC cannot: when the mules it
learned from change, does the model keep citing the same evidence? A detector
whose reasons rewrite themselves between refits cannot be explained to an
analyst, and would produce a different ProofGraph for the same account depending
on which positives happened to be in the training set.

**The 0.77 above is the flat protocol's answer, and it is the wrong one to
quote.** The nested run of the same stress
(`artifacts/metrics/nested_positive_removal.json`, 2026-08-13) reports
`feature_rank_correlation_mean` **0.3944** — roughly half. The gap is structural
rather than noise: the flat stress selects features **once**, over pooled
development data, before the rounds begin, so dropping training positives cannot
disturb a choice that has already been made. The nested stress re-selects inside
every outer fold, which is what a refit in production would actually do, and
under that treatment most of the ranking moves.

So the reassuring reading of this paragraph does not survive. The PR-AUC
robustness holds — the nested drop is 1.77 %, paired 95 % CI
[−0.0213, −0.0063] — but the *explanation* stability does not: when the mules
change, the model largely rewrites what it cites. That is a real limitation of
citing feature importances as evidence at 64 positives, and it is recorded here
rather than left as the higher number.

---

## 6. The robustness badge, read off fixed thresholds

`artifacts/metrics/robustness_grade_v2.json` publishes
**Hidden Validation Robustness: LOW**.

Addendum UPDATE 4 says the status must come from documented thresholds and must
not be invented. The thresholds live in `muleguard.models.robustness`, were fixed
**before** the experiments ran, and were not revised after seeing the results:

| Criterion | HIGH | MEDIUM | Measured | At HIGH? |
|---|---:|---:|---:|:--:|
| positive-removal PR-AUC relative drop | ≤ 0.15 | ≤ 0.30 | **0.0337** | ✅ |
| positive-removal PR-AUC std | ≤ 0.06 | ≤ 0.12 | **0.03545** | ✅ |
| prediction rank stability | ≥ 0.90 | ≥ 0.75 | **0.3694** | ❌ |
| family-dropout worst relative drop | ≤ 0.30 | ≤ 0.50 | **0.1215** | ✅ |

The grade is **the worst of its criteria, not their average**, so three passes at
HIGH and one miss produces LOW. `limiting_criteria` names the one that decided
it, so the badge can be reported without implying the model failed everywhere.

**We publish LOW rather than redefining the metric to earn a better badge.** The
honest caveat is published alongside it, in the artefact itself: about 99 % of
development rows are negatives whose calibrated scores sit in a narrow band near
zero, so their relative order moves freely between rounds without any account
changing review status. All-row Spearman is therefore weak evidence on its own,
which is why `top_budget_rank_stability` (0.61 / 0.68 / 0.45 at the budgets an
analyst actually works) is reported next to it — as a **diagnostic, not as an
input to the badge**. Changing the badge's definition after seeing the result is
precisely the move the fixed-threshold policy exists to prevent.

---

## 7. Promotion rule (addendum UPDATE 13)

`artifacts/metrics/promotion_decision_v2.json`:

```
generalization_score = PR_AUC_mean − 0.5 × PR_AUC_std + 0.1 × Recall@100
```

Applied **only as a tie-break**, within a 0.01 PR-AUC band of the leader; inside
that band the simpler and more stable model wins. It is not a ranking function
applied to the whole leaderboard, because a composite score applied everywhere
would quietly become the thing we optimise.

Champion: `xgboost_top_120`, generalization score **0.83385**, tie-break not
required (`tie_break_applied: false` — it led outright).

UPDATE 4's clause — *"prefer a slightly lower-scoring model if it is materially
more stable"* — is live policy here, and §8 of `UPGRADE_GAP_ANALYSIS.md` records
the case where it bit: the highest-scoring model in the tournament was **not**
promoted.

---

## 8. Generalisation over score: the TabPFN decision

`tabpfn_top_60` scored **0.9110 ± 0.0044** OOF — 0.14 PR-AUC above the champion,
far outside the 0.0905 noise floor, on firewall-admitted columns only, verified
across three independent fold seeds as UPDATE 1 requires. Fold independence
0.1989–0.2026 against 0.2000 chance; zero quarantine overlap; and the control
that settles it — `xgboost_top_60` on the *identical* 60 columns over the
*identical* folds scores 0.7408, so the gap is the model, not the feature set.

It is **not promoted**, and not because the number is doubted. Measured serving
cost (`artifacts/metrics/tabpfn_latency.json`): fit 1.2 s, but **438 s for a
single-row `predict_proba`**, because a prior-fitted transformer carries the
whole training set through the forward pass on every call — the cost is per
*call*, not per row. With no attribution path, a §17 ProofGraph would need
occlusion at ≈ 7.4 hours per case.

Full reasoning and the four UPDATE 1 gates: `UPGRADE_GAP_ANALYSIS.md` §3.1.1,
`artifacts/metrics/challenger_review_v2.json`
(`challenger_status: VERIFIED_NOT_PROMOTED`).

> **This decision is reversible and the conditions are written down.** If the
> organiser's track is batch-scored and needs no per-case explanation, TabPFN is
> the stronger submission and the evidence to promote it is already on disk.

---

## 9. What we refused to do

| Refused | Why |
|---|---|
| Tune against the locked test after viewing it | UPDATE 1 — it stops being a holdout the moment you do |
| Use hidden-validation labels to retrain, recalibrate or re-threshold | UPDATE 3, enforced by the Sealed Validation Protocol |
| Report accuracy as the headline metric | 99.1 % accuracy is achievable by predicting "legitimate" for everyone |
| Fabricate an out-of-time score | §4 above |
| Redefine `prediction_rank_stability` to earn a better badge | §6 above |
| Chase "100 % accuracy" or exploit `F3912` | the leaked variant scores 0.9419 and is published as **REJECTED LEAKAGE — evidence only** |
| Resample to a friendlier prevalence | the hidden set will be at natural prevalence |

---

## 10. The rehearsal

§42's dry-run is the strategy's final check: the locked-test rows, target
removed, uploaded through the Validation Lab over HTTP in **eight** malformed
shapes an organiser's export might actually take. All eight scored; the five that
should be indistinguishable produced **byte-identical predictions**; the
sensitivity control differed; the bundle fingerprint was unchanged before and
after every upload.

Full record: `docs/ORGANISER_DRY_RUN.md`. Enforced by release-gate check
`organiser_dry_run_passed`.
