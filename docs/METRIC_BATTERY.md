# Metric Battery

Master prompt §24–27. Every evaluation metric the system reports, computed on one
set of predictions, with confidence intervals sized for 64 positives.

Authoritative artefact: `artifacts/metrics/metric_battery.json`.
Computation: `src/muleguard/models/metric_battery.py`.
Runner: `src/muleguard/cli/metric_battery.py`. Tests: `tests/unit/test_metric_battery.py`.

---

## 1. Why this document exists

The project already had PR-AUC in one artefact, budget recalls in another, tier
counts in a third and calibration error in a fourth. All four were correct. The
problem is that they were computed on subtly different things — a per-repeat mean
here, a pooled score there, a calibrated vector somewhere else — so assembling a
summary from them meant choosing, metric by metric, which number to quote. That
is how an honest project drifts into an optimistic one without anybody writing a
false number.

This battery computes everything from a single predictions file in a single pass,
records which vector each number came from, and writes down the places where two
legitimate calculations of "the same" metric disagree. Section 8 is that list. It
is the most useful part of this document.

### How it was produced

```
.venv/Scripts/python.exe -m muleguard.cli.metric_battery \
  --source artifacts/predictions/oof_v2.parquet --model xgboost_top_120 \
  --protocol FLAT --n-boot 2000 --fresh

.venv/Scripts/python.exe -m muleguard.cli.metric_battery \
  --source artifacts/predictions/nested_oof.parquet --model xgboost \
  --protocol NESTED_PRELIMINARY --n-boot 2000

.venv/Scripts/python.exe -m muleguard.cli.metric_battery \
  --source artifacts/predictions/holdout_predictions.parquet \
  --model retired_gen1_pre_firewall_stack --protocol HOLDOUT_REFERENCE --n-boot 2000
```

`--source` is required and never defaulted. When the full nested run lands, the
first command is re-run against its predictions with `--protocol NESTED` and the
document gains a fourth entry; nothing here has to be rewritten by hand. Each run
records the source file's SHA-256 prefix, size, mtime and the `generated_utc` of
its companion metrics file, so a number can always be traced to the exact array it
came from.

---

## 2. Which numbers are primary

This is the part a reader should take away before any individual metric.

| Protocol | Status | May be used for | In this document |
|---|---|---|---|
| **NESTED** (full) | **PRIMARY — not yet available** | the honest headline | pending; see §7 |
| NESTED_PRELIMINARY | superseded, under-powered | nothing | quoted only to show the optimism gap |
| **FLAT** | historical development estimate | model selection (already done) | the numbers in §6 |
| HOLDOUT | **reference only, never selection** | a labelled historical footnote | §9, and it is inadmissible |

The full nested run was still executing when this battery was computed — catboost
was mid-way through its folds and xgboost had not started. Its
`artifacts/metrics/nested_cv.json` on disk is still the old 1-repeat / 2-trial
run, so **no nested number in this document is the nested result.** The flat
figures below are labelled historical throughout and the preliminary nested
figures are labelled superseded. Nothing has been quietly promoted.

The reason nested is primary and flat is not: in the flat protocol, feature
selection pooled importance across every development fold, so each row helped
choose the columns before it was scored as held-out. The optimism this introduces
is small in expectation but it is not zero at 64 positives, and it always points
the same way.

---

## 3. Ranking metrics — how good is the ordering

**PR-AUC (average precision)** is the primary metric. At 0.89 % prevalence a
random ranker scores about 0.0089, so the metric has almost the whole unit
interval in which to distinguish real skill, and every point of it is earned in
the region a reviewer actually looks at — the top of the list. It answers the
question the bank is asking: if I walk down this ranking, how dense are the mules?

**ROC-AUC** is reported second and deliberately demoted. It scored **0.95771**
here, which sounds far better than the PR-AUC of 0.76904 and means far less. ROC
integrates over the false-positive rate, and with 7,200 negatives an enormous
number of false positives moves that rate very little. A model can look excellent
on ROC while burying mules under hundreds of alerts. When these two disagree,
PR-AUC is the one describing the reviewer's day.

Both are computed by one tie-grouped `argsort` + `cumsum` pass, verified against
scikit-learn to a maximum absolute deviation of **1.110e-16** across 200
tie-heavy cases (`tests/unit/test_metric_battery.py`, section 1).

---

## 4. Operating metrics — what a reviewer's day looks like

PR-AUC summarises a whole curve. Nobody reviews a whole curve.

**Recall and precision at a fixed alert budget** (top 10 / 18 / 25 / 50 / 73 /
100 / 145 accounts; 1 % and 2 % of the book) answer the operational question
directly: with capacity for *k* reviews, how many of the 64 mules are caught and
how much of the queue is wasted. **Alerts-to-catch-50/75/90 %** inverts it: to
reach a target recall, how many reviews must be funded.

Three supporting numbers are reported alongside every budget:

- **Recall ceiling** — the recall achievable by a perfect ranker at that budget.
  At *k* = 10 with 64 positives the ceiling is 0.15625, so a recall of 0.15625 is
  a *perfect* score, not a bad one. Without the ceiling, small budgets look like
  failures.
- **Needles per review (NNR)** — alerts examined per mule found; the inverse of
  precision, in the unit a reviewer feels.
- **False positives per 1,000 accounts** — the disruption rate for ordinary
  customers, which is what a false-positive budget is actually written in.

**Accuracy is computed and then explicitly labelled uninformative.** Predicting
"no mule" for every account scores 99.11 % here. The STANDARD tier scores 96.118 %
and is far more useful. A metric that rewards silence cannot be allowed to appear
without that sentence next to it; there is a test asserting exactly this
(`test_accuracy_is_uninformative_at_low_prevalence`).

**F1, F2, MCC** are reported at each frozen threshold. F2 is included because a
missed mule costs more than a wasted review, and MCC because it is the one summary
that uses all four cells of the confusion matrix and stays honest under extreme
imbalance.

---

## 5. Probability quality, and the intervals

### 5.1 Calibration

The tiered policy is defined on a probability, so the probability has to mean
something. **Brier score** is reported against a base-rate reference: 0.0031276
versus 0.0087329 for a constant-prevalence predictor, a **Brier skill of 0.6419**.
A raw Brier score at 0.89 % prevalence is unreadable — everything looks excellent
because everything is near zero — so the skill score is the honest form.

**ECE is reported under both binnings** (equal-width 0.0014895, equal-mass
0.000960) because at this prevalence they measure different things. Equal-width
bins put roughly the whole dataset in the first bin and mostly report the base
rate; equal-mass bins actually populate the high-risk region where the thresholds
live. Quoting only one is how a calibration claim gets made without support.

### 5.2 Interval method

**Percentile bootstrap over accounts, B = 2,000, α = 0.05, seed 42**, reported
under two schemes:

- **stratified** — positives and negatives resampled separately, so all 64
  positives are present in every replicate. This isolates ranking uncertainty.
- **resample_accounts** — the whole book resampled, so the mule count varies
  between replicates. This is the wider and more honest interval, because in a
  real portfolio the number of mules is not fixed either.

The axis matters more than the scheme. **The bootstrap resamples accounts, never
CV repeats.** The three repeats are three views of the same 7,264 people; treating
their spread as a confidence interval would describe how stable the training
procedure is, not how uncertain the estimate is. Reporting "± 1 sd over repeats"
would give ± 0.02663 where the bootstrap gives a half-width of 0.0884 — an
interval 3.3 times too narrow, and narrow in the flattering direction. The statistic
bootstrapped is the *mean over repeats*, so each replicate draws one account sample
and evaluates every repeat against it. Two tests enforce this: one feeds three
identical repeats and requires the interval to stay wide despite zero repeat
spread; another re-implements the draw independently and demands agreement to
1e-12.

Ranking and budget intervals are computed on the per-repeat score matrix; threshold
intervals are computed on the calibrated probability vector, because a frozen
threshold is only meaningful on the scale it was frozen on. Every interval row in
the JSON carries a `computed_on` field saying which.

Some intervals are **degenerate and marked so**. Top-10 recall has a stratified CI
of [0.1562, 0.1562]: with all 64 positives always present and the top 10 always
containing the same 10, there is nothing left to vary. That is a real property of
the estimator, not a bug, and the account-resampling scheme gives it the honest
width of [0.1235, 0.2000].

---

## 6. Headline numbers — FLAT protocol (historical)

`xgboost_top_120`, dev OOF, 7,264 accounts, 64 mules (0.881057 %), 3 repeats × 5
folds, leakage-free v2 feature pool. Source `oof_v2.parquet`, SHA `69c73c8f…`,
companion generated 2026-08-12T09:39:29Z.

**One mule is 1.5625 percentage points of recall.** Every recall figure below moves
in steps that size. Differences smaller than one step are not differences.

| Metric | Point | Stratified 95 % CI | Account-resample 95 % CI |
|---|---:|---|---|
| **PR-AUC** | **0.76904** | [0.67630, 0.85310] | [0.67555, 0.85245] |
| ROC-AUC | 0.95771 | [0.92165, 0.98649] | [0.92317, 0.98742] |

Recall at budget:

| Budget | Recall | Stratified CI | Precision | Ceiling | NNR | FP/1k |
|---:|---:|---|---:|---:|---:|---:|
| 25 | 0.3906 | [0.3802, 0.3906] | 1.0000 | 0.3906 | 1.00 | 0.00 |
| 50 | 0.6458 | [0.5729, 0.7240] | 0.8267 | 0.7812 | 1.21 | 1.20 |
| 100 | 0.7812 | [0.6875, 0.8646] | 0.5000 | 1.0000 | 2.00 | 6.94 |
| 145 | 0.8125 | [0.7240, 0.8958] | 0.3586 | 1.0000 | 2.79 | 12.92 |

At the frozen thresholds (`policy_version 1.0`, unchanged by this work):

| Tier | Threshold | Alerts | TP | FP | Recall | Precision | F1 (95 % CI) | MCC (95 % CI) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| CRITICAL | 0.93385 | 25 | 25 | 0 | 0.3906 | 1.0000 | 0.5618 [0.4385, 0.6804] | 0.6233 [0.5283, 0.7165] |
| URGENT | 0.09774 | 100 | 53 | 47 | 0.8281 | 0.5300 | 0.6463 [0.5780, 0.7195] | 0.6589 [0.5897, 0.7322] |
| STANDARD | 0.01318 | 334 | 58 | 276 | 0.9062 | 0.1737 | 0.2915 [0.2624, 0.3228] | 0.3873 [0.3521, 0.4214] |

Probability quality: Brier 0.0031276 (base 0.0087329, **skill 0.6419**), log-loss
0.015777 (base 0.050462), ECE 0.0014895 equal-width / 0.000960 equal-mass, MCE
0.002543. Fold-level AP across 15 folds: mean 0.78174, **sd 0.12562**, min
0.50176, max 1.00000.

The CRITICAL tier's 25 alerts at 100 % precision and the URGENT tier's 100 alerts
at 0.828125 recall reproduce `capacity_curve.json` exactly; PR-AUC 0.76904 ± 0.02663
reproduces `tournament_v2.json` exactly. Those are reproductions, not new claims.

---

## 7. Preliminary nested (superseded — do not quote)

One repeat, two Optuna trials. **PR-AUC 0.66792**, stratified CI [0.55686, 0.77186].
Recorded only so the full run can be compared against it, and because the gap to
the flat 0.76904 is informative in one specific way: the two intervals overlap
heavily, so this run does **not** establish a selection-optimism penalty. It is
too weak to establish anything. The real comparison waits for the full nested run.

One number from it deserves attention when the real run lands: alerts to catch
90 % of mules was **3,569 — 49.13 % of the entire book** — against 579.7 in the
flat run. Deep-tail recall is the least stable thing this system does, and any
claim about catching 90 % of mules should be treated as unsupported until the
nested run says otherwise.

---

## 8. Where two metrics disagreed

Under §67 a conflict is investigated, not resolved by picking the better number.
Six were found.

### 8.1 Four defensible PR-AUCs from the same predictions

| Aggregation | PR-AUC |
|---|---:|
| **mean of per-repeat AP (reported)** | **0.76904** |
| AP of the mean score | 0.80811 |
| AP of the mean rank | 0.79328 |
| AP of the crossfit-Platt calibrated mean score | 0.80465 |

Per-repeat APs are 0.74927, 0.80668, 0.75117. The spread is not noise in the
metric — averaging three independently-fitted models denoises the ranking, and an
ensemble of three is genuinely better than any one of them. **The served bundle is
a single fit.** So 0.76904 describes what ships and 0.80811 describes an ensemble
that does not exist. The headline is the lower number, recorded in the artefact as
`headline_choice: "mean_of_per_repeat"`.

### 8.2 The same conflict, hiding in the budget table

Top-100 recall reads **0.7812** in §6 and the URGENT tier — also 100 alerts —
reads **0.8281**. Same accounts, same budget, 3 mules apart. Per-repeat recalls at
*k* = 100 are 0.7500, 0.7969, 0.7969 (mean 0.7812); on the pooled and calibrated
vectors it is 0.8281. It is §8.1 again, and it propagates to every budget: top-50
is 0.6458 per-repeat against 0.6875 pooled. Both are tagged `computed_on` in the
JSON. The lower one is the one describing a single deployed model.

### 8.3 A spec-named artefact quotes the higher aggregation

`artifacts/metrics/holdout_metrics.json → dev_oof_reference` reports dev OOF
PR-AUC as **0.8046500485848826** with CI [0.71154, 0.88303] — the calibrated
pooled figure from §8.1 — while the tournament artefact and this battery report
0.76904. Neither is wrong; they are different estimators of different things, and
this battery reproduces both from the same file. Anyone comparing the two
artefacts should know they are not comparable, which is why it is written here.

### 8.4 Alerts-to-catch-90 % is a mean of a wildly skewed quantity

The three repeats need **1,149, 342 and 248** alerts to reach 90 % recall; the
mean is 579.7 and the pooled vector needs 298. One repeat placing one mule very
deep moves the mean by hundreds. The stratified CI is [148, 2617] — nearly
useless, and correctly so. **This metric should be read as an order of magnitude,
never as a plan.** The 50 % figure (32.3 alerts, CI [32, 38]) is solid; the 90 %
figure is not.

### 8.5 Isotonic wins on calibration and damages the ranking

Isotonic beats Platt on equal-mass ECE (0.000731 vs 0.000960) but its PR-AUC is
**0.75531 against 0.80465** — fitting a step function to 64 positives collapses
score ties and destroys ordering information. Under the pre-registered rule
(isotonic must beat Platt by ≥ 2 % relative on **both** Brier and ECE) Platt wins
on Brier and therefore wins outright, reproducing the stored selection exactly.
The rule was written before the comparison, which is the entire point of it.

### 8.6 Two artefacts, one `policy_version`

`artifacts/metrics/lens_stack_oof_v2.json` carries thresholds
0.9338487190474695 / 0.09773832436704985 / 0.013182620557866929 and
`artifacts/model_registry/policy_snapshot.json` carries
0.9638304837100166 / 0.048153197236171764 / 0.017246147513913988 — and **both are
labelled `policy_version "1.0"`**. The registry snapshot is the older
generation-1 file (mtime 2026-07-10T20:47:22); the v2 lens artefact matches the
frozen thresholds in use and is what this battery used. This is a versioning
defect, not a metric defect, and it is recorded here because a stale snapshot
sharing a version string with the live policy is exactly the kind of thing that
silently ships. Fixing it was out of scope for this task — no threshold or policy
file was modified.

---

## 9. The holdout is inadmissible, and not for the obvious reason

`docs/LOCKED_TEST_RULING.md` rules the 1,818-row / 17-mule set **HISTORICAL
HOLDOUT — FOR REFERENCE ONLY**: no feature selection, no calibration, no threshold
tuning, no model selection against its labels. This battery honours that — it runs
no threshold search on that split and refuses to fit a calibrator against it.

Running the battery there surfaced something worse than a rule violation. The
stored vector scores **PR-AUC 0.86077**, the highest number in the project, and it
is not the champion's. Its calibrated column reproduces
**0.8242261397312548** — bit-for-bit the retired pre-firewall run's recorded
figure. That identifies the vector positively: it belongs to the **retired
generation-1 stack on a pre-firewall feature pool**, a model that was rejected.

The champion's own reportable holdout figure is **PR-AUC 0.7262714933700882** /
ROC-AUC 0.9664892053434366, with recall 0.8235 at 100 alerts, from the sealed
protocol. **It has no confidence interval and cannot be given one** — the seal
revealed summary metrics and a SHA-256 of the predictions, not the predictions,
so there is no per-row vector to bootstrap. Re-scoring to obtain one would be a
second touch of a split already ruled reference-only.

Quoting 0.86077 would have meant reporting a rejected model's number as the
system's best result. The battery now refuses: `stored_vector_admissible_for_reporting`
is `false` and the CLI logs a warning.

---

## 10. Paired versus unpaired comparisons

The published **seed-noise floor of 0.0905 PR-AUC** applies to *independently-run*
configurations. Applying it to a paired comparison on identical folds is a
statistical error that would erase real effects — a paired design removes the
fold-to-fold variance that makes the unpaired floor so large, and must be judged
against its own much smaller spread.

The missingness ablation run by another agent is a clean worked example, and its
three paired tests disagree:

| Test | p |
|---|---:|
| sign test | 0.11847 |
| Wilcoxon signed-rank | 0.03534 |
| paired *t* (t = 2.5106) | 0.02495 |

Mean paired gain **+0.04945**, 95 % CI **[0.00721, 0.09170]**, improved in 11 of
15 folds (WITHOUT 0.79068 ± 0.01179, WITH 0.84827 ± 0.01921). The tests disagree
because they use different amounts of the data: the sign test sees only 11-of-15
and cannot reach significance with 15 observations, while the *t*-test uses the
magnitudes and is helped by a few large gains. **With 15 folds and 64 positives,
"significant" and "not significant" are both inside the noise of which test you
picked.** The interval is the honest summary: the effect is probably positive,
plausibly as small as +0.007, and the lower bound is what a decision should be
sized against. A follow-up arm excluding the suspect `FEES_AND_CHARGES` family is
still running, so the gain is **not settled** — if it depends on that family, the
constraint against accepting a win driven by a suspicious feature applies.

---

## 11. What these numbers cannot tell a judge

- **64 positives.** Every interval here is wide because the data is small, not
  because the method is timid. PR-AUC's CI spans 0.676 to 0.853; a competing
  system anywhere in that range is not distinguishable from this one.
- **Recall is quantised at 1.5625 pp.** Any comparison finer than one mule is
  arithmetic, not evidence.
- **One dataset, one period.** Nothing here measures drift, seasonality, or
  performance on next quarter's accounts. There is no temporal validation in these
  numbers.
- **Prevalence is fixed at 0.89 %.** PR-AUC is prevalence-dependent; at a different
  base rate these figures do not transfer.
- **No cost model.** F1 and MCC weight a missed mule against a wasted review by
  arithmetic convenience. The bank's real exchange rate is not in this file, and
  the tier structure — not these metrics — is where that judgement lives.
- **The bootstrap resamples the mules we have.** It cannot represent a typology
  absent from these 64. Novel mule behaviour is outside every interval here.
- **These are development estimates.** The primary nested figure is still
  computing; the holdout is reference-only and inadmissible as shown. The strongest
  honest claim available today is the flat estimate, labelled as such.
- **No adversarial adaptation is measured.** Mule operators respond to detection.
  Nothing in a static cross-validation captures that.

---

## 12. Not computed, and why

- **Full nested CV metrics.** The run had not finished; quoting the 1-repeat /
  2-trial file as "the nested result" would misrepresent it. The CLI takes
  `--source` so it can be re-run when the predictions land.
- **A confidence interval for the champion's holdout result.** No per-row vector
  exists (§9), and generating one means touching a reference-only split again.
- **Fresh latency measurement.** Every core was occupied by running
  cross-validation, so a timing would have measured the load. The recorded figures
  (p50 0.3075 s, p95 0.4219 s, n = 15, 687.7 rows/s) are quoted with their
  2026-07-10 date and the caveat that the recorded bundle size (2.370938 MB) does
  not match the bundle on disk today (3.464956 MB) — they describe an earlier
  build of the same pipeline.
- **Peak RAM.** Never instrumented in any run. Adding a figure here would mean
  inventing one.
