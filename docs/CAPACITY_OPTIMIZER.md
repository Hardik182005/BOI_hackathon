# Analyst Capacity Optimizer

**Computation:** `src/muleguard/models/capacity.py` (pure, no I/O)
**Generation CLI:** `src/muleguard/cli/capacity_curve.py`
(`python -m muleguard.cli.capacity_curve`)
**Machine-readable results:** `artifacts/metrics/capacity_curve.json`
(schema 1.0, generated 2026-08-12T12:33:57 UTC, 433 curve points)
**API:** `src/muleguard/api/routes_capacity.py` —
`GET /v1/capacity/curve`, `POST /v1/capacity/plan`
**UI:** `frontend/src/pages/CapacityOptimizer.tsx` (route `/capacity`)
**Tests:** `tests/unit/test_capacity.py`, `frontend/src/capacity.test.tsx`

---

## 1. The question this answers

A reviewing bank does not have an unlimited investigation team. It has a
number — the accounts a desk can work through in a day — and the question that
follows from it is not "what is the model's AUC", it is:

> if my analysts can review 50 accounts a day, how many mules do I catch,
> how much of their day is wasted, and what score cut-off gives me exactly
> that queue?

The panel takes **one** input, in either direction:

- **Accounts analysts can review per day** (a budget `K`), or
- **Maximum acceptable false positives per 1,000 accounts** (a tolerance).

and returns expected recall, expected precision, expected mules caught, mules
missed, false alarms, and a **recommended operational threshold** — with the
uncertainty attached to every one of those numbers.

Nothing is retrained to answer it. The curve is a re-reading of predictions
that already exist on disk: the promoted champion's stored development
out-of-fold scores.

---

## 2. Where the numbers come from

| | |
|---|---|
| Evaluation split | development out-of-fold (leakage-free v2), repeat-averaged |
| Accounts | 7,264 |
| Confirmed mules | 64 (prevalence 0.881%) |
| CV repeats | 3, averaged — the ranking the served bundle produces |
| Champion | `xgboost_top_120`, read from `artifacts/metrics/lens_stack_oof_v2.json` |
| Predictions | `artifacts/predictions/oof_v2.parquet` |
| Calibrator | frozen Platt map from `artifacts/models/final_bundle.joblib`, applied with `.predict` only |
| Locked test used | **no** |
| Retraining performed | **no** |

Two facts about the arithmetic are worth stating, because they decide what the
output can honestly claim.

**Recall and precision at a budget are ranking statistics.** Sort by score,
take the top `K`, count. Any strictly increasing calibration leaves them
unchanged. So the curve is calibration-independent; only the *threshold* is
not, and that is why the threshold is reported on the `calibrated_risk` scale
the scoring API actually emits, using the already-frozen calibrator rather
than a new fit.

**One cumulative sum answers every budget.** `cumsum(y[argsort(-score)])`
gives true positives at every `K` at once, which is why a 1,000-replicate
bootstrap over 433 budgets is cheap.

### Cross-check against independently computed artifacts

The module's output is not trusted on its own. `tests/unit/test_capacity.py`
asserts agreement with numbers produced by a different code path at bundle
freeze time (`muleguard.evaluation.metrics`, stored in
`lens_stack_oof_v2.json`):

- `recall_at_budget` — budgets 25 / 50 / 73 / 100 reproduce exactly
  (0.390625, 0.6875, 0.765625, 0.828125 with matching precision and TP counts).
- `recall_at_fpr` — the false-alarm mode lands on the same operating points:
  ≤5 FP/1,000 → recall 0.796875, ≤10 FP/1,000 → recall 0.84375.

If either ever drifts, the test fails rather than the two being quietly
averaged.

---

## 3. The measured curve

Recall/precision are point estimates on the repeat-averaged ranking. Two
intervals are published for each, from two bootstrap schemes (1,000
replicates, seed 42, 95%):

- **stratified** — positives and negatives resampled separately, so the mule
  count stays at 64. This is uncertainty *in the ranking alone*.
- **resampled accounts** — the whole book resampled, so the number of mules in
  a replicate moves too. Wider, more pessimistic, and **the one to quote** when
  the question is "what would next month look like".

| Reviews/day | Mules caught (of 64) | False alarms | Recall | Recall 95% (stratified) | Recall 95% (whole book) | Precision | FP per 1,000 legit | Recall per repeat |
|---:|---:|---:|---:|---|---|---:|---:|---|
| 10 | 10 | 0 | 15.6% | 15.6 – 15.6% | 12.3 – 20.4% | 100.0% | 0.00 | 15.6 / 15.6 / 15.6% |
| 25 | 25 | 0 | 39.1% | 39.1 – 39.1% | 30.9 – 50.0% | 100.0% | 0.00 | 39.1 / 39.1 / 39.1% |
| 34 | 34 | 0 | 53.1% | 48.4 – 53.1% | 42.0 – 64.0% | 100.0% | 0.00 | 53.1 / 51.6 / 50.0% |
| 50 | 44 | 6 | 68.8% | 62.5 – 76.6% | 58.4 – 78.7% | 88.0% | 0.83 | 64.1 / 68.8 / 60.9% |
| 75 | 49 | 26 | 76.6% | 68.8 – 85.9% | 67.7 – 86.7% | 65.3% | 3.61 | 73.4 / 79.7 / 71.9% |
| 100 | 53 | 47 | 82.8% | 73.4 – 90.6% | 72.4 – 91.2% | 53.0% | 6.53 | 75.0 / 79.7 / 79.7% |
| 150 | 54 | 96 | 84.4% | 75.0 – 93.8% | 74.6 – 93.5% | 36.0% | 13.33 | 78.1 / 79.7 / 85.9% |
| 200 | 55 | 145 | 85.9% | 76.6 – 93.8% | 76.3 – 94.7% | 27.5% | 20.14 | 82.8 / 85.9 / 87.5% |
| 300 | 58 | 242 | 90.6% | 81.3 – 96.9% | 81.8 – 96.9% | 19.3% | 33.61 | 84.4 / 87.5 / 90.6% |

Read from the other direction — a tolerance for false alarms — the panel picks
the **largest** budget that stays inside it:

| Tolerance | Reviews/day it buys | Recall | Precision | Advisory threshold (calibrated risk) |
|---|---:|---:|---:|---:|
| 0 FP per 1,000 | 34 | 53.1% | 100.0% | 0.816326 |
| ≤ 1 FP per 1,000 | 54 | 73.4% | 87.0% | 0.302813 |
| ≤ 5 FP per 1,000 | 87 | 79.7% | 58.6% | 0.123151 |
| ≤ 10 FP per 1,000 | 126 | 84.4% | 42.9% | 0.069943 |
| ≤ 20 FP per 1,000 | 199 | 85.9% | 27.6% | 0.033158 |

`FP per 1,000` is ambiguous in the wild, so the API makes the denominator
explicit with a `basis` parameter: `legitimate` (1,000 × FPR — the repo
convention, matching `fp_per_1000_legit` in the lens artifact) or `screened`
(per 1,000 accounts screened). The two never differ by much at this
prevalence, but they are not the same number and are not silently swapped.

**The two input modes cannot disagree**, by construction: the tolerance mode
selects a point on the curve and returns that point, so asking "what does
K=87 give me" returns byte-identical expectations to asking "what does ≤5
FP/1,000 give me". This is asserted, not assumed
(`test_fp_mode_agrees_with_budget_mode`).

---

## 4. The recommended threshold

For a chosen budget `K`, the recommendation is the score of the `K`-th ranked
development account, mapped through the frozen Platt calibrator so it is
expressed on the same `calibrated_risk` field the scoring API returns. It is
verified — in the artifact, in the API response, and in the tests — by counting
how many accounts it actually alerts on: `alerts_produced_on_evaluation_split`
must equal `K`. (There are no score ties in the top 1,000 of this ranking, so
it does.)

Examples: `K=50` → calibrated risk **0.340127**; ≤5 FP/1,000 → **0.123151**.

**It is advisory and it is not applied.** Every recommendation carries
`status: ADVISORY_REQUIRES_HUMAN_APPROVAL` and `applied: false`, there is no
write path from this feature to any policy artifact, and the UI deliberately
offers no apply/save control (`frontend/src/capacity.test.tsx` asserts the
absence of one). The frozen policy of `policy_version 1.0` is **read**,
compared against, and never written:

| Frozen tier | Calibrated-risk threshold | Accounts at or above | Mules caught | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| CRITICAL_REVIEW | 0.9338487 | 26 | 26 | 40.6% | 100.0% |
| URGENT_REVIEW | 0.0977383 | 101 | 53 | 82.8% | 52.5% |
| STANDARD_REVIEW | 0.0131826 | 339 | 58 | 90.6% | 17.1% |

`test_frozen_policy_thresholds_are_not_mutated` answers a series of capacity
questions and then asserts the whole curve document — policy block included —
is byte-identical to before.

---

## 5. What the number means

> At a review budget of 50 accounts per day, on 7,264 development accounts
> containing 64 confirmed mules, the top 50 by score contained 44 of them.

That is the whole claim. It is a **historical, measured, out-of-fold** count on
one dataset, with an interval around it.

## 6. What the number does **not** mean

1. **It is not a guarantee about your book.** It is what happened on this
   development split. A different portfolio, month, or mule typology moves it.
   The whole-book interval is the honest range: at K=50 the recall is
   68.8%, but 58.4–78.7% is the defensible statement.

2. **It is not precise to the percentage point.** There are 64 positives, so
   **one mule is worth 1.56 percentage points of recall**. "76%" and "77%" are
   not distinguishable here. Quote the interval; the UI shows it beside every
   figure for exactly this reason.

3. **A 100% precision row is not a promise of zero false alarms.** The top 34
   ranked development accounts happen to all be mules
   (`leading_true_positive_run = 34`). That is a property of this ranking on
   this data, not a law. It is also *why* the stratified interval collapses to
   a single point for K ≤ 34 — a scheme that holds the mule count fixed has
   nothing left to vary there. A zero-width interval is a degenerate estimate,
   flagged as `stratified_interval_degenerate: true`, **not** evidence of
   certainty. Read the whole-book interval beside it (at K=25 that is
   30.9–50.0%, not 39.1% exactly).

4. **The repeat-averaged curve is optimistic relative to a single run.**
   Averaging three CV repeats denoises the ranking, so the headline curve sits
   above each individual repeat — at K=100, 82.8% averaged versus 75.0–79.7%
   per repeat. The averaged number is the right headline because it is what the
   *served* bundle scores with, but the per-repeat values are published on every
   point so the gap is visible rather than hidden.

5. **It says nothing about the locked test split.** This feature is built on
   development OOF only (`locked_test_used: false`). No holdout figure appears
   anywhere in the artifact, the API, or the page.

6. **The frozen band sizes are close to, not identical to, their design
   targets.** `configs/thresholds.yaml` set `policy_version 1.0` from a
   *cross-fitted* calibration of these same scores, targeting a 25-account
   CRITICAL band, a 100-account URGENT band, and a 0.90 recall target for
   STANDARD. Section 4 applies the *shipped final* calibrator to the same
   frozen thresholds and finds them at 26, 101 and 339 accounts (recall 90.6%,
   which does clear the 0.90 target). The one-to-two-account drift is the gap
   between the two calibration fits, not a policy change, and it is recorded in
   the artifact's `caveats` rather than smoothed over.

7. **Recall is not "mules in the book".** It is recall against *confirmed
   labels*. Mules never labelled as such cannot appear in the numerator or the
   denominator.

8. **A threshold is not a decision.** The recommendation orders a review queue.
   It carries no verdict about any account holder, and nothing punitive follows
   from it automatically — a person with the authority to do so has to approve
   any change to the operating policy.

---

## 7. Regenerating

```bash
.venv/Scripts/python.exe -m muleguard.cli.capacity_curve            # rewrite the artifact
.venv/Scripts/python.exe -m muleguard.cli.capacity_curve --dry-run  # compute and log, write nothing
.venv/Scripts/python.exe -m pytest tests/unit/test_capacity.py      # 23 tests
```

The CLI reads the champion name and the frozen thresholds out of the artifacts
rather than having them typed in, so a re-promotion changes this page's numbers
by re-running it — there is no constant in the backend, the API layer, or the
frontend to keep in sync. The UI reads the precomputed curve and never computes
a metric of its own.
