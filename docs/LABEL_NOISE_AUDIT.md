# Label Noise Audit

**Machine-readable source:** `artifacts/metrics/label_noise_audit_v2.json`
(generated 2026-08-12T08:39:18 UTC)
**Related:** `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md` §10

---

## 1. What this audit is, and what it is not

This audit answers one question: **are there labelled rows that every model in
the tournament disagrees with, consistently?**

It is a reporting instrument. It does not correct anything.

| The audit does | The audit never does |
|---|---|
| Rank development rows by model consensus | Flip a label |
| Flag positives no model can rank | Remove a row from any set |
| Flag negatives every model ranks at the top | Feed a filter to training, calibration or thresholding |
| Publish the flagged rows for human review | Assert that a flagged account was mislabelled |

The artifact carries three explicit guarantees, quoted verbatim:

> - no label was modified
> - no row was removed from any training, calibration or evaluation set
> - no downstream component reads this artifact as a filter

**A flagged positive is a request for human review. It is not a claim that the
account was mislabelled, and it is not a criticism of the analyst who labelled
it.**

---

## 2. Why "the model cannot rank it" does not mean "the label is wrong"

This is the whole reason the audit reports rather than acts.

When 13 models trained on 64 positives all fail to rank a particular mule
highly, at least three explanations fit the evidence equally well:

1. The label is wrong.
2. The account is a genuine mule whose behaviour is not represented in the
   available feature set — the transaction pattern that gave it away is not one
   of the 3,925 columns in this extract, or it happened outside the 7/14/31-day
   windows the aggregates cover.
3. The account is a genuine mule of a type too rare for 64 training positives to
   teach. With one example of a pattern, cross-validation guarantees that the
   fold holding that example was trained without it.

The audit cannot distinguish these, and it does not pretend to. The artifact
states the consequence directly:

> Removing these rows would raise every metric in this repository while making
> the detector strictly worse on the hidden validation set.

That is the trap this project refuses. Dropping the positives a model finds
hard is a mechanical way to manufacture a better-looking PR-AUC, and it deletes
exactly the cases that most need detecting. The same logic applies in the other
direction: a legitimate account that every model ranks near the top is a
candidate false positive **and** a candidate missed mule, and the artifact says
so rather than choosing the flattering reading:

> these are candidate false positives as much as candidate missed mules; the
> audit does not claim to distinguish them

---

## 3. Method

| Parameter | Value |
|---|---|
| Rows audited | 7,264 (development split only; the locked test is not touched) |
| Positives / negatives | 64 / 7,200 |
| Consensus models | 13 |
| Model-repeat runs | 39 |
| Model eligibility floor | PR-AUC ≥ 0.5 |

The 13 consensus models are `xgboost_top_120`, `xgboost_top_60`,
`lightgbm_top_60`, `catboost_top_60`, `xgboost_top_30`, `lightgbm_top_120`,
`lightgbm_top_30`, `lightgbm_viewA_top_60`, `catboost_top_120`,
`lightgbm_viewB_top_60`, `lightgbm_top_250`, `lightgbm_full_pool` and
`catboost_top_30` — every candidate that cleared the 0.5 PR-AUC floor. Weak
models are excluded because a model that ranks nothing well would flag
everything, and the resulting consensus would measure model weakness rather
than label quality.

Each row's percentile is taken from out-of-fold scores in each of the 39
model-repeat runs. "Agreement" is the share of those runs that place the row in
the flagging region.

### Flag definitions and thresholds

| Flag | Definition | Thresholds |
|---|---|---|
| `POSSIBLE_LABEL_NOISE` | a labelled mule ranked at or below the 50th percentile in at least 80 % of model-repeat runs | percentile ≤ 50.0, agreement ≥ 0.80 |
| `HIGH_SCORING_NEGATIVE` | a labelled legitimate account ranked at or above the 99.5th percentile in at least 80 % of model-repeat runs | percentile ≥ 99.5, agreement ≥ 0.80 |

Both thresholds are deliberately strict. A row must be missed by nearly every
model in nearly every run before it is surfaced at all.

---

## 4. Results

### 4.1 Positives

| Measure | Value |
|---|---:|
| Positives audited | 64 |
| Flagged `POSSIBLE_LABEL_NOISE` | **1** |
| Share of positives | 1.56 % |

The single flagged row:

| Field | Value |
|---|---|
| Row index | 9067 |
| Label | 1 (mule) |
| Mean percentile across 39 runs | 25.076 |
| Min / max percentile | 0.427 / 77.265 |
| Mean score | 0.014237 |
| Agreement | 0.821 |
| Action | **`HUMAN_REVIEW_ONLY`** |

Note the spread: this row reached the 77th percentile in its best run and the
0.4th in its worst. That is not a model unanimously certain the label is wrong;
it is a model that mostly cannot find the account, with at least one fold where
it partly could. Consistent with all three explanations in §2.

### 4.2 Negatives

| Measure | Value |
|---|---:|
| Negatives audited | 7,200 |
| Flagged `HIGH_SCORING_NEGATIVE` | **0** |
| Share of negatives | 0.00 % |

No legitimate account is ranked in the top 0.5 % by 80 % or more of the runs.
Under this threshold there is no population of "suspiciously well-ranked
negatives" to review.

### 4.3 Where the positives actually sit

| Percentile of labelled mules | Value |
|---|---:|
| p10 | 79.88 |
| median | **99.25** |
| p90 | 99.80 |

The median labelled mule sits at the 99.25th percentile of the score
distribution, and 90 % of them sit above the 79.88th. The label set and the
models agree on the overwhelming majority of positives. One row in 64 is the
extent of the disagreement.

---

## 5. What happens to the flagged row

Nothing automatic. Specifically:

1. It stays in the development split, with label 1, in every training fold, in
   every calibration fit and in every metric reported in this repository.
2. It is published here with its row index so a bank analyst can pull the case
   and decide. That decision is a human one and is out of scope for this
   system.
3. If a reviewer concludes the label is correct, that is a finding about the
   feature set — it says the behaviour that identified this mule is not
   captured by the available columns, which is useful information for the next
   data request.
4. Nothing in the API, the scoring path, the policy engine or the report
   generator reads this artifact. It is a document, not a control input.

---

## 6. Limits of this audit

- **64 positives.** Every statistic above is computed from a small positive
  class. A flag rate of 1/64 has a wide interval around it, and this document
  does not attach one because the resampling would be over the same 64 rows.
- **Correlated models.** The 13 consensus models share a feature pool and a
  fold structure. Their agreement is not 13 independent opinions, so "80 % of
  runs agree" is weaker evidence than the number suggests.
- **Development split only.** The locked test is not audited, because reading
  it for this purpose would spend its single touch.
- **Thresholds are choices.** The 50th/99.5th percentile cuts and the 0.80
  agreement floor were fixed before the audit ran, and moving them would change
  the counts. They are recorded in the artifact so a reader can see what was
  chosen rather than inferring it from the result.

---

## 7. Policy statement

Quoted from the artifact, and binding on every component in this repository:

> This audit reports only. Labels are never flipped, rows are never removed,
> and no model in this project is trained, calibrated or thresholded on a
> filtered version of the label set. A flagged positive is a request for human
> review, not a verdict about the account or the analyst who labelled it.
