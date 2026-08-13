# Locked-Test Ruling

Final-validation prompt §9, *Previously Touched Holdout Rule*.

The rule is conditional, so the ruling has to start from evidence rather than
from preference:

> If an existing "locked test" has already been viewed and used to compare
> models: **it is no longer an unbiased model-selection set.** Do not tune
> against it. Label it `HISTORICAL HOLDOUT — FOR REFERENCE ONLY`.
>
> If a truly untouched holdout was created before this testing session and
> **can be proven untouched**, it may be evaluated once.

Two questions therefore decide the outcome. Has it been viewed? Can it be
proven untouched? The repository answers the first *yes* and the second *no*.

---

## Verdict

```text
HISTORICAL HOLDOUT — FOR REFERENCE ONLY
```

The primary selection and evaluation estimate for this project is **nested
repeated cross-validation** on the 7,264 development rows. The locked test's
1,818 rows contribute **no** model choice, **no** hyperparameter, **no**
threshold, and **no** calibration parameter.

---

## 1. What the set is

| Property | Value |
|---|---|
| Created (UTC) | `2026-07-10T18:58:47.246429+00:00` |
| Seed | 42, group-aware |
| Rows | 1,818 of 9,082 |
| Positives | 17 |
| Prevalence | 0.935093509350935 % |
| Development complement | 7,264 rows / 64 positives |
| SHA-256 of the sorted row index | `05d3c6f8af32530c89f2b26de29441db77b7e1319c8f29d9e683117c494d0e7e` |
| `data/splits/locked_test_indices.parquet` last modified | 2026-07-11 00:28 |

The row set has not changed since it was carved. That matters in both
directions: it means every historical number below refers to *these* rows, and
it means any knowledge gained from them in July still applies in August.

---

## 2. Has it been viewed? — Yes, at least four times, on at least three models

`artifacts/metrics/locked_test_touch_log.json` records three touches. A fourth
happened and **was not recorded**; it is reconstructed here from
`artifacts/metrics/holdout_metrics.json` and the sealed-validation artifacts.

| # | UTC | Model whose score was read | PR-AUC | Logged? |
|---|---|---|---|---|
| 1 | `2026-07-10T20:48:36.093594Z` | LightGBM agreement model (bundle `04fafaee25ae82c7`) | 0.8607721794686829 | yes |
| 2 | `2026-07-10T20:52:26.991365Z` | `catboost_tuned_top60` (then production scorer) | 0.8242261397312548 | yes, as "rebuild from saved predictions" |
| 3 | `2026-07-10T20:54:00.897178Z` | `catboost_tuned_top60`, identical rebuild | 0.8242261397312548 | yes, as "rebuild from saved predictions" |
| 4 | `2026-08-12T10:59:04.490653Z` → `10:59:38.459657Z` | `xgboost_top_120` (current champion), Sealed Validation Protocol | 0.7262714933700882 | **no — gap** |

Touches 2 and 3 are annotated *"no model evaluation performed"*, and that
annotation is true as far as compute goes — no model was re-fit and no new
predictions were produced. It is **not** true as far as information goes. A
second model's PR-AUC on these rows became known, and it landed in the same
artifact as the first. Two models' scores on the same held-out rows, side by
side, in one file, is a comparison. The §9 trigger does not care whether the
comparison was cheap.

So the condition is met on its plain reading: the set **has** been viewed and
**has** been used to compare models.

---

## 3. Can it be proven untouched? — No, and the reason is not a timestamp

For the current champion the timing is actually clean, and it is worth stating
precisely because it is the strongest available defence:

| Event | UTC |
|---|---|
| `xgboost_top_120` promoted, on dev-OOF evidence only (`promotion_decision_v2.json`) | `2026-08-12T07:39:08.059521Z` |
| Predictions sealed, SHA-256 `cc9f65a1…a949c4649` | `2026-08-12T10:59:04.490653Z` |
| Labels read, hash re-verified, `seal_verified: true` | `2026-08-12T10:59:38.459657Z` |

The champion was chosen **3 h 20 m before** any locked-test label was read, on
`oof_pr_auc_mean = 0.76904` from the development tournament. The seal proves
the predictions pre-dated the labels. Nothing about that ordering is in doubt.

It is still not enough, for one reason that no timestamp can address: **the
generation-2 pipeline was designed by an operator who already knew
generation 1's locked-test results.** The Feature Availability Firewall, the
feature pools, the tournament grid and the promotion rule were all authored
after 0.86077 and 0.82423 were known. That is an analyst-knowledge channel. It
is unquantifiable, it leaves no artifact, and a sealed prediction file does not
close it.

§9 asks for *proof* of untouched status. Proof is not available, so the
permissive branch — "it may be evaluated once" — does not open. The
conservative branch applies.

---

## 4. What the three numbers may and may not be used for

| Number | Model | Status |
|---|---|---|
| 0.8607721794686829 | LightGBM agreement model | **Inadmissible.** Never a production candidate; pre-firewall feature pool. |
| 0.8242261397312548 | `catboost_tuned_top60` | **Inadmissible.** Retired model, trained on a feature set containing quarantined features. It is nonetheless the value sitting in the two spec-named files `locked_test_metrics.json` and `final_locked_test_metrics.json`. See `HISTORICAL_METRIC_RECONCILIATION.md`. |
| 0.7262714933700882 | `xgboost_top_120` | **Reportable, not selective.** May be quoted as a one-touch reference figure with the HISTORICAL HOLDOUT label attached. May not be used to choose a model, tune a hyperparameter, set a threshold, or fit a calibrator. |

Permitted use of 0.72627 is narrow and worth naming exactly, because a
"reference only" label is often treated as a licence to quote freely:

- **Permitted** — reporting it beside the development estimate so a reader can
  see the direction and rough size of the optimism gap.
- **Permitted** — as a coarse sanity check that the pipeline is not producing
  a dev-only artefact.
- **Forbidden** — model selection, threshold selection, calibration fitting,
  feature selection, early stopping, and any "we tried X and the locked test
  liked it" iteration.

A directional reading is available and is itself informative:

| Estimate | PR-AUC | Positives |
|---|---|---|
| Flat repeated OOF, dev (selection basis) | 0.76904 ± 0.02663 | 64 |
| **Nested CV, `xgboost` family** (primary, finished 2026-08-13) | **0.75393 ± 0.00740** | 64 |
| Locked test, single touch | 0.72627 | **17** |
| ~~Preliminary nested CV (1 repeat, 2 Optuna trials)~~ | ~~0.66792~~ | superseded |

**This table changed when the full nested run landed, and the earlier reading of
it did not survive.** While the only nested evidence was the 1-repeat 0.66792,
the locked test sat between flat and nested, and this document read that as the
ordering selection optimism predicts. It no longer holds: the locked test is now
the **lowest** of the three, below the nested estimate rather than above it.

What is still supported is the narrow claim: nested (0.75393) is below flat
(0.76904) for the shipped family, which is the direction selection optimism
predicts, at a magnitude — 0.015 — far smaller than the preliminary run implied.

What is **not** supported is any reading of where the locked test falls. Seventeen
positives cannot rank three estimators. A single draw of 17 mules moves PR-AUC by
far more than the 0.04 separating these rows, so "the locked test is below the
nested estimate" is a fact about one sample, not evidence that the nested protocol
is still optimistic. It is recorded here because deleting it would look like
tidying, and read as nothing more than that.

---

## 5. No new locked test will be manufactured

§9 closes with:

> Do not manufacture a new "locked" test and then repeatedly tune to it.

The tempting move is to carve a fresh holdout out of the 7,264 development
rows. It is declined, on two grounds:

1. **Arithmetic.** The development set holds 64 positives. A 20 % holdout takes
   roughly 13 of them, leaving ~51 to drive every fold of the nested CV. A
   PR-AUC estimated on 13 positives has a confidence interval wide enough to
   contain almost any claim, and the estimate it degrades is the one actually
   being used for selection. The trade is bad in both directions.
2. **The rule's intent.** A new locked test created *now*, by the same operator,
   inside the same session, with the same knowledge, would carry the identical
   defect described in §3 within a day of its creation. It would produce the
   *appearance* of a clean holdout without the substance — which is precisely
   the failure §9 names.

The nested repeated CV design already provides what a fresh holdout would be
for: every model is scored on outer folds it never saw, and feature selection
runs **inside** each outer fold rather than once over the pooled development
set.

---

## 6. Containment checks

Two mechanical guarantees, both re-verified at the time of writing:

| Check | Result |
|---|---|
| `harness.dev_split()` asserts folds never intersect the locked test | enforced in code — `RuntimeError: CV folds overlap the locked test - refusing to run` |
| Nested CV assignment rows | 7,264 unique |
| Nested CV rows ∩ locked test rows | **0** |
| Locked-test row set changed since 2026-07-10? | no |

`artifacts/splits/nested_cv_assignments.parquet` therefore contains development rows
only. The locked test is outside the entire nested experiment by construction,
not by convention.

---

## 7. Recorded defects

Two problems are recorded rather than quietly repaired, because repairing them
after the fact would destroy the evidence a reader needs.

1. **The touch log is incomplete.** The 2026-08-12 champion evaluation is
   absent from `locked_test_touch_log.json`. The evaluation itself is fully
   evidenced — sealed predictions, verified hash, both timestamps — but the log
   that exists specifically to count touches did not count this one. The log
   should be treated as a lower bound on touches, not a census. No back-dated
   entry has been added.

2. **The spec-named files carry the retired number.** `locked_test_metrics.json`
   and `final_locked_test_metrics.json` both hold `catboost_tuned_top60`'s
   0.82423 — the most flattering locked-test value in the repository, produced
   by a retired model on a contaminated feature pool. Anyone reading the
   spec-named path alone will read the wrong number.
   `artifacts/metrics/holdout_metrics.json` separates `current_champion` from
   `retired_run` and is the file to read.

---

## 8. Provenance

| Source | Path |
|---|---|
| Touch log | `artifacts/metrics/locked_test_touch_log.json` |
| Split definition and creation record | `data/splits/split_metadata.json`, `data/splits/locked_test_indices.parquet` |
| Champion + retired locked-test metrics, separated | `artifacts/metrics/holdout_metrics.json` |
| Seal, hash, reveal timestamps | `artifacts/metrics/organiser_dry_run.json` → `offline_label_comparison`; `artifacts/sealed_validation/` |
| Promotion decision and its basis | `artifacts/metrics/promotion_decision_v2.json` |
| Nested fold assignments | `artifacts/splits/nested_cv_assignments.parquet` |
| Generation reconciliation | `docs/HISTORICAL_METRIC_RECONCILIATION.md` |
