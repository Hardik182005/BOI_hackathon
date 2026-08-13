# Validation Lab Report

Master prompt §19–20, addendum UPDATE 11 and UPDATE 12.

Implementation: `src/muleguard/validation/lab.py` (three steps, compatibility
score), `src/muleguard/validation/sealed.py` (Sealed Validation Protocol),
routes `POST /v1/validation/run`, `POST /v1/validation/{seal_id}/reveal`,
`GET /v1/validation/seals`, `GET /v1/validation/seals/{seal_id}`.
Frontend: the Validation Lab page.

---

## 1. What it is for

The organiser will not run our Makefile. They will open a page, drop in a
spreadsheet, and expect scored rows back. The Validation Lab is that page — and
it is also the mechanism by which we can prove, afterwards, that the score we
report was not obtained by looking at the answers first.

Those are two different jobs and both are load-bearing:

- **Job 1** — accept a file we have never seen, in whatever shape it arrives, and
  score it without breaking.
- **Job 2** — make "we did not peek" a *falsifiable claim* rather than an
  assurance.

---

## 2. The fixed order (UPDATE 11)

```
Step 1   Schema Integrity
Step 2   Hidden Validation Shield
Step 3   Predictions (sealed)
         ── hash written ──
Step 4   Reveal (labels read, metrics computed)   ← only reachable after the hash
```

The order **is** the control. A team that scores first and inspects the file
afterwards has already had the opportunity to tune against it. A team that checks
schema and distribution *before* predicting, then seals the predictions, has not.

The sequence cannot be skipped by calling the functions in a different order:
`step_2_hidden_validation_shield` raises `StepOrderError` if the schema step
failed, and `step_3` will not run without a passing step 2. UPDATE 11's clause —
*"write a timestamped prediction hash before evaluating labels"* — is step 3's
exit condition, not a convention.

---

## 3. Step 1 — Schema Integrity

**Question:** does the file contain what the model needs, in a usable form?

It runs first because every later number is meaningless if the answer is no.

| Outcome | Trigger | Behaviour |
|---|---|---|
| `FAIL` | any **required** feature is absent | scoring is **refused** |
| `WARN` | required features present but entirely empty, or duplicate rows | scoring proceeds; both facts flow into the compatibility score |
| `PASS` | every required feature present and parseable | proceed |

Missing a required feature is a hard failure on purpose. Silently zero-filling it
would produce a confident score for an account the model has **no information
about**, and that is the single most dangerous failure mode in the system — a
false negative that looks exactly like a genuine low-risk result.

### The derived-feature honesty clause

Three of the champion's 120 features are `MG_*` meta-features, computed row-wise
from the file's own raw columns and therefore never uploaded. Step 1 counts them
as present — the model genuinely has them — but reports them **separately**:

```json
"n_present": 120, "n_supplied_by_file": 117,
"n_derived_required": 3,
"derivation_note": "MG_* meta-features are computed here from the file's own raw
                    columns; nothing about that derivation consults training
                    data, labels or other rows."
```

Nobody should be able to read `schema_completeness: 1.0` as a claim that the file
supplied 120 columns when it supplied 117. And a derived feature whose own source
columns are absent lands in `all_null_required` like any other empty column, so
the missingness is surfaced rather than hidden by the derivation.

Also reported: `unexpected_columns`, `unparseable_numeric`, `duplicate_rows`,
`target_column_present`, and the guarantee `"no label was read in this step"`.

---

## 4. Step 2 — Hidden Validation Shield (UPDATE 3)

**Question:** is this file the kind of data the model was fitted on?

An adversarial validator is trained to distinguish training rows from uploaded
rows. If it can separate them easily, the upload is distributionally different
and the absolute probabilities are less trustworthy — the *ranking* usually still
is.

Three things this step is explicitly **forbidden** to do, stated in the payload
itself as `guarantees`:

> - the mule model is not retrained or recalibrated on uploaded rows
> - validation labels are not inspected in this step
> - **no prediction is changed as a result of the adversarial AUC**

The third is UPDATE 3's hardest clause. It is tempting to shrink scores when the
shield reports a shift. We do not. A shifted upload produces a **warning printed
next to unchanged predictions**. Even `SEVERE_SHIFT` warns rather than blocks —
refusing to score the organiser's file because our own detector disliked it would
be a self-inflicted zero.

---

## 5. The Validation Compatibility Score (§20)

One auditable 0–100 number with its workings shown:

| Component | Weight | What it measures |
|---|---:|---|
| `schema_completeness` | 40 | share of required features present |
| `distribution_compatibility` | 30 | adversarial AUC mapped linearly: 0.50 → 1.0, 1.00 → 0.0 |
| `missingness_consistency` | 15 | 1 − mean absolute difference in per-feature null rates |
| `value_range_coverage` | 15 | share of features whose uploaded median sits inside the training p1–p99 |

| Band | Score | Meaning shown to the user |
|---|---:|---|
| `HIGH_COMPATIBILITY` | ≥ 85 | the file looks like the data the model was fitted on; read the metrics at face value |
| `ACCEPTABLE` | ≥ 70 | minor differences; results usable, absolute probabilities slightly less reliable |
| `DEGRADED` | ≥ 50 | material differences; prefer the ranking over the probabilities and check the shifted features |
| `INCOMPATIBLE` | < 50 | scores should be treated as indicative only |

Two design decisions worth stating:

**The weights were fixed before any file was ever scored**, so the number cannot
be reverse-engineered to flatter a particular upload.

**The AUC mapping is deliberately linear rather than tuned.** A threshold curve
fitted to make our own uploads score well would defeat the purpose of having the
check.

`value_range_coverage` exists because adversarial AUC has a blind spot: a monotone
rescaling — someone exporting amounts in thousands instead of rupees — can be
hard for a classifier to separate but is exactly the kind of unit error that
destroys a score. Comparing medians against the training p1–p99 catches it
cheaply and readably.

The score describes **inputs**. It never modifies an output. Every response
carries the note:

> computed from inputs only, with no access to labels. A low score is reported
> alongside the predictions; it never modifies them.

---

## 6. Step 3 and the Sealed Validation Protocol (UPDATE 12)

The problem the seal solves is credibility, not accuracy. When a team uploads a
labelled file, scores it, and reports a number, nothing in the artefact tells you
whether the predictions existed before or after somebody looked at the labels.
*"We didn't peek"* is unfalsifiable.

The seal makes it falsifiable. The order is enforced in code:

1. the target column is located and **withheld** from the scoring frame,
2. predictions are produced from the remaining columns,
3. predictions are written to disk,
4. a SHA-256 of that file is computed,
5. a manifest records the hash, the UTC timestamp, the model version and the
   SHA-256 of the **input**,
6. **only then** may labels be read.

`reveal_metrics` recomputes the prediction hash before it will score anything. If
the predictions were edited after sealing — by anyone, for any reason — the hash
no longer matches, the seal state becomes `SEAL_BROKEN`, and the reveal is
refused. The metrics a judge sees therefore provably belong to predictions that
existed before the labels were opened.

States: `PREDICTIONS_SEALED` → `METRICS_REVEALED`, or `SEAL_BROKEN`.

### Path-traversal hardening

Seal ids become filenames, so they are minted in exactly one shape and validated
against a strict allow-list `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. An id of
`../../config` would resolve to a `.json` outside `SEAL_DIR`, and the reveal path
writes the manifest back — which would make it an arbitrary file write.

Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`,
`LPT1`–`LPT9`) are rejected separately, because containment is not sufficient
there: `CON.json` resolves to the console *while sitting inside* `SEAL_DIR`.
Reading one blocks; writing one silently discards the manifest.

---

## 7. Measured behaviour: the §42 rehearsal

The lab was exercised over HTTP against a live backend with the locked-test rows
(1,818 × 3,924, target removed, 17 positives withheld) in eight shapes:

| Variant | Compatibility | Verdict |
|---|---:|---|
| baseline (.xlsx) | 99.97 | PASS |
| baseline (.csv) | 99.97 | PASS |
| shuffled column order | 99.97 | PASS |
| extra columns (`analyst_notes`, `export_batch`, `Unnamed: 0`) | 99.97 | PASS |
| `F3912` present | 99.97 | PASS |
| resolution fields present (F3898/F3899/F3913/F3914/F3915) | 99.97 | PASS |
| unseen category in a selected feature | 99.85 | PASS |
| 40 non-required columns dropped | 99.97 | PASS |

All eight scored. `n_missing_required: 0`, `schema_completeness: 1.0` throughout.
Sealed `10:59:04.490653Z`, revealed `10:59:38.459657Z`, `seal_verified: true`,
final state `METRICS_REVEALED`.

Revealed locked-test metrics: **PR-AUC 0.7263**, ROC-AUC 0.9665, lift 77.7×,
Recall@100 0.824.

Full record and the invariance/sensitivity analysis: `docs/ORGANISER_DRY_RUN.md`.

---

## 8. One real defect the rehearsal found

The first run rejected all six CSV variants with `422 ComputeError`.

`_parse_upload` inferred the CSV schema from 200 rows. On a 3,924-column export,
a column that is numeric for the first 200 rows and carries a stray `N/A` at row
900 aborted the **entire upload**. That is the organiser's file being refused
over a single cell — the worst failure this route has, and one that no unit test
would have found because it only appears at that width and depth.

Fixed in `src/muleguard/api/routes_upload.py` with `_read_csv_tolerantly`: four
attempts, weakest assumption last — sniff 200 rows, scan the whole file, read
every column as text, then tolerate ragged lines. Coercion downstream turns an
unparseable cell into a null, which the imputer already handles. A 422 turns it
into no submission at all.

Three regression tests were added, including one asserting that a clean file
still gets **numeric** dtypes — a tolerant reader that quietly stringifies every
well-formed export would "work" for every input and be worse than the bug.

---

## 9. What the lab never does

| Never | Enforcement |
|---|---|
| retrain the mule model on uploaded rows | UPDATE 3; no fit call exists on the upload path |
| inspect validation labels before prediction | the target is withheld by `sealed.withhold_target` before scoring |
| change predictions based on adversarial AUC | the shield returns a report; no code path lets it touch a score |
| let a low compatibility score silently adjust results | the score is displayed, never applied |
| write back into the model bundle | verified in §42: bundle fingerprint `afd0dc1d8fc02eb9` identical before the first upload and after the last, checked after **every** variant |

---

## 10. Enforcement

Release-gate check `organiser_dry_run_passed` requires three separate facts —
every variant scored, the quarantine invariances held, and the sensitivity
control differed — because any one of them alone is satisfiable by a broken
harness.

Tests: `tests/unit/test_sealed_validation.py` (seal/reveal, hash mismatch
refusal, seal-id validation, reserved names, and the compatibility bands),
`tests/integration/test_validation_api.py` (seal returns no metric, reveal only
after the seal verifies, row-count mismatch, predictions edited after sealing,
missing features stop at step 1), `tests/integration/test_batch_upload.py`.
The step order and component weights are **not** separately tested — an earlier
draft of this line cited a `test_validation_lab.py` that was never written.
