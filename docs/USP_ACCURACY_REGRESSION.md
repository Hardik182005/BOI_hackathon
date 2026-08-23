# USP Accuracy Regression — Proof the Classifier Is Untouched

> **Verdict: PASS.** `max |Δ probability| = 0.000e+00` across 400 probe accounts.
> Zero tier mismatches. Zero policy-action mismatches. All 20 metrics `Δ = 0.0`.
>
> **Artifact:** [`artifacts/testing/usp_accuracy_regression.json`](../artifacts/testing/usp_accuracy_regression.json)
> **Baseline:** [`artifacts/upgrade_baseline/usp_prechange_baseline.json`](../artifacts/upgrade_baseline/usp_prechange_baseline.json)
> **Reproduce:** `python -m muleguard.cli.usp_baseline --check` (exit code *is* the verdict)
> **Tests:** [`tests/regression/test_score_invariance_after_usp.py`](../tests/regression/test_score_invariance_after_usp.py)
> **Spec:** sections 24, 26, 38 and 41 of the Cohort Radar / Zero-Regression USP brief

---

## 1. What is being proved

Two new capabilities were added to MuleGuard — the
[Mule-Farm Cohort Radar](COHORT_RADAR.md) and the
[Account-Control Ambiguity Guardrail](ACCOUNT_CONTROL_AMBIGUITY.md). Both are
post-model layers. The claim under test is the strongest available version of
"post-model":

> Every probability, every tier, every policy action and every published accuracy
> metric is **identical** after the change — not similar, not within noise,
> identical.

The tolerance is `1e-12`. The measured difference is `0.0`.

The spec is explicit about what to do with a small non-zero difference:
*"Do not explain it away as 'small.' Find the cause."* That instruction did not
need to be exercised, and this document does not claim credit for restraint it
was never asked to show — but the tolerance exists and the tests enforce it.

---

## 2. How the comparison avoids fooling itself

Three design choices matter more than the numbers.

**One module, one code path, two invocations.** `--save` and `--check` are the
same code. If the "after" run used a newer snapshot function than the "before"
run, a difference in the report could be a difference in the *measurement* rather
than in the model — exactly the failure a regression check exists to rule out.

**The baseline cannot be silently regenerated.** `--save` refuses to overwrite an
existing baseline without `--force`. A baseline that can be regenerated after a
change is not a baseline; it is a description of the change agreeing with itself.

**The baseline predates the code it is checking.** Ordering, from the artifacts
themselves:

| Event | Timestamp |
|---|---|
| Baseline captured | `2026-08-23T08:12:44 UTC` |
| `control_attribution.py` first written | `2026-08-23T08:51 UTC` |
| `cohort_radar.py` first written | `2026-08-23T09:15 UTC` |
| Post-change check run | `2026-08-23T10:04:03 UTC` |

The invariants were frozen **before either USP module existed**.

### A note on the git SHA

`baseline_git_sha` and `postchange_git_sha` are both `33ac429d…`, because both
snapshots were taken from the same uncommitted working tree. **The git SHA is
therefore not the evidence here, and is not presented as such.** The evidence is
the frozen bundle hash, the fitted-constant equality, and 400 accounts re-scored
end-to-end through the live path. This is recorded rather than tidied away
because a reader checking the artifact would notice it, and should find it already
explained.

---

## 3. Identity checks — the frozen objects

| Check | Before | After | Equal |
|---|---|---|:--:|
| Champion bundle SHA-256 | `d12914de5abee99a…` | `d12914de5abee99a…` | ✅ |
| Selected feature-list hash | `daed5969411c3316…` | `daed5969411c3316…` | ✅ |
| Calibrator fitted constants | — | — | ✅ |
| Policy thresholds | — | — | ✅ |
| Live leakage quarantine | — | — | ✅ |

```json
"identity_differences": {}
```

No retraining was performed. No hyperparameter, feature, calibrator constant,
threshold, quarantine entry, target definition or split was modified. The
locked test was not re-touched (spec section 26).

---

## 4. Probe comparison — 400 accounts, full precision

400 fixed development accounts are re-scored through the **live scoring path**,
not a replayed cache, and compared at full float precision.

| Measure | Value |
|---|---|
| Rows compared | **400** |
| Rows present in only one snapshot | **0** |
| `max abs Δ` calibrated probability | **`0.0`** |
| `max abs Δ` raw model probability | **`0.0`** |
| Tolerance | `1e-12` |
| Tier mismatches | **0** |
| Policy-action mismatches | **0** |

Both the calibrated probability *and* the raw pre-calibration score are compared.
Comparing only the calibrated output would miss a raw-score change that the
calibrator happened to flatten — the two checks together close that gap.

The test suite additionally walks the probes **row by row** rather than trusting
the aggregate maximum, checks each model family's raw scores separately
(XGBoost / LightGBM / CatBoost), and confirms the supporting signals are
unchanged: conformal outcome, OOD flag, auto-action, merchant-safeguard
application and model agreement.

---

## 5. Metric comparison — all 20, all `Δ = 0.0`

Recomputed from the saved out-of-fold predictions.

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| PR-AUC | 0.7690395605299595 | 0.7690395605299595 | `0.0` |
| ROC-AUC | 0.9577090567129630 | 0.9577090567129630 | `0.0` |
| Recall | 0.8645833333333334 | 0.8645833333333334 | `0.0` |
| Precision | 0.1942277472995442 | 0.1942277472995442 | `0.0` |
| F1 | 0.3120645228153761 | 0.3120645228153761 | `0.0` |
| F2 | 0.4964991648579049 | 0.4964991648579049 | `0.0` |
| MCC | 0.3942170192220957 | 0.3942170192220957 | `0.0` |
| Accuracy | 0.9619585168869310 | 0.9619585168869310 | `0.0` |
| Balanced accuracy | 0.9137037037037037 | 0.9137037037037037 | `0.0` |
| **Brier** | 0.0034581153454714 | 0.0034581153454714 | `0.0` |
| **ECE** | 0.0024080675000670 | 0.0024080675000670 | `0.0` |
| Recall@Top-25 | 0.390625 | 0.390625 | `0.0` |
| Recall@Top-50 | 0.6458333333333334 | 0.6458333333333334 | `0.0` |
| Recall@Top-100 | 0.78125 | 0.78125 | `0.0` |
| Precision@Top-25 | 1.0 | 1.0 | `0.0` |
| Precision@Top-50 | 0.8266666666666667 | 0.8266666666666667 | `0.0` |
| Precision@Top-100 | 0.5 | 0.5 | `0.0` |
| FP per 1,000 legitimate | 37.175925925925924 | 37.175925925925924 | `0.0` |
| Operating threshold | 0.01318262055786693 | 0.01318262055786693 | `0.0` |
| Prevalence | 0.00881057268722467 | 0.00881057268722467 | `0.0` |

```json
"findings": [],
"verdict": "PASS"
```

Calibration (Brier, ECE) is listed with the accuracy metrics deliberately. A
post-model layer that leaked into the score would most plausibly show up as a
calibration shift *before* it showed up as a PR-AUC shift, because a small
monotone perturbation can leave ranking metrics intact while moving the
probabilities. Both are unchanged.

---

## 6. Release blockers (section 41) — status

Section 41 names **23** conditions, any one of which fails the upgrade. All 23,
not just the classifier subset:

| # | Blocker | Status | Evidence |
|---|---|:--:|---|
| 1 | Champion model hash changes | ✅ | identical before and after |
| 2 | Selected feature list changes | ✅ | feature-list hash identical |
| 3 | Calibration changes | ✅ | fitted calibrator constants equal |
| 4 | Policy thresholds change | ✅ | threshold set equal |
| 5 | Classification probability changes | ✅ | `Δ = 0.0`, tolerance `1e-12`, 400 probes |
| 6 | Risk-tier mismatch appears | ✅ | 0 of 400, row by row |
| 7 | PR-AUC changes unexpectedly | ✅ | `Δ = 0.0` |
| 8 | Recall@TopK changes unexpectedly | ✅ | `Δ = 0.0` at K = 25/50/100 |
| 9 | Brier/ECE changes unexpectedly | ✅ | `Δ = 0.0` |
| 10 | Cohort Radar uses `F3924` | ✅ | leakage checks 1 & 8 |
| 11 | Cohort Radar uses `F3912` or any quarantined feature | ✅ | leakage checks 2 & 5, read from the **live** quarantine, not a hardcoded list |
| 12 | Cohort Radar uses `F2230` | ✅ | leakage check 3 |
| 13 | Judge labels affect neighbour retrieval | ✅ | leakage checks 8–10; `labels_used_for_retrieval: false` |
| 14 | Cohort similarity changes a classifier score | ✅ | this document + `test_the_cohort_lookup_does_not_move_the_classifier_risk` |
| 15 | Cohort membership auto-escalates a case | ✅ | `automatic_actions_permitted == []`; `test_no_neighbour_is_escalated_by_appearing_in_the_cohort` |
| 16 | Control Attribution infers criminal intent | ✅ | 8 categories in `NEVER_INFERRED`, asserted absent from the card and the ProofGraph node |
| 17 | Fake account relationships displayed | ✅ | `BEHAVIORALLY_SIMILAR_TO` only; forbidden network phrasing asserted absent; the panel draws no arrows |
| 18 | Existing ProofGraph breaks | ✅ | `test_proofgraph_unchanged.py` — core nodes, edges and courtroom verdict identical |
| 19 | Validation Lab breaks | ✅ | live e2e QA suite 10/10 |
| 20 | Sealed validation breaks | ✅ | `test_validation_cohort_invariance.py` — sealed payload and cohort ranking byte-identical with and without the target column |
| 21 | One-command startup breaks | ✅ | `run.sh` verified live on the current champion, and a **fresh `git clone` booted and served a real cohort query** — see §6.1 below |
| 22 | Offline mode breaks | ✅ | release-gate `scoring_survives_ollama_outage`; `ollama_guardrail` 16/16; `ollama_required: false` |
| 23 | P0/P1 regression | ✅ | full release suite **PASS** — pytest 799 passed, QA 90/90, gate 23/23, **P0 0 open · P1 0 open** |

Cohort-side evidence: [`artifacts/testing/cohort_radar_leakage.json`](../artifacts/testing/cohort_radar_leakage.json)
(11/11 PASS), [`artifacts/testing/cohort_radar_determinism.json`](../artifacts/testing/cohort_radar_determinism.json)
(3/3 PASS) and [`artifacts/testing/cohort_radar_fairness.json`](../artifacts/testing/cohort_radar_fairness.json)
(PASS).

### 6.1 What blocker 21 actually surfaced

The fresh-clone test is the only check here that failed the first time, and it is
worth recording rather than quietly fixing.

A clone booted to a healthy `/health/ready` — and then returned **500** on the
first real request. `data/interim/` is gitignored, so the clone had the frozen
model but not the feature frame the model reads. This was **not** caused by the
Cohort Radar: `routes_proofgraph.py` and `routes_validation.py` call the same
`build_model_frame()`, so ProofGraph and the Validation Lab would have failed
identically. It was a pre-existing gap that only a fresh-clone test can find,
because on the machine that trained the model the file is always already there.

`run.sh` now performs the one-time XLSX → Parquet conversion when the frame is
missing, before starting anything. A stack that reports itself ready and then
fails on first use is a worse failure than one that refuses to start.

Re-tested after the fix, from a clean `git clone`: backend ready, and
`POST /v1/cohort/search` returned five behavioural neighbours with similarity,
percentile, band and shared features, scoped to `development reference
partition, 7264 accounts (locked test excluded)`.

---

## 7. Reproducing this

```bash
# the verdict is the exit code
python -m muleguard.cli.usp_baseline --check

# the cohort-side leakage, determinism and fairness audits
python -m muleguard.cli.cohort_evidence

# the regression suite, as tests
pytest tests/regression -q
```

`tests/regression/test_score_invariance_after_usp.py` skips — rather than
silently passing — if the baseline file is absent, so a missing baseline cannot
be mistaken for a clean result.

---

## 8. Related

- [`docs/USP_PRECHANGE_BASELINE.md`](USP_PRECHANGE_BASELINE.md) — what was frozen, and why each item
- [`docs/COHORT_RADAR.md`](COHORT_RADAR.md) — the retrieval layer this proves is inert
- [`docs/ACCOUNT_CONTROL_AMBIGUITY.md`](ACCOUNT_CONTROL_AMBIGUITY.md) — the guardrail this proves is inert
- [`docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md`](FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md) — where these metrics come from
