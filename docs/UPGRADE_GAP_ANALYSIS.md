# Upgrade Gap Analysis

**Companion to:** `docs/CURRENT_REPO_AUDIT.md`
**Machine-readable evidence:** `artifacts/audit/before_upgrade.json`

This document states, per requirement of the upgrade specification, what was
missing and what was done about it. It is written to be read by someone who is
sceptical of the claim "we fixed it" and wants the number that proves it.

---

## 1. The headline gap: the previous model was scoring on the answer

The pre-upgrade accepted model was `catboost_tuned_top60`, reported at OOF
PR-AUC **0.8077 ± 0.045 — RETIRED, leakage-inflated, never to be quoted as a
result**. It is superseded by `xgboost_top_120` at OOF PR-AUC 0.7690 ± 0.0266
(§3). Its 60 features break down as:

| Availability class | Count |
|---|---:|
| BEHAVIORAL | 51 |
| ALERT_CONTEXT | 4 |
| **POST_RESOLUTION_LEAKAGE** | **3** |
| PROFILE | 1 |
| **PRE_EXISTING_RISK_CONTEXT** | **1** |

The four problem columns, and why each is inadmissible:

| Feature | Variable | Why it cannot be an input |
|---|---|---|
| `F3898` | `MIN_RESOLVE_DAYS` | Measures how long the investigation took. It does not exist until the investigation is over — i.e. until after the decision this model is supposed to make. |
| `F3913` | `OTHER_RESOLUTION` | The alert's closing status. One of the four mutually exclusive outcomes of the review. |
| `F3914` | `FALSE_POSITIVE` | The alert's closing status, and very nearly the negation of the target. |
| `F3916` | `L3_FLG` | Customer risk level. The dictionary does not say whether the level is written during periodic KYC (legitimately pre-decision) or as an output of this investigation. Undetermined timing ⇒ quarantine until proven. |

Not one of these will be populated for an account that has not yet been
investigated, which is the only kind of account the system is ever asked
about in production, and the only kind on the organiser's hidden validation
set. A model leaning on them scores well on a historical extract and does
nothing useful in the field.

**Overlap between the old top-60 and the new leakage-free top-60: 12 of 60.**
Four-fifths of the previous feature set was an artifact of the leak.

### 1.1 Leakage by complement — why removing `F3912` alone was not enough

The pre-upgrade quarantine did remove `F3912` (`FRAUD_SUSPECTED`). That was
insufficient, and the reason is worth stating precisely because it is the kind
of thing an audit misses.

`F3912`, `F3913`, `F3914`, `F3915` are a near-one-hot encoding of the alert
resolution: on 7,606 of 9,082 rows exactly one of the four is set. So the
removed column is reconstructible from the ones that were left:

```
complement := 1 - max(F3913, F3914, F3915)
```

- `complement` agrees with the removed `F3912` on **86.95 %** of rows.
- `complement` alone reaches **AP 0.0387** against a prevalence of 0.0089 — a
  **4.3× lift** from a column that was supposed to have been deleted.

Removing one member of a one-hot family does not remove the information. All
four go, together with the two resolve-day columns.

### 1.2 Measured target association of every quarantined column

| Feature | Variable | Pearson r with target | AP alone |
|---|---|---:|---:|
| `F3912` | FRAUD_SUSPECTED | +0.969 | 0.9398 |
| `F3913` | OTHER_RESOLUTION | −0.063 | — |
| `F3898` | MIN_RESOLVE_DAYS | −0.055 | — |
| `F3914` | FALSE_POSITIVE | −0.055 | — |
| `F3899` | MAX_RESOLVE_DAYS | −0.033 | — |
| `F3915` | UNATTENDED | −0.005 | — |
| `F3916` | L3_FLG | +0.022 | — |
| `F3917` | L2_FLG | +0.008 | — |
| `F3918` | L1_FLG | −0.006 | — |

Note the trap: the individually *weak* correlations of `F3913`/`F3914`/`F3898`
are exactly why they survived the previous audit. A correlation screen does
not catch them. Only a question about **when the value comes into existence**
does. That question is now asked of every one of the 3,924 columns, mechanically,
in `src/muleguard/features/dictionary.py`.

### 1.3 `F2230` (MNTH) — a sampling artifact, not a signal

Every negative row in the supplied extract is the 2025-10 snapshot; every
positive row is a Sep/Nov/Dec snapshot. The month column alone reconstructs the
label. That is an artifact of how the sample was assembled, not a property of
mule accounts, and it also makes any "out-of-time" split on this column
meaningless. Hard-quarantined, and the out-of-time stress track is designed
around this fact rather than pretending the column is a date.

---

## 2. Gap-by-gap status

| # | Gap | Before | After | Evidence |
|---|---|---|---|---|
| 1 | Feature semantics | 3,924 opaque `F####` names | Full semantic registry: variable name, description, family, transform, window, direction, availability class | `artifacts/features/feature_dictionary.json` (3,924 rows, **0 unclassified**) |
| 2 | Leakage policy | 4 columns quarantined, by hand | 9 hard + 3 conditional + 1 fairness exclusion, mechanically derived and independently re-asserted at 3 enforcement points | `configs/feature_availability.yaml`, `src/muleguard/features/firewall.py` |
| 3 | Enforcement | None — training code read a JSON list | `build_model_frame()` is the only way to obtain a matrix, and it calls `assert_clean()` before returning | `src/muleguard/features/frame.py` |
| 4 | Domain features | None | 13 interpretable meta-features (pass-through, retention, burst, rail fragmentation, cash-out pressure, balance drain, new-account activity, profile mismatch, alert convergence, odd-hour ratio, merchant legitimacy), all at 100 % coverage | `src/muleguard/features/meta_features.py` |
| 5 | Feature selection | Single global ranking | Stability selection: ranked independently in each of 10 training folds, reported as selection frequency | `artifacts/features/selection_frequency_v2.csv` |
| 6 | Model tournament | Ran on the leaked pool | Re-run entirely on the admitted pool, across five availability views | `artifacts/metrics/tournament_v2.json` |
| 7 | Ensembling | Logistic stacker only | + rank averaging + logit averaging + Borda aggregation, with the addendum's three acceptance criteria and our stricter per-repeat criterion recorded separately so either reading can be re-derived. Outcome: `SINGLE_MODEL_KEPT` | `artifacts/metrics/ensemble_v2.json` |
| 8 | Rare-positive robustness | Not measured | Positive-removal stress test | `docs/ROBUSTNESS_REPORT.md` |
| 9 | Distribution shift | PSI baseline only | Adversarial validation shield (feature-level flags + train-vs-upload classifier) | Validation Lab |
| 10 | Explanation | One-sided reason codes | Dual-Evidence ProofGraph + Counterfactual Twin + Model Courtroom | ProofGraph API |
| 11 | Hidden-validation trust | Upload → score | Schema integrity → shield → **sealed predictions (SHA-256, timestamped)** → only then labels | Sealed Validation Protocol |
| 12 | Fairness | Gender was selectable (and appeared in fold 2 of the old selection) | Excluded by default policy; demographics contextual-only and never a sole escalation driver | `configs/feature_availability.yaml` §fairness |

---

## 3. What the honest number costs

The leakage-free retrain **lowers** the headline metric, and it should. The
tournament was then re-run in full on the firewall-admitted pool, 3 repeats of
stratified group-aware CV on the 7,264-row development split (64 positives).
Source: `artifacts/metrics/tournament_v2.json`,
`artifacts/metrics/model_comparison_v2.csv`.

### 3.1 The leakage-free leaderboard

| Model | Features | OOF PR-AUC (mean ± std) | ROC-AUC | Recall@100 |
|---|---:|---:|---:|---:|
| `tabpfn_top_60` (challenger — verified, **not promoted**, §3.1.1) | 60 | 0.9110 ± 0.0044 | 0.9924 | 0.9115 |
| **`xgboost_top_120`** (promoted, served) | 120 | **0.7690 ± 0.0266** | 0.9577 | 0.7813 |
| `xgboost_top_60` | 60 | 0.7408 ± 0.0498 | 0.9420 | 0.7604 |
| `lightgbm_top_60` | 60 | 0.6992 ± 0.0273 | 0.9397 | 0.7344 |
| `catboost_top_60` | 60 | 0.6963 ± 0.0417 | 0.9323 | 0.7448 |
| `xgboost_top_30` | 30 | 0.6897 ± 0.0122 | 0.9334 | 0.7500 |
| `lightgbm_top_120` | 120 | 0.6862 ± 0.0405 | 0.9387 | 0.6979 |
| `lightgbm_top_30` | 30 | 0.6628 ± 0.0161 | 0.9357 | 0.6979 |
| `lightgbm_viewA_top_60` (behavioural only) | 60 | 0.6584 ± 0.0234 | 0.9423 | 0.6927 |
| `catboost_top_120` | 120 | 0.6467 ± 0.0699 | 0.9316 | 0.7292 |
| `lightgbm_viewB_top_60` | 60 | 0.6454 ± 0.0485 | 0.9427 | 0.6875 |
| `lightgbm_top_250` | 250 | 0.6442 ± 0.0300 | 0.9393 | 0.6563 |
| `lightgbm_full_pool` | 3,925 | 0.5874 ± 0.0276 | 0.9286 | 0.6198 |
| `catboost_top_30` | 30 | 0.5614 ± 0.1125 | 0.9156 | 0.6146 |
| `lightgbm_top_15` | 15 | 0.3357 ± 0.1399 | 0.8931 | 0.4010 |
| `lightgbm_viewE_top_60` (merchant/business) | 60 | 0.3061 ± 0.0842 | 0.8643 | 0.3646 |
| `lightgbm_freq_ge_0_50` | 13 | 0.1917 ± 0.0247 | 0.8634 | 0.2656 |
| `elasticnet_top30` | 30 | 0.0895 ± 0.0045 | 0.8795 | 0.1927 |
| `logistic_top30` | 30 | 0.0882 ± 0.0045 | 0.8808 | 0.1875 |
| `lightgbm_viewC_top_15` (bank prior only) | 15 | 0.0295 ± 0.0110 | 0.6723 | 0.0573 |
| `lightgbm_viewD_top_15` (alert context only) | 15 | 0.0206 ± 0.0021 | 0.7046 | 0.0313 |
| `dummy_prevalence` | — | 0.0087 | 0.4938 | 0.0000 |

### 3.1.1 The TabPFN challenger: verified, and deliberately not promoted

An earlier revision of this document recorded `tabpfn_top_60` as `FAILED`,
because the installed TabPFN refuses to run on CPU with more than 1,000 samples.
That was true of the first attempt and is no longer true of the repository:
re-run with `ignore_pretraining_limits=True`, it completed all three repeats and
`artifacts/metrics/tournament_v2.json` records it with status `OK`. The old
sentence is corrected here rather than deleted, because a leaderboard that
quietly loses its best row is worse than one that explains it.

The result is large enough to deserve disbelief, so it was attacked before it
was accepted. Evidence in `artifacts/metrics/challenger_review_v2.json`:

| Question | Finding |
|---|---|
| Three independent fold seeds (UPDATE 1)? | Yes — 0.90709 / 0.91709 / 0.90883 across repeats, spread 0.010 |
| Folds actually independent? | Pairwise same-fold fraction 0.1989–0.2026 against a 0.2000 chance level |
| Any quarantined column in the set? | None — zero overlap between `top_60` and the firewall's quarantine list |
| Could the *features* be leaking? | No. `xgboost_top_60` consumes the **identical 60 columns over the identical folds** and scores 0.7408. A leaking column would have lifted both. |
| Did the fold contract hold? | Imputation is fitted on the training fold only; `predict_proba` receives validation rows with no labels; the locked test was untouched |

So the number is real, and the honest reading is that the difference lives in
the estimator, not in the data — which is the one explanation that does not
imply a broken firewall.

**It is still not promoted, and the reason has nothing to do with accuracy.**
`artifacts/metrics/tabpfn_latency.json` measures what serving it would cost.
TabPFN learns no parameters; it carries the whole 7,264-row development split
through the transformer on every forward pass:

| Call | Time |
|---|---:|
| `fit` (memorise the training set) | 1.2 s |
| `predict_proba`, **1 row** | **438 s** |
| `predict_proba`, 100 rows | 531 s |

The cost is per *call*, not per row. One analyst lookup takes 7.3 minutes
against milliseconds for the champion. Worse for this system specifically:
TabPFN exposes no attribution path, so a §17 ProofGraph would have to be built
by occlusion — 61 forward passes, about **7.4 hours per explained case**. §17 is
the product's main claim, "every alert must prove why it was raised", and a
score that cannot be proved is not one this system is allowed to serve. Nor can
the champion's SHAP stand in for it: the two models agree only at Spearman
**0.222**, so presenting one model's attributions as an explanation of the
other's score would be fabricated evidence.

That is a decision about what the system is for, not a dispute about the
metric, and it is recorded as such in `challenger_review_v2.json` under
`decision`, with three things it explicitly is *not*: the PR-AUC is not
disputed, no leakage was found, and the folds were not correlated. It would be
promoted if a GPU brought single-row inference under about a second **and** a
faithful per-feature attribution for TabPFN became available.

Meanwhile the challenger is kept as a recorded second opinion. UPDATE 2's
rank-stable blend of the two scores **0.8627 ± 0.0143** — between the members,
not above them — so there is no ensembling case either. The 0.222 rank
correlation is carried into the Model Courtroom as *uncertainty*, never as a
reason to raise a risk score (UPDATE 6).

Against the 0.8919 % prevalence of the extract, the promoted model's
**0.7690 is roughly an 86× lift** over the prevalence floor, and it is a lift
that is actually about account behaviour. The previously published honest
baseline — "leakage-free LightGBM, full admitted pool, 0.5521, single repeat"
— is superseded: the same configuration measured over 3 repeats scores
**0.5874 ± 0.0276**, and it is now the *worst* of the tree models rather than
the reference point. Compact, stability-selected sets beat the full 3,925-column
pool by a wide margin, which was the tournament's stated job.

### 3.2 Why 0.8077 is retired and must not be quoted

`catboost_tuned_top60` reported **OOF PR-AUC 0.8077 ± 0.045**. Its top-60 set
contained `F3898` (MIN_RESOLVE_DAYS), `F3913` (OTHER_RESOLUTION), `F3914`
(FALSE_POSITIVE) and `F3916` (L3_FLG) — three post-resolution columns and one
whose availability time is undetermined (§1). Those columns do not exist for an
account that has not yet been investigated, which is the only kind of account
the system is ever asked about, so the number they produced is not an estimate
of hidden-validation performance. It is **retired**. It appears in this
repository only as a labelled leaky baseline, with this explanation attached,
and in the `supersedes` block of `artifacts/models/model_manifest.json` and
`artifacts/model_registry/registry.json`, where the retired bundle is recorded
alongside its replacement.

The correct comparison is therefore not "0.8077 fell to 0.7690". It is
"0.8077 was never measuring the task; 0.7690 is the first number that was".

### 3.3 The view ablations — evidence that the model learned behaviour

The two ablation views are the strongest available evidence that the promoted
model generalises from behaviour rather than from a pre-existing verdict:

| View | Content | OOF PR-AUC | Reading |
|---|---|---:|---|
| C — `C_bank_prior` | the bank's 18 finalised variables plus safe profile fields | **0.0295 ± 0.0110** | barely above the 0.0087 prevalence floor |
| D — `D_alert_context` | pre-decision alert evidence only (28 candidates) | **0.0206 ± 0.0021** | barely above the 0.0087 prevalence floor |

If the detector were quietly re-reading a pre-existing bank risk flag, view C
would score well. If it were re-reading analyst alert metadata — "this case was
queued, therefore it is a mule" — view D would score well. Neither does. Both
sit within roughly 2–3× of a model that predicts the base rate for every
account, while the admitted behavioural pool reaches 0.7690. The signal the
promoted model uses is in the transaction and balance aggregates, not in the
bank's prior opinion of the customer and not in the shape of the alert.

We would rather present 0.7690 that survives the hidden validation set than
0.8077 that collapses on it. That trade is the whole point of the upgrade, and
the numbers above are published precisely so a judge can see the trade was made
deliberately.

---

## 4. Deliberate non-changes

- **The locked holdout was not regenerated.** It was created target-blind under
  the previous quarantine list. Rebuilding it now — after model results have
  been seen — would destroy the one property that makes it worth having.
  Documented rather than silently refreshed.
- **`cli/tournament.py` and `models/baselines.py` are retained**, superseded but
  not deleted, because they are the reproducible record of how the leaked
  result was produced.
- **No competitor code, assets or architecture were copied.** Repositories that
  could not be accessed are marked `NOT_VERIFIED` in
  `docs/COMPETITOR_GAP_MATRIX.md`.
- **Nothing from addendum UPDATE 7's exclusion list was added** — no ODE risk
  scoring, no mock GraphSAGE embeddings, no persistent homology, no bandits,
  no synthetic trajectories, no auto-adapting thresholds, no automatic
  freezing.
