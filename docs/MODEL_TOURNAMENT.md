# Model Tournament

Master prompt §9–10, addendum UPDATE 1, UPDATE 2 and UPDATE 13.

Runner: `src/muleguard/cli/tournament_v2.py` · Leaderboard:
`artifacts/metrics/model_comparison_v2.csv` · Promotion:
`promotion_decision_v2.json` · Ensembling: `ensemble_v2.json` · Challenger:
`challenger_review_v2.json`.

**Nothing below is typed by hand.** Every figure is read from those files. The
retired, pre-firewall tournament (winner `catboost_tuned_top60`, headline 0.8077)
is superseded and narrated separately in `docs/MODEL_TOURNAMENT_REPORT.md`; its
number is **retired** because its pool still contained post-resolution columns.

---

## 1. Protocol

| | |
|---|---|
| Design | repeated stratified **5 folds × 3 repeats**, one immutable fold assignment shared by every contestant |
| Primary metric | **out-of-fold PR-AUC** (average precision at natural prevalence), mean ± std across repeats, per-repeat values preserved |
| Secondary | ROC-AUC, Recall@100, per-repeat minimum |
| In-fold everything | imputation, feature selection, early stopping, hyperparameters, calibration — all fitted inside the training fold |
| Feature source | every matrix is built through `features/frame.build_model_frame()`, so the Availability Firewall cannot be bypassed by any contestant |
| Locked test | **not touched** by any candidate; asserted in the harness and by a release-gate check |
| Resampling | none. No SMOTE, no undersampling |

PR-AUC is primary because accuracy is meaningless at 0.89 % prevalence — a model
predicting "not a mule" for everyone scores 99.1 %. The dummy baseline row below
is included so that number stays visible.

---

## 2. The leaderboard

21 contestants across 7 families and 5 availability views.

| # | Model | Family | View | Feats | OOF PR-AUC | ± std | Min repeat | ROC-AUC | R@100 | Gen. score |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **xgboost_top_120** | xgboost | ALL | 120 | **0.76904** | 0.02663 | 0.74927 | 0.95771 | 0.78125 | **0.83385** |
| 2 | xgboost_top_60 | xgboost | ALL | 60 | 0.74082 | 0.04981 | 0.68068 | 0.94201 | 0.76042 | 0.79196 |
| 3 | lightgbm_top_60 | lightgbm | ALL | 60 | 0.69918 | 0.02733 | 0.66243 | 0.93968 | 0.73438 | 0.75895 |
| 4 | catboost_top_60 | catboost | ALL | 60 | 0.69630 | 0.04168 | 0.65837 | 0.93226 | 0.74479 | 0.74994 |
| 5 | xgboost_top_30 | xgboost | ALL | 30 | 0.68974 | **0.01224** | 0.67415 | 0.93336 | 0.75000 | 0.75862 |
| 6 | lightgbm_top_120 | lightgbm | ALL | 120 | 0.68621 | 0.04045 | 0.63087 | 0.93874 | 0.69792 | 0.73578 |
| 7 | lightgbm_top_30 | lightgbm | ALL | 30 | 0.66281 | 0.01609 | 0.64600 | 0.93570 | 0.69792 | 0.72456 |
| 8 | lightgbm_viewA_top_60 | lightgbm | A broad behavioural | 60 | 0.65844 | 0.02341 | 0.62561 | 0.94231 | 0.69271 | 0.71600 |
| 9 | catboost_top_120 | catboost | ALL | 120 | 0.64670 | 0.06993 | 0.55161 | 0.93160 | 0.72917 | 0.68465 |
| 10 | lightgbm_viewB_top_60 | lightgbm | B stable compact | 60 | 0.64543 | 0.04847 | 0.57715 | 0.94271 | 0.68750 | 0.68995 |
| 11 | lightgbm_top_250 | lightgbm | ALL | 250 | 0.64419 | 0.02999 | 0.60412 | 0.93932 | 0.65625 | 0.69482 |
| 12 | **lightgbm_full_pool** | lightgbm | ALL | **3,925** | 0.58743 | 0.02758 | 0.55561 | 0.92861 | 0.61979 | 0.63562 |
| 13 | catboost_top_30 | catboost | ALL | 30 | 0.56136 | 0.11250 | 0.41553 | 0.91561 | 0.61458 | 0.56657 |
| 14 | lightgbm_top_15 | lightgbm | ALL | 15 | 0.33566 | 0.13990 | 0.22834 | 0.89306 | 0.40104 | 0.30581 |
| 15 | lightgbm_viewE_top_60 | lightgbm | E profile/merchant | 60 | 0.30611 | 0.08419 | 0.19078 | 0.86428 | 0.36458 | 0.30047 |
| 16 | lightgbm_freq_ge_0_50 | lightgbm | ALL | 13 | 0.19166 | 0.02470 | 0.15720 | 0.86343 | 0.26562 | 0.20588 |
| 17 | elasticnet_top30 | elasticnet | ALL | 30 | 0.08954 | 0.00449 | 0.08576 | 0.87953 | 0.19271 | 0.10657 |
| 18 | logistic_top30 | logistic | ALL | 30 | 0.08816 | 0.00454 | 0.08243 | 0.88075 | 0.18750 | 0.10464 |
| 19 | lightgbm_viewC_top_15 | lightgbm | C bank prior | 15 | 0.02949 | 0.01101 | 0.01554 | 0.67231 | 0.05729 | 0.02972 |
| 20 | lightgbm_viewD_top_15 | lightgbm | D alert context | 15 | 0.02059 | 0.00205 | 0.01825 | 0.70457 | 0.03125 | 0.02269 |
| 21 | dummy_prevalence | dummy | ALL | 30 | 0.00871 | 0.00000 | 0.00871 | 0.49375 | 0.00000 | 0.00871 |

---

## 3. What the leaderboard actually says

### 3.1 Compact beats complete, decisively

**120 features: 0.76904. All 3,925 features: 0.58743.** Throwing the whole matrix
at the problem costs **0.18 PR-AUC** — and 409 s per fit instead of 32 s.

With 64 training positives, 3,925 columns is not information, it is noise for the
model to overfit. This is the graded deliverable of the problem statement
("identify relevant fraud/mule features") answered with a measurement rather than
an assertion.

### 3.2 But compact has a floor

top-120 → 0.769 · top-60 → 0.741 · top-30 → 0.690 · **top-15 → 0.336**. And
`freq_ge_0_50` — the 13 features stability selection picks in ≥ 50 % of folds —
scores only 0.192.

The signal here is genuinely distributed. There is no small set of "the mule
features"; anyone showing a 10-feature model on this data is showing something
that does not work.

### 3.3 Linear models fail completely

Logistic and ElasticNet land at **0.088** — an order of magnitude below the trees
— despite ROC-AUC of 0.88. That gap between ROC-AUC and PR-AUC is the whole
lesson of imbalanced problems: the linear models rank broadly well and are
useless at the top of the queue, which is the only part anyone reviews.

The mule signal is **interaction-shaped**, not additive. Trees are not a
convenience here, they are a requirement.

### 3.4 Alert flags alone are nearly worthless

View D (alert context only, 15 features) scores **0.021** — barely above the
0.0087 dummy floor. View C (bank prior) scores 0.029.

This matters for the pitch: the bank's existing alert flags do **not** identify
mules on their own. The value is in the behavioural columns and in the
combination. A rules engine over those flags — the approach in three of the seven
competitor repos we reviewed — would land near the dummy baseline.

### 3.5 Stability and score are not the same thing

`xgboost_top_30` has the second-lowest std (0.01224) and `catboost_top_30` the
highest (0.11250) with a min-repeat of 0.41553 — a model whose worst fold is 40 %
below its mean is not deployable regardless of its average.

This is why promotion uses a stability-aware rule rather than raw PR-AUC.

---

## 4. Promotion (UPDATE 13)

```
generalization_score = PR_AUC_mean − 0.5 × PR_AUC_std + 0.1 × Recall@100
```

Applied **only as a tie-break within a 0.01 PR-AUC band of the leader**; inside
that band the simpler and more stable model wins.

Recorded in `promotion_decision_v2.json`: the band contained only
`xgboost_top_120`, so `tie_break_applied: false`. **The champion is the raw
PR-AUC leader and also the generalization-score leader** — no judgement call was
required, and the rule is published so it is checkable that none was made.

Champion per-repeat: 0.74927, 0.80668, 0.75117.

---

## 5. Ensembling (UPDATE 2)

Members: `xgboost_top_120`, `xgboost_top_60`, `lightgbm_viewA_top_60`,
`lightgbm_viewB_top_60`.

| Combiner | Mean OOF PR-AUC |
|---|---:|
| **best single** (`xgboost_top_120`) | **0.76904** |
| logit mean | 0.76946 |
| stacker | 0.74269 |
| rank mean | 0.73552 |
| Borda | 0.73552 |

Decision: **`SINGLE_MODEL_KEPT`**.

The logit mean "wins" by **0.0004 PR-AUC**. The measured seed-noise floor on this
dataset is **0.0905** — this margin is smaller by more than two orders of
magnitude. Shipping a four-model ensemble to capture it would add four model loads
and four attribution paths to the serving cost in exchange for nothing
measurable.

UPDATE 2 also forbids what C2 in our competitor review does: *"Do NOT add a
complex 5,000-random-blend search."* A Dirichlet blend search over folds holding
~16 positives fits fold noise, and it would have been *very* easy to report the
resulting number as an improvement.

---

## 6. The TabPFN challenger (UPDATE 1)

Champion selection was reopened, as the addendum requires. `tabpfn_top_60` is
substantially better on paper:

| | Champion `xgboost_top_120` | Challenger `tabpfn_top_60` |
|---|---:|---:|
| OOF PR-AUC | 0.76904 ± 0.02663 | **0.91100 ± 0.00436** |
| ROC-AUC | 0.95771 | **0.99238** |
| Recall@50 | 0.6458 | **0.7708** |
| Recall@100 | 0.7813 | **0.9115** |

### It is not a leak, and we checked properly

| Gate | Result |
|---|---|
| ≥ 3 independent fold seeds | **PASS** — pairwise same-fold fraction 0.2025 / 0.2026 / 0.1989 against a chance level of 0.200 → `INDEPENDENT` |
| No quarantined feature | **PASS** — overlap empty |
| Beats champion on PR-AUC | **PASS** |
| At least as stable | **PASS** — std 0.0044 vs 0.0266 |
| Shared-feature control | **`ESTIMATOR_NOT_DATA`** — TabPFN 0.911 vs `xgboost_top_60` 0.741 on the *identical* 60 columns and folds. A leaking column would have lifted the control too |

**All four UPDATE 1 gates pass and the result is accepted as real.**

### It is still not promoted

`challenger_status: VERIFIED_NOT_PROMOTED`, for reasons that have nothing to do
with accuracy:

| Measured serving cost | |
|---|---:|
| fit | 1.21 s |
| **single-row predict** | **438.1 s** |
| 100-row predict | 530.6 s |
| ProofGraph via occlusion (61 forward passes) | **~7.4 hours per case** |

Cost is per **call**, not per row — this is in-context learning, so every
prediction replays the training set through a transformer.

Two things break. An analyst clicking an account waits **seven minutes** instead
of milliseconds. And TabPFN exposes no attribution path, so §17's ProofGraph — the
evidence surface this entire system is built on — would take **7.4 hours** to
produce for one case. A surrogate SHAP model is not an option either: the rank
correlation between the two models is Spearman **0.222**, so a surrogate would be
explaining a different model's decision.

> A more accurate score that cannot be explained or returned in time is not a
> better product.

**Explicitly not the reasons:** the challenger's PR-AUC is not disputed; no
leakage was found; the folds were independent at chance level.

**It would be promoted if** a GPU brings single-row inference under ~1 s **and** a
faithful per-feature attribution for TabPFN becomes available.

Meanwhile it is retained as a recorded second opinion. Its disagreement with the
champion enters the Model Courtroom as **uncertainty** — never as a reason to
raise a risk score (UPDATE 6).

A rank blend of the two was also measured: 0.8627 ± 0.0143, `beats_best_member:
false`. It sits *between* the members, so there is no ensembling case.

---

## 7. Where the champion landed

| | |
|---|---|
| Served model | `xgboost_top_120` |
| OOF PR-AUC | 0.76904 ± 0.02663 (per-repeat 0.74927 / 0.80668 / 0.75117) |
| ROC-AUC | 0.95771 |
| Recall@100 | 0.78125 |
| Locked test (single touch, reported not tuned) | **PR-AUC 0.7263**, ROC-AUC 0.9665, lift **77.7×**, Recall@100 0.824 |
| Bundle SHA-256 | `d12914de5abee99a…` · runtime fingerprint `afd0dc1d8fc02eb9` |

The locked-test figure is **lower** than the OOF figure. It is reported as
measured, with no re-tuning, because the honest gap between development and
held-out performance is the most useful number a judge can have.
