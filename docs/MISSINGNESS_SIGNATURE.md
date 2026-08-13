# Missingness Signature

**Status of the verdict: RESOLVED — KEEP, and the suspect-family check passed.** The deciding
ablation is in section 9 (`artifacts/metrics/missingness_ablation.json`) and the confirmation arm
that excludes the untraceable `FEES_AND_CHARGES` columns is in section 10
(`artifacts/metrics/missingness_ablation_no_fees.json`, 2026-08-13). The gain survives the
exclusion, so the KEEP is no longer provisional. It remains a decision about a **feature block for
`histgb` on development folds** — it promotes nothing and changes no served model.

Reproduce the evidence in sections 3–6 with:

```
.venv/Scripts/python.exe -m muleguard.cli.missingness_probe --control-draws 2000 --control-seed 42
```

which writes `artifacts/metrics/missingness_probe.json`. Every number quoted below comes from that
artifact.

---

## 1. Why look at missingness at all

27.6 % of the cells in this dataset are null. That is too much to treat as an accident of
extraction, and too much to impute away without asking what the nulls mean first. In a banking
feature table, a null is rarely "we lost this value" — it usually means *the event never
happened*. An account with no 7-day cheque aggregate did not have cheque activity in the last
seven days.

If that is what nulls mean, then the *shape* of the missingness is behavioural information, and
throwing it into an imputer discards it. The question this document answers is whether encoding
that shape actually makes the model better, or only makes it more complicated.

The instruction being followed here is `updates1` upgrade #1, which is specific about the terms:
build the indicators **inside CV only**, ablate WITHOUT against WITH, and *"keep them only if
repeated CV improves and the indicators remain stable across folds."* Two conditions, not one.

## 2. What was built

`src/muleguard/features/missingness.py` — a fitted encoder, prefix `MX_`, producing **358
columns** at the default cap:

| kind | count | what it encodes |
|---|---|---|
| `MX_NULL__<col>` | 200 | per-column "this field was absent" flags |
| `MX_FAMCNT__<fam>` / `MX_FAMRATIO__<fam>` | 22 × 2 | how much of a dictionary-defined block is absent |
| `MX_SHORTGAP__…` / `MX_LONGGAP__…` | 54 × 2 | short window missing while long window present, and the reverse |
| `MX_CTXCNT__<fam>` / `MX_CTXALL__<fam>` | 2 × 2 | PROFILE and ALERT_CONTEXT absence, and whole-block absence |
| `MX_TOTAL_NULL` / `MX_TOTAL_NULL_RATIO` | 2 | global volume |

Three design choices are load-bearing:

**Flags are banded, not universal.** A column that is null 0.02 % of the time, or 100 % of the
time, yields a near-constant indicator that carries no contrast and only widens the candidate
pool. Only columns whose *training* missing rate lies in [0.01, 0.99] earn a flag, capped at 200
by descending indicator variance `p(1−p)` with column order as a deterministic tie-break.

**Families and window pairs come from the data dictionary, not from string matching.** The
`(L7D, L31D)` asymmetry is only meaningful between two variables that differ *only* in their
window suffix, which requires the registry to confirm they are the same measurement.

**The transform is row-local.** Everything that involves a *decision* — which columns get a flag,
which families are large enough to summarise, which window pairs exist — is fitted on training
rows in `fit()`. The transform itself is arithmetic on `isnan(X)` for a single row. Two
consequences: applying it to held-out rows cannot move information across a fold boundary, and a
single account can be scored alone at serving time without reference to any other account.

`MX_CTXALL__<fam>` exists separately from `MX_CTXCNT__<fam>` because "the entire PROFILE block is
absent" is a categorically different event from "some of PROFILE is absent" — the first says the
bank did not have or did not attach the record at all.

## 3. Is the missingness structural or incidental?

Measured on the 7,264 development rows (64 positives) across the 3,925 admitted columns:

| quantity | value |
|---|---|
| fraction of cells null | **0.276213** |
| columns with at least one null | 3,834 |
| columns entirely null | 63 |
| columns in the 1–99 % band | **1,403** |
| mean row-null ratio, legitimate | 0.27613 |
| mean row-null ratio, mules | 0.28600 |
| ROC-AUC of the naive null count | **0.61865** |

**Interpretation.** The missingness is structural: it touches 3,834 of 3,925 columns, and 1,403 of
them sit in a range where an indicator could carry real contrast. But the *volume* of missingness
is nearly useless on its own — mules are null in 28.6 % of their fields against 27.6 % for
everyone else, and a count of nulls reaches only 0.619 ROC-AUC. That gap is the entire reason to
encode shape rather than amount. It is also a warning: if shape turns out to be worth only a
little more than 0.619, it is not worth 358 columns.

**A number that looks alarming and is not.** The admitted matrix has 3,925 columns, which is
exactly the raw column count — it reads as though nothing was removed. It is a coincidence: the
firewall drops the 13 quarantined columns and the `MG_*` meta-feature block adds 13. The probe
artifact therefore carries the check directly, and
`quarantined_columns_found_in_admitted_set` is empty: none of `F3924, F3912, F3913, F3914,
F3898, F3899, F3915, F3916, F3917, F3918, F2230, F3892, __UNNAMED__0` is present as a model input.

## 4. The strongest indicator, and why it was treated as a suspect

In-sample univariate ROC-AUC against the target, top of 358:

| column | ROC-AUC |
|---|---|
| `MX_FAMCNT__FEES_AND_CHARGES` | **0.74060** |
| `MX_FAMRATIO__FEES_AND_CHARGES` | 0.74060 |
| `MX_NULL__F3348`, `F3351`, `F3672`, `F3675` | 0.70924 |
| `MX_NULL__F3240`, `F3243` | 0.70910 |

The count and ratio columns for a family score identically because the ratio is the count divided
by a constant, so they carry the same ranking. That is expected, and it means the block contains
more redundancy than its width suggests.

**These numbers are in-sample and must not be used to pick columns.** They are the maximum of 358
noisy statistics computed on the same rows they are reported for. The probe artifact is stamped
`EXPLORATORY - NOT A SELECTION INSTRUMENT` and the probe lives in a different file from the
ablation specifically so that "this column looked good in the probe, let me add it" has to be
typed out deliberately rather than happening by drift.

The reason to look at all is that 0.741 from a *missingness count* is high enough to be
suspicious. A field that is absent because the bank already resolved the case would score like
this, and would be worthless — or worse than worthless — on a live account. So the next step was
not to celebrate it. It was to try to kill it.

## 5. Leakage audit: the correlation that looked damning was the target restated

The leading indicator was scored against the target and against every quarantined
post-resolution marker:

| marker | prevalence | AUC from `MX_FAMCNT__FEES_AND_CHARGES` | agreement with target | positives shared with target |
|---|---|---|---|---|
| `F3924` (the target) | 0.00881 | 0.74060 | 1.00000 | 64 / 64 |
| `F3912` | 0.00881 | **0.73072** | **0.99945** | **62 / 64** |
| `F3915` | 0.00234 | 0.54149 | 0.98885 | 0 / 64 |
| `F3899` | 0.75289 | 0.52278 | 0.25069 | 45 |
| `F3898` | 0.69700 | 0.52029 | 0.30383 | 35 |
| `F3913` | 0.57971 | 0.50565 | 0.41561 | 15 |
| `F3914` | 0.31938 | 0.48615 | 0.67291 | 4 |

The 0.73072 against `F3912` is what a workflow leak looks like, so it was checked rather than
assumed. `F3912` agrees with the target on **99.945 %** of development rows — 7,260 of 7,264 —
and shares 62 of the target's 64 positives. It is a near-perfect copy of the label.

**Therefore the correlation with `F3912` is the correlation with the target, restated.** It is not
a second, independent leak. Predicting a 99.945 % copy of `y` at 0.731 when you predict `y` itself
at 0.741 is arithmetic, not evidence.

The markers that *are* independent of the target are the ones that could reveal workflow leakage,
and they sit between **0.48615 and 0.54149** — indistinguishable from chance. The leading
missingness indicator does not track the bank's investigation process.

**A flaw in this audit, found and fixed.** The first version of the classifier called any marker
with ≥ 0.99 agreement a proxy. At 0.88 % prevalence that rule is close to meaningless: two
unrelated rare columns agree on ~99 % of rows purely by both being almost always zero. `F3915`
demonstrates it — 98.885 % agreement with the target while sharing **none** of its 64 positives.
It was classified correctly only because it fell 0.001 below an arbitrary cut. The rule now
requires agreement ≥ 0.99 **and** coverage of at least half the target's positives, and the
artifact reports co-rare-but-not-proxy markers separately. The conclusions did not change; the
grounds for them did.

## 6. A real cohort fingerprint, and a null model for it

`MX_FAMCNT__FEES_AND_CHARGES` is not spread across its range for mules the way it is for everyone
else:

| group | rows | distinct values | modal value | modal share |
|---|---|---|---|---|
| mules | 64 | **6** | 178 | **0.70312** |
| legitimate | 7,200 | 67 | 178 | 0.35806 |

Mules occupy 6 distinct values where legitimate accounts occupy 67, and 70.3 % of them sit on a
single value against 35.8 % of legitimate accounts — on the *same* modal value, which matters: the
mules are not somewhere unusual, they are unusually *concentrated* in the ordinary place.

With 64 positives this is exactly the kind of claim that dissolves under scrutiny, because any 64
draws from a wide distribution look concentrated. So a null model was built: 2,000 draws of 64
legitimate accounts (seed 42), each scored the same way.

| statistic | observed in mules | null model over 2,000 draws of 64 | empirical p |
|---|---|---|---|
| distinct values | 6 | min 10, mean 16.311 | **0.0000** |
| modal share | 0.70312 | max 0.54688, mean 0.35667 | **0.0000** |

Not one of 2,000 equally-sized samples of ordinary accounts got as concentrated as the mules on
either statistic — the *most* concentrated draw still had 10 distinct values against the mules' 6,
and the highest modal share reached 0.547 against the mules' 0.703. The fingerprint is not a
sample-size artifact.

**What this does and does not establish.** It establishes that the concentration is real. It does
**not** establish what causes it. Two explanations remain live and this evidence cannot separate
them:

1. mule accounts genuinely share a fee profile — plausible, since mule accounts are typically new,
   low-balance, and fee-exempt in similar ways; or
2. the mule rows were extracted, joined, or labelled through a path that homogenised this block —
   an artifact of dataset construction that would not reproduce on live data.

Explanation 2 is not paranoia; it is the ordinary failure mode of a curated fraud dataset, and no
amount of staring at the column distinguishes it from explanation 1. That is precisely why the
decision is delegated to an ablation that can return NO.

## 7. An independent negative result that belongs here

The Missed-Mule Error Atlas (`docs/ERROR_ATLAS.md`, `artifacts/metrics/error_atlas.json`) examined
the 11 mules the champion misses at K = 100 (recall 0.828125) and tested each against two
missingness criteria. Its finding, quoted:

> Zero misses met either missingness test… missing data does not explain these misses.
> Missingness-structure work is worth doing on its own merits but should not be justified by this
> Atlas.

The missed mules have missing fractions of 0.100–0.217 — *less* missing than the dataset average
of 0.276, not more.

This is a different question from the one the ablation asks. The Atlas asks whether missingness
explains the 11 specific failures; the ablation asks whether the missingness block improves
ranking across all 7,264 rows. A feature can do the second without doing the first. But the Atlas
is a genuine caution against the most attractive over-claim available here — "missingness will
recover the mules we are losing" — and that claim is not supported. It is recorded before the
verdict, not after, so it cannot be quietly dropped if the ablation comes back positive.

## 8. The deciding experiment

`src/muleguard/cli/missingness_ablation.py`. Two arms, WITHOUT and WITH, on **byte-identical
outer folds** — the same 3 × 5 assignments the nested tournament uses.

What is held fixed between the arms: the folds, the preprocessing, the inner-fold selector, the
model family, **the hyperparameters**, the seeds, and the feature-set size. The only difference is
that the WITH arm calls `MissingnessSignature.fit()` on each outer fold's training rows and
appends the block before preprocessing.

Four choices in that design are there to stop the WITH arm from winning for the wrong reason:

- **Fitted inside each outer fold**, on training rows only — never once on the full development
  set. This is the `updates1` requirement and also the only way the comparison means anything.
- **Hyperparameters fixed, not tuned per arm.** Tuning both arms would let WITH win on a luckier
  hyperparameter draw rather than on the new columns.
- **Feature budget fixed at 120.** The new columns do not get extra room; to be used they must
  *displace* base columns, which is the honest test of whether they carry information the existing
  features do not.
- **Paired.** Because the arms differ in one thing on identical folds, the yardstick is the spread
  of the paired difference, which is far tighter than the project's 0.0905 *unpaired* seed-noise
  floor. That floor is the right test for "is model A better than model B"; applying it to a
  paired one-variable ablation would be so blunt it could never detect anything.

The verdict rule, fixed before the run and requiring **all three** conditions:

| condition | threshold |
|---|---|
| mean paired gain positive | `> 0` |
| improved in at least 70 % of folds | `≥ 11` of 15 |
| at least 3 indicators stable in ≥ 80 % of folds | `≥ 3` |

A positive mean with unstable indicators is **rejected as noise**. The third condition is the one
`updates1` asks for and the one most likely to fail: it requires the *same* missingness columns to
keep surviving the selector across folds, not merely that some missingness column helps somewhere.

## 9. Primary result

`--family histgb --repeats 3 --n-feat 120 --max-flags 200`, 15 outer folds, ~51 training positives
and ~13 validation positives per fold. `artifacts/metrics/missingness_ablation.json`.

| arm | PR-AUC (mean of 3 repeats) | per-repeat |
|---|---|---|
| WITHOUT | 0.79068 ± 0.01179 | 0.77699, 0.80577, 0.78928 |
| WITH | **0.84827 ± 0.01921** | 0.82241, 0.85400, 0.86840 |

**Mean paired gain +0.04945**, median +0.06504, improved in **11 of 15** folds.

All three pre-registered conditions passed, so the rule returns **KEEP**. But the rule is not the
whole story, and two things in this output need explaining rather than reporting.

### 9.1 The three significance tests disagree

| test | p (two-sided) | uses magnitude? |
|---|---|---|
| exact sign test | **0.11847** | no |
| Wilcoxon signed-rank | 0.03534 | ranks only |
| paired t | **0.02495** (t = 2.5106) | yes |

95 % CI on the paired gain: **[0.00721, 0.09170]**.

Per §67 this conflict is investigated rather than resolved in favour of the better number. The
cause is visible in the per-fold gains:

```
+0.040  +0.011  -0.122  +0.100  +0.127
+0.070  +0.065  +0.000  +0.112  -0.014
-0.008  +0.083  +0.081  +0.200  -0.003
```

Three of the four losing folds lose by **−0.003, −0.008 and −0.014** — differences of no practical
size at all — while the winning folds include +0.200, +0.127 and +0.112. The sign test scores a
−0.003 fold as a full loss, which is why it is the most conservative of the three and why it
fails to clear 0.05.

**The honest reading is that the effect is real but the evidence is moderate, not strong.** The
lower bound of the interval is +0.007, which is close enough to zero that a small true effect
cannot be excluded. Reporting only the paired t (p = 0.025) would overstate this; reporting only
the sign test (p = 0.118) would understate it. Both are recorded in the artifact.

### 9.2 One fold loses badly, and it is the one that used the most new columns

Fold r0 f2 falls from 0.89502 to 0.77264, a loss of 0.122 — larger than any single gain except
r2 f3. That fold also selected **20** missingness columns, the highest of any fold (range 11–20).
Across the 15 folds the correlation between "missingness columns selected" and "gain" is **−0.301**,
which with n = 15 is far from significant and is reported as an observation, not a result.

The mechanism it suggests is nonetheless the one the design anticipated: the feature budget is
fixed at 120, so every missingness column admitted **displaces** a base column. When the selector
admits many of them into a fold where they do not pay, the fold loses twice — it gains little and
gives up features that were working. This is the intended cost of a fixed budget and the reason
the budget was fixed.

Fold-level spread is large in both arms — WITHOUT ranges from 0.519 to 0.989 across folds — which
is what average precision on ~13 positives looks like. No fold-level number here should be read
as meaningful on its own.

### 9.3 Indicator stability

31 distinct missingness columns were selected at least once; 11–20 per fold.

**Survived every one of the 15 folds (5):**
`MX_FAMCNT__AADHAAR_PAYMENT_BRIDGE`, `MX_FAMCNT__LOAN`, `MX_FAMRATIO__LOAN`, `MX_NULL__F3240`,
`MX_NULL__F3348`.

**Survived at least 80 % of folds (11):** the five above plus
`MX_FAMCNT__ELEC_XFER`, `MX_FAMCNT__FEES_AND_CHARGES`, `MX_FAMCNT__NET_BANKING`,
`MX_FAMRATIO__AADHAAR_PAYMENT_BRIDGE`, `MX_FAMRATIO__FEES_AND_CHARGES`, `MX_NULL__F3395`.

This is the condition `updates1` cares most about and it passes convincingly — the requirement was
3 stable indicators and there are 11. The survivors are not scattered noise: they concentrate on
whole-block absence for specific payment rails (Aadhaar payment bridge, electronic transfer, net
banking, loan), which is a coherent story rather than an arbitrary set. It is consistent with mule
accounts being narrow-purpose accounts that never touch most of a normal customer's rails.

That coherence is suggestive, not confirmatory. It is exactly the kind of post-hoc narrative that
sounds convincing about any stable feature set, and it is recorded as an observation.

### 9.4 The result is provisional pending a suspect-family check

`MX_FAMCNT__FEES_AND_CHARGES` and `MX_FAMRATIO__FEES_AND_CHARGES` are both among the 80 %
survivors — and those are precisely the columns whose cohort fingerprint (section 6) could not be
traced to a cause. §67 requires that a model winning *only* because of a suspicious feature be
rejected.

So the KEEP above cannot stand on its own. A third arm was run:

```
.venv/Scripts/python.exe -m muleguard.cli.missingness_ablation \
  --family histgb --repeats 3 --exclude-contains FEES_AND_CHARGES \
  --out missingness_ablation_no_fees.json
```

If the gain largely survives without the FEES columns, the block is carrying broad missingness
structure and the suspect family is incidental to it. If the gain collapses, the win *was* the
suspect feature and the honest outcome is REJECT regardless of what the headline number says.
Section 10 records the outcome.

## 10. The confirmation arm: the gain survives the exclusion

`artifacts/metrics/missingness_ablation_no_fees.json`, run 2026-08-13 on the same 15 outer folds
with the same fixed hyperparameters, `--exclude-contains FEES_AND_CHARGES`.

| arm | PR-AUC (mean of 3 repeats) | per-repeat |
|---|---|---|
| WITHOUT | 0.79068 ± 0.01179 | 0.77699, 0.80577, 0.78928 |
| WITH, FEES excluded | **0.86330 ± 0.02307** | 0.83169, 0.87210, 0.88612 |

**Mean paired gain +0.0611**, median +0.07101, improved in **13 of 15** folds.

| test | p (two-sided) | verdict against §9.1 |
|---|---|---|
| exact sign test | **0.00739** | was 0.11847 — now significant |
| Wilcoxon signed-rank | **0.01025** | was 0.03534 |
| paired *t* (t = 3.1337) | **0.00733** | was 0.02495 |

The three tests disagreed in §9.1 and agree here, which is the opposite of the direction a
suspect-feature story predicts: removing the untraceable columns made the effect **larger and
more consistent**, not smaller. `MX_FAMCNT__FEES_AND_CHARGES` and `MX_FAMRATIO__FEES_AND_CHARGES`
were not carrying the result; with the budget fixed at 120, dropping them freed slots for
indicators that paid more.

That is the §67 condition satisfied, and it is worth being precise about what was and was not
shown: the win does not *depend* on a feature whose cohort fingerprint could not be traced. It is
not a demonstration that the fingerprint of section 6 is benign. Those columns remain unexplained;
they are simply not load-bearing.

## 11. What is not concluded, whatever the arms say

- The champion is **not** re-promoted on this evidence. The ablation used `histgb`, which places
  third in the finished nested tournament (catboost 0.80653, histgb 0.76735, xgboost 0.75393) —
  not the served champion's `xgboost`, and not the nested leader either. A confirmation arm on
  `--family xgboost` is required before any change to the served model is even considered, and
  promotion is a separate decision made by the nested tournament.
- Nothing here licenses a hand-written rule keyed to a missingness pattern.
- The Atlas result in section 7 stands: this block is not evidence that the 11 missed mules become
  catchable.

## 12. What this work does not license

- It does not license adding the leading indicator, or any indicator, on the strength of section 4.
  Those AUCs are in-sample maxima over 358 candidates.
- It does not license the claim that missingness recovers missed mules. Section 7 is evidence
  against that.
- It does not license treating the section 6 fingerprint as an explanation. The cause is unresolved
  and stated as unresolved.
- It does not license a hand-written rule keyed to `FEES_AND_CHARGES` absence. Any new feature must
  beat the existing model through nested cross-validation, which is what section 8 exists to do.
