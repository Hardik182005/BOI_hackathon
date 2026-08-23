# Trinetra Mule-Farm Cohort Radar

> **Status:** shipped · post-model retrieval layer · zero effect on any score
> **Radar version:** 1.0 · transform `e88d91fa1e41d2e8…` · fingerprint hash `397784b11e13ad70…`
> **Code:** [`src/muleguard/usp/cohort_radar.py`](../src/muleguard/usp/cohort_radar.py) · [`cohort_eval.py`](../src/muleguard/usp/cohort_eval.py) · [`cohort_audit.py`](../src/muleguard/usp/cohort_audit.py)
> **Config:** [`configs/cohort_radar.yaml`](../configs/cohort_radar.yaml) · **Manifest:** [`artifacts/models/cohort_radar_manifest.json`](../artifacts/models/cohort_radar_manifest.json)
> **Spec:** sections 6–19 of the Cohort Radar / Zero-Regression USP brief

---

## 1. What it is, in one paragraph

The classifier tells an investigator *how unusual this account is*. It does not
tell them *whether this account is one of forty that all became unusual the same
way in the same fortnight* — and that is the question that turns a single alert
into a farm sweep. The Cohort Radar answers exactly that, and nothing else: given
one account, **which other accounts in the portfolio behave unusually similarly to
it?** It reads a score that has already been produced and never feeds anything
back.

**It is a retrieval layer, not a classifier.** No output of this module reaches
the model, the calibrator, a threshold, or a tier. There is no
`final_risk = 0.8 * model + 0.2 * cohort` anywhere in this repo, and section 6
below explains why that formula is the thing this design exists to avoid.

---

## 2. Three graphs, three different things

This project contains three graph-shaped objects and they are routinely confused,
so they are defined here once:

| | **ProofGraph** | **Cohort Radar** | **Transaction graph** |
|---|---|---|---|
| **Scope** | one account | many accounts | many accounts |
| **Edges mean** | "this evidence raised this decision" | "these two behave similarly" | "this account sent money to that one" |
| **Edge label** | `RAISED_BY`, `REQUIRES_HUMAN_VERIFICATION` | `BEHAVIORALLY_SIMILAR_TO` | `SENT_MONEY_TO` |
| **Built from** | SHAP attributions + patterns | a weighted feature fingerprint | sender/receiver ledger rows |
| **In this project?** | **Yes** | **Yes** | **No — the dataset has no counterparty edges** |

The third column is the one that matters for honesty. A real transaction graph
requires sender/receiver pairs. This dataset is account-level aggregates; it does
not contain them. **So MuleGuard does not draw one, and does not imply one.** The
Radar's edges are similarity edges and are labelled as such at every layer — in
the manifest, the API payload, the graph object and the UI.

---

## 3. What the Radar will never say (section 5)

`BEHAVIORALLY_SIMILAR_TO` is the only edge label. The following strings are
enumerated in `artifacts/models/cohort_radar_manifest.json` under `forbidden`
and asserted absent from every serialised payload by the test suite:

```
same criminal network · criminal network · mule handler · same handler
connected mule ring  · mule ring        · same syndicate · syndicate
controlled by        · controlled by same person        · same owner
sent money to        · transacted with  · money flow
same person          · same gang        · crime ring
```

The product language is **"Behaviourally similar accounts"**, and the panel
carries a mandatory disclaimer:

> Behavioural similarity is not proof of a shared owner, handler, or criminal
> network. It indicates accounts whose transaction patterns resemble one another
> and merits human review.

This is not squeamishness. Similarity between two accounts has many innocent
generators — the same salary cycle, the same merchant category, the same product,
the same month-end. Calling that a "ring" in an investigator's UI manufactures a
conclusion the data cannot support, and the guardrail is structural rather than
editorial: the forbidden strings cannot be reintroduced by a copy edit without
failing a test.

---

## 4. How an account is fingerprinted

### 4.1 Which features

The fingerprint reuses **the champion's own 120 selected features** — 119 numeric,
1 categorical (`F3886`, product name). Those columns already passed the leakage
firewall to become model inputs, so reusing them guarantees the Radar cannot see
anything the classifier could not.

They are nevertheless **re-checked through `firewall.assert_clean()` at build time
and at query time.** "It was admitted once" is a claim; the firewall is the check.
The Radar calls the same admission logic as the core pipeline — not a copy of it —
so a change to the quarantine policy propagates to both or fails loudly in
neither.

The target `F3924` is absent. So are all 13 live-quarantined columns, the
explicitly named `F3912`–`F3918`, `F3898`, `F3899`, `F2230`, and the single
`fairness_excluded` attribute. See §8.

### 4.2 How the weights are set

No weight is hand-chosen. The order is fixed by spec section 10:

| Source | Features | Mass |
|---|---:|---:|
| **Exact TreeSHAP global importance** (out-of-fold) | top 30 | 0.4714 |
| **Stability-selection gain share** (the tail) | remaining 90 | 0.5286 |
| Equal weights (fallback, unused) | — | — |

SHAP ranks only the top 30 of the 120, holding 47.14 % of total |SHAP| mass, so it
fixes those 30 weights *and the size of the remaining tail*; the tail is split
across the other 90 by their stability-selection gain share.

```
weight sum = 0.9999999999999999   min = 0.000458   max = 0.105505
weights hash = ada180a46d42974b…
```

### 4.3 The similarity function

A weighted Gower-style distance over mixed and partially-missing data:

```
S(x, y) = 1 − Σ_j  w_j · δ_j          clamped to [0, 1]

numeric      δ_j = min( |x_j − y_j| / (4 · IQR_j), 1 )
categorical  δ_j = 0 if the category matches, else 1
missing      both missing → 0 ; exactly one missing → 1
```

**There is no imputation.** Filling a missing value with a median would invent a
behaviour the account never exhibited and then match on it. "One side missing"
is treated as maximal dissimilarity on that feature, which is the honest reading:
we do not know, so we do not claim resemblance.

Numeric scale is `4 × IQR` computed on the **development partition only**. Where
an IQR is degenerate the scale falls back to the documented p1–p99 training range
rather than to an invented constant:

| Scale source | Features |
|---|---:|
| `IQR` | 112 |
| `TRAINING_RANGE_p1_p99` | 6 |
| `CONSTANT_UNIT_SCALE` | 1 |

---

## 5. Turning a similarity into a band

A raw similarity of `0.88` means nothing on its own. It is converted to a
percentile against an **empirical null distribution** of 199,971 uniformly random
development pairs (200,000 drawn, seed 42):

```
null mean 0.6407 · median 0.6562 · std 0.1337 · range [0.1829, 0.9759]
```

The bands are **percentiles of that null**, not hardcoded similarity values. The
similarity numbers the Radar actually compares against are the empirical values
at those percentiles, frozen into the manifest:

| Band | Percentile | Frozen similarity threshold |
|---|---:|---:|
| `VERY_HIGH_SIMILARITY` | 99.5 | 0.8945920529970107 |
| `HIGH_SIMILARITY` | 99.0 | 0.8825004289815930 |
| `MODERATE_SIMILARITY` | 95.0 | 0.8375502773886088 |
| `TYPICAL_SIMILARITY` | below 95 | — |

**The null was sampled without consulting a single label** (`label_used: false`),
and the bands were **never tuned on the locked test** (spec section 12).

### 5.1 The label-conditioned diagnostic, reported for honesty

A reasonable objection: *if you had built the null from negatives only, would the
bands have moved?* Rather than assert not, it was computed — as a **diagnostic
only**; the frozen bands remain the label-free ones.

| Band | Frozen (label-free) | If conditioned on negatives | Δ |
|---|---:|---:|---:|
| `VERY_HIGH` | 0.8945920 | 0.8930930 | 0.0014990 |
| `HIGH` | 0.8825004 | 0.8805716 | 0.0019289 |
| `MODERATE` | 0.8375503 | 0.8360226 | 0.0015277 |

Max band difference: **0.00193**. The label-free construction costs essentially
nothing, which is the outcome you want — it means the choice was made on principle
rather than on performance.

### 5.2 Data-availability floor

Neighbour coverage — the share of fingerprint features present on both sides —
is measured, not assumed. The empirical `min_reference_coverage` is **0.4866**
with a median of **0.8784**. A candidate below the floor is not returned: a
similarity computed on a handful of shared features is a coincidence, not a
resemblance.

---

## 6. Why cohort membership never touches the score

This is the central design decision, and the spec is blunt about it (section 3:
*hard model freeze*). The tempting version of this feature is a blended score:

```
final_risk = 0.8 · model_score + 0.2 · cohort_score        ← FORBIDDEN
```

That formula is forbidden here for three reasons:

1. **It destroys the calibration.** The champion's probabilities are Platt-
   calibrated with Brier 0.0031 and ECE 0.0015. A blend produces a number that is
   no longer a probability of anything, while still being displayed as a
   percentage.
2. **It is circular.** Neighbours are found by feature similarity; the features
   are the model's own inputs. Blending re-counts the same evidence and calls the
   second count corroboration.
3. **It escalates by association.** A legitimate high-volume merchant that
   resembles a suspicious account would gain risk *because of who it resembles* —
   which is the precise mechanism a fraud system must never have.

So the separation is enforced, not promised:

| Guarantee | Enforced by | Tolerance |
|---|---|---|
| Probabilities identical before/after the USP | `tests/regression/test_score_invariance_after_usp.py` | ≤ `1e-12` |
| Tiers identical, row by row | same | exact |
| Policy actions identical | same | exact |
| Cohort lookup leaves the scored payload bit-identical | `tests/regression/test_false_positive_safety.py` | exact |
| No `combined` / `final_risk` / `adjusted_risk` key exists | same | key absence |
| The experimental path stays off | `EXPERIMENTAL_COHORT_FEATURES: false` | config |

That last flag (section 27) is a switch for a code path **that does not exist**.
It is present so that anyone who later builds the forbidden thing has to turn it
on deliberately and explain themselves in a diff.

---

## 7. Retrieval quality (section 18)

### 7.1 What this measures — and what it does not

> **This is a retrieval-quality diagnostic, not the mule classifier accuracy.**

It answers: *when I look at a known mule account, do its behavioural neighbours
tend to be mule accounts too?* A high number means the fingerprint captures
something real about mule behaviour. It does **not** mean cohort similarity proves
common ownership, and it is not comparable to PR-AUC or Recall@TopK.

### 7.2 Protocol

Evaluation runs **outer-fold-safe**, on the model's own nested-CV fold map
(`artifacts/splits/nested_cv_assignments.parquet`, repeat 0, 5 outer folds). For
each fold the similarity transform is **refitted on that fold's training rows
only**; queries come from the held-out validation rows; retrieval is by similarity
alone, and **labels are read only after the neighbours are fixed**. Train/validation
overlap is asserted empty. Reference positive prevalence: **0.008811**.

### 7.3 Results — pooled, query-count weighted

**Positive queries** (n = 64) — known mule accounts:

| k | Hit@k | Mean positive neighbours | Neighbour positive prevalence | Lift over base prevalence |
|---:|---:|---:|---:|---:|
| 5 | **0.8906** | 3.67 / 5 | 0.7344 | **83.34 ×** |
| 10 | **0.9375** | 6.83 / 10 | 0.6828 | **77.47 ×** |
| 25 | 0.9375 | 14.44 / 25 | 0.5775 | 65.54 × |

**Legitimate queries** (n = 7,200) — the control:

| k | Hit@k | Mean positive neighbours | Neighbour positive prevalence | Lift |
|---:|---:|---:|---:|---:|
| 5 | 0.0554 | 0.098 / 5 | 0.0196 | 2.23 × |
| 10 | 0.0889 | 0.217 / 10 | 0.0217 | 2.46 × |
| 25 | 0.1639 | 0.603 / 25 | 0.0241 | 2.74 × |

**Reading:** for 15 in 16 mule accounts, at least one of the top 10 behavioural
neighbours is also a mule account, and on average **6.8 of those 10 are** — a
77× enrichment over the 0.88 % base rate. For a legitimate account the same
lookup returns 0.22 positives in 10. The gap between the two tables is the
evidence that the fingerprint is tracking mule-like behaviour rather than
retrieving arbitrary look-alikes.

Full per-fold breakdown: [`artifacts/metrics/cohort_radar_retrieval.json`](../artifacts/metrics/cohort_radar_retrieval.json).
Reproduce: `python -m muleguard.cli.cohort_evidence`.

---

## 8. Leakage audit (section 19) — 11/11 PASS

[`artifacts/testing/cohort_radar_leakage.json`](../artifacts/testing/cohort_radar_leakage.json) · **release-blocking**

| # | Check | Result |
|---|---|---|
| 1 | `F3924` is not one of the 120 fingerprint features | PASS |
| 2 | The live firewall quarantines 13 columns; 0 are in the fingerprint | PASS |
| 3 | The columns section 4 names explicitly are absent | PASS |
| 4 | None of the fairness-excluded attributes contributes to similarity | PASS |
| 5 | The quarantine hash recorded at fit time still matches the live one | PASS |
| 6 | Reference partition (7,264 rows) shares none of the 1,818 locked-test rows | PASS |
| 7 | The empirical null was built from unlabelled random pairs | PASS |
| 8 | Setting `F3924 = 1` on the query leaves the ranking unchanged | PASS |
| 9 | Setting 11 forbidden columns to an extreme value leaves the ranking unchanged | PASS |
| 10 | Label-shaped keys added to an uploaded row are ignored entirely | PASS |
| 11 | Weights and scaling statistics are byte-identical after an adversarial query | PASS |

Checks 8–11 are the important ones: they are **adversarial**, not structural. They
do not ask whether a forbidden column is in a list — they set it to an extreme
value and confirm the returned neighbours do not move.

## 9. Determinism audit — 3/3 PASS

[`artifacts/testing/cohort_radar_determinism.json`](../artifacts/testing/cohort_radar_determinism.json)

| # | Check | Result |
|---|---|---|
| 1 | 25 probe rows queried twice, in order, with identical results | PASS |
| 2 | Querying by row id and by re-uploading that row's own values agree | PASS |
| 3 | Shuffling the reference rows handed to `build_index` changes nothing | PASS |

Check 3 encodes an invariant worth stating: **the reference frame is a set, not a
sequence.** Ties break on reference position so the ordering is total, and a
re-ingest in a different order cannot silently reorder an investigator's cohort.

## 10. Fairness audit — PASS

Cohort neighbours agree on customer occupation **0.998 ×** as often as two random
accounts; area, age-decade and gender land between 1.09 × and 1.12 ×. Behavioural
features carry **94.04 %** of the similarity weight; the only profile feature in
the fingerprint is product name at **0.05 %**.

Full method and numbers: [`docs/COHORT_RADAR_FAIRNESS_AUDIT.md`](COHORT_RADAR_FAIRNESS_AUDIT.md).

---

## 11. The cohort graph

Edges are **mutual k-NN** (`mutual_knn_k: 10`): an edge exists only where each
account appears in the other's top-10. A one-directional resemblance is not a
cohort — a hub account that everything looks a bit like would otherwise generate
a fake ring on its own.

The graph is computed **lazily per query**, never precomputed portfolio-wide.
`default_k: 10`, `max_k: 25`.

There is no Neo4j, no vector database, no FAISS, no GNN, no embedding model. The
retrieval is a weighted distance over 120 columns computed with NumPy, which is
both sufficient at this portfolio size and inspectable — an investigator can be
shown *which features* drove a match (`main_shared_features`), which no learned
embedding would allow.

---

## 12. Performance (section 33)

Measured, not estimated: [`artifacts/metrics/cohort_radar_performance.json`](../artifacts/metrics/cohort_radar_performance.json).
200 seeded queries at k = 10 over 7,264 reference accounts, the two arms
interleaved per query so any thermal or cache drift hits both equally.

| Measure | Value |
|---|---:|
| Index build (cold, full refit of the reference block) | 11.44 s |
| Index memory | 6.94 MB |
| **Single query p50** | **174.94 ms** |
| Single query p95 | 224.93 ms |
| Single query p99 | 248.26 ms |
| Batch of 50, per query | 178.18 ms |
| Explanation overhead (p50) | **2.38 ms** |

Two things worth reading off that table. The index is **6.94 MB and builds in
eleven seconds** — at this portfolio size an exact comparison against every
reference account is simply the right answer, and a vector database would add an
approximation, a dependency and a service without buying anything. And the
**explanation costs 2.38 ms**: showing an investigator *which features* drove a
match is essentially free here, which it would not be with a learned embedding.

The first measurement of this was wrong and is worth recording. Timing
similarity-only *after* the explained arm made it look ~19 ms slower than the
arm that does strictly more work. That is an ordering artifact, not a finding;
interleaving the arms collapsed the gap to 2.38 ms in the direction physics
requires. The published numbers are the interleaved ones.

---

## 13. What appears in the UI

A panel of 5–10 behaviourally similar accounts, each with:

- account reference, similarity value and band
- **`main_shared_features`** — the features that actually drove the match
- the neighbour's own risk and tier, **unchanged**, straight from the index
- the mandatory disclaimer beneath the panel

Rendered in the existing white/black Trinetra visual system. **No 3D graph. No
directional arrows** — arrows imply money movement, and there is no money-movement
data. **No automatic action** is offered from this panel: cohort membership never
escalates a neighbour, and `automatic_actions_permitted` is `[]` regardless of
what the cohort looks like.

---

## 14. Related

- [`docs/COHORT_RADAR_FAIRNESS_AUDIT.md`](COHORT_RADAR_FAIRNESS_AUDIT.md) — cohorts group by behaviour, not by people
- [`docs/ACCOUNT_CONTROL_AMBIGUITY.md`](ACCOUNT_CONTROL_AMBIGUITY.md) — why similarity is never converted into an accusation
- [`docs/USP_ACCURACY_REGRESSION.md`](USP_ACCURACY_REGRESSION.md) — proof the classifier is bit-identical before and after
- [`docs/FEATURE_AVAILABILITY_FIREWALL.md`](FEATURE_AVAILABILITY_FIREWALL.md) — the admission policy the Radar shares with the classifier
- [`docs/PROOFGRAPH_DESIGN.md`](PROOFGRAPH_DESIGN.md) — the single-account evidence graph
