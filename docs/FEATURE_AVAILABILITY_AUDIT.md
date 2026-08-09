# Feature Availability Audit

Master prompt §6. Implementation: `src/muleguard/features/firewall.py`, policy
`configs/feature_availability.yaml` (**policy version 2.0**), evidence
`artifacts/features/quarantined_features.json`.

---

## 1. The one question

The firewall asks a single question of every column, and refuses everything that
cannot answer it:

> **Was this value knowable at the moment the analyst had to decide?**

Not "does it correlate with the label" — that test admits the label itself. A
column recording how long a fraud review took is perfectly correlated and
perfectly useless, because it does not exist until after the decision the model
is supposed to make.

`admitted_features()` is deliberately the **narrow** path: it starts from the
full column list and only ever removes.

```python
ADMISSIBLE_CLASSES = (BEHAVIORAL, PROFILE, ALERT_CONTEXT)
FORBIDDEN_CLASSES  = (POST_RESOLUTION_LEAKAGE, TARGET, INDEX_OR_ID)
```

Two guarantees follow, both asserted by the release gate:

1. No accepted artefact contains a `POST_RESOLUTION_LEAKAGE`, `TARGET` or
   `INDEX_OR_ID` column.
2. Every admitted column carries a recorded availability class, so the model
   manifest can state exactly which classes of evidence produced a score.

---

## 2. The census

3,924 described columns, classified from `Description.xlsx`
(sha-256 `7d30652b72d4b79b…`):

| Availability class | Columns | Admissible? |
|---|---:|---|
| `BEHAVIORAL` | 3,884 | ✅ transaction aggregates over L7D/L14D/L31D windows |
| `ALERT_CONTEXT` | 20 | ✅ pre-decision alert evidence |
| `PROFILE` | 9 | ✅ except `F3892` (fairness policy, §5) |
| `POST_RESOLUTION_LEAKAGE` | 6 | ❌ hard quarantine |
| `PRE_EXISTING_RISK_CONTEXT` | 3 | ⚠️ conditional quarantine |
| `INDEX_OR_ID` | 1 | ❌ hard quarantine |
| `TARGET` | 1 | ❌ hard quarantine |

Leakage disposition: **3,913 SAFE**, **8 QUARANTINED**, **3 REVIEW**.

---

## 3. The quarantine, column by column

Thirteen columns are excluded. All thirteen are present in the dataset — none of
these entries is hypothetical.

| Column | Variable | Class | Why it is excluded | Disposition |
|---|---|---|---|---|
| `F3924` | `FRAUD_TGT` | TARGET | the outcome being predicted | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3912` | `FRAUD_SUSPECTED` | POST_RESOLUTION_LEAKAGE | resolution status flag — the outcome being predicted | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3913` | `OTHER_RESOLUTION` | POST_RESOLUTION_LEAKAGE | post-decision outcome | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3914` | `FALSE_POSITIVE` | POST_RESOLUTION_LEAKAGE | post-decision outcome | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3915` | `UNATTENDED` | POST_RESOLUTION_LEAKAGE | post-decision outcome | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3898` | `MIN_RESOLVE_DAYS` | POST_RESOLUTION_LEAKAGE | measures the duration of the review being predicted | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3899` | `MAX_RESOLVE_DAYS` | POST_RESOLUTION_LEAKAGE | measures the duration of the review being predicted | `EXCLUDED_FROM_ALL_TRAINING` |
| `F2230` | `MNTH` | INDEX_OR_ID | snapshot month — **deterministically reconstructs the label** in this extract | `EXCLUDED_FROM_ALL_TRAINING` |
| `__UNNAMED__0` | — | UNKNOWN_REVIEW | unnamed leading row-index column | `EXCLUDED_FROM_ALL_TRAINING` |
| `F3916` | `L3_FLG` | PRE_EXISTING_RISK_CONTEXT | customer risk level, time-of-availability undetermined | `QUARANTINE_UNTIL_PROVEN_PRE_DECISION` |
| `F3917` | `L2_FLG` | PRE_EXISTING_RISK_CONTEXT | customer risk level, time-of-availability undetermined | `QUARANTINE_UNTIL_PROVEN_PRE_DECISION` |
| `F3918` | `L1_FLG` | PRE_EXISTING_RISK_CONTEXT | customer risk level, time-of-availability undetermined | `QUARANTINE_UNTIL_PROVEN_PRE_DECISION` |
| `F3892` | `GENDER` | PROFILE | protected attribute with no defensible causal link to mule behaviour | `EXCLUDED_BY_FAIRNESS_POLICY` |

### 3.1 The three that are hardest to give up

**`F3912 FRAUD_SUSPECTED`** is worth 0.33 PR-AUC on its own. A LightGBM allowed
to see it scores **0.9419 ± 0.0119**; the same configuration under the firewall
scores 0.6142 ± 0.0474 (`artifacts/metrics/with_vs_without_f3912.json`). That
gap is the entire argument for the firewall's existence, and the ablation is kept
on record labelled **REJECTED LEAKAGE — evidence only**. A submission built on it
would top a local leaderboard and collapse on a hidden set that does not carry a
resolution flag.

**`F2230 MNTH`** is subtler and more dangerous, because it does not look like a
label. It is a snapshot month. But in *this extract* all 9,001 negatives sit in
2025-10 while all 81 positives sit in 2025-09/11/12 — balanced label
reconstruction from the month alone is **1.0**. It is an artefact of how the
sample was collected, and any model given it learns the collection process.

**`F3916/F3917/F3918`** (L1/L2/L3 risk flags) are the honest hard case. They
might be pre-existing customer risk ratings, in which case they are legitimate
pre-decision evidence and among the most useful columns in the file. Or they
might be set by the same review that produced the label. The description does not
say, and we could not establish it from the data. Default per §6:
`QUARANTINE_UNTIL_PROVEN_PRE_DECISION` — **the burden of proof is on admission,
not on exclusion.** They enter only an explicitly labelled ablation
(`include_conditional=True`), never the accepted model.

---

## 4. Leakage-by-complement

A column can leak without being a resolution flag, by being the complement of one.
If `FRAUD_SUSPECTED + OTHER_RESOLUTION + FALSE_POSITIVE + UNATTENDED` partition
the reviewed population, then any three of them reconstruct the fourth, and a
model that has three has effectively been handed the label.

The audit therefore checks reconstruction rather than names: for each candidate,
can the target be recovered at above-chance balanced accuracy from that column
and its neighbours alone? The columns above are quarantined **as a set**, so
removing four and keeping two is not an available shortcut.

Full derivation: `docs/LEAKAGE_AUDIT.md`, `docs/FINAL_DATA_AND_LEAKAGE_AUDIT.md`.

---

## 5. Fairness exclusion is separate from leakage exclusion

`F3892 GENDER` is not quarantined for leakage. It is excluded by policy: a
protected attribute with no defensible causal link to mule behaviour, whose only
possible contribution to a mule model is proxy discrimination.

The firewall keeps this distinct — `fairness_excluded` is its own config
section with its own flag (`include_gender`, default `False`, requiring a
documented justification to flip). Conflating the two would mean a future
leakage review could silently readmit a protected attribute.

Four columns are marked sensitive in the registry: `F3890`, `F3891`, `F3892`,
`F3894`. Their disposition is in `docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`.

---

## 6. Views: the same firewall, five admissible lenses

`configs/feature_availability.yaml` defines five views, each a *further*
restriction of the admitted pool — never a widening:

| View | Classes | Purpose |
|---|---|---|
| `A_broad_behavioral` | BEHAVIORAL | all cleaned behavioural aggregates, data-driven selection |
| `B_stable_compact` | BEHAVIORAL, PROFILE | top-K by cross-fold stability selection |
| `C_bank_prior` | BEHAVIORAL, PROFILE, restricted to the bank's 18 finalized variables | does the bank's own shortlist hold up? |
| `D_alert_context` | ALERT_CONTEXT, PROFILE | pre-decision alert evidence only |
| `E_profile_merchant` | PROFILE + POS_MERCHANT/GST/BALANCE/LOAN/STANDING_INSTRUCTION/FEES_AND_CHARGES/TOTAL_ALL_RAILS | merchant / business legitimacy context — feeds the Merchant Legitimacy Verifier (UPDATE 9) |

The champion is trained on `ALL_ADMISSIBLE`. Views exist so that a claim like
"this model works because of alert context" can be tested by removing everything
else, rather than argued.

---

## 7. Verification at inference, not just at training

A firewall argued from the training configuration is a promise. This one is
measured from the output.

In the §42 organiser rehearsal, the same 1,818-row file was uploaded with
`F3912` present, and again with all five resolution fields present. Both
produced **byte-identical prediction files** — SHA-256
`cc9f65a1945b407b…` — against the clean baseline.

The control that makes those results meaningful: a variant that injected an
unseen category into a feature the champion genuinely reads produced a
**different** hash. Without it, identical hashes would be equally consistent with
a harness that never reached the scorer.

Full record: `docs/ORGANISER_DRY_RUN.md` §3.

---

## 8. Enforcement summary

| Where | Check |
|---|---|
| `admitted_features()` | removes forbidden classes before any candidate list is built |
| training | selection, calibration and thresholding operate on admitted columns only |
| release gate | fails if any accepted artefact contains a forbidden-class column |
| release gate | fails if the §42 rehearsal's quarantine invariances stop holding |
| inference | uploads carrying quarantined columns are accepted and **ignored**, provably |
| `artifacts/models/model_manifest.json` | records `leakage_status: FIREWALL_ADMITTED_ONLY` |

The cost of all this is on record and is not small: the champion scores
**0.7690** where the leaked variant scores **0.9419**. We publish both, and serve
the first.
