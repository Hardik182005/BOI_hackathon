# Feature Dictionary Report

Master prompt §5. The semantic registry built from `Description.xlsx`, which
turns 3,924 opaque `F####` codes into named, classified, explainable evidence.

Builder: `src/muleguard/features/dictionary.py` ·
Artefacts: `artifacts/features/feature_dictionary.json`, `feature_dictionary.csv` ·
Source: `Description.xlsx`, sha-256 `7d30652b72d4b79b3feeb3e14ebe67177988b3155912ea33f284f140e8774b20`

---

## 1. Why this exists

The modelling file ships as `F1 … F3924`. Without a registry, three things are
impossible:

1. **Leakage cannot be found by name.** `F3898` is indistinguishable from
   `F3897` until you know it is `MIN_RESOLVE_DAYS` — the duration of the review
   being predicted.
2. **An alert cannot be explained.** A ProofGraph node reading *"F3642 is high"*
   is not evidence a reviewer can act on. *"Deviation of average UPI credit
   amount over the last 7 days"* is.
3. **Domain meta-features cannot be constructed.** Building a pass-through ratio
   requires knowing which columns are credits, which are debits, and over which
   window.

The registry is parsed once, hashed, and every downstream component reads it
rather than re-deriving semantics. Nothing in the pipeline guesses what a column
means.

---

## 2. What is recorded per column

```json
"F1": {
  "feature": "F1",
  "variable_name": "R_CASH_TXN_L7_14D",
  "description": "Ratio of Cash Total txns - last 7 to 14D",
  "bank_finalized": false,
  "feature_family": "CASH",
  "transform_family": "RATIO_WINDOW",
  "window": "L7_14D",
  "direction": "BOTH",
  "availability_class": "BEHAVIORAL",
  "sensitive": false,
  "sensitive_kind": null,
  "leakage_status": "SAFE",
  "semantic_tags": ["CASH", "RATIO_WINDOW", "SHORT_WINDOW",
                    "TRANSACTION_COUNT", "WINDOW_L7_14D"]
}
```

Five of these fields are load-bearing:

- **`availability_class`** feeds the Feature Availability Firewall (§6). Nothing
  reaches a model without one.
- **`leakage_status`** records the audit disposition: `SAFE` / `QUARANTINED` /
  `REVIEW`.
- **`feature_family`** and **`window`** drive family-dropout robustness testing
  and the meta-feature constructors.
- **`sensitive`** flags protected attributes for the fairness policy.
- **`description`** is what a reviewer actually reads in a ProofGraph node.

---

## 3. The census

**3,924 described columns.**

### Availability

| Class | Columns |
|---|---:|
| `BEHAVIORAL` | 3,884 |
| `ALERT_CONTEXT` | 20 |
| `PROFILE` | 9 |
| `POST_RESOLUTION_LEAKAGE` | 6 |
| `PRE_EXISTING_RISK_CONTEXT` | 3 |
| `INDEX_OR_ID` | 1 |
| `TARGET` | 1 |

Leakage disposition: **3,913 SAFE**, **8 QUARANTINED**, **3 REVIEW**.

### Feature families (25 in total, top 20)

| Family | Columns | | Family | Columns |
|---|---:|---|---|---:|
| `NON_CASH_CHQ` | 432 | | `CHEQUE` | 215 |
| `FEES_AND_CHARGES` | 430 | | `ATM` | 208 |
| `CASH` | 216 | | `POS_MERCHANT` | 208 |
| `CASH_INTENSIVE` | 216 | | `AADHAAR_PAYMENT_BRIDGE` | 205 |
| `GST` | 216 | | `BBPS` | 152 |
| `ELEC_XFER` | 216 | | `UPI` | 148 |
| `NET_BANKING` | 216 | | `UPI_XFER` | 72 |
| `MOBILE_BANKING` | 216 | | `TOTAL_ALL_RAILS` | 45 |
| `LOAN` | 216 | | `BALANCE` | 45 |
| `STANDING_INSTRUCTION` | 216 | | `ALERT_CONTEXT` | 16 |

This is the file's real structure: **the same ~216 measurements repeated across
each payment rail.** That single fact shapes everything downstream — it is why
family-dropout testing is meaningful (remove an entire rail and see what
happens), and why raw correlation-based selection produces near-duplicate
feature sets.

### Observation windows

| Window | Columns |
|---|---:|
| `NONE` (static / profile / totals) | 720 |
| `L7D` | 654 |
| `L14D` | 652 |
| `L31D` | 651 |
| `L7_31D` | 424 |
| `L14_31D` | 415 |
| `L7_14D` | 408 |

The `Lx_yD` forms are **ratios between windows** — recent activity measured
against its own recent baseline. They turn out to matter: 32 of the champion's
120 features are window-ratio or deviation-of-average transforms.

### Transform families

| Transform | Columns | | Transform | Columns |
|---|---:|---|---|---:|
| `DEVIATION_OF_AVERAGE` | 631 | | `RATIO_WINDOW` | 251 |
| `DEVIATION` | 627 | | `RATIO_OF_AVERAGES` | 250 |
| `MAXIMUM` | 335 | | `RAW_OR_META` | 233 |
| `MINIMUM` | 332 | | `COUNT` | 184 |
| `AVERAGE` | 330 | | `RATIO_CREDIT_INTENSITY` | 108 |
| `RANGE_MAX_MINUS_MIN` | 324 | | | |
| `DEV_TOTAL_VS_AVG` | 319 | | | |

### Marked subsets

- **`bank_finalized`** — 18 columns the bank itself shortlisted: `F115`, `F321`,
  `F527`, `F531`, `F670`, `F1692`, `F2082`, `F2122`, `F2582`, `F2678`, `F2737`,
  `F2956`, `F3043`, `F3836`, `F3887`, `F3889`, `F3891`, `F3894`. These form
  firewall view `C_bank_prior`, so "does the bank's own shortlist hold up?" is a
  question with a measured answer rather than an opinion.
- **`sensitive`** — 4 columns: `F3890`, `F3891`, `F3892`, `F3894`. Disposition in
  `FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`.

---

## 4. What the champion actually selected

`xgboost_top_120` — the 120 features, resolved through the registry:

| Family | Selected | | Window | Selected |
|---|---:|---|---|---:|
| `NON_CASH_CHQ` | 46 | | `L31D` | 29 |
| `UPI` | 16 | | `L14_31D` | 26 |
| `TOTAL_ALL_RAILS` | 11 | | `NONE` | 25 |
| `UPI_XFER` | 10 | | `L7D` | 16 |
| `LOAN` | 8 | | `L14D` | 12 |
| `GST` | 7 | | `L7_31D` | 6 |
| `CASH` | 4 | | `L7_14D` | 6 |
| `NET_BANKING` | 3 | | | |
| `MOBILE_BANKING` | 3 | | | |
| **`MG_DERIVED`** | **3** | | | |
| `ELEC_XFER`, `ATM`, `BALANCE` | 2 each | | | |
| `FEES_AND_CHARGES`, `PROFILE`, `ALERT_CONTEXT` | 1 each | | | |

Transform mix: `DEVIATION_OF_AVERAGE` 32, `RAW_OR_META` 17, `COUNT` 12,
`MAXIMUM` 10, `RATIO_WINDOW` 9, `RATIO_OF_AVERAGES` 9, `DEV_TOTAL_VS_AVG` 9,
`AVERAGE` 8, `RANGE_MAX_MINUS_MIN` 6, `DEVIATION` 4, `MG` 3,
`RATIO_CREDIT_INTENSITY` 1.

This reads plausibly for mule detection and was not designed to: **electronic
rails dominate** (`NON_CASH_CHQ` 46, `UPI` + `UPI_XFER` 26), cash is nearly
absent (4), and more than half the features are **deviation or ratio transforms**
rather than raw levels — the model keys on *change against an account's own
baseline*, not on absolute size. A large merchant and a mule account can both
move a lot of money; only one of them started doing it last week.

Examples, with the registry doing the translating:

| Column | Variable | Description |
|---|---|---|
| `F114` | `R_NON_CASH_CHQ_AMT_DB_L14_31D` | Ratio of non-cash non-cheque debit amount, last 14 to 31 days |
| `F120` | `R_CI_NON_CASH_CHQ_AMT_DB_L14_31D` | Ratio of customer-induced non-cash non-cheque debit amount, last 14 to 31 days |
| `F148` | `R_GST_AMT_L14_31D` | Ratio of GST total amount, last 14 to 31 days |
| `F158` | `R_UPI_TXN_CR_L14_31D` | Ratio of UPI credit transactions, last 14 to 31 days |
| `F193` | `R_LOAN_TXN_L14_31D` | Ratio of loan transactions, last 14 to 31 days |

---

## 5. The three derived features

Three of the champion's 120 features do not exist in any file:
`MG_PASSTHROUGH_7D`, `MG_RAIL_FRAGMENTATION`, `MG_ALERT_CONVERGENCE`
(§11, `src/muleguard/features/meta_features.py`).

They are **row-wise functions of raw columns**, computed from the uploaded row at
scoring time. They are never uploaded and never expected in an organiser's file —
which is why the §42 rehearsal reports 117 features read from the file plus 3
derived, and still records `schema_completeness: 1.0`.

Because they are absent from the persisted registry, `_registry_lookup` in the
ProofGraph resolves them from `META_DESCRIPTIONS` instead. Without that, the
three most interpretable features in the model would be the only ones rendering
without an explanation.

---

## 6. Integrity

| Property | How |
|---|---|
| Source immutability | `Description.xlsx` is read-only; sha-256 recorded in the registry and re-checked on load |
| No inferred semantics | every field is parsed from the workbook or derived by documented rule; nothing is guessed |
| Availability completeness | every one of the 3,924 columns carries an `availability_class`; the firewall refuses an unclassified column |
| Round-trip | `feature_dictionary.csv` and `.json` are generated from the same parse |
| Tests | `tests/unit/test_dictionary.py` — parse, classification rules, `MG_*` resolution, sensitive-column marking |

Downstream consumers: the Feature Availability Firewall (§6), meta-feature
construction (§11), pattern cards (§12), ProofGraph node labels (§17), the
Validation Lab compatibility score (§20), and the fairness audit (§24).
