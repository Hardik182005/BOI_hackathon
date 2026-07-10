# Data Dictionary

Everything below is **verified from the raw file** (`DataSet.xlsx`, SHA-256 in
`data/interim/data_fingerprint.json`). Anonymised features are documented
statistically only — no invented business meaning anywhere.

## File structure

| Item | Verified value |
|---|---|
| Rows (accounts) | 9,082 |
| Columns | 3,925 = 1 unnamed index + features F1–F3923 + target F3924 |
| Missing marker | literal string `"NA"` (canonicalised to null) |
| Row order | **File is physically sorted by label** — all 81 positives are rows 9001–9081 (0-based). Row order must never be used as a feature; the index column is quarantined. |

## Target

| Column | Meaning |
|---|---|
| `F3924` | Binary target per Topic.pdf. `1` = suspicious/mule (81 rows), `0` = legitimate/not currently labelled (9,001 rows). Prevalence 0.8919% (~1:111). No nulls, no out-of-range values. |

## Quarantined columns (never features)

| Column | Evidence |
|---|---|
| `F3924` | the target itself |
| `F3912` | two-valued column; \|corr\| = 0.969 with target, single-feature **cross-validated** PR-AUC = 0.943, balanced label reconstruction = 0.988 → post-outcome/leak field |
| `F2230` | datetime snapshot month with 4 values. **All 9,001 negatives are 2025-10; all 81 positives are 2025-09/11/12.** The month reconstructs the label perfectly (balanced reconstruction 1.0). Dataset-construction artifact; also invalidates any out-of-time split (time ≡ label). |
| `__UNNAMED__0` | row index 1..9082; file sorted by label (single-feature CV PR-AUC 1.0) |

## Interpretable columns (meaning verified from raw values)

| Column | Type | Values / range | Missing |
|---|---|---|---|
| `F3886` | categorical (17 levels) | account/product type: Savings/Current variants, `Gold Loan`, `PL`, `ML`, `Agri Adv`, `MSME Small`, `Staff Loans`, `Term Deposit`, `All Others`, … | 0% |
| `F3888` | date (mixed text formats, parsed with 0 failures) | account opening date, 1900-01-25 → 2025-12-11 | 0% |
| `F3889` | categorical (7) | tenure/vintage bucket: `L7D`, `L14D`, `L31D`, `L90D`, `L180D`, `L365D`, `G365D` | 0% |
| `F3890` | categorical (4) | region/locality: `R`, `SU`, `M`, `U` (rural/semi-urban/metro/urban — indicative) | 0% |
| `F3891` | categorical (7) | occupation: salaried, selfemployed, student, agriculture, housewife, retired, others | 0% |
| `F3892` | categorical (3) | gender: `M`, `F`, `O` | ~28.6% `"NA"` |
| `F3893` | categorical (2) | customer segment: `RETAIL`, `CORPORATE` | 0% |

## Anonymised numeric features (all remaining columns)

| Property | Measured value |
|---|---|
| Numeric after canonicalisation | 3,915 columns (128 of them arrived as text-with-"NA" and were coerced with 0 parse failures) |
| Constant columns | 359 |
| Quasi-constant (≥99% one value or missing) | 1,078 |
| Exact duplicate columns | 977 (content-hash groups; one representative kept **per training fold**) |
| Cells missing | ~27.6% of feature cells; 97%+ of columns have some missing |
| Bank-hint features (Topic.pdf) | all 18 present: F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894 |

Full per-column statistics: `artifacts/reports/data_profile.csv` (3,925 rows) and
per-feature leakage statistics: `artifacts/reports/leakage_audit_table.csv`.

## Explanation language rule

For anonymised features the only permitted statement form is comparative:
> "F1702 is at the 99.2nd percentile of the legitimate cohort and increases the model score."

Forbidden: assigning semantics ("F1702 is fan-in") or intent ("proves laundering").
