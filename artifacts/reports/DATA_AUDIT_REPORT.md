# Data Audit Report

Generated 2026-07-10T18:56:17.931019+00:00 by `muleguard.cli.audit_data`.
Dataset fingerprint: raw SHA-256 `7d1be90fe23b5746…`, parquet SHA-256 `8ffeead97cfd0e87…`.
All numbers below are **measured from the raw file by this pipeline** (split: full dataset — audit only, no modelling decision selects features from these numbers).

## Shape and target

| Item | Value |
|---|---|
| Rows (accounts) | 9,082 |
| Columns (total incl. target) | 3,925 |
| Target column | `F3924` |
| Positives (1 = suspicious/mule) | 81 |
| Negatives (0) | 9,001 |
| Null targets | 0 |
| Positive prevalence | 0.8919% (~1 : 111) |
| Independent engine validation | PASSED |

## Data quality

| Item | Value |
|---|---|
| Missing cells (features) | 9,847,086 (27.63%) |
| Columns with any missing | 3,835 |
| Constant columns | 359 |
| Quasi-constant columns (≥99% one value or missing) | 1,078 |
| Exact duplicate columns | 977 |
| Dtypes | {'Float64': 3916, 'String': 6, 'Date': 1, "Datetime(time_unit='ms', time_zone=None)": 1} |
| Index-like columns detected | ['__UNNAMED__0'] |

## Non-numeric / interpretable columns

| feature | dtype | n_unique | missing_rate | top_values |
|---|---|---|---|---|
| F2230 | Datetime(time_unit='ms', time_zone=None) | 4 | 0.0 | datetime.datetime(2025, 10, 1, 0, 0):9001; datetime.datetime(2025, 9, 1, 0, 0):48; datetime.datetime(2025, 11, 1, 0, 0):23; datetime.datetime(2025, 12, 1, 0, 0):10 |
| F3886 | String | 17 | 0.0 | 'Savings':5956; 'Current':2051; 'MSME Micro':337; 'MSME Small':242; 'Staff Loans':108 |
| F3888 | Date | 4292 | 0.0 | None |
| F3889 | String | 7 | 0.0 | 'G365D':7544; 'L365D':397; 'L7D':386; 'L180D':313; 'L90D':207 |
| F3890 | String | 4 | 0.0 | 'M':2900; 'SU':2390; 'R':2015; 'U':1777 |
| F3891 | String | 7 | 0.0 | 'selfemployed':3951; 'salaried':1909; 'student':1185; 'agriculture':1112; 'housewife':660 |
| F3892 | String | 3 | 0.28606 | 'M':5007; 'F':1416; 'O':61 |
| F3893 | String | 2 | 0.0 | 'RETAIL':6437; 'CORPORATE':2645 |


## Strongest single-feature signals (top 15 by |corr|)

Single-feature PR-AUC is **cross-validated** (direction chosen on train folds, scored on held-out folds).

| feature | target_corr | single_feature_cv_pr_auc | mutual_information_bits | suspicious |
|---|---|---|---|---|
| F3912 | 0.9691 | 0.9433 | 0.0 | True |
| F2506 | 0.1845 | 0.0123 | 0.0004 | False |
| F2507 | 0.1845 | 0.0123 | 0.0004 | False |
| __UNNAMED__0 | 0.1628 | 1.0 | 0.0301 | True |
| F2408 | 0.1571 | 0.0113 | 0.0004 | False |
| F2409 | 0.1571 | 0.0113 | 0.0004 | False |
| F515 | 0.137 | 0.0089 | 0.0012 | False |
| F518 | 0.1269 | 0.0089 | 0.0012 | False |
| F2578 | 0.119 | 0.0095 | 0.0008 | False |
| F81 | 0.1169 | 0.0098 | 0.0003 | False |
| F82 | 0.1169 | 0.0098 | 0.0003 | False |
| F83 | 0.1164 | 0.0098 | 0.0003 | False |
| F84 | 0.1164 | 0.0098 | 0.0003 | False |
| F255 | 0.1131 | 0.0143 | 0.0021 | False |
| F2285 | 0.1126 | 0.009 | 0.0018 | False |


Full per-feature table: `artifacts/reports/data_profile.csv` and `artifacts/reports/leakage_audit_table.csv`.
