# Current Repository Audit — pre-upgrade state

**Captured:** 2026-08-12, before any Trinetra upgrade code was executed.
**Machine-readable companion:** `artifacts/audit/before_upgrade.json`

This document records what existed in the MuleGuard repository before the
Trinetra upgrade, and classifies each part as **KEEP**, **FIX**, **EXTEND**,
**REMOVE** or **NOT_IMPLEMENTED**. The point of writing it down first is that
the upgrade is an *in-place* upgrade: anything not listed here as FIX or
REMOVE is expected to survive unchanged, and a reviewer can check that claim.

---

## 1. Inventory

| Area | Count | Detail |
|---|---:|---|
| Python modules (`src/`) | 62 | 9,347 lines |
| Test files | 11 | 1,054 lines, 93 passed / 3 skipped |
| Frontend components (`.tsx`) | 11 | 7 pages + App + main |
| Config files (`configs/`) | 7 | base, data, train, thresholds, leakage_quarantine, ollama, feature_availability* |
| Docs | 35 | `docs/*.md` |
| CLI entry points | 15 | `muleguard.cli.*` |

\* `feature_availability.yaml` is new in this upgrade; it is listed because the
inventory was captured after the firewall config landed.

### Package layout

```
src/muleguard/
  settings.py          path + config resolution, TARGET_COLUMN, GLOBAL_SEED
  utils.py             hashing, atomic JSON, seeding, git info, timers
  logging.py           structured logger
  data/                ingest (XLSX -> parquet, independent-engine validation)
                       split  (group-aware locked test + repeated stratified CV)
  features/            preprocessing (fold-safe encode/impute/dedupe)
  models/              baselines, core_models, calibration, conformal,
                       anomaly, hard_negative, scoring
  evaluation/          metrics (PR-AUC first, no accuracy), plots
  explain/             reason_codes, evidence_packet
  action/              policy (tiering, human approval)
  monitoring/          drift (PSI baseline)
  llm/                 ollama_client, prompts, schemas, validator,
                       deterministic_fallback
  api/                 FastAPI app, upload routes, SQLite persistence
  cli/                 audit_data, audit_env, make_splits, train, tournament,
                       advanced, build_lenses, evaluate, release_gate, demo,
                       export_submission, final_report, qa_harness, ...
```

---

## 2. Classification

### KEEP — correct, defensible, reused as-is

| Component | Why it survives |
|---|---|
| `data/ingest.py` | The raw workbook is copied read-only, hashed, converted once, and the conversion is re-validated with an **independent engine** (openpyxl streaming) on row count, column count, header equality, target distribution and random spot cells. This is exactly the evidence a judge should ask for. |
| `data/split.py` | Locked test and repeated CV folds are **group-aware**: rows with identical feature vectors are hashed into groups and never split across the dev/test boundary or across folds. Prevents twin leakage, which is a real risk at 9,082 rows. |
| `features/preprocessing.py` | `FoldPreprocessor` learns constants, duplicates, medians and scaling **inside the training fold only**. Tree models receive NaN untouched. |
| `evaluation/metrics.py` | PR-AUC primary, stratified bootstrap CIs that resample positives and negatives separately, Recall/Precision@budget, recall@FPR, ECE, Brier. No plain accuracy anywhere. |
| `models/calibration.py`, `conformal.py`, `anomaly.py` | Platt/isotonic selection, Mondrian conformal, isolation forest. |
| `llm/` | Ollama is schema-validated and confined to narration; a deterministic fallback exists. |
| `api/main.py` | Health/ready split, rate limiting, batch cap, `RECOMMEND_FREEZE` requires `approved_by`. |
| `action/policy.py` | No automatic freezing; human approval gate present. |

### FIX — present but wrong, must change

| Component | Defect | Action taken |
|---|---|---|
| `configs/leakage_quarantine.yaml` | Quarantined only `F3924`, `F3912`, `F2230`, `__UNNAMED__0`. Left `F3898`, `F3899`, `F3913`, `F3914`, `F3915` (resolution outcome + resolution duration) available to the model. | Superseded by `configs/feature_availability.yaml` (policy v2.0) with 9 hard-quarantine and 3 conditional-quarantine entries plus a fairness exclusion. |
| **Accepted model `catboost_tuned_top60`** | Its top-60 feature set contains `F3898` (MIN_RESOLVE_DAYS), `F3913` (OTHER_RESOLUTION), `F3914` (FALSE_POSITIVE) and `F3916` (L3_FLG). Reported OOF PR-AUC **0.8077** is therefore not an estimate of hidden-validation performance. | **Done.** Full leakage-free retrain completed. `catboost_tuned_top60` is `status: "retired"` in the registry and 0.8077 is retired with it; the promoted model is **`xgboost_top_120`, OOF PR-AUC 0.7690 ± 0.0266**, 120 firewall-admitted features, 0 quarantined columns. See `docs/UPGRADE_GAP_ANALYSIS.md` §3 and `docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md`. |
| `cli/tournament.py::_dev_matrices` | Builds the candidate pool from `load_quarantine_list()` + `candidate_feature_columns()`, i.e. the four-entry list. Any rerun reproduces the leak. | Superseded by `cli/tournament_v2.py`, which sources every matrix from `features/frame.build_model_frame()` and therefore cannot bypass the firewall. |
| `models/baselines.py::run_oof` | Same data path. | Superseded by `models/harness.py::run_oof`, which takes a `ModelFrame`. `baselines.py`'s *scorers* are kept and reused. |
| `frontend/src/App.tsx` | Tagline contains a mojibake `Â·` from a bad encoding round-trip. | Corrected during the frontend pass. |

### EXTEND — right idea, insufficient coverage

| Component | Gap |
|---|---|
| `explain/evidence_packet.py` | Produces one-sided evidence (why flagged). The upgrade requires **dual** evidence — why it might be a false positive — plus a counterfactual twin. |
| `models/hard_negative.py` | A verifier exists but does not yet feed an auditable *escalation-confidence* decision (addendum UPDATE 9). |
| `monitoring/drift.py` | PSI baseline exists; no adversarial train-vs-upload shift classifier (addendum UPDATE 3). |
| `evaluation/metrics.py` | No MCC; no Recall@TopK-driven promotion score. |
| `api/routes_upload.py` | Accepts an upload and scores it; no sealed-prediction protocol, no schema-integrity step, no compatibility score. |
| `configs/thresholds.yaml` | Tiering exists; abstention band and escalation-confidence semantics need to be explicit. |

### NOT_IMPLEMENTED — required by the upgrade, absent before it

- Description.xlsx semantic registry (all 3,924 features carried no meaning).
- Feature Availability Firewall and its enforcement gate.
- Interpretable domain meta-features.
- Dual-Evidence ProofGraph and Counterfactual Twin.
- Model Courtroom panel.
- Validation Lab page, Validation Compatibility Score, Sealed Validation Protocol.
- Hidden Validation Shield (adversarial validation).
- Mule Stability Stress Test (positive-removal resampling).
- Label-noise audit.
- Transaction Graph Adapter (edge-file-only activation).
- Business Value simulator page.
- Out-of-time stress track (Track C).
- Competitor gap matrix.

### REMOVE — nothing

No component is deleted. The two superseded modules (`cli/tournament.py`,
`models/baselines.py::run_oof`) are retained: they are the reproducible record
of how the leaked result was produced, and `docs/LEAKAGE_AUDIT.md` cites them.
Deleting them would destroy the evidence that the leak was found and fixed.

---

## 3. Environment

```
Python      3.13.2 (venv at .venv)
Platform    Windows 11, Intel 4C/8T, 16.9 GB RAM
CUDA        not available -> compute mode cpu_16gb
Seed        42 (global)
```

Consequence for the upgrade: model families are CPU-bounded. A full-pool
LightGBM OOF repeat costs ~121 s; a compact-set repeat costs ~15 s; TabPFN
costs ~52 min per repeat. The tournament design reflects that budget
explicitly rather than pretending it does not exist.

---

## 4. Test baseline

`pytest tests -q` → **93 passed, 3 skipped** before the upgrade.
(`pytest-timeout` is not installed; the `--timeout` flag documented elsewhere
in the repo is unavailable and was dropped from the command.)

This is the regression floor: the upgrade may add tests, and must not reduce
this number.
