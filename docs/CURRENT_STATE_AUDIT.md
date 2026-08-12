# Current State Audit

**Date:** 2026-07-10 (session start ~23:30 IST) · **Auditor:** Claude Code (autonomous build agent)

## Repository state at audit time

| Item | Finding |
|---|---|
| Git | Repository initialised on branch `main` with **zero commits** (unborn branch) |
| Source code | **None** — no Python, no notebooks, no frontend, no backend |
| Dependency files | None |
| Docker files | None |
| Tests | **None exist.** "Run existing tests" step: nothing to run — recorded as 0 tests, 0 passed, 0 failed |
| Model artifacts | None |
| CI | None |

## Files present (before this build)

| File | Size | SHA-256 (first 16) | Role |
|---|---|---|---|
| `DataSet.xlsx` | 147,053,196 B | recorded in `data/interim/data_fingerprint.json` | Raw dataset (authoritative, never modified) |
| `Topic.pdf` | 250,939 B | — | Official problem statement (PS2) |
| `MuleGuard_Trinetra_PS2_Submission.pdf` | 138,218 B | — | Team design document (claims to re-verify, not truth) |
| `MuleGuard_Trinetra_Master_Claude_Code_Prompt.md` | 46,981 B | — | Build specification |

## Authoritative facts extracted

- **Topic.pdf (PS2):** binary classification of suspicious/mule accounts from portal data; **feature 3924 is the target variable**; 18 "commonly used" bank features listed (F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894).
- **Submission PDF claims requiring re-verification from raw data:** 9,082 rows × 3,924 columns; 81 positives vs 9,001 negatives (0.89%); 8 interpretable categorical columns (F2230, F3886, F3888–F3893); F3912 suspected target leak (|r|≈0.969, single-feature PR-AUC≈0.94, inflates full model to PR-AUC≈1.000); leakage-free LightGBM PR-AUC≈0.908±0.016; top-15 features ≈0.872.

None of the submission numbers are treated as results. Every one is recomputed by this pipeline and only pipeline-generated values are reported. In particular the design document's "leakage-free LightGBM PR-AUC≈0.908±0.016" did not survive recomputation: it was measured before the Feature Availability Firewall quarantined the post-resolution columns. The recomputed leakage-free figure is **0.7690 ± 0.0266** (`xgboost_top_120`, `artifacts/metrics/tournament_v2.json`); the pre-firewall accepted model's 0.8077 is retired as leakage-inflated.

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language 10.0.22631 |
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40GHz (4 cores / 8 threads) |
| RAM | 16.9 GB (16 GB class) |
| GPU | Intel Iris Xe (no CUDA) |
| Disk | E: 1.9 TB total, ~1.6 TB free; C: only ~37 GB free → **all installs/artifacts on E:** |
| Python | 3.13.2 (`C:\Python313\python.exe`); project venv at `E:\BOI\BOI_hackathon\.venv` |
| Node | v24.16.0, npm 11.13.0 |
| Ollama | 0.31.2 with local models incl. `qwen3:8b`, `llama3.2:3b`, `phi4-mini` (no new pulls needed) |
| Docker | 29.6.1 |

**Chosen compute mode: A (`cpu_16gb`)** — CPU-only tree models, controlled Optuna budgets, no automatic TabICL/TabPFN (attempted only behind memory/time guard), thread cap `n_jobs=4`, ≥2–3 GB RAM headroom preserved.

## Decisions taken at audit time

1. Repo was empty of code → nothing to preserve; the target structure is created fresh (no destructive rewrites possible).
2. `DataSet.xlsx` stays untouched at repo root and is `.gitignore`d (147 MB binary); a **read-only copy** goes to `data/raw/` and both hashes are recorded.
3. All Python dependencies live in a project venv on E: (user requirement: nothing installed to C:). `requirements.lock` frozen from this venv.
4. No existing tests → the "run existing tests" gate records `command: none available, 0 collected` and the new test suites are built from scratch.
