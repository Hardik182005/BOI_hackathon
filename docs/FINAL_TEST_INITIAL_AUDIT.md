# Final Test — Initial Audit

Captured before QA fixes began (2026-07-11, commit `cd6f9c8`, branch `main`).

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.22631 |
| Python | 3.13.2 (venv `E:\BOI\BOI_hackathon\.venv`) |
| Node / npm | v24.16.0 / 11.13.0 |
| CPU / RAM | i5-1135G7 (4C/8T) / 16.9 GB |
| GPU | Intel Iris Xe — no CUDA (compute mode `cpu_16gb`) |
| Disk free (E:) | ~1.6 TB |
| Dataset hash | see `data/interim/data_fingerprint.json` (raw SHA-256 `9f8f0a4933…`-prefix recorded there) |
| Model bundle | `artifacts/models/final_bundle.joblib`, SHA-256 in `model_manifest.json` |

## Repository state at QA start

- Backend: FastAPI (`src/muleguard/api/`) with scoring, cases, feedback,
  metrics, drift, reports endpoints; SQLite audit DB (append-only trigger).
- Frontend: React/TS (Vite), 7 pages, white-background/black-text theme.
- ML: full pipeline complete — audit → splits → baselines → stability
  selection → Optuna tournament → lens stack → locked test (single touch).
- Docs: 21-report set present; plots present; model registry populated.
- Docker: `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.frontend`.
- Startup at QA start: **two manual commands** (uvicorn + vite) — one-command
  startup did not exist yet (built in this phase as `run.sh`).

## Existing-test baseline (run before any QA fixes)

| Suite | Command | Result |
|---|---|---|
| Backend (unit/model/integration/security/e2e-robustness) | `.venv/Scripts/python.exe -m pytest -q` | **83 passed, 0 failed** (exit 0) |
| Frontend | `cd frontend && npx vitest run` | **3 passed, 0 failed** |

Machine-readable copy: `artifacts/testing/initial_test_results.json`.

## Known gaps at QA start (worked in this phase)

1. No `run.sh` one-command startup (blocker per QA spec §4).
2. No CSV/XLSX batch-upload endpoint (spec §14) — batch scoring was JSON-only.
3. `artifacts/testing/` evidence files not yet emitted.
4. TabPFN challenger run in progress (CPU); result lands in
   `artifacts/metrics/advanced_models.json`.
