# One-Command Run Guide

## Canonical command

```bash
./run.sh
```

(Windows: run from **Git Bash** — installed with Git for Windows. `make run`
is an equivalent alias.)

What it does, in order:

1. Verifies the Python venv, ML packages and frontend `node_modules` exist
   (fails loudly with the exact fix command if not).
2. Creates missing local directories (`artifacts/`, `logs/`) and a `.env`
   from `.env.example` with safe local defaults if absent.
3. Verifies the approved model bundle is present (`artifacts/models/final_bundle.joblib`).
   **It never retrains** — startup loads the frozen, hash-recorded bundle.
4. Checks the API port is free.
5. Starts the FastAPI backend (model loaded once at startup; SQLite schema
   auto-migrates; append-only audit trail active).
6. **Waits for `/health/ready` to actually pass** before any ready banner;
   if the backend dies, it prints the log tail and exits non-zero.
7. Starts the Vite frontend and waits for it to respond.
8. Prints the exact URLs, model version, and whether the optional local LLM
   (Ollama) is available — scoring never depends on it.
9. `Ctrl+C` stops both child processes cleanly.

## URLs

| Service | URL |
|---|---|
| Frontend dashboard | http://localhost:5173 |
| Backend API | http://127.0.0.1:8001 |
| OpenAPI docs | http://127.0.0.1:8001/docs |
| Health | http://127.0.0.1:8001/health/ready |

## Variants

```bash
./run.sh backend      # backend only (headless scoring service)
API_PORT=8010 ./run.sh # override ports via env
./scripts/stop.sh     # stop services started in a detached shell
```

## First-time setup (one-off, before the one command)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]
cd frontend && npm install && cd ..
# training pipeline (produces the model bundle) - see docs/DEPLOYMENT_GUIDE.md
```

## What it never requires

No MCP, no Claude-in-Chrome, no browser agents, no external LLM APIs, no
internet. Ollama is optional; with it stopped, evidence reports use the
deterministic narrative and scoring is unchanged.

Startup evidence from this machine: `artifacts/testing/one_command_startup.log`.
