# Deployment Guide

## Local development (Windows/Linux/macOS)

```bash
# 1. environment
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]        # Windows
# .venv/bin/python -m pip install -e .[dev]          # Linux/macOS

# 2. data pipeline (DataSet.xlsx at repo root; raw file is never modified)
.venv/Scripts/python -m muleguard.cli.audit_env
.venv/Scripts/python -m muleguard.cli.audit_data     # ~10 min (independent-engine validation)
.venv/Scripts/python -m muleguard.cli.make_splits

# 3. training (CPU Mode A timings on i5-1135G7 / 16 GB)
.venv/Scripts/python -m muleguard.cli.train baselines        # ~20 min
.venv/Scripts/python -m muleguard.cli.tournament select      # ~40 min
.venv/Scripts/python -m muleguard.cli.tournament tune        # ~2-3 h (resumable; Optuna SQLite)
.venv/Scripts/python -m muleguard.cli.tournament finalists   # ~1-2 h
.venv/Scripts/python -m muleguard.cli.advanced               # guarded challengers
.venv/Scripts/python -m muleguard.cli.build_lenses           # ~15 min
.venv/Scripts/python -m muleguard.cli.evaluate               # locked test - ONCE
.venv/Scripts/python -m muleguard.evaluation.plots
.venv/Scripts/python -m muleguard.cli.demo

# 4. tests + release gate
.venv/Scripts/python -m pytest
.venv/Scripts/python -m muleguard.cli.release_gate

# 5. serve
.venv/Scripts/python -m uvicorn muleguard.api.main:app --port 8001
cd frontend && npm install && npm run dev            # http://localhost:5173
```

Every training command is resumable: Optuna studies persist in
`artifacts/optuna/studies.db`; completed models are skipped by rerunning the
stage (OOF rows are replaced per model, never duplicated).

## Ollama (optional)

```bash
ollama serve                  # if not already running
ollama pull qwen3:8b          # or any model in configs/ollama.yaml preference list
```
Scoring and evidence reports work with Ollama absent (deterministic
narratives). No configuration change needed either way.

## Docker

```bash
docker compose up --build
# frontend: http://localhost:5173  api: http://localhost:8001/docs
```

The API container mounts `./artifacts` (bundle + metrics + SQLite DB) and the
canonical parquet read-only. Ollama on the host is reachable via
`host.docker.internal:11434` and remains optional.

## Rollback

Every bundle is registered in `artifacts/model_registry/registry.json` with
SHA-256 and status (champion/superseded). To roll back: point
`MODELS_DIR/final_bundle.joblib` at the archived bundle (or restore from the
registry copy) and restart the API — health checks confirm the loaded version.

## Configuration

| File | Purpose |
|---|---|
| `configs/base.yaml` | seed, target, bank-hint list |
| `configs/data.yaml` | audit thresholds, split sizes |
| `configs/train.yaml` | budgets, Optuna trials, stability-selection knobs |
| `configs/thresholds.yaml` | policy tier derivation, conformal alpha, OOD/anomaly knobs |
| `configs/ollama.yaml` | narrator model preference, timeouts, breaker |
| `configs/leakage_quarantine.yaml` | generated quarantine list (audit output) |
| `.env` | DB path, ports (see `.env.example`) |
