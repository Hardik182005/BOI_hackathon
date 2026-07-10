#!/usr/bin/env bash
# Offline validation: scoring + guardrails with the LLM narrator disabled
# and no network dependency (Ollama circuit intentionally pointed nowhere).
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
OLLAMA_BASE_URL="http://127.0.0.1:59999" "$PY" -m pytest tests/unit/test_llm_guardrails.py tests/e2e -q \
  && "$PY" -m muleguard.cli.qa_harness ollama
