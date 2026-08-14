#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
"$PY" -m pytest -q && (cd frontend && npm test --silent)
