#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
"$PY" -m pytest tests/security -q && "$PY" -m muleguard.cli.qa_harness security
