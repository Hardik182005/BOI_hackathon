#!/usr/bin/env bash
# Offline validation: scoring + guardrails with the LLM narrator disabled
# and no network dependency (Ollama circuit intentionally pointed nowhere).
#
# Writes artifacts/testing/offline_results.json - §57 requires the evidence as a
# file, not only as console output a reader has to be shown.
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
mkdir -p artifacts/testing logs
LOG="logs/test_offline.log"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

OLLAMA_BASE_URL="http://127.0.0.1:59999" "$PY" -m pytest tests/unit/test_llm_guardrails.py tests/e2e -q 2>&1 | tee "$LOG"
PYTEST_RC="${PIPESTATUS[0]}"

QA_RC=0
if [ "$PYTEST_RC" -eq 0 ]; then
  "$PY" -m muleguard.cli.qa_harness ollama 2>&1 | tee -a "$LOG"
  QA_RC="${PIPESTATUS[0]}"
else
  QA_RC=-1   # not run: the guardrail tests must pass before the harness means anything
fi

SUMMARY="$(grep -Eo '[0-9]+ (passed|failed|error)[^,]*' "$LOG" | tail -3 | tr '\n' ' ')"
[ "$PYTEST_RC" -eq 0 ] && [ "$QA_RC" -eq 0 ] && VERDICT="PASS" || VERDICT="FAIL"

cat > artifacts/testing/offline_results.json <<JSON
{
  "started_utc": "$STARTED",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "network_policy": "OLLAMA_BASE_URL pointed at 127.0.0.1:59999 (nothing listens there)",
  "guardrail_and_e2e_pytest_rc": $PYTEST_RC,
  "qa_harness_ollama_rc": $QA_RC,
  "pytest_summary": "$(echo "$SUMMARY" | sed 's/"/\\"/g')",
  "verdict": "$VERDICT",
  "log": "$LOG"
}
JSON

echo "offline verdict: $VERDICT (evidence: artifacts/testing/offline_results.json)"
[ "$VERDICT" = "PASS" ]
