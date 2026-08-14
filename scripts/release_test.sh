#!/usr/bin/env bash
# MuleGuard - Trinetra: canonical full release test suite.
# Runs every required suite, aggregates evidence into artifacts/testing/,
# regenerates the final reports and returns non-zero on any failure.
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
FAIL=0
step() { echo; echo "=== [$1] ==="; }

step "backend pytest (unit/model/integration/security/e2e)"
"$PY" -m pytest -q --tb=short || FAIL=1

step "frontend tests"
(cd frontend && npm test --silent) || FAIL=1

step "live QA harness (backend/data/ollama/perf/e2e/consistency/security/frontend/batch)"
if ! curl -s -o /dev/null --max-time 3 "http://127.0.0.1:${API_PORT:-8001}/health/ready"; then
  echo "backend not running - starting headless for harness"
  bash run.sh backend > artifacts/testing/harness_backend_start.log 2>&1 &
  RUNNER_PID=$!
  for i in $(seq 1 60); do
    curl -s -o /dev/null "http://127.0.0.1:${API_PORT:-8001}/health/ready" && break
    sleep 1
  done
fi
"$PY" -m muleguard.cli.qa_harness all || FAIL=1

step "ML release gate (leakage/split/metric-trace/determinism blockers)"
"$PY" -m muleguard.cli.release_gate || FAIL=1

step "final reports"
"$PY" -m muleguard.cli.final_report || FAIL=1
"$PY" -m muleguard.cli.qa_final_summary || FAIL=1

echo
if [ "$FAIL" = 0 ]; then
  echo "RELEASE TEST SUITE: PASS"
else
  echo "RELEASE TEST SUITE: FAIL (see artifacts/testing/ and docs/FINAL_RELEASE_TEST_REPORT.md)"
fi
exit $FAIL
