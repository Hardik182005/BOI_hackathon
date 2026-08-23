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
# The harness also checks that the shipped UI is actually served, so the dev
# server has to be up for that check to measure anything. Without this the
# suite reported "frontend_test_results: FAIL (6/7), dev_server_serves_html:
# no dev server on :5173" on every run - a red blocker that described the
# harness, not the product.
# Killing $! is not enough: it is the subshell, npm is its child and the vite
# node process is its grandchild, so the port stays held. Free the port the
# same way scripts/stop.sh does, and only that port - the backend is still
# wanted by whatever runs next.
stop_vite() {
  [ -n "${VITE_PID:-}" ] || return 0
  VITE_PID=""
  local pids
  pids=$(netstat -ano 2>/dev/null | grep ":5173 " | grep LISTENING | awk '{print $5}' | sort -u)
  for pid in $pids; do
    taskkill //PID "$pid" //F >/dev/null 2>&1 || kill -9 "$pid" 2>/dev/null
  done
}
if ! curl -s -o /dev/null --max-time 3 "http://localhost:5173"; then
  echo "frontend dev server not running - starting for harness"
  (cd frontend && npm run dev) > artifacts/testing/harness_frontend_start.log 2>&1 &
  VITE_PID=$!
  trap stop_vite EXIT INT TERM
  for i in $(seq 1 60); do
    curl -s -o /dev/null "http://localhost:5173" && break
    sleep 1
  done
fi

"$PY" -m muleguard.cli.qa_harness all || FAIL=1

stop_vite

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
