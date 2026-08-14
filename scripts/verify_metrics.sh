#!/usr/bin/env bash
# MuleGuard - Trinetra: one command to check the metrics and the accuracy.
#
#   bash scripts/verify_metrics.sh
#
# Read-only. Nothing here retrains a model, rebuilds a feature set, or reads the
# locked test set. It re-derives the headline accuracy from the saved
# predictions, checks it against what the registry and the reports claim, and
# runs the test suites that guard those claims.
#
# The distinction that matters: this is not a report of stored numbers. Every
# metric below is recomputed from the prediction files and compared to the
# artifact that claims it, so a stale artifact describing a model that has since
# changed fails the run instead of printing a number nobody rechecked.
#
# Exit code is 0 only if every step passed.
#
# For the full release rehearsal - which does need a running backend, does
# exercise the API over HTTP, and takes far longer - use:
#   bash scripts/final_validation.sh
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
export PYTHONIOENCODING=utf-8

FAIL=0
PASSED=0
FAILED_NAMES=""

step() {
  local name="$1"; shift
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "  $name"
  echo "──────────────────────────────────────────────────────────────"
  if "$@"; then
    PASSED=$((PASSED + 1))
  else
    FAILED_NAMES="$FAILED_NAMES
    - $name"
    FAIL=1
  fi
}

echo "MuleGuard - Trinetra :: metric & accuracy verification"
echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- 1. do the claims hold? ------------------------------------------------
# Ten independent checks: bundle signature, champion identity across three
# sources, PR-AUC recomputed from saved predictions, the leakage firewall, that
# no reported metric touched a locked-test row, threshold ordering, calibration,
# the 27-column table's self-consistency, that every design decision has a
# recorded experiment behind it, and that any experiment which beat the shipped
# configuration is written up rather than left sitting in a JSON file.
step "accuracy claims re-derived from source" "$PY" -m muleguard.cli.verify_metrics

# --- 2. the definitive table ----------------------------------------------
# Rebuilt rather than printed, so what you read is what the predictions say
# right now. All 27 columns of section 55.
step "definitive model comparison (27 columns)" "$PY" -m muleguard.cli.final_accuracy_table

# --- 3. no forgotten experiments ------------------------------------------
# Fails if any metrics artifact has no ledger entry. This is the check that
# stops a discouraging result from quietly not being mentioned: a run that
# produced a file must account for it, including as NOT_AN_EXPERIMENT.
step "experiment ledger complete" "$PY" -m muleguard.cli.experiment_ledger

# --- 4. the guards --------------------------------------------------------
# The leakage, firewall and hallucination-guardrail tests are the ones that
# would catch a number being right for the wrong reason.
step "test suite (unit, integration, security)" \
  "$PY" -m pytest tests/unit tests/integration tests/security -q

# --- 5. verdict -----------------------------------------------------------
echo
echo "══════════════════════════════════════════════════════════════"
if [ "$FAIL" = 0 ]; then
  echo "  VERDICT: PASS  ($PASSED/4 steps)"
  echo
  echo "  champion, metrics and guarantees all agree with their sources."
  echo "  details: artifacts/metrics/verify_metrics.json"
  echo "           artifacts/metrics/final_accuracy_table.csv"
else
  echo "  VERDICT: FAIL  ($PASSED/4 steps passed)"
  echo "  failed steps:$FAILED_NAMES"
  echo
  echo "  details: artifacts/metrics/verify_metrics.json"
fi
echo "══════════════════════════════════════════════════════════════"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$FAIL"
