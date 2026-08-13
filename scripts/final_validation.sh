#!/usr/bin/env bash
# MuleGuard - Trinetra: one-command final validation (final-validation prompt §59).
#
#   ./scripts/final_validation.sh                                 # default
#   ./scripts/final_validation.sh --reuse-verified-model-artifacts # same, explicit
#   ./scripts/final_validation.sh --full-retrain                   # + retrain everything
#
# The default mode NEVER retrains. It verifies the shipped artifacts, re-derives
# every metric from stored predictions, and runs the release suite. That is the
# path a judge demo takes, and §59 requires the demo not to retrain.
#
# --full-retrain additionally rebuilds the nested evidence from the raw
# workbooks. It costs hours, not minutes, and is not part of a demo.
#
# Exit code is 0 only if every step passed.
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"

MODE="reuse"
for arg in "$@"; do
  case "$arg" in
    --reuse-verified-model-artifacts) MODE="reuse" ;;
    --full-retrain)                   MODE="retrain" ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "unknown argument: $arg (try --help)" >&2
      exit 2 ;;
  esac
done

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p artifacts/testing logs
LOG="logs/final_validation.log"
: > "$LOG"

FAIL=0
STEPS_PASSED=0
STEPS_FAILED=0
FAILED_NAMES=""

step() {
  local name="$1"; shift
  echo | tee -a "$LOG"
  echo "=== [$name] ===" | tee -a "$LOG"
  # PIPESTATUS, not $?: piping through tee would otherwise report tee's status
  # and a failing step would be recorded as a pass.
  "$@" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    STEPS_PASSED=$((STEPS_PASSED + 1))
  else
    STEPS_FAILED=$((STEPS_FAILED + 1))
    FAILED_NAMES="$FAILED_NAMES $name"
    FAIL=1
  fi
}

echo "MuleGuard final validation - mode=$MODE - started $STARTED_UTC" | tee -a "$LOG"

# --- 1. truth about the inputs -------------------------------------------
# Fingerprints both workbooks and re-verifies shape/target/prevalence. Cheap,
# and every later number is meaningless if this disagrees.
step "environment fingerprint"      "$PY" -m muleguard.cli.audit_env
step "dataset + description integrity" "$PY" -m muleguard.cli.audit_data

# --- 2. retrain, only when asked -----------------------------------------
if [ "$MODE" = "retrain" ]; then
  echo | tee -a "$LOG"
  echo "--- FULL RETRAIN: hours, not minutes ---" | tee -a "$LOG"
  step "nested repeated CV tournament" "$PY" -m muleguard.cli.nested_cv --repeats 3 --inner 4
  step "stability / ensemble / shift"  "$PY" -m muleguard.cli.nested_ses --stages all --n-jobs 2
  step "missingness signature ablation" "$PY" -m muleguard.cli.missingness_ablation --family histgb --repeats 3
  step "champion tournament + lenses"  bash -c '"$0" -m muleguard.cli.tournament_v2 && "$0" -m muleguard.cli.build_lenses_v2' "$PY"
  step "shield + robustness"           bash -c '"$0" -m muleguard.cli.shield_v2 && "$0" -m muleguard.cli.robustness_v2' "$PY"
  step "merchant verifier + label audit" bash -c '"$0" -m muleguard.cli.merchant_verifier && "$0" -m muleguard.cli.audit_labels' "$PY"
fi

# --- 3. metrics re-derived from stored predictions ------------------------
# Read-only in both modes. The battery is pointed at the nested predictions when
# they exist, because nested is the primary protocol; the flat store is the
# documented fallback and is labelled as historical wherever it is used.
if [ -f artifacts/predictions/nested_oof.parquet ]; then
  BATTERY_SOURCE="artifacts/predictions/nested_oof.parquet"
  BATTERY_PROTOCOL="NESTED"
else
  BATTERY_SOURCE="artifacts/predictions/oof_v2.parquet"
  BATTERY_PROTOCOL="FLAT"
  echo "note: nested predictions absent, battery falls back to $BATTERY_SOURCE (historical)" | tee -a "$LOG"
fi
step "metric battery ($BATTERY_PROTOCOL)" "$PY" -m muleguard.cli.metric_battery \
  --source "$BATTERY_SOURCE" --protocol "$BATTERY_PROTOCOL"
step "analyst capacity curve" "$PY" -m muleguard.cli.capacity_curve
# §56. Redraws the required figures from whatever predictions are current, so a
# plot can never show a model the metrics no longer describe.
step "required plots"         "$PY" -m muleguard.evaluation.plots_final
# §57/§58. Re-derives every spec-named artifact from its real source, so the
# required names always hold current content rather than a stale copy.
step "artifact reconciliation" "$PY" -m muleguard.cli.reconcile_artifacts
# §60. Rebuilt from the artifacts themselves, and it exits non-zero if any
# artifact under artifacts/metrics/ is not claimed by a ledger entry - so an
# experiment cannot quietly drop out of the record between runs.
step "experiment ledger"      "$PY" -m muleguard.cli.experiment_ledger

# --- 4. the release suite -------------------------------------------------
# pytest + frontend tests + live QA harness + release gate + report regeneration.
step "release test suite" bash scripts/release_test.sh

# --- 5. hidden-validation readiness --------------------------------------
# The organiser dry-run goes over HTTP against the running backend, exactly as a
# judge would. release_test.sh leaves one on :8001.
step "organiser dry run"      "$PY" -m muleguard.cli.dry_run
step "competition export"     "$PY" -m muleguard.cli.export_submission

# --- 5b. the section 65 answer ------------------------------------------
# Assembled from the artifacts the steps above just refreshed. It fails while
# any blocker is unresolved or any pass criterion is unmet, which is the point:
# this script cannot report success over evidence that does not exist.
step "final verdict (A-L)"    "$PY" -m muleguard.cli.final_verdict

# --- 6. verdict -----------------------------------------------------------
FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$FAIL" = 0 ]; then VERDICT="PASS"; else VERDICT="FAIL"; fi

cat > artifacts/testing/final_validation.json <<JSON
{
  "mode": "$MODE",
  "retraining_performed": $([ "$MODE" = "retrain" ] && echo true || echo false),
  "started_utc": "$STARTED_UTC",
  "finished_utc": "$FINISHED_UTC",
  "steps_passed": $STEPS_PASSED,
  "steps_failed": $STEPS_FAILED,
  "failed_steps": "$(echo $FAILED_NAMES)",
  "metric_battery_source": "$BATTERY_SOURCE",
  "metric_battery_protocol": "$BATTERY_PROTOCOL",
  "verdict": "$VERDICT",
  "log": "$LOG"
}
JSON

echo | tee -a "$LOG"
echo "FINAL VALIDATION: $VERDICT ($STEPS_PASSED passed, $STEPS_FAILED failed)" | tee -a "$LOG"
[ "$FAIL" = 0 ] || echo "failed steps:$FAILED_NAMES" | tee -a "$LOG"
echo "evidence: artifacts/testing/final_validation.json" | tee -a "$LOG"
exit $FAIL
