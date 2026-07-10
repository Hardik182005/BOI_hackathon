#!/usr/bin/env bash
# MuleGuard - Trinetra: one-command local startup.
#   ./run.sh          start backend + frontend, wait for health, print URLs
#   ./run.sh backend  start backend only
# Requires: the project venv (.venv) and frontend/node_modules (see README
# quick start / docs/ONE_COMMAND_RUN_GUIDE.md). Ollama is OPTIONAL.
set -u

cd "$(dirname "$0")"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
MODE="${1:-all}"

PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
log() { printf '[run] %s\n' "$*"; }
fail() { printf '[run] FATAL: %s\n' "$*" >&2; exit 1; }

# ---- 1. dependency checks -------------------------------------------------
[ -f "$PY" ] || fail "venv missing - create it with: python -m venv .venv && $PY -m pip install -e .[dev]"
"$PY" -c "import muleguard, fastapi, lightgbm, catboost, xgboost" 2>/dev/null \
  || fail "python dependencies missing - run: $PY -m pip install -e .[dev]"
if [ "$MODE" = "all" ]; then
  command -v npm >/dev/null 2>&1 || fail "npm not found - install Node.js 18+"
  [ -d frontend/node_modules ] || fail "frontend deps missing - run: cd frontend && npm install"
fi

# ---- 2. directories + env --------------------------------------------------
mkdir -p artifacts data/interim logs
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null && log "created .env from .env.example (safe local defaults)"
fi
[ -f artifacts/models/final_bundle.joblib ] \
  || fail "model bundle missing (artifacts/models/final_bundle.joblib) - run the training pipeline first (docs/DEPLOYMENT_GUIDE.md)"

# ---- 3. port availability ---------------------------------------------------
if curl -s -o /dev/null --max-time 2 "http://$API_HOST:$API_PORT/health/live"; then
  fail "port $API_PORT already serving something - stop it or set API_PORT"
fi

# ---- 4. start backend -------------------------------------------------------
PIDS=()
cleanup() {
  log "shutting down..."
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
  wait 2>/dev/null
  log "stopped."
}
trap cleanup INT TERM

log "starting backend on http://$API_HOST:$API_PORT ..."
"$PY" -m uvicorn muleguard.api.main:app --host "$API_HOST" --port "$API_PORT" \
  > logs/backend.log 2>&1 &
PIDS+=($!)

# ---- 5. wait for backend health (fail loud, no premature ready banner) ------
for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://$API_HOST:$API_PORT/health/ready"; then break; fi
  kill -0 "${PIDS[0]}" 2>/dev/null || { tail -5 logs/backend.log >&2; fail "backend process died during startup (logs/backend.log)"; }
  sleep 1
  [ "$i" = 60 ] && { tail -5 logs/backend.log >&2; fail "backend not healthy after 60s (logs/backend.log)"; }
done
READY_JSON=$(curl -s "http://$API_HOST:$API_PORT/health/ready")
MODEL_VERSION=$(printf '%s' "$READY_JSON" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d.get('winner','?'), 'v'+d.get('model_version','?'))")
OLLAMA_MODEL=$(printf '%s' "$READY_JSON" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d.get('ollama_model') or 'unavailable (deterministic narratives active)')")

# ---- 6. start frontend -------------------------------------------------------
if [ "$MODE" = "all" ]; then
  log "starting frontend on http://localhost:$FRONTEND_PORT ..."
  (cd frontend && npm run dev -- --port "$FRONTEND_PORT" --strictPort > ../logs/frontend.log 2>&1) &
  PIDS+=($!)
  for i in $(seq 1 60); do
    if curl -s -o /dev/null "http://localhost:$FRONTEND_PORT"; then break; fi
    sleep 1
    [ "$i" = 60 ] && { tail -5 logs/frontend.log >&2; fail "frontend not responding after 60s (logs/frontend.log)"; }
  done
fi

# ---- 7. ready banner ----------------------------------------------------------
log "-------------------------------------------------------------"
log "MuleGuard - Trinetra is READY"
log "  Frontend:   http://localhost:$FRONTEND_PORT"
log "  Backend:    http://$API_HOST:$API_PORT"
log "  API docs:   http://$API_HOST:$API_PORT/docs"
log "  Health:     http://$API_HOST:$API_PORT/health/ready"
log "  Model:      $MODEL_VERSION"
log "  Local LLM:  $OLLAMA_MODEL  (scoring never depends on it)"
log "  Logs:       logs/backend.log  logs/frontend.log"
log "Press Ctrl+C to stop all services."
log "-------------------------------------------------------------"

wait
