#!/usr/bin/env bash
# Stop MuleGuard local services (backend uvicorn on API_PORT, frontend vite).
set -u
API_PORT="${API_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

kill_port() {
  local port=$1 name=$2
  local pids
  pids=$(netstat -ano 2>/dev/null | grep ":$port " | grep LISTENING | awk '{print $5}' | sort -u)
  if [ -n "$pids" ]; then
    for pid in $pids; do
      taskkill //PID "$pid" //F >/dev/null 2>&1 || kill -9 "$pid" 2>/dev/null
    done
    echo "[stop] $name (port $port) stopped"
  else
    echo "[stop] $name (port $port) not running"
  fi
}
kill_port "$API_PORT" "backend"
kill_port "$FRONTEND_PORT" "frontend"
