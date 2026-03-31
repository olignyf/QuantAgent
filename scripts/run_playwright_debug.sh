#!/usr/bin/env bash
set -euo pipefail

# Runs the Playwright UI debug test against a server started in Conda.
#
# Usage:
#   bash scripts/run_playwright_debug.sh
#
# Optional env vars:
#   CONDA_ENV=quantagents
#   SERVER_HOST=127.0.0.1
#   QUANTAGENT_PORT=5000          # if unset, an ephemeral free port is chosen
#   TEST_FILE=tests/test_playwright_gld_debug.py
#   STARTUP_TIMEOUT_SEC=60
#   SERVER_LOG=server_playwright_debug.log

CONDA_ENV="${CONDA_ENV:-quantagents}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
TEST_FILE="${TEST_FILE:-tests/test_playwright_gld_debug.py}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-60}"
SERVER_LOG="${SERVER_LOG:-server_playwright_debug.log}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found. Install/init conda first." >&2
  exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
  echo "ERROR: test file not found: $TEST_FILE" >&2
  exit 1
fi

# Prefer explicit port; otherwise bind an ephemeral port (avoids stale :5000).
if [[ -z "${QUANTAGENT_PORT:-}" ]]; then
  QUANTAGENT_PORT="$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")"
fi
export QUANTAGENT_PORT
BASE_URL="http://${SERVER_HOST}:${QUANTAGENT_PORT}"

echo "[runner] Starting server in conda env '${CONDA_ENV}' on port ${QUANTAGENT_PORT}..."
: >"$SERVER_LOG"
conda run -n "$CONDA_ENV" --no-capture-output \
  env QUANTAGENT_SMOKE_ANALYZE=1 "QUANTAGENT_PORT=${QUANTAGENT_PORT}" \
  python web_interface.py >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[runner] Stopping server (pid=$SERVER_PID)"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

sleep 1
if grep -q "Address already in use" "$SERVER_LOG" 2>/dev/null; then
  echo "ERROR: port ${QUANTAGENT_PORT} is already in use. Set QUANTAGENT_PORT to a free port." >&2
  tail -n 40 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "[runner] Waiting for ${BASE_URL}/demo (timeout ${STARTUP_TIMEOUT_SEC}s)..."
start_ts="$(date +%s)"
until curl -fsS "${BASE_URL}/demo" >/dev/null 2>&1; do
  now_ts="$(date +%s)"
  elapsed=$(( now_ts - start_ts ))
  if (( elapsed >= STARTUP_TIMEOUT_SEC )); then
    echo "ERROR: server did not become ready in ${STARTUP_TIMEOUT_SEC}s." >&2
    echo "[runner] Last server logs:" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "[runner] Server is ready. Running Playwright test..."
export QUANTAGENT_BASE_URL="$BASE_URL"

# Auto-install Playwright if missing in the selected conda env.
if ! conda run -n "$CONDA_ENV" --no-capture-output python -c "import playwright" >/dev/null 2>&1; then
  echo "[runner] Playwright not found in env '${CONDA_ENV}'. Installing..."
  conda run -n "$CONDA_ENV" --no-capture-output python -m pip install playwright
  conda run -n "$CONDA_ENV" --no-capture-output playwright install chromium
fi

conda run -n "$CONDA_ENV" --no-capture-output python "$TEST_FILE"

echo "[runner] Done. Test passed."
