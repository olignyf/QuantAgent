#!/usr/bin/env bash
set -euo pipefail

# Runs the Playwright UI debug test against a server started in Conda.
#
# Usage:
#   bash scripts/run_playwright_debug.sh
#
# Optional env vars:
#   CONDA_ENV=quantagents
#   BASE_URL=http://127.0.0.1:5000
#   SERVER_PORT=5000
#   SERVER_HOST=127.0.0.1
#   TEST_FILE=tests/test_playwright_gld_debug.py
#   STARTUP_TIMEOUT_SEC=60

CONDA_ENV="${CONDA_ENV:-quantagents}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5000}"
BASE_URL="${BASE_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
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

echo "[runner] Starting server in conda env '${CONDA_ENV}'..."
conda run -n "$CONDA_ENV" --no-capture-output python web_interface.py >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[runner] Stopping server (pid=$SERVER_PID)"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

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
