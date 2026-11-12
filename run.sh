#!/usr/bin/env bash
# Start the TADASHI server.
#   ./run.sh              -> http://127.0.0.1:8000
#   PORT=12000 ./run.sh   -> http://127.0.0.1:12000
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
exec python3 -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "$PORT" "$@"
