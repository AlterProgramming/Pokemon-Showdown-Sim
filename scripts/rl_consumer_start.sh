#!/usr/bin/env bash
# Start (or restart) the rl_consumer daemon.
set -euo pipefail

REPO="/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Agents-Go-Brrrr"
VENV_PY="$REPO/.venv/bin/python3"
SCRIPT="$REPO/rl_consumer.py"
LOG="/tmp/rl_consumer.log"

# Kill any existing consumer. `pkill -f` matches the argv line;
# the || true keeps the script from failing when nothing is running.
pkill -f rl_consumer.py 2>/dev/null || true
sleep 1

cd "$REPO"
nohup "$VENV_PY" "$SCRIPT" >"$LOG" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true

echo "rl_consumer started PID=$PID log=$LOG"
