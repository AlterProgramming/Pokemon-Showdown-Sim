#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$ROOT_DIR/pokemon-showdown-model-feature/dist/sim/examples/statistical-runner.js"
ARTIFACT_ROOT="$ROOT_DIR/artifacts/word_prediction_model/recorded_runs"

RUN_NAME="${1:-}"
if [[ -z "$RUN_NAME" ]]; then
  echo "usage: $0 <run-name> [total-games] [concurrency]"
  exit 1
fi

TOTAL_GAMES="${2:-100}"
CONCURRENCY="${3:-5}"
RUN_DIR="$ARTIFACT_ROOT/$RUN_NAME"
LOG_PATH="$RUN_DIR/benchmark.log"
REPLAY_DIR="$RUN_DIR/replays"

mkdir -p "$REPLAY_DIR"

pkill -f 'dist/sim/examples/statistical-runner.js' >/dev/null 2>&1 || true
pkill -f 'word_prediction_model/ipc_policy_worker.py' >/dev/null 2>&1 || true

echo "run_dir=$RUN_DIR"
echo "log_path=$LOG_PATH"
echo "replay_dir=$REPLAY_DIR"

(
  cd "$ROOT_DIR"
  TOTAL_GAMES="$TOTAL_GAMES" \
  CONCURRENCY="$CONCURRENCY" \
  BENCHMARK_QUIET=false \
  RL_MODEL_TRANSPORT=ipc \
  RL_MODEL_PROFILE=joint-policy \
  RL_ALLOW_VOLUNTARY_SWITCHES=false \
  RL_MODEL_IPC_PYTHON=python3 \
  REPLAY_CAPTURE_MODE=loss \
  REPLAY_OUTPUT_DIR="$REPLAY_DIR" \
  node "$RUNNER"
) | tee "$LOG_PATH"
