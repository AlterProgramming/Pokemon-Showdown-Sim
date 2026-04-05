#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Model 5 training launcher — adds --predict-turn-sequence (Session 2 feature)
#
# Generation lineage:
#   model2  — joint action + predict_turn_outcome
#   model3  — (same, re-run with improved data)
#   model4  — model3 + predict_value
#   model5  — model4 + predict_turn_sequence   <-- THIS SCRIPT
#
# Registry key: model5
# Entry point:  train_policy.py  (sequence head lives here)
# ---------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$LOGS_DIR"

MODEL_NAME="${MODEL_NAME:-model_5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts}"
MAX_BATTLES="${MAX_BATTLES:-5000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
PATIENCE="${PATIENCE:-3}"

# Sequence head hyperparameters (Session 2 defaults)
SEQUENCE_WEIGHT="${SEQUENCE_WEIGHT:-0.1}"
SEQUENCE_HIDDEN_DIM="${SEQUENCE_HIDDEN_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32}"

# ---------------------------------------------------------------------------
# Python interpreter resolution:
#   1. $PS_AGENT_PYTHON env override
#   2. repo-local .venv
#   3. python3 on PATH
# ---------------------------------------------------------------------------
if [ -n "${PS_AGENT_PYTHON:-}" ]; then
    PYTHON="$PS_AGENT_PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python3" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python3"
elif [ -x "$REPO_ROOT/venv/bin/python3" ]; then
    PYTHON="$REPO_ROOT/venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

if [ -z "$PYTHON" ]; then
    echo "[launcher] ERROR: No usable Python interpreter found." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="$LOGS_DIR/${MODEL_NAME}-${TIMESTAMP}.train.out.log"
STATE_PATH="$LOGS_DIR/${MODEL_NAME}-${TIMESTAMP}.train.state.json"

# ---------------------------------------------------------------------------
# Data path resolution (in priority order):
#   1. Explicit positional args passed to this script
#   2. $PS_AGENT_DATA environment variable
#   3. kagglehub cache (public dataset — no credentials needed)
#      Run scripts/fetch_training_data.sh once to pre-warm the cache.
# ---------------------------------------------------------------------------
DATASET="thephilliplin/pokemon-showdown-battles-gen9-randbats"
DATA_PATH=""

if [ $# -gt 0 ]; then
    DATA_PATH="$1"
elif [ -n "${PS_AGENT_DATA:-}" ]; then
    DATA_PATH="$PS_AGENT_DATA"
else
    # Check the predictable kagglehub cache path first (avoids stdout contamination
    # from the version-warning that kagglehub 0.3.x prints to stdout on Python 3.9).
    # kagglehub cache layout: ~/.cache/kagglehub/datasets/{owner}/{name}/versions/{N}/
    KAGGLE_CACHE="${HOME}/.cache/kagglehub/datasets/${DATASET}/versions"
    if [ -d "$KAGGLE_CACHE" ]; then
        # Pick the highest version directory present
        KAGGLE_PATH="$(ls -d "$KAGGLE_CACHE"/[0-9]* 2>/dev/null | sort -t/ -k1 -V | tail -1)"
    fi

    if [ -n "$KAGGLE_PATH" ] && [ -d "$KAGGLE_PATH" ]; then
        DATA_PATH="$KAGGLE_PATH"
        echo "[launcher] using kagglehub cache: $DATA_PATH"
    else
        echo "[launcher] cache not found — run scripts/fetch_training_data.sh first"
        echo "[launcher] falling back: train_policy.py will auto-download via kagglehub"
    fi
fi

echo "[launcher] model_name=$MODEL_NAME"
echo "[launcher] python=$PYTHON"
echo "[launcher] log=$LOG_PATH"
echo "[launcher] data=${DATA_PATH:-<kaggle_default>}"

# ---------------------------------------------------------------------------
# Write lightweight state file
# ---------------------------------------------------------------------------
"$PYTHON" -c "
import json, sys
state = {
    'model_name': '$MODEL_NAME',
    'started_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'repo_root': '$REPO_ROOT',
    'log_path': '$LOG_PATH',
    'output_dir': '$OUTPUT_DIR',
    'python_executable': '$PYTHON',
    'generation': 5,
    'new_flags': ['--predict-turn-sequence'],
    'data_path': '$DATA_PATH',
}
print(json.dumps(state, indent=2))
" > "$STATE_PATH"

# ---------------------------------------------------------------------------
# Run training — stdout+stderr teed to log file
# ---------------------------------------------------------------------------
if [ -n "$DATA_PATH" ]; then
    "$PYTHON" -u "$REPO_ROOT/train_policy.py" \
        "$DATA_PATH" \
        --model-name            "$MODEL_NAME" \
        --output-dir            "$OUTPUT_DIR" \
        --max-battles           "$MAX_BATTLES" \
        --epochs                "$EPOCHS" \
        --batch-size            "$BATCH_SIZE" \
        --learning-rate         "$LEARNING_RATE" \
        --patience              "$PATIENCE" \
        --include-switches \
        --predict-turn-outcome \
        --predict-value \
        --predict-turn-sequence \
        --sequence-weight       "$SEQUENCE_WEIGHT" \
        --sequence-hidden-dim   "$SEQUENCE_HIDDEN_DIM" \
        --max-seq-len           "$MAX_SEQ_LEN" \
        --transition-weight     0.25 \
        --value-weight          0.25 \
        --action-embed-dim      32 \
        --transition-hidden-dim 256 \
        --hidden-dim            256 \
        --depth                 3 \
        --dropout               0.1 \
        --policy-return-weighting     exp \
        --policy-return-weight-scale  0.75 \
        --policy-return-weight-min    0.25 \
        --policy-return-weight-max    4.0 \
        --seed                  42 \
        2>&1 | tee "$LOG_PATH"
else
    "$PYTHON" -u "$REPO_ROOT/train_policy.py" \
        --model-name            "$MODEL_NAME" \
        --output-dir            "$OUTPUT_DIR" \
        --max-battles           "$MAX_BATTLES" \
        --epochs                "$EPOCHS" \
        --batch-size            "$BATCH_SIZE" \
        --learning-rate         "$LEARNING_RATE" \
        --patience              "$PATIENCE" \
        --include-switches \
        --predict-turn-outcome \
        --predict-value \
        --predict-turn-sequence \
        --sequence-weight       "$SEQUENCE_WEIGHT" \
        --sequence-hidden-dim   "$SEQUENCE_HIDDEN_DIM" \
        --max-seq-len           "$MAX_SEQ_LEN" \
        --transition-weight     0.25 \
        --value-weight          0.25 \
        --action-embed-dim      32 \
        --transition-hidden-dim 256 \
        --hidden-dim            256 \
        --depth                 3 \
        --dropout               0.1 \
        --policy-return-weighting     exp \
        --policy-return-weight-scale  0.75 \
        --policy-return-weight-min    0.25 \
        --policy-return-weight-max    4.0 \
        --seed                  42 \
        2>&1 | tee "$LOG_PATH"
fi

EXIT_CODE=${PIPESTATUS[0]}
echo "[launcher] exit_code=$EXIT_CODE"
exit $EXIT_CODE
