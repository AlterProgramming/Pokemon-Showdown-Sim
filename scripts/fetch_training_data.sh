#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch_training_data.sh
#
# Downloads the Gen 9 Random Battles dataset from Kaggle (public — no
# credentials required) via kagglehub and caches it locally.
# Prints the resolved dataset path for use in training launchers.
#
# Usage:
#   ./scripts/fetch_training_data.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="thephilliplin/pokemon-showdown-battles-gen9-randbats"

# ---------------------------------------------------------------------------
# Python interpreter
# ---------------------------------------------------------------------------
if [[ -n "${PS_AGENT_PYTHON:-}" ]]; then
    PYTHON="$PS_AGENT_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python3"
elif [[ -x "$REPO_ROOT/venv/bin/python3" ]]; then
    PYTHON="$REPO_ROOT/venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

echo "[fetch] python=$PYTHON"
echo "[fetch] dataset=$DATASET (public — no credentials needed)"

# Ensure kagglehub is available
if ! "$PYTHON" -c "import kagglehub" 2>/dev/null; then
    echo "[fetch] installing kagglehub..."
    "$PYTHON" -m pip install --quiet kagglehub
fi

# Download (no-op if already cached in ~/.cache/kagglehub)
DATASET_PATH="$("$PYTHON" -c "
import kagglehub, warnings
warnings.filterwarnings('ignore')
print(kagglehub.dataset_download('$DATASET'))
")"

FILE_COUNT="$(find "$DATASET_PATH" -name '*.json' | wc -l | tr -d ' ')"
echo "[fetch] dataset_path=$DATASET_PATH"
echo "[fetch] json_files=$FILE_COUNT"
echo "[fetch] done — use this path in training:"
echo "  PS_AGENT_DATA=\"$DATASET_PATH\" ./launchers/start_model5_sequence_training.sh"
echo "  # or just run the launcher directly — it auto-resolves the kagglehub cache"
