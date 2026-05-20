from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHOWDOWN_MOVES_JS = PROJECT_ROOT / "pokemon-showdown-model-feature" / "dist" / "data" / "moves.js"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "word_prediction_model" / "cache"
CACHE_PATH = CACHE_DIR / "showdown_move_metadata.json"


def _to_id(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _build_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    script = f"""
const {{Moves}} = require({json.dumps(str(SHOWDOWN_MOVES_JS))});
const payload = {{}};
for (const [id, move] of Object.entries(Moves)) {{
  payload[id] = {{
    type: move.type || null,
    category: move.category || null,
    basePower: Number(move.basePower || 0),
    accuracy: move.accuracy === true ? 101 : Number(move.accuracy || 0),
    priority: Number(move.priority || 0),
    target: move.target || null,
    selfSwitch: move.selfSwitch || null,
    status: move.status || null,
    volatileStatus: move.volatileStatus || null,
    sideCondition: move.sideCondition || null,
    pseudoWeather: move.pseudoWeather || null
  }};
}}
process.stdout.write(JSON.stringify(payload));
"""
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    CACHE_PATH.write_text(result.stdout, encoding="utf-8")


@lru_cache(maxsize=1)
def _move_metadata() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        _build_cache()
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def move_metadata(move_name_or_id: str | None) -> dict[str, Any]:
    sid = _to_id(move_name_or_id)
    if not sid:
        return {}
    return _move_metadata_for_id(sid)


@lru_cache(maxsize=2048)
def _move_metadata_for_id(sid: str) -> dict[str, Any]:
    return dict(_move_metadata().get(sid) or {})
