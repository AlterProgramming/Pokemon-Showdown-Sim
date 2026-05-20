from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POKEDEX_PATH = PROJECT_ROOT / "Pokemon-Showdown-Agents-Go-Brrrr" / "data" / "pokedex.json"
DIAGNOSE_TYPES_PATH = PROJECT_ROOT / "Pokemon-Showdown-Agents-Go-Brrrr" / "tools" / "diagnose_state_failures.py"


def _to_id(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _pokedex() -> dict[str, Any]:
    return json.loads(POKEDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _diagnostic_module() -> Any:
    spec = importlib.util.spec_from_file_location("pokemon_diagnose_state_failures", DIAGNOSE_TYPES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load type utilities from {DIAGNOSE_TYPES_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def species_types(species_name: str | None) -> list[str]:
    sid = _to_id(species_name)
    if not sid:
        return []
    return list(_species_types_for_id(sid))


@lru_cache(maxsize=1024)
def _species_types_for_id(sid: str) -> tuple[str, ...]:
    entry = _pokedex().get(sid) or {}
    return tuple(str(type_name) for type_name in (entry.get("types") or []))


def base_speed(species_name: str | None) -> int | None:
    sid = _to_id(species_name)
    if not sid:
        return None
    return _base_speed_for_id(sid)


@lru_cache(maxsize=1024)
def _base_speed_for_id(sid: str) -> int | None:
    entry = _pokedex().get(sid) or {}
    base_stats = entry.get("baseStats") or {}
    raw_speed = base_stats.get("spe")
    if raw_speed is None:
        return None
    try:
        return int(raw_speed)
    except (TypeError, ValueError):
        return None


def move_type(move_name_or_id: str | None) -> str | None:
    sid = _to_id(move_name_or_id)
    if not sid:
        return None
    return _move_type_for_id(sid)


@lru_cache(maxsize=1024)
def _move_type_for_id(sid: str) -> str | None:
    return _diagnostic_module().get_move_type(sid)


def type_effectiveness(move_type_name: str | None, defending_types: list[str]) -> float:
    if not move_type_name or not defending_types:
        return 1.0
    return _type_effectiveness_cached(move_type_name, tuple(defending_types))


@lru_cache(maxsize=2048)
def _type_effectiveness_cached(move_type_name: str, defending_types: tuple[str, ...]) -> float:
    return float(_diagnostic_module().type_effectiveness(move_type_name, list(defending_types)))
