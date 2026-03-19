from __future__ import annotations

from typing import Any


TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", ""}
REVIVE_HINT_KEYS = {
    "reviving",
    "revive",
    "needsReviveTarget",
    "chooseRevive",
    "chooseFainted",
    "canRevive",
}


def coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_STRINGS:
            return True
        if lowered in FALSE_STRINGS:
            return False
        return None
    return bool(value)


def any_true(value: Any) -> bool:
    if isinstance(value, list):
        return any(any_true(entry) for entry in value)
    coerced = coerce_optional_bool(value)
    return bool(coerced)


def _iter_active_entries(data: dict) -> list[dict]:
    active = data.get("active")
    if isinstance(active, list):
        return [entry for entry in active if isinstance(entry, dict)]
    if isinstance(active, dict):
        return [active]
    return []


def _iter_active_side_pokemon(data: dict) -> list[dict]:
    side = data.get("side")
    if not isinstance(side, dict):
        return []
    pokemon = side.get("pokemon")
    if not isinstance(pokemon, list):
        return []
    return [
        mon
        for mon in pokemon
        if isinstance(mon, dict) and coerce_optional_bool(mon.get("active")) is True
    ]


def _flag_present(mapping: dict, key: str) -> bool:
    return key in mapping and mapping.get(key) is not None


def _has_legality_signal(data: dict, legal_switches: list[dict]) -> bool:
    keys = {
        "forceSwitch",
        "trapped",
        "maybeTrapped",
        "canSwitch",
        "switching_allowed",
        "switchAllowed",
    }
    if any(_flag_present(data, key) for key in keys):
        return True
    if _iter_active_entries(data):
        return True
    if _iter_active_side_pokemon(data):
        return True
    return any(
        isinstance(sw, dict)
        and any(
            _flag_present(sw, key)
            for key in ("canSwitch", "disabled", "trapped", "maybeTrapped", "fainted")
        )
        for sw in legal_switches
    )


def _switch_entry_allowed(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if any_true(entry.get("disabled")):
        return False
    if any_true(entry.get("trapped")) or any_true(entry.get("maybeTrapped")):
        return False
    if coerce_optional_bool(entry.get("canSwitch")) is False:
        return False
    if any_true(entry.get("fainted")):
        return False
    return entry.get("slot") is not None


def _request_has_revive_hint(data: dict) -> bool:
    if any(any_true(data.get(key)) for key in REVIVE_HINT_KEYS):
        return True

    for entry in _iter_active_entries(data):
        if any(any_true(entry.get(key)) for key in REVIVE_HINT_KEYS):
            return True
    return False


def _all_candidates_marked_fainted(entries: list[dict]) -> bool:
    if not entries:
        return False
    fainted_flags = []
    for entry in entries:
        if not isinstance(entry, dict) or "fainted" not in entry:
            return False
        fainted_flags.append(coerce_optional_bool(entry.get("fainted")) is True)
    return all(fainted_flags)


def _revive_entry_allowed(entry: dict, require_fainted_flag: bool) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("slot") is None:
        return False
    if any_true(entry.get("disabled")):
        return False
    if coerce_optional_bool(entry.get("canRevive")) is False:
        return False

    fainted = coerce_optional_bool(entry.get("fainted"))
    if fainted is False:
        return False
    if require_fainted_flag and fainted is not True:
        return False
    return True


def filter_legal_revive_targets(data: dict | None, revive_targets: list[dict] | None) -> tuple[list[dict], str | None]:
    request_data = data if isinstance(data, dict) else {}
    options = [entry for entry in (revive_targets or []) if isinstance(entry, dict)]
    if not options:
        return [], None

    revive_request = _request_has_revive_hint(request_data) or _all_candidates_marked_fainted(options)
    if not revive_request:
        return [], None

    require_fainted_flag = _all_candidates_marked_fainted(options)
    candidates = [entry for entry in options if _revive_entry_allowed(entry, require_fainted_flag)]
    if not candidates:
        return [], "no_revive_candidates"
    return candidates, "revive_target_selection"


def filter_legal_switches(data: dict | None, legal_switches: list[dict] | None) -> tuple[list[dict], str | None]:
    request_data = data if isinstance(data, dict) else {}
    candidates = [sw for sw in (legal_switches or []) if _switch_entry_allowed(sw)]

    if not candidates:
        return [], "no_switch_candidates"

    if any_true(request_data.get("forceSwitch")):
        return candidates, "force_switch"

    active_entries = _iter_active_entries(request_data)
    active_side_pokemon = _iter_active_side_pokemon(request_data)

    trapped = any(
        [
            any_true(request_data.get("trapped")),
            any_true(request_data.get("maybeTrapped")),
            any(
                any_true(entry.get("trapped")) or any_true(entry.get("maybeTrapped"))
                for entry in active_entries
            ),
            any(
                any_true(mon.get("trapped")) or any_true(mon.get("maybeTrapped"))
                for mon in active_side_pokemon
            ),
        ]
    )
    if trapped:
        return [], "trapped"

    top_level_can_switch = coerce_optional_bool(request_data.get("canSwitch"))
    if top_level_can_switch is False:
        return [], "top_level_can_switch_false"

    if any(coerce_optional_bool(entry.get("canSwitch")) is False for entry in active_entries):
        return [], "active_can_switch_false"

    if any(coerce_optional_bool(mon.get("canSwitch")) is False for mon in active_side_pokemon):
        return [], "side_can_switch_false"

    explicit_allow = any(
        value is True
        for value in (
            coerce_optional_bool(request_data.get("switching_allowed")),
            coerce_optional_bool(request_data.get("switchAllowed")),
            top_level_can_switch,
        )
    )
    if explicit_allow:
        return candidates, "explicit_allow"

    if active_entries or active_side_pokemon:
        return candidates, "request_active_context"

    if _has_legality_signal(request_data, candidates):
        return candidates, "request_legality_signal"

    if any_true(request_data.get("assume_legal_switches")):
        return candidates, "assumed_legal"

    # If the caller only provides a bare switch list, be conservative and avoid
    # recommending illegal voluntary switches such as trapped turns.
    return [], "missing_legality_context"
