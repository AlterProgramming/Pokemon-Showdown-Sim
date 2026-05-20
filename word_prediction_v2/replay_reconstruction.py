from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .battle_questions import default_question_for_state
from .loss_analysis import extract_battle_log
from .text import normalize_token


def _empty_boosts() -> Dict[str, int]:
    return {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def _normalize_species(species_text: str) -> str:
    return "".join(ch for ch in species_text.lower() if ch.isalnum())


def _parse_ref(ref: str) -> tuple[str | None, str | None]:
    if ":" not in ref:
        return None, None
    player_ref, species_text = ref.split(":", 1)
    player = player_ref[:2] if player_ref[:2] in {"p1", "p2"} else None
    species = species_text.strip()
    return player, species or None


def _resolve_uid_from_ref(state: Dict[str, Any], ref: str) -> tuple[str | None, str | None]:
    player, species = _parse_ref(ref)
    if player is None:
        return None, None
    active_uid = state[player]["active_uid"]
    if active_uid is not None:
        return active_uid, player
    if species is None:
        return None, player
    return _ensure_mon(state, player, species), player


def _parse_switch_parts(raw_species: str) -> str:
    return raw_species.split(",", 1)[0].strip()


def _parse_hp_status(text: str) -> tuple[int | None, int | None, str | None, bool]:
    value = str(text or "").strip()
    if not value:
        return None, None, None, False
    if value == "0 fnt":
        return 0, None, None, True
    parts = value.split()
    hp_part = parts[0]
    status = parts[1] if len(parts) > 1 and not parts[1].startswith("[") else None
    if "/" not in hp_part:
        return None, None, status, False
    hp_raw, max_raw = hp_part.split("/", 1)
    try:
        hp = int(hp_raw)
    except ValueError:
        hp = None
    try:
        max_hp = int(max_raw)
    except ValueError:
        max_hp = None
    fainted = hp == 0
    return hp, max_hp, status, fainted


def _new_state() -> Dict[str, Any]:
    return {
        "turn_index": 0,
        "field": {"weather": None, "global_conditions": []},
        "p1": {"active_uid": None, "slots": [None] * 6, "side_conditions": {}},
        "p2": {"active_uid": None, "slots": [None] * 6, "side_conditions": {}},
        "mons": {},
    }


def _ensure_mon(state: Dict[str, Any], player: str, species: str) -> str:
    uid = f"{player}_{_normalize_species(species)}"
    mons = state["mons"]
    if uid not in mons:
        mons[uid] = {
            "uid": uid,
            "player": player,
            "species": species,
            "hp": None,
            "max_hp": None,
            "hp_frac": None,
            "status": None,
            "ability": None,
            "item": None,
            "tera_type": None,
            "terastallized": False,
            "public_revealed": True,
            "fainted": False,
            "boosts": _empty_boosts(),
            "observed_moves": [],
        }
        slots = state[player]["slots"]
        for index, current in enumerate(slots):
            if current is None:
                slots[index] = uid
                break
    return uid


def _set_hp(mon: Dict[str, Any], hp: int | None, max_hp: int | None) -> None:
    if hp is not None:
        mon["hp"] = hp
    if max_hp is not None:
        mon["max_hp"] = max_hp
    if mon["hp"] is not None and mon["max_hp"] not in {None, 0}:
        mon["hp_frac"] = max(0.0, min(1.0, float(mon["hp"]) / float(mon["max_hp"])))
    elif mon["hp"] == 0:
        mon["hp_frac"] = 0.0


@dataclass(frozen=True)
class DecisionPoint:
    turn_index: int
    player: str
    question: str
    battle_state: Dict[str, Any]
    legal_moves: tuple[Dict[str, Any], ...]
    legal_switches: tuple[Dict[str, Any], ...]
    actual_action_type: str
    actual_action_id: str
    actual_switch_species: str | None = None


def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(state)


def _active_mon(state: Dict[str, Any], player: str) -> Dict[str, Any] | None:
    uid = state[player]["active_uid"]
    if uid is None:
        return None
    return state["mons"].get(uid)


def _legal_moves_from_observed(active_mon: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not active_mon:
        return []
    return [
        {"move": move_name, "id": normalize_token(move_name), "slot": index + 1}
        for index, move_name in enumerate(active_mon.get("observed_moves") or [])
    ]


def _legal_switches_from_state(state: Dict[str, Any], player: str) -> List[Dict[str, Any]]:
    switches: List[Dict[str, Any]] = []
    active_uid = state[player]["active_uid"]
    for slot_index, uid in enumerate(state[player]["slots"], start=1):
        if uid is None or uid == active_uid:
            continue
        mon = state["mons"].get(uid) or {}
        if mon.get("fainted"):
            continue
        switches.append({"slot": slot_index, "hp_frac": mon.get("hp_frac")})
    return switches


def _slot_for_uid(state: Dict[str, Any], player: str, uid: str) -> int | None:
    for slot_index, slot_uid in enumerate(state[player]["slots"], start=1):
        if slot_uid == uid:
            return slot_index
    return None


def reconstruct_battle_state(log_text: str) -> tuple[Dict[str, Any], List[DecisionPoint]]:
    state = _new_state()
    decision_points: List[DecisionPoint] = []

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == "turn" and len(parts) >= 3:
            try:
                state["turn_index"] = int(parts[2])
            except ValueError:
                pass
            continue

        if tag in {"switch", "drag"} and len(parts) >= 5:
            player, ref_species = _parse_ref(parts[2])
            species = _parse_switch_parts(parts[3])
            if player is None:
                continue
            previous_active_uid = state[player]["active_uid"]
            if player == "p2" and tag == "switch" and int(state["turn_index"] or 0) > 0 and previous_active_uid is not None:
                pre_snapshot = _snapshot(state)
            uid = _ensure_mon(state, player, species or ref_species or "unknown")
            mon = state["mons"][uid]
            mon["species"] = species or mon["species"]
            hp, max_hp, status, fainted = _parse_hp_status(parts[4])
            _set_hp(mon, hp, max_hp)
            if status:
                mon["status"] = status
            mon["fainted"] = fainted
            state[player]["active_uid"] = uid
            if player == "p2" and tag == "switch" and int(state["turn_index"] or 0) > 0 and previous_active_uid is not None:
                slot = _slot_for_uid(state, player, uid)
                legal_switches = tuple(_legal_switches_from_state(pre_snapshot, player))
                decision_points.append(
                    DecisionPoint(
                        turn_index=int(state["turn_index"] or 0),
                        player=player,
                        question="which switch is best here?",
                        battle_state=pre_snapshot,
                        legal_moves=tuple(),
                        legal_switches=legal_switches,
                        actual_action_type="switch",
                        actual_action_id=f"slot:{slot}" if slot is not None else "",
                        actual_switch_species=species or ref_species or None,
                    )
                )
            continue

        if tag == "move" and len(parts) >= 4:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            move_name = parts[3].strip()
            if player is None or uid is None:
                continue
            mon = state["mons"][uid]
            state[player]["active_uid"] = uid
            if move_name and move_name not in mon["observed_moves"]:
                mon["observed_moves"].append(move_name)
            if player == "p2":
                snapshot = _snapshot(state)
                active_mon = snapshot["mons"].get(uid)
                legal_moves = tuple(_legal_moves_from_observed(active_mon))
                legal_switches = tuple(_legal_switches_from_state(snapshot, player))
                payload = {
                    "battle_state": snapshot,
                    "perspective_player": player,
                    "legal_moves": list(legal_moves),
                    "legal_switches": list(legal_switches),
                    "active": [{}],
                }
                question = default_question_for_state(payload)
                decision_points.append(
                    DecisionPoint(
                        turn_index=int(snapshot["turn_index"] or 0),
                        player=player,
                        question=question,
                        battle_state=snapshot,
                        legal_moves=legal_moves,
                        legal_switches=legal_switches,
                        actual_action_type="move",
                        actual_action_id=normalize_token(move_name),
                        actual_switch_species=None,
                    )
                )
            continue

        if tag == "-immune" and len(parts) >= 3:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if uid is None or player is None:
                continue
            mon = state["mons"][uid]
            for extra in parts[3:]:
                marker = str(extra).strip()
                if marker.startswith("[from] ability:"):
                    mon["ability"] = marker.split(":", 1)[1].strip()
                    break
            continue

        if tag in {"-damage", "-heal"} and len(parts) >= 4:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            mon = state["mons"][uid]
            hp, max_hp, status, fainted = _parse_hp_status(parts[3])
            _set_hp(mon, hp, max_hp)
            if status is not None:
                mon["status"] = status
            mon["fainted"] = fainted
            if fainted and state[player]["active_uid"] == uid:
                state[player]["active_uid"] = None
            continue

        if tag == "-status" and len(parts) >= 4:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            state["mons"][uid]["status"] = parts[3].strip()
            continue

        if tag == "-curestatus" and len(parts) >= 3:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            state["mons"][uid]["status"] = None
            continue

        if tag in {"-boost", "-unboost"} and len(parts) >= 5:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            stat = normalize_token(parts[3])
            try:
                amount = int(parts[4])
            except ValueError:
                continue
            if tag == "-unboost":
                amount *= -1
            boosts = state["mons"][uid]["boosts"]
            if stat in boosts:
                boosts[stat] = max(-6, min(6, int(boosts.get(stat, 0)) + amount))
            continue

        if tag == "faint" and len(parts) >= 3:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            mon = state["mons"][uid]
            mon["fainted"] = True
            mon["hp"] = 0
            mon["hp_frac"] = 0.0
            if state[player]["active_uid"] == uid:
                state[player]["active_uid"] = None
            continue

        if tag == "-sidestart" and len(parts) >= 4:
            player = parts[2].strip()
            if player in {"p1", "p2"}:
                condition = normalize_token(parts[3].replace("move:", ""))
                current = int(state[player]["side_conditions"].get(condition, 0))
                state[player]["side_conditions"][condition] = current + 1
            continue

        if tag == "-sideend" and len(parts) >= 4:
            player = parts[2].strip()
            if player in {"p1", "p2"}:
                condition = normalize_token(parts[3].replace("move:", ""))
                state[player]["side_conditions"].pop(condition, None)
            continue

        if tag in {"-item", "-enditem"} and len(parts) >= 3:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            if tag == "-enditem":
                state["mons"][uid]["item"] = None
            elif len(parts) >= 4:
                state["mons"][uid]["item"] = parts[3].strip()
            continue

        if tag == "-ability" and len(parts) >= 4:
            uid, player = _resolve_uid_from_ref(state, parts[2])
            if player is None or uid is None:
                continue
            state["mons"][uid]["ability"] = parts[3].strip()
            continue

        if tag == "-weather" and len(parts) >= 3:
            weather = normalize_token(parts[2])
            state["field"]["weather"] = None if weather == "none" else weather
            source_ability = None
            source_ref = None
            for extra in parts[3:]:
                marker = str(extra).strip()
                if marker.startswith("[from] ability:"):
                    source_ability = marker.split(":", 1)[1].strip()
                elif marker.startswith("[of] "):
                    source_ref = marker[5:].strip()
            if source_ability and source_ref:
                uid, player = _resolve_uid_from_ref(state, source_ref)
                if uid is not None and player is not None:
                    state["mons"][uid]["ability"] = source_ability
            continue

    return state, decision_points


def reconstruct_replay_html(html_text: str) -> tuple[Dict[str, Any], List[DecisionPoint]]:
    return reconstruct_battle_state(extract_battle_log(html_text))


def reconstruct_replay_file(path: Path) -> tuple[Dict[str, Any], List[DecisionPoint]]:
    return reconstruct_replay_html(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct battle states from a replay HTML file.")
    parser.add_argument("replay_html", type=Path)
    args = parser.parse_args()
    final_state, decision_points = reconstruct_replay_file(args.replay_html)
    print(
        json.dumps(
            {
                "final_state": final_state,
                "decision_points": [
                    {
                        "turn_index": point.turn_index,
                        "player": point.player,
                        "question": point.question,
                        "actual_action_type": point.actual_action_type,
                        "actual_action_id": point.actual_action_id,
                        "legal_moves": list(point.legal_moves),
                        "legal_switches": list(point.legal_switches),
                    }
                    for point in decision_points
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
