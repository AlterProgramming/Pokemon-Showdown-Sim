from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .replay_reconstruction import reconstruct_replay_file
from .text import normalize_token


PROGRESS_MOVE_IDS = {
    "stealthrock", "spikes", "stickyweb", "toxicspikes", "stoneaxe", "ceaselessedge",
    "roar", "whirlwind", "dragontail", "circlethrow", "haze", "clearsmog",
    "toxic", "poisonpowder", "stunspore", "thunderwave", "spore", "hypnosis",
    "sleeppowder", "willowisp", "leechseed", "yawn", "encore",
}

RECOVER_MOVE_IDS = {
    "recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun",
    "synthesis", "rest", "wish", "strengthsap",
}

SETUP_MOVE_IDS = {
    "swordsdance", "nastyplot", "bulkup", "calmmind", "dragondance",
    "quiverdance", "shellsmash", "agility", "curse", "growth", "trailblaze",
}

PRIORITY_MOVE_IDS = {
    "iceshard", "machpunch", "bulletpunch", "jetpunch", "quickattack",
    "extremespeed", "shadowsneak", "suckerpunch", "accelerock", "vacuumwave",
    "aquajet", "fakeout", "firstimpression",
}

WEATHER_ENGINE_ABILITIES = {
    "drought", "drizzle", "sandstream", "snowwarning",
    "protosynthesis", "chlorophyll", "solarpower", "harvest",
    "swiftswim", "raindish", "sandrush", "slushrush",
}

WEATHER_ENGINE_MOVES = {
    "weatherball", "solarbeam", "solarblade", "thunder", "hurricane",
    "morningsun", "moonlight", "synthesis",
}


def _active_mon(state: Dict[str, Any], player: str) -> Dict[str, Any] | None:
    uid = state[player]["active_uid"]
    if uid is None:
        return None
    return state["mons"].get(uid)


def _other_player(player: str) -> str:
    return "p1" if player == "p2" else "p2"


def _alive_bench_count(state: Dict[str, Any], player: str) -> int:
    active_uid = state[player]["active_uid"]
    total = 0
    for uid in state[player]["slots"]:
        if not uid or uid == active_uid:
            continue
        mon = state["mons"].get(uid) or {}
        if not mon.get("fainted") and float(mon.get("hp_frac") or 0.0) > 0.0:
            total += 1
    return total


def _slot_uid(state: Dict[str, Any], player: str, slot_ref: str) -> str | None:
    if not slot_ref.startswith("slot:"):
        return None
    try:
        slot_index = int(slot_ref.split(":", 1)[1]) - 1
    except ValueError:
        return None
    slots = state[player]["slots"]
    if 0 <= slot_index < len(slots):
        return slots[slot_index]
    return None


def _bench_priority_count(state: Dict[str, Any], player: str) -> int:
    active_uid = state[player]["active_uid"]
    total = 0
    for uid in state[player]["slots"]:
        if not uid or uid == active_uid:
            continue
        mon = state["mons"].get(uid) or {}
        if mon.get("fainted") or float(mon.get("hp_frac") or 0.0) <= 0.0:
            continue
        observed = {normalize_token(str(move)) for move in (mon.get("observed_moves") or [])}
        if observed & PRIORITY_MOVE_IDS:
            total += 1
    return total


def classify_decision_point(point: Any) -> Dict[str, Any]:
    state = point.battle_state
    player = point.player
    opponent = _other_player(player)
    my_active = _active_mon(state, player) or {}
    opp_active = _active_mon(state, opponent) or {}
    move_id = str(point.actual_action_id or "")
    bench_count = _alive_bench_count(state, player)
    bench_priority = _bench_priority_count(state, player)
    my_hp = float(my_active.get("hp_frac") or 0.0)
    opp_hp = float(opp_active.get("hp_frac") or 0.0)
    opp_observed = {normalize_token(str(move)) for move in (opp_active.get("observed_moves") or [])}
    opp_ability = normalize_token(str(opp_active.get("ability") or ""))
    weather = normalize_token(str((state.get("field") or {}).get("weather") or ""))
    opp_farming = bool(opp_observed & (SETUP_MOVE_IDS | RECOVER_MOVE_IDS))
    weather_engine_live = bool(
        weather
        and weather != "none"
        and (
            opp_ability in WEATHER_ENGINE_ABILITIES
            or bool(opp_observed & WEATHER_ENGINE_MOVES)
        )
    )
    has_progress_move = move_id in PROGRESS_MOVE_IDS
    has_recover_move = move_id in RECOVER_MOVE_IDS
    has_setup_move = move_id in SETUP_MOVE_IDS
    action_type = str(point.actual_action_type or "none")

    node_tags: List[str] = []
    edge_tags: List[str] = []

    if opp_farming:
        node_tags.append("opponent_farming")
    if weather_engine_live:
        node_tags.append("weather_engine_live")
    if my_hp <= 0.3:
        node_tags.append("low_hp_active")
    if bench_count > 0:
        node_tags.append("bench_available")
    if bench_priority > 0:
        node_tags.append("priority_bench")
    if opp_hp <= 0.35:
        node_tags.append("closeout_window")

    if action_type == "switch":
        if opp_farming and my_hp <= 0.35:
            edge_tags.append("escape_edge")
        elif my_hp <= 0.35 and bench_priority > 0:
            edge_tags.append("anti_sack_edge")
        else:
            edge_tags.append("switch_edge")
    elif action_type == "move":
        if has_progress_move:
            edge_tags.append("progress_edge")
        elif has_recover_move and opp_farming:
            edge_tags.append("feed_loop_edge")
        elif has_setup_move and opp_farming:
            edge_tags.append("feed_setup_edge")
        elif weather_engine_live and not has_progress_move and opp_hp > 0.2:
            edge_tags.append("engine_roll_edge")
        elif my_hp <= 0.35 and bench_priority > 0 and opp_hp > 0.2:
            edge_tags.append("sack_value_edge")
        else:
            edge_tags.append("attack_edge")

    return {
        "turn_index": int(point.turn_index),
        "question": point.question,
        "actual_action_type": action_type,
        "actual_action_id": move_id,
        "actual_switch_species": getattr(point, "actual_switch_species", None),
        "active_species": str(my_active.get("species") or ""),
        "opponent_species": str(opp_active.get("species") or ""),
        "weather": weather or None,
        "opponent_ability": str(opp_active.get("ability") or ""),
        "my_hp": my_hp,
        "opp_hp": opp_hp,
        "bench_count": bench_count,
        "bench_priority_count": bench_priority,
        "node_tags": node_tags,
        "edge_tags": edge_tags,
    }


def _annotate_switch_outcomes(rows: List[Dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if row["actual_action_type"] != "switch":
            continue
        switched_in_uid = _slot_uid(row["battle_state"], "p2", row["actual_action_id"])
        switched_in_species = str(row.get("actual_switch_species") or "")
        if switched_in_uid and not switched_in_species:
            switched_in_species = str((row["battle_state"]["mons"].get(switched_in_uid) or {}).get("species") or "")
        if not switched_in_species:
            continue
        acted_with_species = False
        for future_index in range(index + 1, len(rows)):
            future = rows[future_index]
            if future["actual_action_type"] == "move" and future["active_species"] == switched_in_species:
                acted_with_species = True
                break
            if future["actual_action_type"] == "switch" and future["active_species"] != row["active_species"]:
                break
        if not acted_with_species:
            row["edge_tags"].append("failed_conversion_switch")
            if row["my_hp"] <= 0.4:
                row["edge_tags"].append("sack_value_loss_edge")


def analyze_replay_graph(replay_path: Path, *, limit: int | None = None) -> Dict[str, Any]:
    _, decision_points = reconstruct_replay_file(replay_path)
    rows: List[Dict[str, Any]] = []
    node_counter = Counter()
    edge_counter = Counter()

    for index, point in enumerate(decision_points):
        if limit is not None and index >= limit:
            break
        row = classify_decision_point(point)
        row["battle_state"] = point.battle_state
        rows.append(row)
        node_counter.update(row["node_tags"])
        edge_counter.update(row["edge_tags"])

    _annotate_switch_outcomes(rows)
    node_counter = Counter()
    edge_counter = Counter()
    for row in rows:
        node_counter.update(row["node_tags"])
        edge_counter.update(row["edge_tags"])

    return {
        "replay_file": str(replay_path),
        "decision_count": len(rows),
        "replay_tags": ["weather_engine_loss"] if edge_counter.get("engine_roll_edge", 0) > 0 else [],
        "node_tag_counts": dict(node_counter),
        "edge_tag_counts": dict(edge_counter),
        "rows": [
            {key: value for key, value in row.items() if key != "battle_state"}
            for row in rows
        ],
    }


def analyze_replay_graphs(replay_paths: Sequence[Path]) -> Dict[str, Any]:
    payloads = [analyze_replay_graph(path) for path in replay_paths]
    node_counter = Counter()
    edge_counter = Counter()
    for payload in payloads:
        node_counter.update(payload["node_tag_counts"])
        edge_counter.update(payload["edge_tag_counts"])
    return {
        "replay_count": len(payloads),
        "node_tag_counts": dict(node_counter),
        "edge_tag_counts": dict(edge_counter),
        "replays": payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify replay decision points as graph nodes and edges.")
    parser.add_argument("replay_paths", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if len(args.replay_paths) == 1:
        payload = analyze_replay_graph(args.replay_paths[0], limit=args.limit)
    else:
        payload = analyze_replay_graphs(args.replay_paths)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
