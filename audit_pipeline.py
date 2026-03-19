from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from BattleStateTracker import BattleStateTracker


def iter_json_paths(inputs: List[str]) -> Iterable[Path]:
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        elif path.suffix.lower() == ".json":
            yield path


def analyze_battles(json_paths: Iterable[Path], max_battles: int) -> Dict[str, int]:
    tracker = BattleStateTracker(form_change_species={"Palafin"})
    stats = {
        "battles": 0,
        "turns": 0,
        "move_examples": 0,
        "examples_missing_my_active": 0,
        "examples_missing_opp_active": 0,
        "examples_missing_either_active": 0,
        "raw_weather_events": 0,
        "raw_side_condition_events": 0,
        "raw_field_condition_events": 0,
        "raw_tera_events": 0,
        "raw_status_end_events": 0,
        "tracker_turns_with_weather": 0,
        "tracker_turns_with_side_conditions": 0,
        "tracker_turns_with_field_conditions": 0,
    }

    for path in json_paths:
        if stats["battles"] >= max_battles:
            break

        try:
            battle = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if "turns" not in battle:
            continue

        stats["battles"] += 1
        tracker.load_battle(battle)

        for turn in battle.get("turns", []) or []:
            stats["turns"] += 1
            events = turn.get("events", []) or []

            tracker._backfill_visible_actives(turn)
            state_before = tracker.snapshot()

            for player in ("p1", "p2"):
                action = tracker.extract_action_for_player(events, player)
                if action is None or action[0] != "move":
                    continue

                other = "p2" if player == "p1" else "p1"
                stats["move_examples"] += 1
                my_missing = state_before[player]["active_uid"] is None
                opp_missing = state_before[other]["active_uid"] is None
                stats["examples_missing_my_active"] += int(my_missing)
                stats["examples_missing_opp_active"] += int(opp_missing)
                stats["examples_missing_either_active"] += int(my_missing or opp_missing)

            for ev in events:
                if ev.get("type") == "form_change":
                    stats["raw_tera_events"] += 1
                elif ev.get("type") == "status_end":
                    stats["raw_status_end_events"] += 1
                elif ev.get("type") == "effect":
                    effect_type = ev.get("effect_type")
                    if effect_type == "weather":
                        stats["raw_weather_events"] += 1
                    elif effect_type in {"sidestart", "sideend"}:
                        stats["raw_side_condition_events"] += 1
                    elif effect_type in {"fieldstart", "fieldend"}:
                        stats["raw_field_condition_events"] += 1

            tracker.apply_turn(turn)
            state_after = tracker.snapshot()
            if state_after["field"]["weather"]:
                stats["tracker_turns_with_weather"] += 1
            if state_after["field"]["global_conditions"]:
                stats["tracker_turns_with_field_conditions"] += 1
            if state_after["p1"]["side_conditions"] or state_after["p2"]["side_conditions"]:
                stats["tracker_turns_with_side_conditions"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tracker/vectorization readiness on battle JSONs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON file(s) or directories containing battle logs.",
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        default=200,
        help="Maximum number of battles to scan.",
    )
    args = parser.parse_args()

    stats = analyze_battles(iter_json_paths(args.paths), max_battles=args.max_battles)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
