from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


LOG_BLOCK_RE = re.compile(
    r'<script type="text/plain" class="battle-log-data">(.*?)</script>',
    re.DOTALL,
)

HAZARD_NAMES = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web"}
RECOVER_NAMES = {
    "Recover", "Roost", "Slack Off", "Soft-Boiled", "Moonlight", "Morning Sun",
    "Synthesis", "Rest",
}
SETUP_NAMES = {
    "Swords Dance", "Nasty Plot", "Bulk Up", "Calm Mind", "Dragon Dance",
    "Quiver Dance", "Shell Smash", "Agility", "Curse", "Growth", "Trailblaze",
}


def extract_battle_log(html_text: str) -> str:
    match = LOG_BLOCK_RE.search(html_text)
    if not match:
        return ""
    return html.unescape(match.group(1)).strip()


def parse_battle_log(log_text: str) -> dict[str, object]:
    turns = 0
    p1_switches = 0
    p2_switches = 0
    p1_hazard_events = 0
    p2_hazard_events = 0
    p1_status_events = 0
    p2_status_events = 0
    p1_recover_moves = 0
    p2_recover_moves = 0
    p1_setup_moves = 0
    p2_setup_moves = 0
    winner = ""
    leading_tags: set[str] = set()

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|turn|"):
            try:
                turns = max(turns, int(line.split("|")[2]))
            except (IndexError, ValueError):
                pass
            continue
        if line.startswith("|win|"):
            winner = line.split("|", 2)[2]
            continue
        if line.startswith("|switch|") or line.startswith("|drag|"):
            if "|p1a:" in line:
                p1_switches += 1
            elif "|p2a:" in line:
                p2_switches += 1
            continue
        if line.startswith("|move|"):
            parts = line.split("|")
            if len(parts) >= 4:
                actor = parts[2]
                move_name = parts[3]
                if actor.startswith("p1"):
                    if move_name in RECOVER_NAMES:
                        p1_recover_moves += 1
                    if move_name in SETUP_NAMES:
                        p1_setup_moves += 1
                elif actor.startswith("p2"):
                    if move_name in RECOVER_NAMES:
                        p2_recover_moves += 1
                    if move_name in SETUP_NAMES:
                        p2_setup_moves += 1
            continue
        if line.startswith("|-sidestart|"):
            parts = line.split("|")
            if len(parts) >= 4:
                side = parts[2]
                effect = parts[3].replace("move: ", "")
                if effect in HAZARD_NAMES:
                    if side == "p1":
                        p1_hazard_events += 1
                    elif side == "p2":
                        p2_hazard_events += 1
            continue
        if line.startswith("|-status|"):
            parts = line.split("|")
            if len(parts) >= 3:
                actor = parts[2]
                if actor.startswith("p1"):
                    p1_status_events += 1
                elif actor.startswith("p2"):
                    p2_status_events += 1

    if turns >= 30:
        leading_tags.add("long_game")
    if p1_hazard_events > 0 or p2_hazard_events > 0:
        leading_tags.add("hazards_present")
    if p1_status_events > 0 or p2_status_events > 0:
        leading_tags.add("status_game")
    if p1_recover_moves + p2_recover_moves >= 4:
        leading_tags.add("recovery_loop")
    if p2_setup_moves >= 2:
        leading_tags.add("opponent_setup_pressure")
    if p1_switches >= 10:
        leading_tags.add("high_switch_loss")
    if p1_setup_moves >= 3 and turns <= 25:
        leading_tags.add("setup_spiral")
    if p1_setup_moves >= 5:
        leading_tags.add("setup_spiral")
    if turns <= 22 and p1_setup_moves + p1_recover_moves == 0 and p1_switches <= 6:
        leading_tags.add("short_tempo_loss")

    return {
        "winner": winner,
        "turns": turns,
        "p1_switches": p1_switches,
        "p2_switches": p2_switches,
        "p1_hazard_events": p1_hazard_events,
        "p2_hazard_events": p2_hazard_events,
        "p1_status_events": p1_status_events,
        "p2_status_events": p2_status_events,
        "p1_recover_moves": p1_recover_moves,
        "p2_recover_moves": p2_recover_moves,
        "p1_setup_moves": p1_setup_moves,
        "p2_setup_moves": p2_setup_moves,
        "tags": sorted(leading_tags),
    }


def analyze_replay_files(paths: Iterable[Path]) -> dict[str, object]:
    replay_summaries: list[dict[str, object]] = []
    tag_counter: Counter[str] = Counter()
    total_turns = 0
    total_switches = 0

    for path in sorted(paths):
        html_text = path.read_text(encoding="utf-8")
        log_text = extract_battle_log(html_text)
        summary = parse_battle_log(log_text)
        summary["file"] = str(path)
        replay_summaries.append(summary)
        total_turns += int(summary["turns"])
        total_switches += int(summary["p1_switches"]) + int(summary["p2_switches"])
        tag_counter.update(summary["tags"])

    count = len(replay_summaries)
    return {
        "loss_count": count,
        "avg_turns": (total_turns / count) if count else 0.0,
        "avg_total_switches": (total_switches / count) if count else 0.0,
        "tag_counts": dict(tag_counter.most_common()),
        "losses": replay_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze captured random-vs-model loss replays.")
    parser.add_argument("replay_dir", help="Directory containing saved replay HTML files.")
    parser.add_argument("--output-json", help="Optional path to write the summary JSON.")
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    paths = sorted(replay_dir.glob("*.html"))
    summary = analyze_replay_files(paths)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
