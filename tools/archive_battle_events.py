#!/usr/bin/env python3
"""
archive_battle_events.py
========================
Extract and archive battle events from Pokemon Showdown battle JSON files
for perfect reconstruction.

Captures every state-affecting event (moves, damage, crits, status, terrain,
side conditions, ability triggers, switches, faints, stat changes, heals) in
a compact-but-complete format.  Archived battles can be replayed with the
exact same outcome: same crits, same damage rolls, same effects.

Output structure (per run_id)::

    runs/<run_id>/
      ├── battle_logs/
      │   ├── gen9randombattle-2390494424.json
      │   └── ...
      ├── run_metadata.json
      └── event_index.csv

Input JSON schema: schema_version 1.0.0 with ``turns[].events[]``.

Usage::

    python tools/archive_battle_events.py \\
      --input-dir /path/to/battles/ \\
      --output-dir runs/entity_action_bc_v1_20260406_seq_run1/ \\
      --run-id entity_action_bc_v1_20260406_seq_run1

    # Single file
    python tools/archive_battle_events.py \\
      --input-dir data/gen9randombattle-2390494424.json \\
      --output-dir runs/test_run/ \\
      --run-id test_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
ARCHIVE_VERSION = "1.0.0"

# Event types that carry state-affecting information we must preserve.
# Every event in turns[].events[] is kept — this list is for documentation.
_STATE_AFFECTING_EVENT_TYPES = {
    "move",
    "damage",
    "heal",
    "switch",
    "faint",
    "status_start",
    "status_end",
    "stat_change",
    "weather",
    "field",
    "side_condition",
    "forme_change",
    "effect",       # ability / item / crit markers live here
    "transform",
    "mega",
    "tera",
    "revive",
    "replace",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_json_paths(inputs: List[str]) -> List[Path]:
    """Expand a list of files/directories into individual JSON paths."""
    result: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            result.extend(sorted(p.rglob("*.json")))
        elif p.suffix.lower() == ".json" and p.exists():
            result.append(p)
        else:
            print(f"  [warn] skipping {raw}: not a JSON file or directory", file=sys.stderr)
    return result


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on parse error."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [warn] failed to load {path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# HP tracking: resolve hp_before from sequential state
# ---------------------------------------------------------------------------


class _HpTracker:
    """Tracks current HP for each pokemon_uid to fill hp_before on damage/heal events."""

    def __init__(self) -> None:
        self._hp: Dict[str, Optional[int]] = {}
        self._max_hp: Dict[str, Optional[int]] = {}

    def seed(self, uid: str, hp: Optional[int], max_hp: Optional[int]) -> None:
        """Seed initial HP from team revelation data."""
        if uid not in self._hp:
            self._hp[uid] = hp
        if uid not in self._max_hp:
            self._max_hp[uid] = max_hp

    def get(self, uid: str) -> Tuple[Optional[int], Optional[int]]:
        """Return (current_hp, max_hp) for a uid."""
        return self._hp.get(uid), self._max_hp.get(uid)

    def update(self, uid: str, hp_after: Optional[int], max_hp: Optional[int]) -> Optional[int]:
        """Record hp_after; returns previous hp (hp_before). Updates max_hp if provided."""
        hp_before = self._hp.get(uid)
        self._hp[uid] = hp_after
        if max_hp is not None:
            self._max_hp[uid] = max_hp
        return hp_before

    def max_hp(self, uid: str) -> Optional[int]:
        return self._max_hp.get(uid)


# ---------------------------------------------------------------------------
# Crit marker tracking: an "effect/crit" event precedes its damage event
# ---------------------------------------------------------------------------


class _CritMarker:
    """Tracks pending crit markers (effect events with effect_type=="crit")."""

    def __init__(self) -> None:
        self._pending: bool = False

    def mark(self) -> None:
        self._pending = True

    def consume(self) -> bool:
        """Return True and clear if a crit was pending."""
        if self._pending:
            self._pending = False
            return True
        return False

    def reset(self) -> None:
        self._pending = False


# ---------------------------------------------------------------------------
# Core event extraction
# ---------------------------------------------------------------------------


def _extract_team_info(battle: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build teams dict from team_revelation data.

    Returns::

        {
            "p1": [{"uid": ..., "species": ..., "level": ...,
                    "ability": ..., "item": ..., "tera_type": ..., "hp": ...}],
            "p2": [...]
        }
    """
    team_rev = battle.get("team_revelation", {}) or {}
    teams_raw = team_rev.get("teams", {}) or {}
    teams: Dict[str, List[Dict[str, Any]]] = {}

    for player, roster in teams_raw.items():
        entries: List[Dict[str, Any]] = []
        for mon in (roster or []):
            entry: Dict[str, Any] = {
                "uid": mon.get("pokemon_uid"),
                "species": mon.get("species"),
                "level": mon.get("level"),
                "ability": mon.get("known_ability"),
                "item": mon.get("known_item"),
                "tera_type": mon.get("known_tera_type"),
            }
            # Include base HP for reconstruction reference
            base_stats = mon.get("base_stats") or {}
            hp = base_stats.get("hp")
            if hp is not None:
                entry["base_hp"] = hp
            entries.append(entry)
        teams[player] = entries

    return teams


def _seed_hp_tracker(tracker: _HpTracker, teams: Dict[str, List[Dict[str, Any]]]) -> None:
    """Pre-seed the HP tracker with max_hp values from team revelation."""
    for roster in teams.values():
        for mon in roster:
            uid = mon.get("uid")
            max_hp = mon.get("base_hp")
            if uid:
                tracker.seed(uid, max_hp, max_hp)


def _normalize_event(
    raw: Dict[str, Any],
    hp_tracker: _HpTracker,
    crit_marker: _CritMarker,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one raw event dict into the compact archive format.

    Returns None for events that carry no state information (e.g. pure
    display events with no mechanical effect).  All state-affecting events
    are preserved with full fidelity.

    Crit information is sourced from:
      1. The ``crit`` field directly on damage events (schema 1.0.0 populates this).
      2. Preceding ``effect`` events with ``effect_type == "crit"`` (belt-and-suspenders).
    """
    etype = raw.get("type", "")

    # ------------------------------------------------------------------ move
    if etype == "move":
        ev: Dict[str, Any] = {
            "type": "move",
            "seq": raw.get("seq"),
            "player": raw.get("player"),
            "uid": raw.get("pokemon_uid"),
            "move": raw.get("move_id"),
            "target_uid": raw.get("target_uid"),
        }
        # Preserve miss / no-effect flags if present
        if raw.get("missed"):
            ev["missed"] = True
        if raw.get("no_effect"):
            ev["no_effect"] = True
        return {k: v for k, v in ev.items() if v is not None}

    # ---------------------------------------------------------------- damage
    if etype == "damage":
        target_uid = raw.get("target_uid")
        hp_after = raw.get("hp_after")
        max_hp_raw = raw.get("max_hp")

        # Resolve hp_before from tracker state
        hp_before, tracked_max = hp_tracker.get(target_uid)

        # Use the event's max_hp when available; fall back to tracked
        max_hp = max_hp_raw if max_hp_raw is not None else tracked_max

        # Update tracker
        hp_tracker.update(target_uid, hp_after, max_hp_raw)

        # Crit: honour explicit field first, then consume pending marker
        is_crit = bool(raw.get("crit", False)) or crit_marker.consume()

        ev = {
            "type": "damage",
            "seq": raw.get("seq"),
            "target_uid": target_uid,
            "hp_before": hp_before,
            "hp_after": hp_after,
            "max_hp": max_hp,
            "is_crit": is_crit,
            "effectiveness": raw.get("effectiveness"),
            "source": raw.get("source"),
        }
        # Compute raw damage amount when both HP values are known
        if hp_before is not None and hp_after is not None:
            ev["damage"] = hp_before - hp_after
        return {k: v for k, v in ev.items() if v is not None}

    # ------------------------------------------------------------------ heal
    if etype == "heal":
        target_uid = raw.get("target_uid")
        hp_after = raw.get("hp_after")
        max_hp_raw = raw.get("max_hp")

        hp_before, tracked_max = hp_tracker.get(target_uid)
        max_hp = max_hp_raw if max_hp_raw is not None else tracked_max
        hp_tracker.update(target_uid, hp_after, max_hp_raw)

        ev = {
            "type": "heal",
            "seq": raw.get("seq"),
            "target_uid": target_uid,
            "hp_before": hp_before,
            "hp_after": hp_after,
            "max_hp": max_hp,
            "source": raw.get("source"),
        }
        if hp_before is not None and hp_after is not None:
            ev["heal_amount"] = hp_after - hp_before
        return {k: v for k, v in ev.items() if v is not None}

    # ---------------------------------------------------------------- switch
    if etype == "switch":
        return {
            "type": "switch",
            "seq": raw.get("seq"),
            "player": raw.get("player"),
            "uid": raw.get("pokemon_uid"),
            "into_uid": raw.get("into_uid"),
        }

    # ----------------------------------------------------------------- faint
    if etype == "faint":
        return {
            "type": "faint",
            "seq": raw.get("seq"),
            "target_uid": raw.get("target_uid"),
        }

    # ----------------------------------------------------------- status_start
    if etype == "status_start":
        return {
            "type": "status_start",
            "seq": raw.get("seq"),
            "target_uid": raw.get("target_uid"),
            "status": raw.get("status"),
            "source": raw.get("source"),
        }

    # ------------------------------------------------------------- status_end
    if etype == "status_end":
        return {
            "type": "status_end",
            "seq": raw.get("seq"),
            "target_uid": raw.get("target_uid"),
            "status": raw.get("status"),
        }

    # ----------------------------------------------------------- stat_change
    if etype == "stat_change":
        return {
            "type": "stat_change",
            "seq": raw.get("seq"),
            "target_uid": raw.get("target_uid"),
            "stat": raw.get("stat"),
            "amount": raw.get("amount"),
        }

    # ------------------------------------------------------------ weather
    if etype == "weather":
        return {
            "type": "weather",
            "seq": raw.get("seq"),
            "weather": raw.get("weather"),
            "is_removal": bool(raw.get("is_removal", False)),
            "source": raw.get("source"),
        }

    # ------------------------------------------------------------- field
    if etype == "field":
        return {
            "type": "field",
            "seq": raw.get("seq"),
            "condition": raw.get("condition") or raw.get("field"),
            "is_removal": bool(raw.get("is_removal", False)),
        }

    # -------------------------------------------------------- side_condition
    if etype == "side_condition":
        return {
            "type": "side_condition",
            "seq": raw.get("seq"),
            "player": raw.get("player"),
            "condition": raw.get("condition"),
            "is_removal": bool(raw.get("is_removal", False)),
        }

    # ---------------------------------------------------------- forme_change / tera
    if etype in ("forme_change", "tera", "mega", "transform", "replace"):
        ev = {
            "type": etype,
            "seq": raw.get("seq"),
            "target_uid": raw.get("target_uid") or raw.get("pokemon_uid"),
        }
        for field_name in ("species", "forme", "tera_type", "into_uid"):
            if raw.get(field_name) is not None:
                ev[field_name] = raw[field_name]
        return {k: v for k, v in ev.items() if v is not None}

    # ---------------------------------------------------------------- effect
    # Ability triggers, item effects, crit markers, misc effects.
    if etype == "effect":
        effect_type = raw.get("effect_type", "")
        raw_parts = raw.get("raw_parts") or []

        # Crit marker: record it so next damage event is tagged is_crit=True
        if effect_type == "crit":
            crit_marker.mark()
            # Still emit the event so the crit context is preserved in the log
            return {
                "type": "effect",
                "seq": raw.get("seq"),
                "effect_type": "crit",
                "raw_parts": raw_parts,
            }

        # Ability / item effect triggers
        if effect_type in ("ability", "item"):
            return {
                "type": "effect",
                "seq": raw.get("seq"),
                "effect_type": effect_type,
                "raw_parts": raw_parts,
            }

        # Other effect types (animating, display) — preserve if they carry
        # a non-empty raw_parts payload; discard pure display noise.
        if raw_parts:
            return {
                "type": "effect",
                "seq": raw.get("seq"),
                "effect_type": effect_type or "unknown",
                "raw_parts": raw_parts,
            }

        # No payload — skip
        return None

    # ---------------------------------------------------------------- revive
    if etype == "revive":
        return {
            "type": "revive",
            "seq": raw.get("seq"),
            "target_uid": raw.get("target_uid") or raw.get("pokemon_uid"),
        }

    # --------------------------------- anything else with a non-empty payload
    # Preserve unknown event types rather than silently dropping them —
    # the goal is perfect reconstruction fidelity.
    if raw:
        ev = dict(raw)
        ev.setdefault("type", etype or "unknown")
        return ev

    return None


# ---------------------------------------------------------------------------
# Per-battle archiver
# ---------------------------------------------------------------------------


def _archive_battle(battle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce the compact archive representation of one battle.

    All state-affecting information is preserved at full precision
    (exact HP integers, exact crit flags, exact stat change amounts).
    """
    battle_id = battle.get("battle_id", "unknown")
    format_id = battle.get("format_id", "unknown")
    metadata = battle.get("metadata", {}) or {}
    outcome_raw = metadata.get("outcome", {}) or {}
    total_turns = metadata.get("total_turns")
    timestamp = metadata.get("timestamp_unix")

    # Outcome
    outcome = {
        "winner": outcome_raw.get("winner"),
        "reason": outcome_raw.get("reason"),
        "result": outcome_raw.get("result"),
    }

    # Teams
    teams = _extract_team_info(battle)

    # HP tracker seeded from team revelation
    hp_tracker = _HpTracker()
    _seed_hp_tracker(hp_tracker, teams)

    crit_marker = _CritMarker()

    archived_turns: List[Dict[str, Any]] = []
    total_event_count = 0

    for turn_raw in (battle.get("turns") or []):
        turn_number = turn_raw.get("turn_number")
        raw_events = turn_raw.get("events") or []

        # Reset crit marker at turn boundary (crits don't span turns)
        crit_marker.reset()

        normalized: List[Dict[str, Any]] = []
        for raw_ev in raw_events:
            result = _normalize_event(raw_ev, hp_tracker, crit_marker)
            if result is not None:
                normalized.append(result)

        total_event_count += len(normalized)
        archived_turns.append({
            "turn_number": turn_number,
            "events": normalized,
        })

    archived = {
        "archive_version": ARCHIVE_VERSION,
        "battle_id": battle_id,
        "format": format_id,
        "timestamp_unix": timestamp,
        "total_turns": total_turns,
        "outcome": outcome,
        "teams": teams,
        "turns": archived_turns,
        "total_archived_events": total_event_count,
    }

    return archived


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _compute_win_rates(index_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win/loss/draw counts per player from index rows."""
    wins: Dict[str, int] = {}
    draws = 0
    total = len(index_rows)

    for row in index_rows:
        winner = row.get("winner")
        if winner and winner != "none":
            wins[winner] = wins.get(winner, 0) + 1
        else:
            draws += 1

    rates: Dict[str, float] = {}
    if total > 0:
        for player, count in wins.items():
            rates[player] = round(count / total, 4)

    return {
        "wins_by_player": wins,
        "draws": draws,
        "win_rates": rates,
    }


def _compute_event_stats(index_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate event count statistics across all archived battles."""
    counts = [r["total_events"] for r in index_rows if r.get("total_events") is not None]
    if not counts:
        return {}
    total = sum(counts)
    return {
        "total_events_all_battles": total,
        "mean_events_per_battle": round(total / len(counts), 2),
        "min_events": min(counts),
        "max_events": max(counts),
    }


# ---------------------------------------------------------------------------
# Main archiving pipeline
# ---------------------------------------------------------------------------


def archive_battles(
    input_paths: List[str],
    output_dir: Path,
    run_id: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Process all battle JSON files and write the archive structure.

    Parameters
    ----------
    input_paths:
        File paths or directories containing battle JSON files.
    output_dir:
        Root output directory (``runs/<run_id>/``).
    run_id:
        Identifier string for this collection run.
    model_info:
        Optional dict with model metadata (name, checkpoint, etc.)
        to embed in run_metadata.json.
    """
    json_paths = _iter_json_paths(input_paths)
    if not json_paths:
        print("[error] no JSON files found in input paths", file=sys.stderr)
        sys.exit(1)

    logs_dir = output_dir / "battle_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    archived_count = 0
    skipped_count = 0

    print(f"Archiving {len(json_paths)} battle file(s) -> {output_dir}")

    for path in json_paths:
        battle = _safe_load_json(path)
        if battle is None:
            skipped_count += 1
            continue

        # Validate schema version
        schema_ver = battle.get("schema_version", "")
        if schema_ver and not schema_ver.startswith("1."):
            print(
                f"  [warn] {path.name}: unsupported schema_version={schema_ver!r}, attempting anyway",
                file=sys.stderr,
            )

        battle_id = battle.get("battle_id") or path.stem

        try:
            archived = _archive_battle(battle)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {path.name}: extraction failed: {exc}", file=sys.stderr)
            skipped_count += 1
            continue

        # Write compact battle log
        out_path = logs_dir / f"{battle_id}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(archived, fh, separators=(",", ":"))

        # Accumulate index row
        outcome = archived.get("outcome", {})
        index_rows.append({
            "battle_id": battle_id,
            "format": archived.get("format", ""),
            "winner": outcome.get("winner") or "none",
            "reason": outcome.get("reason") or "",
            "turn_count": archived.get("total_turns") or 0,
            "total_events": archived.get("total_archived_events", 0),
        })

        archived_count += 1

    if archived_count == 0:
        print("[error] no battles were successfully archived", file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Write event_index.csv
    # ----------------------------------------------------------------
    index_path = output_dir / "event_index.csv"
    fieldnames = ["battle_id", "format", "winner", "reason", "turn_count", "total_events"]
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    # ----------------------------------------------------------------
    # Write run_metadata.json
    # ----------------------------------------------------------------
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    win_rates_info = _compute_win_rates(index_rows)
    event_stats = _compute_event_stats(index_rows)

    run_metadata: Dict[str, Any] = {
        "archive_version": ARCHIVE_VERSION,
        "run_id": run_id,
        "created_at_utc": now_iso,
        "battle_count": archived_count,
        "skipped_count": skipped_count,
        "win_rates": win_rates_info,
        "event_statistics": event_stats,
    }
    if model_info:
        run_metadata["model_info"] = model_info

    meta_path = output_dir / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"Done. Archived {archived_count} battles ({skipped_count} skipped).")
    print(f"  battle_logs/ : {logs_dir}")
    print(f"  event_index  : {index_path}")
    print(f"  run_metadata : {meta_path}")
    print(f"  total events : {event_stats.get('total_events_all_battles', 'n/a')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and archive battle events for perfect reconstruction. "
            "Produces compact JSON logs, an event index CSV, and run metadata."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "One or more paths to battle JSON files or directories "
            "containing battle JSON files (searched recursively)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for the archive (e.g. runs/entity_action_bc_v1_20260406_seq_run1/).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        metavar="RUN_ID",
        help="Identifier for this collection run, embedded in run_metadata.json.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        metavar="NAME",
        help="Optional model name to embed in run_metadata.json.",
    )
    parser.add_argument(
        "--model-checkpoint",
        default=None,
        metavar="CKPT",
        help="Optional model checkpoint path to embed in run_metadata.json.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)

    model_info: Optional[Dict[str, Any]] = None
    if args.model_name or args.model_checkpoint:
        model_info = {}
        if args.model_name:
            model_info["name"] = args.model_name
        if args.model_checkpoint:
            model_info["checkpoint"] = args.model_checkpoint

    archive_battles(
        input_paths=args.input_dir,
        output_dir=output_dir,
        run_id=args.run_id,
        model_info=model_info,
    )


if __name__ == "__main__":
    main()
