"""Unit tests for :mod:`core.format_adapter`.

These tests use fabricated JSONL content only — no tensorflow, no keras, no
real shard paths. They run in <1s and verify:

* sharded records are grouped by battleId and sorted by recordedAt
* a game whose result is ``tie`` is dropped entirely
* a decision flagged ``usedFallback: true`` is dropped in ``"decision"`` mode
  without killing the rest of the game
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path("/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Agents-Go-Brrrr")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.format_adapter import sharded_to_per_game


def _make_record(
    *,
    battle_id: str,
    recorded_at: str,
    turn_index: int,
    action_token: str,
    result: str,
    used_fallback: bool = False,
    model_id: str = "entity_action_v2_20260409_1811",
    perspective: str = "p1",
) -> dict:
    """Fabricate a single sharded-record dict matching the league schema."""
    return {
        "recordedAt": recorded_at,
        "perspectivePlayer": perspective,
        "requestKind": "move",
        "modelRequest": {
            "model_id": model_id,
            "battle_state": {
                "turn_index": turn_index,
                "field": {},
                "p1": {},
                "p2": {},
                "mons": [],
            },
        },
        "modelResponse": {"action_token": action_token},
        "chosenAction": "move 1",
        "usedFallback": used_fallback,
        "modelCheckpointId": model_id,
        "battleId": battle_id,
        "format": "gen9customgame@@@!Team Preview",
        "seed": [0, 0, 0, 0],
        "teamId": "team-a",
        "opponentModelId": "model1",
        "opponentTeamId": "team-b",
        "result": result,
    }


def _write_shard(tmpdir: Path, name: str, records: list[dict]) -> Path:
    path = tmpdir / name
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


class FormatAdapterTests(unittest.TestCase):
    def test_two_turns_same_battle_grouped_and_sorted(self) -> None:
        """Two sharded records for one battle should merge into one game."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            # Write records OUT OF ORDER to prove the adapter sorts by recordedAt.
            shard = _write_shard(
                tmp,
                "shard.jsonl",
                [
                    _make_record(
                        battle_id="battle-X",
                        recorded_at="2026-04-23T05:03:44.948Z",
                        turn_index=2,
                        action_token="move:thunderbolt",
                        result="win",
                    ),
                    _make_record(
                        battle_id="battle-X",
                        recorded_at="2026-04-23T05:03:44.937Z",
                        turn_index=1,
                        action_token="move:earthquake",
                        result="win",
                    ),
                ],
            )

            games, drop_counts = sharded_to_per_game([shard])

            self.assertEqual(len(games), 1)
            game = games[0]
            self.assertEqual(game["game_id"], "battle-X")
            self.assertEqual(game["model_id"], "entity_action_v2_20260409_1811")
            self.assertEqual(game["outcome"], 1.0)
            self.assertEqual(game["perspective_player"], "p1")
            self.assertEqual(len(game["decisions"]), 2)

            # Sorted ascending by recordedAt => turn 1 first, turn 2 second.
            self.assertEqual(game["decisions"][0]["turn"], 1)
            self.assertEqual(game["decisions"][1]["turn"], 2)
            self.assertEqual(
                game["decisions"][0]["modelResponse"]["action_token"],
                "move:earthquake",
            )
            self.assertEqual(
                game["decisions"][1]["modelResponse"]["action_token"],
                "move:thunderbolt",
            )
            # State_json must be the battle_state dict (not the whole modelRequest).
            self.assertIsInstance(game["decisions"][0]["state_json"], dict)
            self.assertEqual(
                game["decisions"][0]["state_json"]["turn_index"], 1
            )
            # usedFallback must survive into the emitted decision.
            self.assertFalse(game["decisions"][0]["usedFallback"])
            self.assertEqual(drop_counts["used_fallback"], 0)
            self.assertEqual(drop_counts["ties"], 0)
            self.assertEqual(drop_counts["records_seen"], 2)

    def test_tie_game_is_dropped(self) -> None:
        """A game whose result is ``tie`` must be dropped and counted."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            shard = _write_shard(
                tmp,
                "shard.jsonl",
                [
                    _make_record(
                        battle_id="battle-tie",
                        recorded_at="2026-04-23T05:03:44.937Z",
                        turn_index=1,
                        action_token="move:earthquake",
                        result="tie",
                    ),
                    _make_record(
                        battle_id="battle-tie",
                        recorded_at="2026-04-23T05:03:44.948Z",
                        turn_index=2,
                        action_token="move:thunderbolt",
                        result="tie",
                    ),
                    # Also include a winning game alongside to show tie is the
                    # only one dropped.
                    _make_record(
                        battle_id="battle-win",
                        recorded_at="2026-04-23T05:03:44.960Z",
                        turn_index=1,
                        action_token="move:surf",
                        result="win",
                    ),
                ],
            )

            games, drop_counts = sharded_to_per_game([shard])

            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]["game_id"], "battle-win")
            self.assertEqual(drop_counts["ties"], 1)
            self.assertEqual(drop_counts["used_fallback"], 0)
            self.assertEqual(drop_counts["records_seen"], 3)

    def test_fallback_decision_drops_only_that_decision(self) -> None:
        """In default ``"decision"`` mode a fallback decision is dropped in place."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            shard = _write_shard(
                tmp,
                "shard.jsonl",
                [
                    _make_record(
                        battle_id="battle-mixed",
                        recorded_at="2026-04-23T05:03:44.937Z",
                        turn_index=1,
                        action_token="move:earthquake",
                        result="win",
                        used_fallback=False,
                    ),
                    _make_record(
                        battle_id="battle-mixed",
                        recorded_at="2026-04-23T05:03:44.948Z",
                        turn_index=2,
                        action_token="move:thunderbolt",
                        result="win",
                        used_fallback=True,  # this one must be dropped
                    ),
                    _make_record(
                        battle_id="battle-mixed",
                        recorded_at="2026-04-23T05:03:44.957Z",
                        turn_index=3,
                        action_token="move:surf",
                        result="win",
                        used_fallback=False,
                    ),
                ],
            )

            games, drop_counts = sharded_to_per_game(
                [shard], drop_fallback_mode="decision"
            )

            self.assertEqual(len(games), 1)
            game = games[0]
            self.assertEqual(game["game_id"], "battle-mixed")
            # Kept turns are 1 and 3; turn 2 was dropped.
            kept_turns = [d["turn"] for d in game["decisions"]]
            self.assertEqual(kept_turns, [1, 3])
            # And the retained decisions' action tokens match, in order.
            self.assertEqual(
                [d["modelResponse"]["action_token"] for d in game["decisions"]],
                ["move:earthquake", "move:surf"],
            )
            self.assertEqual(drop_counts["used_fallback"], 1)
            self.assertEqual(drop_counts["ties"], 0)

    def test_fallback_game_mode_drops_whole_game(self) -> None:
        """In ``"game"`` mode any fallback-flagged decision drops the entire game."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            shard = _write_shard(
                tmp,
                "shard.jsonl",
                [
                    _make_record(
                        battle_id="battle-bad",
                        recorded_at="2026-04-23T05:03:44.937Z",
                        turn_index=1,
                        action_token="move:earthquake",
                        result="win",
                        used_fallback=False,
                    ),
                    _make_record(
                        battle_id="battle-bad",
                        recorded_at="2026-04-23T05:03:44.948Z",
                        turn_index=2,
                        action_token="move:thunderbolt",
                        result="win",
                        used_fallback=True,
                    ),
                ],
            )

            games, drop_counts = sharded_to_per_game(
                [shard], drop_fallback_mode="game"
            )
            self.assertEqual(games, [])
            self.assertEqual(drop_counts["used_fallback"], 1)


if __name__ == "__main__":
    unittest.main()
