from __future__ import annotations

import json
import unittest
from pathlib import Path

from BattleStateTracker import BattleStateTracker
from core.TurnEventV1 import (
    ALL_EVENT_TYPES,
    EVENT_BOOST,
    EVENT_DAMAGE,
    EVENT_FAINT,
    EVENT_FIELD,
    EVENT_HEAL,
    EVENT_MOVE,
    EVENT_SIDE_CONDITION,
    EVENT_STATUS_END,
    EVENT_STATUS_START,
    EVENT_SWITCH,
    EVENT_TURN_END,
    EVENT_TYPE_VOCAB,
    EVENT_UNBOOST,
    EVENT_WEATHER,
    EVENT_FORME_CHANGE,
    TurnEventV1,
    hp_delta_to_bin,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BATTLE_PATH = ROOT / "data" / "gen9randombattle-2390494424.json"


# ---------------------------------------------------------------------------
# Unit tests: hp_delta_to_bin
# ---------------------------------------------------------------------------


class TestHpDeltaToBin(unittest.TestCase):
    def test_hp_delta_to_bin_zero_delta(self) -> None:
        """Same before and after fraction yields 0."""
        self.assertEqual(hp_delta_to_bin(0.5, 0.5), 0)

    def test_hp_delta_to_bin_full_to_zero(self) -> None:
        """Full HP to zero is -20 (the minimum bin)."""
        self.assertEqual(hp_delta_to_bin(1.0, 0.0), -20)

    def test_hp_delta_to_bin_zero_to_full(self) -> None:
        """Zero HP to full is +20 (the maximum bin)."""
        self.assertEqual(hp_delta_to_bin(0.0, 1.0), 20)

    def test_hp_delta_to_bin_small_damage(self) -> None:
        """1.0 -> 0.72 is a delta of -0.28, which is -5.6 bins, rounded to -6."""
        self.assertEqual(hp_delta_to_bin(1.0, 0.72), -6)

    def test_hp_delta_to_bin_none_inputs(self) -> None:
        """Both None inputs return 0."""
        self.assertEqual(hp_delta_to_bin(None, None), 0)

    def test_hp_delta_to_bin_clamp(self) -> None:
        """Values that would exceed the [-20, 20] range are clamped."""
        # before=2.0, after=0.0 => delta=-2.0 => raw_bin=-40 => clamped to -20
        self.assertEqual(hp_delta_to_bin(2.0, 0.0), -20)
        # before=0.0, after=2.0 => delta=+2.0 => raw_bin=+40 => clamped to +20
        self.assertEqual(hp_delta_to_bin(0.0, 2.0), 20)


# ---------------------------------------------------------------------------
# Unit tests: TurnEventV1 dataclass
# ---------------------------------------------------------------------------


class TestTurnEventV1Dataclass(unittest.TestCase):
    def test_turn_event_to_dict_minimal(self) -> None:
        """A turn_end event with no extra fields should only contain event_type."""
        ev = TurnEventV1(event_type=EVENT_TURN_END)
        d = ev.to_dict()
        self.assertEqual(d, {"event_type": "turn_end"})

    def test_turn_event_to_dict_populated(self) -> None:
        """Non-default fields appear in the dict; default-valued fields are omitted."""
        ev = TurnEventV1(
            event_type=EVENT_MOVE,
            actor_side="p1",
            target_side="p2",
            move_id="thunderbolt",
        )
        d = ev.to_dict()
        self.assertEqual(d["event_type"], EVENT_MOVE)
        self.assertEqual(d["actor_side"], "p1")
        self.assertEqual(d["target_side"], "p2")
        self.assertEqual(d["move_id"], "thunderbolt")
        # Default-valued fields must be absent.
        self.assertNotIn("hp_delta_bin", d)
        self.assertNotIn("species_id", d)
        self.assertNotIn("boost_stat", d)
        self.assertNotIn("boost_delta", d)
        self.assertNotIn("status", d)
        self.assertNotIn("weather", d)
        self.assertNotIn("terrain", d)
        self.assertNotIn("side_condition", d)
        self.assertNotIn("slot_index", d)

    def test_turn_event_round_trip(self) -> None:
        """to_dict -> from_dict should yield an equivalent TurnEventV1."""
        original = TurnEventV1(
            event_type=EVENT_BOOST,
            target_side="p1",
            boost_stat="atk",
            boost_delta=2,
        )
        d = original.to_dict()
        restored = TurnEventV1.from_dict(d)
        self.assertEqual(restored.event_type, original.event_type)
        self.assertEqual(restored.target_side, original.target_side)
        self.assertEqual(restored.boost_stat, original.boost_stat)
        self.assertEqual(restored.boost_delta, original.boost_delta)
        # Fields not in the dict should revert to defaults.
        self.assertEqual(restored.actor_side, "")
        self.assertEqual(restored.move_id, "")
        self.assertEqual(restored.hp_delta_bin, 0)

    def test_event_type_vocab_completeness(self) -> None:
        """Every entry in ALL_EVENT_TYPES must be in EVENT_TYPE_VOCAB with a unique ID."""
        self.assertEqual(len(ALL_EVENT_TYPES), len(EVENT_TYPE_VOCAB))
        for event_type in ALL_EVENT_TYPES:
            self.assertIn(event_type, EVENT_TYPE_VOCAB)
        # All IDs must be unique.
        ids = list(EVENT_TYPE_VOCAB.values())
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# Integration tests: iter_turn_examples with turn_events_v1
# ---------------------------------------------------------------------------


def _load_sample_battle() -> dict:
    return json.loads(SAMPLE_BATTLE_PATH.read_text(encoding="utf-8"))


def _collect_examples(battle: dict, player: str = "p1") -> list:
    tracker = BattleStateTracker(form_change_species={"Palafin"})
    return list(tracker.iter_turn_examples(battle, player=player, include_switches=True))


class TestIterTurnExamplesHasTurnEventsV1(unittest.TestCase):
    def setUp(self) -> None:
        self.battle = _load_sample_battle()
        self.examples = _collect_examples(self.battle)

    def test_iter_turn_examples_has_turn_events_v1(self) -> None:
        """Every yielded example must contain a non-empty 'turn_events_v1' list."""
        self.assertGreater(len(self.examples), 0, "No examples were generated")
        for ex in self.examples:
            self.assertIn("turn_events_v1", ex)
            self.assertIsInstance(ex["turn_events_v1"], list)
            self.assertGreater(len(ex["turn_events_v1"]), 0)

    def test_turn_events_end_with_turn_end(self) -> None:
        """The last event in every turn's event list must be 'turn_end'."""
        for ex in self.examples:
            events = ex["turn_events_v1"]
            last_event = events[-1]
            self.assertEqual(
                last_event["event_type"],
                EVENT_TURN_END,
                f"Turn {ex['turn_number']}: last event is "
                f"{last_event['event_type']!r}, expected 'turn_end'",
            )

    def test_turn_events_contain_expected_types(self) -> None:
        """Across all turns we expect to see at least 'move' and 'damage' events."""
        all_types: set = set()
        for ex in self.examples:
            for ev in ex["turn_events_v1"]:
                all_types.add(ev["event_type"])

        self.assertIn(
            EVENT_MOVE, all_types, "Expected at least one 'move' event across all turns"
        )
        self.assertIn(
            EVENT_DAMAGE,
            all_types,
            "Expected at least one 'damage' event across all turns",
        )

    def test_turn_events_valid_structure(self) -> None:
        """Every event dict must be deserializable back into TurnEventV1."""
        for ex in self.examples:
            for ev_dict in ex["turn_events_v1"]:
                try:
                    ev = TurnEventV1.from_dict(ev_dict)
                except Exception as exc:
                    self.fail(
                        f"Failed to deserialize event {ev_dict!r} "
                        f"at turn {ex['turn_number']}: {exc}"
                    )
                # The event_type must be a recognized constant.
                self.assertIn(
                    ev.event_type,
                    EVENT_TYPE_VOCAB,
                    f"Unknown event_type {ev.event_type!r} at turn {ex['turn_number']}",
                )


if __name__ == "__main__":
    unittest.main()
