from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.TurnEventTokenizer import (
    PAD_ID,
    BOS_ID,
    EOS_ID,
    UNK_ID,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    event_to_composite_key,
    build_sequence_vocab,
    encode_turn_event_sequence,
    decode_turn_event_sequence,
)
from BattleStateTracker import BattleStateTracker

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BATTLE_PATH = ROOT / "data" / "gen9randombattle-2390494424.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_examples() -> list:
    battle = json.loads(SAMPLE_BATTLE_PATH.read_text(encoding="utf-8"))
    tracker = BattleStateTracker(form_change_species={"Palafin"})
    return list(tracker.iter_turn_examples(battle, player="p2"))


# ---------------------------------------------------------------------------
# Unit tests: event_to_composite_key
# ---------------------------------------------------------------------------


class TestEventToCompositeKey(unittest.TestCase):
    def test_move_event(self) -> None:
        key = event_to_composite_key(
            {"event_type": "move", "actor_side": "p2", "move_id": "thunderbolt", "target_side": "p1"}
        )
        self.assertEqual(key, "move:p2:thunderbolt:p1")

    def test_damage_event(self) -> None:
        key = event_to_composite_key(
            {"event_type": "damage", "target_side": "p1", "hp_delta_bin": -6}
        )
        self.assertEqual(key, "damage:p1:hp_bin_-6")

    def test_switch_event(self) -> None:
        key = event_to_composite_key(
            {"event_type": "switch", "actor_side": "p1", "species_id": "Pikachu", "slot_index": 3}
        )
        self.assertEqual(key, "switch:p1:Pikachu:3")

    def test_turn_end_event(self) -> None:
        key = event_to_composite_key({"event_type": "turn_end"})
        self.assertEqual(key, "turn_end")

    def test_faint_event(self) -> None:
        key = event_to_composite_key({"event_type": "faint", "target_side": "p1"})
        self.assertEqual(key, "faint:p1")

    def test_weather_event(self) -> None:
        key = event_to_composite_key({"event_type": "weather", "weather": "raindance"})
        self.assertEqual(key, "weather:raindance")

    def test_side_condition_start(self) -> None:
        key = event_to_composite_key(
            {"event_type": "side_condition", "actor_side": "p1", "side_condition": "stealthrock", "is_removal": False}
        )
        self.assertIn("start", key)
        self.assertNotIn("end", key)

    def test_side_condition_end(self) -> None:
        key = event_to_composite_key(
            {"event_type": "side_condition", "actor_side": "p1", "side_condition": "stealthrock", "is_removal": True}
        )
        self.assertIn("end", key)
        self.assertNotIn("start", key)

    def test_field_event(self) -> None:
        key = event_to_composite_key({"event_type": "field", "terrain": "electricterrain"})
        self.assertEqual(key, "field:electricterrain:start")

    def test_forme_change_tera(self) -> None:
        key = event_to_composite_key(
            {"event_type": "forme_change", "target_side": "p1", "forme_change_kind": "tera", "status": "Electric"}
        )
        self.assertEqual(key, "forme_change:p1:tera:Electric")


# ---------------------------------------------------------------------------
# Integration tests: build_sequence_vocab
# ---------------------------------------------------------------------------


class TestBuildSequenceVocab(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = _load_examples()
        cls.vocab = build_sequence_vocab(cls.examples)

    def test_special_tokens_at_start(self) -> None:
        self.assertEqual(self.vocab[PAD_TOKEN], 0)
        self.assertEqual(self.vocab[BOS_TOKEN], 1)
        self.assertEqual(self.vocab[EOS_TOKEN], 2)
        self.assertEqual(self.vocab[UNK_TOKEN], 3)

    def test_vocab_is_deterministic(self) -> None:
        vocab2 = build_sequence_vocab(self.examples)
        self.assertEqual(self.vocab, vocab2)

    def test_vocab_contains_turn_end(self) -> None:
        self.assertIn("turn_end", self.vocab)


# ---------------------------------------------------------------------------
# Unit tests: encode / decode sequence
# ---------------------------------------------------------------------------


class TestEncodeDecodeSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = _load_examples()
        cls.vocab = build_sequence_vocab(cls.examples)
        # Grab a short event list from the first example.
        cls.short_events = cls.examples[0]["turn_events_v1"]

    def test_encode_starts_with_bos(self) -> None:
        encoded = encode_turn_event_sequence(self.short_events, self.vocab, max_len=64)
        self.assertEqual(encoded[0], BOS_ID)

    def test_encode_ends_before_pad_with_eos(self) -> None:
        encoded = encode_turn_event_sequence(self.short_events, self.vocab, max_len=20)
        # Find last non-PAD element.
        last_non_pad_idx = len(encoded) - 1
        while last_non_pad_idx >= 0 and encoded[last_non_pad_idx] == PAD_ID:
            last_non_pad_idx -= 1
        self.assertGreaterEqual(last_non_pad_idx, 0, "Sequence is entirely PAD")
        self.assertEqual(encoded[last_non_pad_idx], EOS_ID)

    def test_encode_pads_to_max_len(self) -> None:
        encoded = encode_turn_event_sequence(self.short_events, self.vocab, max_len=32)
        self.assertEqual(len(encoded), 32)

    def test_encode_truncates_long_sequence(self) -> None:
        # Create 40 dummy move events to guarantee exceeding max_len.
        dummy_events = [
            {"event_type": "move", "actor_side": "p1", "move_id": "tackle", "target_side": "p2"}
        ] * 40
        encoded = encode_turn_event_sequence(dummy_events, self.vocab, max_len=10)
        self.assertEqual(len(encoded), 10)
        self.assertEqual(encoded[-1], EOS_ID)

    def test_decode_round_trip(self) -> None:
        encoded = encode_turn_event_sequence(self.short_events, self.vocab, max_len=64)
        decoded = decode_turn_event_sequence(encoded, self.vocab)
        # decoded should start with BOS_TOKEN, then the composite keys.
        # The first element is BOS_TOKEN.
        self.assertEqual(decoded[0], BOS_TOKEN)
        # The remaining decoded tokens (excluding BOS) should match composite keys.
        expected_keys = [event_to_composite_key(ev) for ev in self.short_events]
        # decoded may be truncated, so compare up to the shorter length.
        decoded_keys = decoded[1:]  # strip BOS
        compare_len = min(len(decoded_keys), len(expected_keys))
        self.assertGreater(compare_len, 0, "No event tokens decoded")
        for i in range(compare_len):
            self.assertEqual(decoded_keys[i], expected_keys[i])


# ---------------------------------------------------------------------------
# Integration tests: vectorize_multitask_dataset with include_sequence
# ---------------------------------------------------------------------------


class TestVectorizeMultitaskSequence(unittest.TestCase):
    def test_vectorize_includes_sequence_target(self) -> None:
        from StateVectorization import (
            build_action_vocab,
            build_action_context_vocab,
            iter_turn_examples_both_players,
            vectorize_multitask_dataset,
        )

        battle = json.loads(SAMPLE_BATTLE_PATH.read_text(encoding="utf-8"))
        tracker = BattleStateTracker(form_change_species={"Palafin"})
        examples = list(iter_turn_examples_both_players(tracker, battle, include_switches=True))
        vocab = build_sequence_vocab(examples)
        policy_vocab = build_action_vocab(examples, include_switches=True)
        action_context_vocab = build_action_context_vocab(examples)

        max_seq_len = 32
        _, targets = vectorize_multitask_dataset(
            examples,
            policy_vocab,
            include_switches=True,
            use_action_tokens=True,
            action_context_vocab=action_context_vocab,
            include_sequence=True,
            sequence_vocab=vocab,
            max_seq_len=max_seq_len,
        )

        self.assertIn("sequence", targets)
        sequences = targets["sequence"]
        self.assertGreater(len(sequences), 0, "No sequence targets produced")
        # Each sequence should have length == max_seq_len.
        for seq in sequences:
            self.assertEqual(len(seq), max_seq_len)


if __name__ == "__main__":
    unittest.main()
