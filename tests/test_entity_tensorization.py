from __future__ import annotations

import unittest

import numpy as np

from BattleStateTracker import STAT_ORDER
from EntityTensorization import (
    MAX_GLOBAL_CONDITIONS,
    MAX_OBSERVED_MOVES,
    build_entity_training_bundle,
    encode_entity_state,
    vectorize_entity_multitask_dataset,
)
from TurnEventTokenizer import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def make_boosts() -> dict[str, int]:
    return {stat: 0 for stat in STAT_ORDER}


def make_examples() -> list[dict]:
    state = {
        "turn_index": 3,
        "field": {
            "weather": "raindance",
            "global_conditions": ["trickroom"],
        },
        "p1": {
            "active_uid": "p1a",
            "slots": ["p1a", "p1b", None, None, None, None],
            "side_conditions": {"stealthrock": 1},
        },
        "p2": {
            "active_uid": "p2a",
            "slots": ["p2a", "p2b", None, None, None, None],
            "side_conditions": {},
        },
        "mons": {
            "p1a": {
                "uid": "p1a",
                "player": "p1",
                "species": "Pikachu",
                "hp_frac": 0.75,
                "status": None,
                "ability": "static",
                "item": "lightball",
                "tera_type": "electric",
                "terastallized": False,
                "public_revealed": True,
                "fainted": False,
                "boosts": make_boosts(),
                "observed_moves": ["thunderbolt", "volttackle"],
            },
            "p1b": {
                "uid": "p1b",
                "player": "p1",
                "species": "Bulbasaur",
                "hp_frac": 1.0,
                "status": None,
                "ability": None,
                "item": None,
                "tera_type": None,
                "terastallized": False,
                "public_revealed": True,
                "fainted": False,
                "boosts": make_boosts(),
                "observed_moves": [],
            },
            "p2a": {
                "uid": "p2a",
                "player": "p2",
                "species": "Squirtle",
                "hp_frac": 0.5,
                "status": "brn",
                "ability": None,
                "item": None,
                "tera_type": None,
                "terastallized": False,
                "public_revealed": True,
                "fainted": False,
                "boosts": make_boosts(),
                "observed_moves": ["surf"],
            },
            "p2b": {
                "uid": "p2b",
                "player": "p2",
                "species": None,
                "hp_frac": None,
                "status": None,
                "ability": None,
                "item": None,
                "tera_type": None,
                "terastallized": False,
                "public_revealed": False,
                "fainted": False,
                "boosts": make_boosts(),
                "observed_moves": [],
            },
        },
    }
    next_state = {
        **state,
        "mons": {
            **state["mons"],
            "p2a": {
                **state["mons"]["p2a"],
                "hp_frac": 0.25,
            },
        },
    }
    return [
        {
            "battle_id": "b1",
            "turn_number": 3,
            "player": "p1",
            "state": state,
            "next_state": next_state,
            "action": ("move", "thunderbolt"),
            "action_token": "move:thunderbolt",
            "opponent_action_token": "move:surf",
            "terminal_result": 1.0,
            "turn_events_v1": [
                {"event_type": "move", "actor_side": "p1", "move_id": "thunderbolt", "target_side": "p2"},
                {"event_type": "damage", "target_side": "p2", "hp_delta_bin": -2},
                {"event_type": "move", "actor_side": "p2", "move_id": "surf", "target_side": "p1"},
                {"event_type": "turn_end"},
            ],
        }
    ]


class EntityTensorizationTests(unittest.TestCase):
    def test_encode_entity_state_produces_fixed_shapes(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=True,
            include_value=True,
        )

        encoded = encode_entity_state(
            examples[0]["state"],
            perspective_player="p1",
            token_vocabs=bundle["token_vocabs"],
        )

        self.assertEqual(len(encoded["pokemon_species"]), 12)
        self.assertEqual(len(encoded["pokemon_observed_moves"]), 12)
        self.assertEqual(len(encoded["pokemon_observed_moves"][0]), MAX_OBSERVED_MOVES)
        self.assertEqual(len(encoded["global_conditions"]), MAX_GLOBAL_CONDITIONS)
        self.assertEqual(len(encoded["weather"]), 1)

    def test_vectorize_entity_multitask_dataset_tracks_policy_transition_and_value(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=True,
            include_value=True,
        )

        X, targets = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=bundle["policy_vocab"],
            token_vocabs=bundle["token_vocabs"],
            action_context_vocab=bundle["action_context_vocab"],
            include_switches=True,
            include_transition=True,
            include_value=True,
        )

        self.assertEqual(len(X["pokemon_species"]), 1)
        self.assertEqual(len(X["my_action"]), 1)
        self.assertEqual(len(targets["policy"]), 1)
        self.assertEqual(len(targets["transition"]), 1)
        self.assertEqual(targets["value"], [1.0])


    def test_bundle_includes_sequence_vocab_when_requested(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_sequence=True,
        )

        self.assertIn("sequence_vocab", bundle)
        seq_vocab = bundle["sequence_vocab"]
        # Special tokens must be present
        self.assertIn(PAD_TOKEN, seq_vocab)
        self.assertIn(BOS_TOKEN, seq_vocab)
        self.assertIn(EOS_TOKEN, seq_vocab)
        self.assertIn(UNK_TOKEN, seq_vocab)
        # Vocab must have more entries than just the 4 special tokens
        self.assertGreater(len(seq_vocab), 4)

    def test_vectorization_emits_sequence_target(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_sequence=True,
            max_seq_len=16,
        )

        X, targets = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=bundle["policy_vocab"],
            token_vocabs=bundle["token_vocabs"],
            action_context_vocab=bundle["action_context_vocab"],
            include_switches=True,
            include_transition=False,
            include_value=False,
            include_sequence=True,
            sequence_vocab=bundle["sequence_vocab"],
            max_seq_len=16,
        )

        self.assertIn("sequence", targets)
        seq_arr = np.asarray(targets["sequence"], dtype=np.int64)
        self.assertEqual(seq_arr.shape, (1, 16))
        # All token ids should be non-negative
        self.assertTrue(np.all(seq_arr >= 0))

    def test_action_context_in_X_when_sequence_without_transition(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_sequence=True,
        )

        # action_context_vocab must be built even when transition is off
        self.assertIsNotNone(bundle["action_context_vocab"])

        X, targets = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=bundle["policy_vocab"],
            token_vocabs=bundle["token_vocabs"],
            action_context_vocab=bundle["action_context_vocab"],
            include_switches=True,
            include_transition=False,
            include_value=False,
            include_sequence=True,
            sequence_vocab=bundle["sequence_vocab"],
        )

        self.assertIn("my_action", X)
        self.assertIn("opp_action", X)
        self.assertEqual(len(X["my_action"]), 1)
        self.assertEqual(len(X["opp_action"]), 1)


if __name__ == "__main__":
    unittest.main()
