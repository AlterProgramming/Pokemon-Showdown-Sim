from __future__ import annotations

import unittest
from unittest import mock

from BattleStateTracker import STAT_ORDER
from EntityTensorizationV2 import (
    ACTION_TYPE_MOVE,
    ACTION_TYPE_SWITCH,
    MAX_LEGAL_ACTIONS,
    build_entity_v2_training_bundle,
    encode_entity_state_with_candidates,
    to_single_example_entity_v2_inputs,
    vectorize_entity_v2_policy_dataset,
)


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
            "slots": ["p1a", "p1b", "p1c", None, None, None],
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
            "p1c": {
                "uid": "p1c",
                "player": "p1",
                "species": "Charizard",
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
    return [
        {
            "battle_id": "b1",
            "turn_number": 3,
            "player": "p1",
            "state": state,
            "legal_moves": [{"id": "thunderbolt"}, {"id": "quickattack"}],
            "legal_switches": [{"slot": 2}, {"slot": 3}],
            "action": ("switch", "p1b"),
            "action_token": "switch:2",
            "terminal_result": 1.0,
        }
    ]


class EntityTensorizationV2Tests(unittest.TestCase):
    def test_encode_entity_state_with_candidates_tracks_legal_actions(self) -> None:
        examples = make_examples()
        bundle = build_entity_v2_training_bundle(examples)

        encoded = encode_entity_state_with_candidates(
            examples[0]["state"],
            perspective_player="p1",
            token_vocabs=bundle["token_vocabs"],
            legal_moves=examples[0]["legal_moves"],
            legal_switches=examples[0]["legal_switches"],
            chosen_action=examples[0]["action"],
            chosen_action_token=examples[0]["action_token"],
        )

        self.assertEqual(len(encoded["candidate_type"]), MAX_LEGAL_ACTIONS)
        self.assertEqual(encoded["candidate_type"][:4], [ACTION_TYPE_MOVE, ACTION_TYPE_MOVE, ACTION_TYPE_SWITCH, ACTION_TYPE_SWITCH])
        self.assertEqual(encoded["candidate_switch_slot"][2], 2)
        self.assertEqual(encoded["candidate_switch_slot"][3], 3)
        self.assertEqual(encoded["chosen_candidate_index"], 2)
        self.assertEqual(encoded["candidate_tokens"][:4], ["move:thunderbolt", "move:quickattack", "switch:2", "switch:3"])

    def test_vectorize_entity_v2_policy_dataset_emits_candidate_targets(self) -> None:
        examples = make_examples()
        bundle = build_entity_v2_training_bundle(examples)

        X, targets = vectorize_entity_v2_policy_dataset(
            examples,
            token_vocabs=bundle["token_vocabs"],
            include_value=True,
        )

        self.assertEqual(len(X["candidate_type"]), 1)
        self.assertEqual(targets["policy"], [2])
        self.assertEqual(targets["value"], [1.0])
        self.assertEqual(X["candidate_mask"][0][:4], [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(all(value == 0.0 for value in X["candidate_mask"][0][4:]))

    def test_single_example_inputs_use_int32_for_integer_tensors(self) -> None:
        examples = make_examples()
        bundle = build_entity_v2_training_bundle(examples)
        encoded = encode_entity_state_with_candidates(
            examples[0]["state"],
            perspective_player="p1",
            token_vocabs=bundle["token_vocabs"],
            legal_moves=examples[0]["legal_moves"],
            legal_switches=examples[0]["legal_switches"],
        )

        batched = to_single_example_entity_v2_inputs(encoded)
        self.assertEqual(str(batched["pokemon_species"].dtype), "int32")
        self.assertEqual(str(batched["pokemon_observed_moves"].dtype), "int32")
        self.assertEqual(str(batched["candidate_type"].dtype), "int32")
        self.assertEqual(str(batched["candidate_switch_slot"].dtype), "int32")
        self.assertEqual(str(batched["candidate_mask"].dtype), "float32")

    def test_encode_entity_state_with_candidates_reuses_state_view_once(self) -> None:
        examples = make_examples()
        bundle = build_entity_v2_training_bundle(examples)
        state_view_calls = 0

        original_build = encode_entity_state_with_candidates.__globals__["build_entity_state_view"]

        def counting_build(*args, **kwargs):
            nonlocal state_view_calls
            state_view_calls += 1
            return original_build(*args, **kwargs)

        with mock.patch(
            "core.EntityTensorizationV2.build_entity_state_view",
            side_effect=counting_build,
        ), mock.patch(
            "core.EntityActionV1.build_entity_state_view",
            side_effect=AssertionError("candidate encoding should reuse the prebuilt state view"),
        ):
            encode_entity_state_with_candidates(
                examples[0]["state"],
                perspective_player="p1",
                token_vocabs=bundle["token_vocabs"],
                legal_moves=examples[0]["legal_moves"],
                legal_switches=examples[0]["legal_switches"],
            )

        self.assertEqual(state_view_calls, 1)


if __name__ == "__main__":
    unittest.main()
