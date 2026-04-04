from __future__ import annotations

import unittest

from BattleStateTracker import STAT_ORDER
from EntityTensorization import (
    MAX_GLOBAL_CONDITIONS,
    MAX_OBSERVED_MOVES,
    build_entity_training_bundle,
    encode_entity_state,
    vectorize_entity_multitask_dataset,
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


if __name__ == "__main__":
    unittest.main()
