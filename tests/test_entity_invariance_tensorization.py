from __future__ import annotations

import unittest

from BattleStateTracker import STAT_ORDER
from EntityInvarianceTensorization import (
    build_identity_remap,
    invariance_tensor_layout,
    vectorize_entity_invariance_dataset,
)
from EntityTensorization import PAD_TOKEN, UNK_TOKEN, build_entity_training_bundle


def make_boosts() -> dict[str, int]:
    return {stat: 0 for stat in STAT_ORDER}


def make_examples() -> list[dict]:
    state_turn1 = {
        "turn_index": 1,
        "field": {"weather": "raindance", "global_conditions": []},
        "p1": {"active_uid": "p1a", "slots": ["p1a", "p1b", None, None, None, None], "side_conditions": {}},
        "p2": {"active_uid": "p2a", "slots": ["p2a", None, None, None, None, None], "side_conditions": {}},
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
                "ability": "overgrow",
                "item": None,
                "tera_type": None,
                "terastallized": False,
                "public_revealed": True,
                "fainted": False,
                "boosts": make_boosts(),
                "observed_moves": ["gigadrain"],
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
        },
    }
    state_turn2 = {
        **state_turn1,
        "turn_index": 2,
        "mons": {
            **state_turn1["mons"],
            "p2a": {
                **state_turn1["mons"]["p2a"],
                "hp_frac": 0.25,
            },
        },
    }
    return [
        {
            "battle_id": "b1",
            "turn_number": 1,
            "player": "p1",
            "state": state_turn1,
            "next_state": state_turn2,
            "action": ("move", "thunderbolt"),
            "action_token": "move:thunderbolt",
            "opponent_action_token": "move:surf",
            "terminal_result": 1.0,
        },
        {
            "battle_id": "b1",
            "turn_number": 2,
            "player": "p1",
            "state": state_turn2,
            "next_state": state_turn2,
            "action": ("move", "volttackle"),
            "action_token": "move:volttackle",
            "opponent_action_token": "move:surf",
            "terminal_result": 1.0,
        },
    ]


class EntityInvarianceTensorizationTests(unittest.TestCase):
    def test_invariance_tensor_layout_contains_previous_turn_views(self) -> None:
        names = {entry["name"] for entry in invariance_tensor_layout()}
        self.assertIn("pokemon_species", names)
        self.assertIn("prev_pokemon_species", names)
        self.assertIn("prev_global_numeric", names)

    def test_build_identity_remap_preserves_pad_and_unknown_ids(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=True,
            include_value=True,
        )
        remap = build_identity_remap(bundle["token_vocabs"], seed=13)

        species_vocab = bundle["token_vocabs"]["species"]
        self.assertEqual(remap["species"][species_vocab[PAD_TOKEN]], species_vocab[PAD_TOKEN])
        self.assertEqual(remap["species"][species_vocab[UNK_TOKEN]], species_vocab[UNK_TOKEN])

    def test_vectorize_entity_invariance_dataset_adds_previous_turn_inputs(self) -> None:
        examples = make_examples()
        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=True,
            include_value=True,
        )

        X, targets = vectorize_entity_invariance_dataset(
            examples,
            policy_vocab=bundle["policy_vocab"],
            token_vocabs=bundle["token_vocabs"],
            action_context_vocab=bundle["action_context_vocab"],
            include_switches=True,
            include_transition=True,
            include_value=True,
            include_history=True,
            identity_regime="placeholder_id",
            placeholder_seed=13,
        )

        self.assertEqual(len(X["pokemon_species"]), 2)
        self.assertEqual(len(X["prev_pokemon_species"]), 2)
        self.assertEqual(len(X["prev_pokemon_species"][0]), 12)
        self.assertEqual(len(targets["policy"]), 2)
        self.assertEqual(targets["value"], [1.0, 1.0])
        self.assertTrue(all(value == 0 for value in X["prev_pokemon_species"][0]))
        self.assertEqual(X["prev_pokemon_species"][1], X["pokemon_species"][0])


if __name__ == "__main__":
    unittest.main()
