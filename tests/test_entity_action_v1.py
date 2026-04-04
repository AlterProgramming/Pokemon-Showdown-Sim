from __future__ import annotations

import unittest

from BattleStateTracker import STAT_ORDER
from EntityActionV1 import UNK_SPECIES_TOKEN, build_entity_action_graph


def make_boosts() -> dict[str, int]:
    return {stat: 0 for stat in STAT_ORDER}


class EntityActionV1Tests(unittest.TestCase):
    def test_build_entity_action_graph_creates_entities_candidates_and_edges(self) -> None:
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
                    "observed_moves": ["thunderbolt"],
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

        graph = build_entity_action_graph(
            state=state,
            perspective_player="p1",
            legal_moves=[{"id": "thunderbolt"}, {"id": "quickattack"}],
            chosen_action=("move", "thunderbolt"),
            chosen_action_token="move:thunderbolt",
        )

        self.assertEqual(graph["family_id"], "entity_action_bc")
        self.assertEqual(graph["family_version"], 1)
        self.assertEqual(graph["summary"]["pokemon_entity_count"], 12)
        self.assertEqual(graph["summary"]["move_candidate_count"], 2)
        self.assertEqual(graph["summary"]["switch_candidate_count"], 1)

        pokemon_entities = [entity for entity in graph["entities"] if entity["entity_type"] == "pokemon"]
        self.assertEqual(len(pokemon_entities), 12)
        opp_hidden = next(entity for entity in pokemon_entities if entity["id"] == "pokemon:opponent:slot2")
        self.assertEqual(opp_hidden["token_inputs"]["species"], UNK_SPECIES_TOKEN)

        move_candidates = [candidate for candidate in graph["action_candidates"] if candidate["action_type"] == "move"]
        switch_candidates = [candidate for candidate in graph["action_candidates"] if candidate["action_type"] == "switch"]
        self.assertEqual(len(move_candidates), 2)
        self.assertEqual(len(switch_candidates), 1)
        self.assertTrue(any(candidate["is_chosen"] for candidate in move_candidates))
        self.assertEqual(switch_candidates[0]["target_entity_id"], "pokemon:self:slot2")

        edge_types = {edge["edge_type"] for edge in graph["edges"]}
        self.assertIn("active_matchup", edge_types)
        self.assertIn("global_context", edge_types)
        self.assertIn("move_to_self_active", edge_types)

    def test_fallback_move_candidates_include_chosen_move(self) -> None:
        state = {
            "turn_index": 1,
            "field": {"weather": None, "global_conditions": []},
            "p1": {
                "active_uid": "p1a",
                "slots": ["p1a", None, None, None, None, None],
                "side_conditions": {},
            },
            "p2": {
                "active_uid": "p2a",
                "slots": ["p2a", None, None, None, None, None],
                "side_conditions": {},
            },
            "mons": {
                "p1a": {
                    "uid": "p1a",
                    "player": "p1",
                    "species": "Charmander",
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
                    "species": "Eevee",
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
            },
        }

        graph = build_entity_action_graph(
            state=state,
            perspective_player="p1",
            legal_moves=None,
            chosen_action=("move", "flamethrower"),
            chosen_action_token="move:flamethrower",
        )

        move_tokens = {candidate["token"] for candidate in graph["action_candidates"] if candidate["action_type"] == "move"}
        self.assertEqual(move_tokens, {"move:flamethrower"})


if __name__ == "__main__":
    unittest.main()
