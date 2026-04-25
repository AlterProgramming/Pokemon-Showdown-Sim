from __future__ import annotations

import json
import unittest
from pathlib import Path

from BattleStateTracker import BattleStateTracker
from StateVectorization import (
    build_action_context_vocab,
    build_action_vocab,
    encode_turn_outcome,
    iter_turn_examples_both_players,
    turn_outcome_dim,
    vectorize_action_dataset,
    vectorize_action_transition_dataset,
    vectorize_multitask_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BATTLE_PATH = ROOT / "data" / "gen9randombattle-2390494424.json"


class BattleStateTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = BattleStateTracker(form_change_species={"Palafin"})
        self.sample_battle = json.loads(SAMPLE_BATTLE_PATH.read_text(encoding="utf-8"))

    def test_turn_one_actives_are_backfilled_from_first_move(self) -> None:
        examples = list(self.tracker.iter_turn_examples(self.sample_battle, player="p2"))
        first_example = examples[0]

        self.assertEqual(first_example["turn_number"], 1)
        self.assertEqual(first_example["state"]["p1"]["active_uid"], "p1-1")
        self.assertEqual(first_example["state"]["p2"]["active_uid"], "p2-0")

    def test_pre_turn_state_does_not_leak_voluntary_switches(self) -> None:
        examples = list(self.tracker.iter_turn_examples(self.sample_battle, player="p1"))
        turn_three = next(ex for ex in examples if ex["turn_number"] == 3)

        self.assertEqual(turn_three["state"]["p1"]["active_uid"], "p1-2")
        self.assertEqual(turn_three["state"]["p2"]["active_uid"], "p2-0")

    def test_switch_actions_are_labeled_by_stable_slot(self) -> None:
        examples = list(self.tracker.iter_turn_examples(self.sample_battle, player="p2", include_switches=True))
        turn_three = next(ex for ex in examples if ex["turn_number"] == 3)

        self.assertEqual(turn_three["action"], ("switch", "p2-1"))
        self.assertEqual(turn_three["action_token"], "switch:2")
        self.assertEqual(turn_three["opponent_action_token"], "move:double-edge")

    def test_examples_include_winner_and_terminal_result(self) -> None:
        p1_examples = list(self.tracker.iter_turn_examples(self.sample_battle, player="p1"))
        p2_examples = list(self.tracker.iter_turn_examples(self.sample_battle, player="p2"))

        self.assertEqual(p1_examples[0]["winner"], "p1")
        self.assertEqual(p2_examples[0]["winner"], "p1")
        self.assertEqual(p1_examples[0]["terminal_result"], 1.0)
        self.assertEqual(p2_examples[0]["terminal_result"], 0.0)

    def test_effects_status_end_and_tera_are_tracked(self) -> None:
        synthetic_battle = {
            "battle_id": "synthetic",
            "team_revelation": {
                "teams": {
                    "p1": [
                        {
                            "pokemon_uid": "p1-0",
                            "species": "Pikachu",
                            "base_stats": {"hp": 211},
                        }
                    ],
                    "p2": [
                        {
                            "pokemon_uid": "p2-0",
                            "species": "Pelipper",
                            "base_stats": {"hp": 244},
                        }
                    ],
                }
            },
            "turns": [
                {
                    "turn_number": 1,
                    "events": [
                        {
                            "type": "move",
                            "player": "p1",
                            "pokemon_uid": "p1-0",
                            "move_id": "thunderbolt",
                            "target_uid": "p2-0",
                        },
                        {
                            "type": "move",
                            "player": "p2",
                            "pokemon_uid": "p2-0",
                            "move_id": "scald",
                            "target_uid": "p1-0",
                        },
                        {
                            "type": "effect",
                            "effect_type": "weather",
                            "raw_parts": ["-weather", "RainDance", "[from] ability: Drizzle", "[of] p2a: Pelipper"],
                        },
                        {
                            "type": "effect",
                            "effect_type": "sidestart",
                            "raw_parts": ["-sidestart", "p1: Alice", "move: Spikes"],
                        },
                        {
                            "type": "effect",
                            "effect_type": "fieldstart",
                            "raw_parts": ["-fieldstart", "move: Electric Terrain"],
                        },
                        {
                            "type": "effect",
                            "effect_type": "ability",
                            "raw_parts": ["-ability", "p2a: Pelipper", "Drizzle"],
                        },
                        {
                            "type": "effect",
                            "effect_type": "item",
                            "raw_parts": ["-item", "p1a: Pikachu", "Air Balloon"],
                        },
                        {
                            "type": "status_start",
                            "target_uid": "p1-0",
                            "status": "brn",
                            "source": "move",
                        },
                        {
                            "type": "form_change",
                            "target_uid": "p1-0",
                            "mechanic": "terastallize",
                            "tera_type": "Ghost",
                        },
                    ],
                },
                {
                    "turn_number": 2,
                    "events": [
                        {
                            "type": "move",
                            "player": "p1",
                            "pokemon_uid": "p1-0",
                            "move_id": "voltswitch",
                            "target_uid": "p2-0",
                        },
                        {"type": "status_end", "target_uid": "p1-0", "status": "brn"},
                        {
                            "type": "effect",
                            "effect_type": "sideend",
                            "raw_parts": ["-sideend", "p1: Alice", "move: Spikes", "[of] p1a: Pikachu"],
                        },
                        {
                            "type": "effect",
                            "effect_type": "fieldend",
                            "raw_parts": ["-fieldend", "move: Electric Terrain"],
                        },
                        {
                            "type": "effect",
                            "effect_type": "enditem",
                            "raw_parts": [
                                "-enditem",
                                "p1a: Pikachu",
                                "Air Balloon",
                                "[from] move: Knock Off",
                                "[of] p2a: Pelipper",
                            ],
                        },
                    ],
                },
            ],
        }

        self.tracker.load_battle(synthetic_battle)
        self.tracker.apply_turn(synthetic_battle["turns"][0])
        after_turn_one = self.tracker.snapshot()
        p1_mon = after_turn_one["mons"]["p1-0"]
        p2_mon = after_turn_one["mons"]["p2-0"]

        self.assertEqual(after_turn_one["field"]["weather"], "raindance")
        self.assertIn("electricterrain", after_turn_one["field"]["global_conditions"])
        self.assertEqual(after_turn_one["p1"]["side_conditions"].get("spikes"), 1)
        self.assertEqual(p1_mon["status"], "brn")
        self.assertTrue(p1_mon["terastallized"])
        self.assertEqual(p1_mon["tera_type"], "Ghost")
        self.assertEqual(p1_mon["item"], "Air Balloon")
        self.assertEqual(p2_mon["ability"], "Drizzle")

        self.tracker.apply_turn(synthetic_battle["turns"][1])
        after_turn_two = self.tracker.snapshot()
        p1_after = after_turn_two["mons"]["p1-0"]

        self.assertIsNone(p1_after["status"])
        self.assertNotIn("electricterrain", after_turn_two["field"]["global_conditions"])
        self.assertNotIn("spikes", after_turn_two["p1"]["side_conditions"])
        self.assertIsNone(p1_after["item"])

    def test_action_vocab_and_joint_vectorization_include_switches(self) -> None:
        examples = list(
            iter_turn_examples_both_players(
                self.tracker,
                self.sample_battle,
                include_switches=True,
            )
        )
        action_vocab = build_action_vocab(examples, include_switches=True)
        X, y = vectorize_action_dataset(examples, action_vocab, include_switches=True)

        self.assertIn("switch:2", action_vocab)
        self.assertIn("move:double-edge", action_vocab)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(y), 37)

    def test_transition_targets_align_with_examples(self) -> None:
        examples = list(
            iter_turn_examples_both_players(
                self.tracker,
                self.sample_battle,
                include_switches=True,
            )
        )
        action_vocab = build_action_vocab(examples, include_switches=True)
        action_context_vocab = build_action_context_vocab(examples)
        X, y_policy, y_transition = vectorize_action_transition_dataset(
            examples,
            action_vocab,
            action_context_vocab,
            include_switches=True,
        )

        self.assertIn("<NONE>", action_context_vocab)
        self.assertEqual(len(X["state"]), len(y_policy))
        self.assertEqual(len(y_policy), len(y_transition))
        self.assertEqual(len(y_transition[0]), turn_outcome_dim())

        first = examples[0]
        encoded = encode_turn_outcome(first["state"], first["next_state"], first["player"])
        self.assertEqual(len(encoded), turn_outcome_dim())
        self.assertEqual(encoded, y_transition[0])

    def test_multitask_vectorization_emits_value_targets(self) -> None:
        examples = list(
            iter_turn_examples_both_players(
                self.tracker,
                self.sample_battle,
                include_switches=True,
            )
        )
        action_vocab = build_action_vocab(examples, include_switches=True)
        action_context_vocab = build_action_context_vocab(examples)
        X, targets = vectorize_multitask_dataset(
            examples,
            action_vocab,
            include_switches=True,
            use_action_tokens=True,
            action_context_vocab=action_context_vocab,
            include_transition=True,
            include_value=True,
        )

        self.assertIn("state", X)
        self.assertIn("policy", targets)
        self.assertIn("transition", targets)
        self.assertIn("value", targets)
        self.assertEqual(len(X["state"]), len(targets["policy"]))
        self.assertEqual(len(targets["policy"]), len(targets["transition"]))
        self.assertEqual(len(targets["policy"]), len(targets["value"]))
        self.assertIn(1.0, targets["value"])
        self.assertIn(0.0, targets["value"])


if __name__ == "__main__":
    unittest.main()
