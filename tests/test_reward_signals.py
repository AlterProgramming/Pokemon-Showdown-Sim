from __future__ import annotations

import unittest

from RewardSignals import (
    RewardConfig,
    build_move_reward_profile,
    compute_reward_components,
    reward_total_from_components,
)


def make_state(
    *,
    p1_hp: float,
    p2_hp: float,
    p1_fainted: bool = False,
    p2_fainted: bool = False,
) -> dict:
    return {
        "turn_index": 1,
        "field": {"weather": None, "global_conditions": []},
        "p1": {"active_uid": "p1-0", "slots": ["p1-0", None, None, None, None, None], "side_conditions": {}},
        "p2": {"active_uid": "p2-0", "slots": ["p2-0", None, None, None, None, None], "side_conditions": {}},
        "mons": {
            "p1-0": {
                "uid": "p1-0",
                "player": "p1",
                "hp_frac": p1_hp,
                "public_revealed": True,
                "fainted": p1_fainted,
            },
            "p2-0": {
                "uid": "p2-0",
                "player": "p2",
                "hp_frac": p2_hp,
                "public_revealed": True,
                "fainted": p2_fainted,
            },
        },
    }


class RewardSignalsTests(unittest.TestCase):
    def test_hp_ko_and_terminal_components_are_weighted(self) -> None:
        before = make_state(p1_hp=1.0, p2_hp=1.0)
        after = make_state(p1_hp=0.8, p2_hp=0.0, p2_fainted=True)
        reward_config = RewardConfig()

        components = compute_reward_components(
            before,
            after,
            ("move", "tackle"),
            "p1",
            {"tackle": {"is_offensive": True}},
            is_terminal_example=True,
            terminal_result=1.0,
        )

        self.assertAlmostEqual(components["hp_swing"], (1.0 - 0.2) / 6.0)
        self.assertEqual(components["ko_swing"], 1.0)
        self.assertEqual(components["wasted_offensive_move"], 0.0)
        self.assertEqual(components["terminal"], 1.0)
        self.assertGreater(reward_total_from_components(components, reward_config), 0.0)

    def test_wasted_offensive_move_penalty_requires_offensive_profile(self) -> None:
        before = make_state(p1_hp=1.0, p2_hp=1.0)
        after = make_state(p1_hp=1.0, p2_hp=1.0)

        offensive = compute_reward_components(
            before,
            after,
            ("move", "tackle"),
            "p1",
            {"tackle": {"is_offensive": True}},
        )
        non_offensive = compute_reward_components(
            before,
            after,
            ("move", "splash"),
            "p1",
            {"splash": {"is_offensive": False}},
        )
        switch_action = compute_reward_components(
            before,
            after,
            ("switch", "p1-1"),
            "p1",
            {"tackle": {"is_offensive": True}},
        )

        self.assertEqual(offensive["wasted_offensive_move"], 1.0)
        self.assertEqual(non_offensive["wasted_offensive_move"], 0.0)
        self.assertEqual(switch_action["wasted_offensive_move"], 0.0)

    def test_build_move_reward_profile_marks_offensive_moves_from_damage_rate(self) -> None:
        reward_config = RewardConfig(
            offensive_move_min_uses=20,
            offensive_move_min_damage_rate=0.5,
        )
        examples = []
        damaging_after = make_state(p1_hp=1.0, p2_hp=0.8)
        inert_after = make_state(p1_hp=1.0, p2_hp=1.0)
        before = make_state(p1_hp=1.0, p2_hp=1.0)

        for idx in range(20):
            examples.append(
                {
                    "player": "p1",
                    "state": before,
                    "next_state": damaging_after if idx < 10 else inert_after,
                    "action": ("move", "tackle"),
                }
            )

        profile = build_move_reward_profile(examples, reward_config)
        self.assertIn("tackle", profile)
        self.assertEqual(profile["tackle"]["uses"], 20)
        self.assertEqual(profile["tackle"]["damage_turns"], 10)
        self.assertAlmostEqual(profile["tackle"]["damage_turn_rate"], 0.5)
        self.assertTrue(profile["tackle"]["is_offensive"])


if __name__ == "__main__":
    unittest.main()
