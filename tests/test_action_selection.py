from __future__ import annotations

import unittest

import numpy as np

from core.ActionSelection import pick_best_action


class ActionSelectionTests(unittest.TestCase):
    def test_switch_bias_can_flip_a_voluntary_choice_to_a_move(self) -> None:
        action_vocab = {
            "<UNK>": 0,
            "move:tackle": 1,
            "switch:2": 2,
        }
        logits = np.asarray([0.0, 0.2, 0.9], dtype=np.float32)
        legal_moves = [{"move": "tackle", "id": "tackle"}]
        legal_switches = [{"slot": 2}]

        best_action, _ = pick_best_action(
            action_vocab,
            logits,
            legal_moves,
            legal_switches,
            switch_logit_bias=1.0,
        )

        self.assertIsNotNone(best_action)
        self.assertEqual(best_action["type"], "move")

    def test_forced_switch_reason_ignores_switch_bias(self) -> None:
        action_vocab = {
            "<UNK>": 0,
            "move:tackle": 1,
            "switch:2": 2,
        }
        logits = np.asarray([0.0, 0.2, 0.9], dtype=np.float32)
        legal_moves = [{"move": "tackle", "id": "tackle"}]
        legal_switches = [{"slot": 2}]

        best_action, _ = pick_best_action(
            action_vocab,
            logits,
            legal_moves,
            legal_switches,
            switch_reason="force_switch",
            switch_logit_bias=1.0,
        )

        self.assertIsNotNone(best_action)
        self.assertEqual(best_action["type"], "switch")


if __name__ == "__main__":
    unittest.main()
