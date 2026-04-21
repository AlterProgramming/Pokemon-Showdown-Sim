from __future__ import annotations

import unittest

from core.SequencePlanning import (
    combine_policy_and_auxiliary_scores,
    decode_greedy_sequence_tokens,
    score_sequence_tokens,
)


class SequencePlanningTests(unittest.TestCase):
    def test_decode_greedy_sequence_tokens_stops_at_eos(self) -> None:
        reverse_vocab = {
            0: "<PAD>",
            1: "<BOS>",
            2: "<EOS>",
            3: "move:p1:tackle:p2",
            4: "damage:p2:hp_bin_-4",
        }
        sequence_probs = [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]

        decoded = decode_greedy_sequence_tokens(sequence_probs, reverse_vocab)

        self.assertEqual(decoded, ["move:p1:tackle:p2", "damage:p2:hp_bin_-4"])

    def test_score_sequence_tokens_rewards_good_outcomes_for_p1(self) -> None:
        tokens = [
            "damage:p2:hp_bin_-4",
            "boost:p1:spa:2",
            "faint:p2",
        ]

        score = score_sequence_tokens(tokens, perspective_player="p1")

        self.assertGreater(score, 0.0)

    def test_score_sequence_tokens_penalizes_bad_outcomes_for_p1(self) -> None:
        tokens = [
            "damage:p1:hp_bin_-5",
            "status_start:p1:psn",
            "faint:p1",
        ]

        score = score_sequence_tokens(tokens, perspective_player="p1")

        self.assertLess(score, 0.0)

    def test_combine_policy_and_auxiliary_scores_adds_centered_value_and_sequence(self) -> None:
        combined = combine_policy_and_auxiliary_scores(
            policy_logit=0.25,
            value_prediction=0.9,
            sequence_score=1.5,
            auxiliary_scale=0.5,
            value_scale=1.0,
            sequence_scale=0.25,
        )

        self.assertAlmostEqual(combined, 0.25 + 0.5 * 0.4 + 0.5 * 0.25 * 1.5)


if __name__ == "__main__":
    unittest.main()
