from __future__ import annotations

import unittest

from word_prediction_model.attention_loss_analysis import analyze_loss_attention


class AttentionLossAnalysisTests(unittest.TestCase):
    def test_analyze_loss_attention_aggregates_attention_by_tag(self) -> None:
        summary = {
            "losses": [
                {
                    "file": "game-1.html",
                    "tags": ["status_game", "recovery_loop"],
                    "turns": 34,
                    "p1_status_events": 1,
                    "p2_status_events": 2,
                    "p1_recover_moves": 4,
                    "p1_setup_moves": 0,
                    "p2_recover_moves": 1,
                },
                {
                    "file": "game-2.html",
                    "tags": ["setup_spiral"],
                    "turns": 22,
                    "p1_status_events": 0,
                    "p2_status_events": 0,
                    "p1_recover_moves": 0,
                    "p1_setup_moves": 5,
                    "p2_recover_moves": 0,
                },
            ]
        }

        result = analyze_loss_attention(summary, top_k=2)

        self.assertEqual(result["loss_count"], 2)
        self.assertEqual(len(result["losses"]), 2)
        self.assertIn("status_game", result["tag_attention_focus"])
        self.assertIn("setup_spiral", result["tag_attention_focus"])
        self.assertTrue(result["losses"][0]["prompt_tokens"])
        self.assertEqual(result["losses"][0]["primary_word"], result["losses"][0]["attention_row"]["word"])


if __name__ == "__main__":
    unittest.main()
