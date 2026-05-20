from __future__ import annotations

import unittest

from word_prediction_model.sweep_profiles import _parse_last_summary


class SweepProfilesTests(unittest.TestCase):
    def test_parse_last_summary_extracts_final_block(self) -> None:
        output = """
noise
===== PARTIAL RESULTS =====
Completed Games: 10
RL Wins: 7
RL Win Rate: 70.00%
===========================
more noise
===== PARTIAL RESULTS =====
Completed Games: 20
RL Wins: 15
Random Wins: 5
Failed Games: 0
Timed Out Games: 0
RL Win Rate: 75.00%
Avg RL Decision Time: 12.34 ms (p95 20.00 ms, max 80.00 ms)
Avg Model Request Latency: 10.00 ms (p95 18.00 ms, max 50.00 ms)
===========================
"""
        summary = _parse_last_summary(output)
        self.assertEqual(summary["completed_games"], 20)
        self.assertEqual(summary["rl_wins"], 15)
        self.assertEqual(summary["random_wins"], 5)
        self.assertEqual(summary["rl_win_rate_percent"], 75.0)
        self.assertEqual(summary["avg_rl_decision_ms"], 12.34)


if __name__ == "__main__":
    unittest.main()
