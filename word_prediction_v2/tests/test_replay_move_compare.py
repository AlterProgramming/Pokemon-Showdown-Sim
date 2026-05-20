from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from word_prediction_model.replay_move_compare import analyze_replay_move_choices


class ReplayMoveCompareTests(unittest.TestCase):
    def test_analyze_replay_move_choices_reports_decoder_comparison(self) -> None:
        html_text = (
            '<script type="text/plain" class="battle-log-data">'
            '|turn|1\n'
            '|switch|p1a: Noctowl|Noctowl, L95, M|344/344\n'
            '|switch|p2a: Hitmontop|Hitmontop, L88, M|231/231\n'
            '|move|p1a: Noctowl|Hurricane|p2a: Hitmontop|[miss]\n'
            '|move|p2a: Hitmontop|Close Combat|p1a: Noctowl\n'
            '|-damage|p1a: Noctowl|164/344\n'
            '|move|p1a: Noctowl|Air Slash|p2a: Hitmontop\n'
            '|move|p2a: Hitmontop|Mach Punch|p1a: Noctowl\n'
            '|-damage|p1a: Noctowl|100/344\n'
            "</script>"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "sample.html"
            replay_path.write_text(html_text, encoding="utf-8")
            payload = analyze_replay_move_choices(replay_path, top_k=2, top_move_k=2)

        self.assertEqual(payload["decision_count"], 2)
        self.assertEqual(payload["compared_decision_count"], 2)
        self.assertIn("attack", payload["primary_word_counts"])
        self.assertEqual(payload["rows"][0]["actual_action_id"], "closecombat")
        self.assertEqual(payload["rows"][0]["decoded_action_type"], "move")
        self.assertTrue(payload["rows"][0]["top_scored_moves"])
        self.assertIsNotNone(payload["rows"][0]["actual_move_rank"])


if __name__ == "__main__":
    unittest.main()
