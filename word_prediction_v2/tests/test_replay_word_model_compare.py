from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from word_prediction_model.replay_word_model_compare import compare_replay_word_models


class ReplayWordModelCompareTests(unittest.TestCase):
    def test_compare_replay_word_models_reports_counts(self) -> None:
        html_text = (
            '<script type="text/plain" class="battle-log-data">'
            '|turn|1\n'
            '|switch|p1a: Noctowl|Noctowl, L95, M|344/344\n'
            '|switch|p2a: Hitmontop|Hitmontop, L88, M|231/231\n'
            '|move|p1a: Noctowl|Hurricane|p2a: Hitmontop|[miss]\n'
            '|move|p2a: Hitmontop|Close Combat|p1a: Noctowl\n'
            '|-damage|p1a: Noctowl|164/344\n'
            "</script>"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "sample.html"
            replay_path.write_text(html_text, encoding="utf-8")
            payload = compare_replay_word_models(
                [replay_path],
                old_run_name="battle_inquiry_v2",
                new_run_name="battle_inquiry_v3",
                top_k=2,
                late_turn_threshold=1,
            )

        self.assertEqual(payload["replay_count"], 1)
        self.assertIn("attack", payload["overall_old_primary_counts"])
        self.assertIn("attack", payload["overall_new_primary_counts"])
        self.assertIn("changed_primary_counts", payload["replays"][0])


if __name__ == "__main__":
    unittest.main()
