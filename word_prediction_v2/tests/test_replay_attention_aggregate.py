from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from word_prediction_model.replay_attention_aggregate import aggregate_replay_attention


class ReplayAttentionAggregateTests(unittest.TestCase):
    def test_aggregate_counts_primary_words(self) -> None:
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
            payload = aggregate_replay_attention([replay_path], top_k=2, late_turn_threshold=1)

        self.assertEqual(payload["replay_count"], 1)
        self.assertEqual(payload["overall_primary_counts"]["attack"], 1)
        self.assertEqual(payload["overall_late_primary_counts"]["attack"], 1)


if __name__ == "__main__":
    unittest.main()
