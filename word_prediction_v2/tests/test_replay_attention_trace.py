from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from word_prediction_model.replay_attention_trace import analyze_replay_attention


class ReplayAttentionTraceTests(unittest.TestCase):
    def test_analyze_replay_attention_produces_rows(self) -> None:
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
            payload = analyze_replay_attention(replay_path, limit=1, top_k=2)

        self.assertEqual(payload["decision_count"], 1)
        self.assertEqual(payload["traced_decision_count"], 1)
        self.assertEqual(payload["rows"][0]["actual_action_id"], "closecombat")
        self.assertEqual(payload["rows"][0]["attention_report"]["tokens"], ["damage", "offense", "pressure", "strike"])


if __name__ == "__main__":
    unittest.main()
