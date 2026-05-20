from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from word_prediction_model.replay_graph_diagnostic import analyze_replay_graph


class ReplayGraphDiagnosticTests(unittest.TestCase):
    def test_progress_edge_for_hazard_move(self) -> None:
        replay_text = "\n".join(
            [
                "|switch|p1a: Jirachi|Jirachi, L80|291/291",
                "|switch|p2a: Kricketune|Kricketune, L88, M|240/240",
                "|turn|1",
                "|move|p2a: Kricketune|Sticky Web|p1a: Jirachi",
            ]
        )
        with TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "hazard.html"
            replay_path.write_text(f"<script type=\"text/plain\" class=\"battle-log-data\">{replay_text}</script>", encoding="utf-8")
            payload = analyze_replay_graph(replay_path)

        self.assertEqual(payload["decision_count"], 1)
        self.assertEqual(payload["rows"][0]["edge_tags"], ["progress_edge"])

    def test_escape_edge_for_low_hp_switch_against_farming_opponent(self) -> None:
        replay_text = "\n".join(
            [
                "|switch|p1a: Bellossom|Bellossom, L80|263/263",
                "|switch|p2a: Carbink|Carbink, L90|50/236",
                "|turn|1",
                "|move|p1a: Bellossom|Quiver Dance|p1a: Bellossom",
                "|turn|2",
                "|switch|p2a: Charizard|Charizard, L80|240/240",
            ]
        )
        with TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "escape.html"
            replay_path.write_text(f"<script type=\"text/plain\" class=\"battle-log-data\">{replay_text}</script>", encoding="utf-8")
            payload = analyze_replay_graph(replay_path)

        self.assertEqual(payload["decision_count"], 1)
        self.assertIn("escape_edge", payload["rows"][0]["edge_tags"])

    def test_weather_engine_loss_tag_for_passive_line_under_sun_engine(self) -> None:
        replay_text = "\n".join(
            [
                "|switch|p1a: Ninetales|Ninetales, L85, F|263/263",
                "|switch|p2a: Tentacruel|Tentacruel, L84, M|272/272",
                "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Ninetales",
                "|-ability|p1a: Ninetales|Drought",
                "|turn|1",
                "|move|p2a: Recover|p2a: Tentacruel",
            ]
        )
        with TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "weather.html"
            replay_path.write_text(f"<script type=\"text/plain\" class=\"battle-log-data\">{replay_text}</script>", encoding="utf-8")
            payload = analyze_replay_graph(replay_path)

        self.assertIn("weather_engine_live", payload["rows"][0]["node_tags"])
        self.assertIn("engine_roll_edge", payload["rows"][0]["edge_tags"])
        self.assertIn("weather_engine_loss", payload["replay_tags"])


if __name__ == "__main__":
    unittest.main()
