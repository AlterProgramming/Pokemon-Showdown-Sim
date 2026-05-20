from __future__ import annotations

import unittest

from word_prediction_model.loss_analysis import analyze_replay_files, extract_battle_log, parse_battle_log


class LossAnalysisTests(unittest.TestCase):
    def test_extract_battle_log_from_replay_html(self) -> None:
        html_text = (
            '<html><body><script type="text/plain" class="battle-log-data">'
            '|turn|1\n|win|RandomBot'
            "</script></body></html>"
        )
        self.assertEqual(extract_battle_log(html_text), "|turn|1\n|win|RandomBot")

    def test_parse_battle_log_tags_long_hazard_status_game(self) -> None:
        log_text = "\n".join(
            [
                "|turn|31",
                "|-sidestart|p1|move: Stealth Rock",
                "|-status|p1a: Pikachu|tox",
                "|move|p2a: Blastoise|Recover|p2a: Blastoise",
                "|move|p2a: Blastoise|Recover|p2a: Blastoise",
                "|move|p2a: Blastoise|Recover|p2a: Blastoise",
                "|move|p2a: Blastoise|Recover|p2a: Blastoise",
                "|win|RandomBot",
            ]
        )
        summary = parse_battle_log(log_text)
        self.assertEqual(summary["winner"], "RandomBot")
        self.assertIn("long_game", summary["tags"])
        self.assertIn("hazards_present", summary["tags"])
        self.assertIn("status_game", summary["tags"])
        self.assertIn("recovery_loop", summary["tags"])

    def test_parse_battle_log_tags_setup_spiral(self) -> None:
        log_text = "\n".join(
            [
                "|turn|1",
                "|move|p1a: Bronzong|Swords Dance|p1a: Bronzong",
                "|turn|2",
                "|move|p1a: Bronzong|Nasty Plot|p1a: Bronzong",
                "|turn|3",
                "|move|p1a: Bronzong|Calm Mind|p1a: Bronzong",
                "|turn|18",
                "|win|RandomBot",
            ]
        )
        summary = parse_battle_log(log_text)
        self.assertEqual(summary["p1_setup_moves"], 3)
        self.assertIn("setup_spiral", summary["tags"])

    def test_parse_battle_log_tags_short_tempo_loss(self) -> None:
        log_text = "\n".join(
            [
                "|turn|1",
                "|move|p1a: Crabominable|Drain Punch|p2a: Wugtrio",
                "|turn|2",
                "|move|p1a: Crabominable|Earthquake|p2a: Pachirisu",
                "|turn|3",
                "|move|p1a: Crabominable|Ice Hammer|p2a: Giratina",
                "|turn|18",
                "|win|RandomBot",
            ]
        )
        summary = parse_battle_log(log_text)
        self.assertEqual(summary["p1_setup_moves"], 0)
        self.assertEqual(summary["p1_recover_moves"], 0)
        self.assertIn("short_tempo_loss", summary["tags"])

    def test_analyze_replay_files_aggregates_tags(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            replay_path = Path(tmp_dir) / "loss.html"
            replay_path.write_text(
                '<script type="text/plain" class="battle-log-data">|turn|5\n|win|RandomBot</script>',
                encoding="utf-8",
            )
            summary = analyze_replay_files([replay_path])
            self.assertEqual(summary["loss_count"], 1)
            self.assertEqual(summary["avg_turns"], 5.0)


if __name__ == "__main__":
    unittest.main()
