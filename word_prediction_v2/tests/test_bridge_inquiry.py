from __future__ import annotations

import unittest

from word_prediction_model.battle_inquiry import sample_battle_state
from word_prediction_model.bridge_inquiry import build_bridge_response


class BridgeInquiryTests(unittest.TestCase):
    def test_bridge_response_has_primary_and_backup_words(self) -> None:
        payload = {
            "battle_state": sample_battle_state(),
            "perspective_player": "p1",
            "legal_moves": [{"move": "thunderbolt"}],
            "legal_switches": [{"slot": 2}],
        }
        response = build_bridge_response(payload, question="should I switch out here?")

        self.assertEqual(response["primary_word"], "switch")
        self.assertIn("prompt_tokens", response)
        self.assertIn("predictions", response)
        self.assertIsInstance(response["backup_words"], list)


if __name__ == "__main__":
    unittest.main()
