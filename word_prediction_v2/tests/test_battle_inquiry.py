from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile

from word_prediction_model.battle_inquiry import (
    answer_battle_inquiry,
    build_battle_prompt_tokens,
    get_battle_model,
    load_inquiry_context,
    sample_battle_state,
)


class BattleInquiryTests(unittest.TestCase):
    def test_switch_question_prefers_switch_family_word(self) -> None:
        state = sample_battle_state()
        answer = answer_battle_inquiry(
            "should I switch out here?",
            state,
            legal_switches=[{"slot": 2}],
            legal_moves=[{"move": "thunderbolt"}],
            top_k=1,
        )

        self.assertEqual(answer.predictions[0].word, "switch")
        self.assertIn(answer.model_source, {"cache", "trained"})

    def test_finishing_question_prefers_finish_family_word(self) -> None:
        state = sample_battle_state()
        state["mons"]["p2a"]["hp_frac"] = 0.12
        answer = answer_battle_inquiry(
            "can I knock it out now?",
            state,
            legal_moves=[{"move": "thunderbolt"}],
            top_k=1,
        )

        self.assertEqual(answer.predictions[0].word, "finish")

    def test_priority_question_prefers_priority_family_word(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.22
        state["mons"]["p2a"]["hp_frac"] = 0.18
        answer = answer_battle_inquiry(
            "do I need priority here?",
            state,
            legal_moves=[
                {"move": "Quick Attack", "id": "quickattack", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            top_k=3,
        )

        self.assertIn(answer.predictions[0].word, {"priority", "revenge", "safeko"})

    def test_prompt_token_builder_uses_battle_state_signals(self) -> None:
        state = sample_battle_state()
        tokens = build_battle_prompt_tokens(
            "what is the safe play?",
            state,
            legal_switches=[{"slot": 2}],
        )

        self.assertIn("safe", tokens)
        self.assertTrue(any(token in tokens for token in ("pivot", "recover", "swap")))

    def test_prompt_token_builder_exposes_priority_and_safe_ko_signals(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.22
        state["mons"]["p2a"]["hp_frac"] = 0.18
        tokens = build_battle_prompt_tokens(
            "can I get a safe knockout here?",
            state,
            legal_moves=[
                {"move": "Quick Attack", "id": "quickattack", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertTrue(any(token in tokens for token in ("quick", "cleanup", "reliable", "clean", "certain")))

    def test_battle_model_is_cached_after_first_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _, first_source = get_battle_model(output_root)
            _, second_source = get_battle_model(output_root)

            self.assertEqual(first_source, "trained")
            self.assertEqual(second_source, "cache")

    def test_load_inquiry_context_accepts_normalized_bridge_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "battle_state": sample_battle_state(),
                        "perspective_player": "p2",
                        "legal_moves": [{"move": "surf"}],
                        "legal_switches": [{"slot": 2}],
                    }
                ),
                encoding="utf-8",
            )
            battle_state, perspective, legal_moves, legal_switches = load_inquiry_context(payload_path)

            self.assertEqual(perspective, "p2")
            self.assertEqual(len(legal_moves), 1)
            self.assertEqual(len(legal_switches), 1)
            self.assertIn("mons", battle_state)

    def test_risk_question_prefers_defensive_word_family(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.15
        state["mons"]["p2a"]["hp_frac"] = 0.82
        answer = answer_battle_inquiry(
            "is this too risky?",
            state,
            legal_moves=[{"move": "thunderbolt"}],
            legal_switches=[],
            top_k=3,
        )

        self.assertIn(answer.predictions[0].word, {"risk", "stabilize", "preserve", "wall"})

    def test_attention_report_is_present_for_battle_answer(self) -> None:
        state = sample_battle_state()
        answer = answer_battle_inquiry(
            "should I switch out here?",
            state,
            legal_switches=[{"slot": 2}],
            legal_moves=[{"move": "thunderbolt"}],
            top_k=2,
        )

        self.assertIsNotNone(answer.attention_report)
        assert answer.attention_report is not None
        self.assertEqual(answer.attention_report["tokens"], list(answer.prompt_tokens))
        self.assertEqual(answer.attention_report["matrix_shape"], [2, len(answer.prompt_tokens)])
        self.assertEqual(answer.attention_report["rows"][0]["word"], answer.predictions[0].word)
        self.assertAlmostEqual(sum(answer.attention_report["rows"][0]["token_weights"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
