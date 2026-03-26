from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from BattleStateTracker import BattleStateTracker
from RewardSignals import RewardConfig, build_move_reward_profile
from StateVectorization import build_action_vocab, iter_turn_examples_both_players
from replay_value_trace import generate_battle_trace_rows


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BATTLE_PATH = ROOT / "gen9randombattle-2390494424.json"


class FakePolicyValueModel:
    output_names = ["policy", "value"]

    def predict(self, state_vectors, verbose: int = 0):
        arr = np.asarray(state_vectors, dtype=np.float32)
        logits = np.stack(
            [
                arr[:, 0] + 0.1,
                arr[:, 1] + 0.2,
                arr[:, -1] + 0.3,
            ],
            axis=1,
        ).astype(np.float32)
        values = np.clip(arr.mean(axis=1, keepdims=True), 0.0, 1.0).astype(np.float32)
        return {"policy": logits, "value": values}


class ReplayValueTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = BattleStateTracker(form_change_species={"Palafin"})
        self.sample_battle = json.loads(SAMPLE_BATTLE_PATH.read_text(encoding="utf-8"))

    def test_generate_battle_trace_rows_emits_required_fields(self) -> None:
        examples = list(
            iter_turn_examples_both_players(
                self.tracker,
                self.sample_battle,
                include_switches=True,
            )
        )
        action_vocab = build_action_vocab(examples, include_switches=True)
        idx_to_token = {idx: token for token, idx in action_vocab.items() if idx < 3}
        fake_model = FakePolicyValueModel()
        reward_profile = build_move_reward_profile(
            examples,
            RewardConfig(
                offensive_move_min_uses=1,
                offensive_move_min_damage_rate=0.0,
            ),
        )

        rows = generate_battle_trace_rows(
            self.sample_battle,
            tracker=self.tracker,
            policy_value_model=fake_model,
            idx_to_token=idx_to_token,
            include_switches=True,
            label_format="action_tokens",
            top_k=3,
            reward_config=RewardConfig(),
            move_reward_profile=reward_profile,
        )

        self.assertGreater(len(rows), 0)
        first = rows[0]
        self.assertIn("battle_id", first)
        self.assertIn("turn_number", first)
        self.assertIn("player", first)
        self.assertIn("chosen_action_token", first)
        self.assertIn("policy_topk", first)
        self.assertIn("value_pre", first)
        self.assertIn("value_post", first)
        self.assertIn("value_delta", first)
        self.assertIn("reward_components", first)
        self.assertIn("reward_total", first)
        self.assertIn("cumulative_reward", first)
        self.assertIn("terminal_result", first)
        self.assertIsInstance(first["value_pre"], float)
        self.assertIsInstance(first["value_post"], float)
        self.assertIsInstance(first["value_delta"], float)
        self.assertIsInstance(first["policy_topk"], list)


if __name__ == "__main__":
    unittest.main()
