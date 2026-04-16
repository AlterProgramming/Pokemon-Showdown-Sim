from __future__ import annotations

import unittest

import numpy as np

from BattleStateTracker import STAT_ORDER
from StateVectorization import SIDE_CONDITION_ORDER
from EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES
from EntityTensorizationV2 import MAX_LEGAL_ACTIONS

try:
    from tensorflow import keras
    from EntityModelV2 import build_entity_action_v2_models
except ModuleNotFoundError:  # pragma: no cover - depends on local training env
    keras = None
    build_entity_action_v2_models = None


POKEMON_NUMERIC_DIM = 6 + len(STAT_ORDER)
GLOBAL_NUMERIC_DIM = 1 + 2 * len(SIDE_CONDITION_ORDER)


def fake_inputs(batch_size: int = 8) -> dict[str, np.ndarray]:
    candidate_mask = np.zeros((batch_size, MAX_LEGAL_ACTIONS), dtype=np.float32)
    candidate_mask[:, :4] = 1.0
    candidate_type = np.zeros((batch_size, MAX_LEGAL_ACTIONS), dtype=np.int64)
    candidate_type[:, 0:2] = 1
    candidate_type[:, 2:4] = 2
    candidate_switch_slot = np.zeros((batch_size, MAX_LEGAL_ACTIONS), dtype=np.int64)
    candidate_switch_slot[:, 2] = 2
    candidate_switch_slot[:, 3] = 3
    return {
        "pokemon_species": np.random.randint(0, 12, size=(batch_size, 12), dtype=np.int64),
        "pokemon_item": np.random.randint(0, 10, size=(batch_size, 12), dtype=np.int64),
        "pokemon_ability": np.random.randint(0, 10, size=(batch_size, 12), dtype=np.int64),
        "pokemon_tera": np.random.randint(0, 8, size=(batch_size, 12), dtype=np.int64),
        "pokemon_status": np.random.randint(0, 8, size=(batch_size, 12), dtype=np.int64),
        "pokemon_side": np.tile(np.array([[1] * 6 + [2] * 6], dtype=np.int64), (batch_size, 1)),
        "pokemon_slot": np.tile(np.array([[1, 2, 3, 4, 5, 6] * 2], dtype=np.int64), (batch_size, 1)),
        "pokemon_observed_moves": np.random.randint(0, 20, size=(batch_size, 12, MAX_OBSERVED_MOVES), dtype=np.int64),
        "pokemon_numeric": np.random.random((batch_size, 12, POKEMON_NUMERIC_DIM)).astype(np.float32),
        "weather": np.random.randint(0, 6, size=(batch_size, 1), dtype=np.int64),
        "global_conditions": np.random.randint(0, 6, size=(batch_size, MAX_GLOBAL_CONDITIONS), dtype=np.int64),
        "global_numeric": np.random.random((batch_size, GLOBAL_NUMERIC_DIM)).astype(np.float32),
        "candidate_type": candidate_type,
        "candidate_move": np.random.randint(0, 20, size=(batch_size, MAX_LEGAL_ACTIONS), dtype=np.int64),
        "candidate_switch_slot": candidate_switch_slot,
        "candidate_mask": candidate_mask,
    }


@unittest.skipIf(keras is None, "TensorFlow is not installed in this test environment")
class EntityModelV2Tests(unittest.TestCase):
    def test_policy_only_v2_model_fits(self) -> None:
        model, policy_model, policy_value_model = build_entity_action_v2_models(
            vocab_sizes={
                "species": 12,
                "item": 10,
                "ability": 10,
                "tera": 8,
                "status": 8,
                "move": 20,
                "weather": 6,
                "global_condition": 6,
            },
            hidden_dim=24,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
        )
        X = fake_inputs()
        y = np.random.randint(0, 4, size=(8,), dtype=np.int64)
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        self.assertIn("loss", history.history)
        self.assertIsNone(policy_value_model)
        preds = policy_model.predict(X, verbose=0)
        self.assertEqual(preds.shape, (8, MAX_LEGAL_ACTIONS))
        self.assertTrue(np.all(preds[:, 4:] < -1e8))

    def test_policy_value_v2_model_fits(self) -> None:
        model, _, policy_value_model = build_entity_action_v2_models(
            vocab_sizes={
                "species": 12,
                "item": 10,
                "ability": 10,
                "tera": 8,
                "status": 8,
                "move": 20,
                "weather": 6,
                "global_condition": 6,
            },
            hidden_dim=24,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
            predict_value=True,
            value_weight=0.25,
        )
        X = fake_inputs()
        y = {
            "policy": np.random.randint(0, 4, size=(8,), dtype=np.int64),
            "value": np.random.random((8, 1)).astype(np.float32),
        }
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        self.assertIn("value_brier", history.history)
        self.assertIsNotNone(policy_value_model)


if __name__ == "__main__":
    unittest.main()
