from __future__ import annotations

import unittest

import numpy as np

from EntityInvarianceModelV1 import build_entity_invariance_models
from EntityModelV1 import GLOBAL_NUMERIC_DIM, POKEMON_NUMERIC_DIM
from EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES

try:
    from tensorflow import keras
except ModuleNotFoundError:  # pragma: no cover - depends on local training env
    keras = None


def fake_inputs(batch_size: int = 8) -> dict[str, np.ndarray]:
    current = {
        "pokemon_species": np.random.randint(0, 12, size=(batch_size, 12), dtype=np.int64),
        "pokemon_item": np.random.randint(0, 10, size=(batch_size, 12), dtype=np.int64),
        "pokemon_ability": np.random.randint(0, 10, size=(batch_size, 12), dtype=np.int64),
        "pokemon_tera": np.random.randint(0, 8, size=(batch_size, 12), dtype=np.int64),
        "pokemon_status": np.random.randint(0, 8, size=(batch_size, 12), dtype=np.int64),
        "pokemon_side": np.tile(np.array([[1] * 6 + [2] * 6], dtype=np.int64), (batch_size, 1)),
        "pokemon_slot": np.tile(np.array([[1, 2, 3, 4, 5, 6] * 2], dtype=np.int64), (batch_size, 1)),
        "pokemon_observed_moves": np.random.randint(
            0,
            20,
            size=(batch_size, 12, MAX_OBSERVED_MOVES),
            dtype=np.int64,
        ),
        "pokemon_numeric": np.random.random((batch_size, 12, POKEMON_NUMERIC_DIM)).astype(np.float32),
        "weather": np.random.randint(0, 6, size=(batch_size, 1), dtype=np.int64),
        "global_conditions": np.random.randint(0, 6, size=(batch_size, MAX_GLOBAL_CONDITIONS), dtype=np.int64),
        "global_numeric": np.random.random((batch_size, GLOBAL_NUMERIC_DIM)).astype(np.float32),
    }
    previous = {
        f"prev_{key}": np.copy(value)
        for key, value in current.items()
    }
    current.update(previous)
    return current


@unittest.skipIf(keras is None, "TensorFlow is not installed in this test environment")
class EntityInvarianceModelV1Tests(unittest.TestCase):
    def test_policy_only_invariance_model_fits(self) -> None:
        model, policy_model, policy_value_model = build_entity_invariance_models(
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
            num_policy_classes=7,
            hidden_dim=24,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
            latent_dim=12,
        )
        X = fake_inputs()
        y = np.random.randint(0, 7, size=(8,), dtype=np.int64)
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        self.assertIn("loss", history.history)
        self.assertIsNone(policy_value_model)
        preds = policy_model.predict(X, verbose=0)
        self.assertEqual(preds.shape, (8, 7))

    def test_policy_transition_value_invariance_model_fits(self) -> None:
        model, _, policy_value_model = build_entity_invariance_models(
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
            num_policy_classes=7,
            hidden_dim=24,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
            latent_dim=12,
            transition_dim=6,
            action_context_vocab_size=9,
            action_embed_dim=8,
            transition_hidden_dim=16,
            transition_weight=0.25,
            predict_value=True,
            value_weight=0.25,
        )
        X = fake_inputs()
        X["my_action"] = np.random.randint(0, 9, size=(8,), dtype=np.int64)
        X["opp_action"] = np.random.randint(0, 9, size=(8,), dtype=np.int64)
        y = {
            "policy": np.random.randint(0, 7, size=(8,), dtype=np.int64),
            "transition": np.random.random((8, 6)).astype(np.float32),
            "value": np.random.random((8, 1)).astype(np.float32),
        }
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        self.assertIn("transition_mae", history.history)
        self.assertIn("value_brier", history.history)
        self.assertIsNotNone(policy_value_model)


if __name__ == "__main__":
    unittest.main()
