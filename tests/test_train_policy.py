from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from train_policy import DEFAULT_KAGGLE_DATASET, build_policy_models, resolve_data_paths

try:
    from tensorflow import keras
except ModuleNotFoundError:  # pragma: no cover - depends on local training env
    keras = None


class TrainPolicyCliTests(unittest.TestCase):
    def test_resolve_data_paths_returns_explicit_paths_unchanged(self) -> None:
        paths = ["data/one", "data/two"]
        self.assertEqual(resolve_data_paths(paths), paths)

    def test_resolve_data_paths_downloads_default_kaggle_dataset(self) -> None:
        fake_kagglehub = mock.Mock()
        fake_kagglehub.dataset_download.return_value = "downloaded/path"

        with mock.patch.dict("sys.modules", {"kagglehub": fake_kagglehub}):
            self.assertEqual(resolve_data_paths([]), ["downloaded/path"])

        fake_kagglehub.dataset_download.assert_called_once_with(DEFAULT_KAGGLE_DATASET)


@unittest.skipIf(keras is None, "TensorFlow is not installed in this test environment")
class TrainPolicyBuilderTests(unittest.TestCase):
    def test_predict_value_state_only_model_fits_and_policy_artifact_stays_loadable(self) -> None:
        model, policy_model, policy_value_model = build_policy_models(
            input_dim=8,
            num_classes=4,
            hidden_dim=16,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
            predict_value=True,
            value_weight=0.25,
        )

        X = np.random.random((8, 8)).astype(np.float32)
        y = {
            "policy": np.random.randint(0, 4, size=(8,), dtype=np.int64),
            "value": np.random.random((8, 1)).astype(np.float32),
        }
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=0)

        self.assertIn("value_mae", history.history)
        self.assertIsNotNone(policy_value_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.keras"
            policy_model.save(policy_path)
            reloaded = keras.models.load_model(policy_path)
            preds = reloaded.predict(X[:1], verbose=0)
            self.assertEqual(preds.shape, (1, 4))

    def test_predict_value_and_transition_fit_together(self) -> None:
        model, _, policy_value_model = build_policy_models(
            input_dim=8,
            num_classes=5,
            hidden_dim=16,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
            transition_dim=6,
            action_context_vocab_size=9,
            action_embed_dim=8,
            transition_hidden_dim=12,
            transition_weight=0.25,
            predict_value=True,
            value_weight=0.25,
        )

        X = {
            "state": np.random.random((8, 8)).astype(np.float32),
            "my_action": np.random.randint(0, 9, size=(8,), dtype=np.int64),
            "opp_action": np.random.randint(0, 9, size=(8,), dtype=np.int64),
        }
        y = {
            "policy": np.random.randint(0, 5, size=(8,), dtype=np.int64),
            "transition": np.random.random((8, 6)).astype(np.float32),
            "value": np.random.random((8, 1)).astype(np.float32),
        }
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=0)

        self.assertIn("transition_mae", history.history)
        self.assertIn("value_brier", history.history)
        self.assertIsNotNone(policy_value_model)


if __name__ == "__main__":
    unittest.main()
