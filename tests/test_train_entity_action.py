from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from RewardSignals import RewardConfig
from train_entity_action import make_training_metadata
from train_entity_action import apply_keras_warm_start, resolve_transfer_source_metadata

try:
    from tensorflow import keras
except ModuleNotFoundError:  # pragma: no cover - depends on local training env
    keras = None


class TrainEntityActionTransferTests(unittest.TestCase):
    def test_make_training_metadata_records_switch_bias_and_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()

            artifact_paths = {
                "policy_model": artifacts_dir / "entity_model.keras",
                "policy_vocab": artifacts_dir / "policy_vocab.json",
                "entity_token_vocabs": artifacts_dir / "entity_token_vocabs.json",
                "metadata": artifacts_dir / "training_metadata_entity.json",
                "reward_profile": artifacts_dir / "reward_profile.json",
                "training_history": artifacts_dir / "training_history_entity.json",
                "epoch_metrics": artifacts_dir / "epoch_metrics_entity.csv",
                "evaluation_summary": artifacts_dir / "evaluation_summary_entity.json",
                "run_manifest": artifacts_dir / "run_manifest_entity.json",
            }
            history = SimpleNamespace(history={"loss": [1.0, 0.75]})
            args = SimpleNamespace(
                predict_turn_outcome=False,
                predict_value=False,
                move_only=False,
                policy_return_weighting="exp",
                policy_return_weight_scale=0.75,
                policy_return_weight_min=0.25,
                policy_return_weight_max=4.0,
                switch_logit_bias=0.3,
                epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                hidden_dim=64,
                depth=2,
                dropout=0.1,
                token_embed_dim=24,
                min_move_count=1,
                max_battles=0,
                val_ratio=0.2,
                seed=7,
                action_embed_dim=16,
                transition_hidden_dim=64,
                transition_weight=0.25,
                value_weight=0.25,
                value_hidden_dim=32,
            )
            metadata = make_training_metadata(
                args,
                model_name="entity_action_bc_test",
                train_size=10,
                val_size=2,
                history=history,
                artifact_paths=artifact_paths,
                policy_vocab={"<UNK>": 0, "move:tackle": 1, "switch:2": 2},
                action_context_vocab=None,
                token_vocabs={
                    "species": {"<UNK>": 0, "bulbasaur": 1},
                    "move": {"<UNK>": 0, "tackle": 1},
                },
                reward_config=RewardConfig(),
                move_reward_profile={"tackle": {"is_offensive": True}},
                policy_weight_stats={"count": 10, "mean": 1.0, "min": 0.8, "max": 1.2},
                raw_data_paths=["/tmp/raw.json"],
                resolved_data_paths=["/tmp/resolved.json"],
                json_paths=["/tmp/battles.json"],
            )

            self.assertEqual(metadata["training_regime"], "offline_entity_bc_weighted")
            self.assertEqual(metadata["policy_return_weighting"], "exp")
            self.assertAlmostEqual(metadata["switch_logit_bias"], 0.3)
            self.assertTrue(metadata["action_selection"]["voluntary_switch_only"])
            self.assertAlmostEqual(metadata["action_selection"]["switch_logit_bias"], 0.3)
            self.assertEqual(metadata["policy_weight_stats"]["count"], 10)

    def test_resolve_transfer_source_metadata_by_release_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()
            payload = {
                "family_id": "entity_action_bc",
                "family_version": 1,
                "model_release_id": "entity_action_bc_v1_20260327_run2",
                "model_name": "entity_action_bc_v1_20260327_run2",
                "training_model_path": "artifacts/entity_action_bc_v1_20260327_run2.keras",
            }
            (artifacts_dir / "training_metadata_entity_action_bc_v1_20260327_run2.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            resolved = resolve_transfer_source_metadata(
                repo_path,
                init_from_metadata=None,
                init_from_release="entity_action_bc_v1_20260327_run2",
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved["model_release_id"], payload["model_release_id"])
            self.assertIn("metadata_path", resolved)

    def test_resolve_transfer_source_metadata_by_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()
            metadata_path = artifacts_dir / "training_metadata_source.json"
            payload = {
                "family_id": "entity_action_bc",
                "family_version": 1,
                "model_release_id": "entity_action_bc_v1_source",
            }
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")

            resolved = resolve_transfer_source_metadata(
                repo_path,
                init_from_metadata="artifacts/training_metadata_source.json",
                init_from_release=None,
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved["model_release_id"], payload["model_release_id"])
            self.assertEqual(Path(resolved["metadata_path"]), metadata_path.resolve())

    def test_resolve_transfer_source_metadata_rejects_ambiguous_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            with self.assertRaises(SystemExit):
                resolve_transfer_source_metadata(
                    repo_path,
                    init_from_metadata="artifacts/source.json",
                    init_from_release="entity_action_bc_v1",
                )


@unittest.skipIf(keras is None, "TensorFlow is not installed in this test environment")
class KerasWarmStartTests(unittest.TestCase):
    def build_model(self):
        inputs = keras.Input(shape=(4,), name="features")
        hidden = keras.layers.Dense(3, activation="relu", name="dense_shared")(inputs)
        outputs = keras.layers.Dense(2, name="policy")(hidden)
        return keras.Model(inputs, outputs)

    def test_apply_keras_warm_start_copies_matching_layer_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.build_model()
            target_model = self.build_model()

            source_model(np.zeros((1, 4), dtype=np.float32))
            target_model(np.zeros((1, 4), dtype=np.float32))

            source_model.get_layer("dense_shared").set_weights(
                [
                    np.full((4, 3), 0.5, dtype=np.float32),
                    np.full((3,), 0.25, dtype=np.float32),
                ]
            )
            source_model.get_layer("policy").set_weights(
                [
                    np.full((3, 2), 0.75, dtype=np.float32),
                    np.full((2,), -0.5, dtype=np.float32),
                ]
            )
            target_model.get_layer("dense_shared").set_weights(
                [
                    np.zeros((4, 3), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                ]
            )
            target_model.get_layer("policy").set_weights(
                [
                    np.zeros((3, 2), dtype=np.float32),
                    np.zeros((2,), dtype=np.float32),
                ]
            )

            checkpoint_path = Path(tmpdir) / "source.keras"
            source_model.save(checkpoint_path)

            report = apply_keras_warm_start(
                keras=keras,
                checkpoint_path=checkpoint_path,
                target_models=[target_model],
            )

            self.assertEqual(report["applied_layer_count"], 2)
            np.testing.assert_allclose(
                target_model.get_layer("dense_shared").get_weights()[0],
                np.full((4, 3), 0.5, dtype=np.float32),
            )
            np.testing.assert_allclose(
                target_model.get_layer("policy").get_weights()[1],
                np.full((2,), -0.5, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
