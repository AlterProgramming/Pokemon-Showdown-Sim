from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import EntityServerRuntime
from EntityServerRuntime import load_entity_runtime_artifacts
from server import EntityServerRuntime as EntityServerRuntimeImpl


class FakeModel:
    def __init__(self) -> None:
        self.loaded_weights: Path | None = None

    def load_weights(self, path: Path) -> None:
        self.loaded_weights = Path(path)

    def predict(self, batched_inputs, verbose: int = 0):  # pragma: no cover - only used if test expands
        return np.asarray([[0.25, 0.75]], dtype=np.float32)


class EntityServerRuntimeTests(unittest.TestCase):
    def write_metadata(self, repo_path: Path, metadata: dict[str, object]) -> Path:
        artifacts_dir = repo_path / "artifacts"
        artifacts_dir.mkdir()
        metadata_path = artifacts_dir / "training_metadata_entity.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (artifacts_dir / "entity_policy.keras").write_text("weights", encoding="utf-8")
        (artifacts_dir / "policy_vocab.json").write_text(json.dumps({"<UNK>": 0, "move:foo": 1}), encoding="utf-8")
        token_vocabs = {
            "species": {"<UNK>": 0, "bulbasaur": 1},
            "item": {"<UNK>": 0, "berry": 1},
            "ability": {"<UNK>": 0, "overgrow": 1},
            "tera": {"<UNK>": 0, "grass": 1},
            "status": {"<UNK>": 0, "brn": 1},
            "move": {"<UNK>": 0, "tackle": 1},
            "weather": {"<UNK>": 0, "raindance": 1},
            "global_condition": {"<UNK>": 0, "trickroom": 1},
        }
        (artifacts_dir / "entity_token_vocabs.json").write_text(json.dumps(token_vocabs), encoding="utf-8")
        return metadata_path

    def test_load_entity_action_runtime_builds_action_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            metadata_path = self.write_metadata(
                repo_path,
                {
                    "family_id": "entity_action_bc",
                    "family_version": 1,
                    "family_name": "entity_action_bc_v1",
                    "model_release_id": "entity_action_bc_test",
                    "model_name": "entity_action_bc_test",
                    "policy_model_path": "artifacts/entity_policy.keras",
                    "policy_vocab_path": "artifacts/policy_vocab.json",
                    "entity_token_vocab_path": "artifacts/entity_token_vocabs.json",
                    "num_action_classes": 2,
                    "hidden_dim": 8,
                    "depth": 1,
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "token_embed_dim": 4,
                    "action_selection": {"switch_logit_bias": 0.45},
                },
            )

            fake_model = FakeModel()
            captured_kwargs: dict[str, object] = {}

            def fake_builder(**kwargs):
                captured_kwargs.update(kwargs)
                return None, fake_model, None

            with mock.patch.object(EntityServerRuntimeImpl, "build_entity_action_models", side_effect=fake_builder), mock.patch.object(
                EntityServerRuntimeImpl,
                "build_entity_invariance_models",
                side_effect=AssertionError("invariance builder should not be called"),
            ):
                runtime = load_entity_runtime_artifacts(metadata_path, repo_path=repo_path)

            self.assertEqual(runtime["kind"], "entity")
            self.assertEqual(runtime["input_mode"], "entity_action")
            self.assertEqual(runtime["model_id"], "entity_action_bc_test")
            self.assertEqual(fake_model.loaded_weights, (repo_path / "artifacts/entity_policy.keras").resolve())
            self.assertEqual(captured_kwargs["num_policy_classes"], 2)
            self.assertEqual(captured_kwargs["token_embed_dim"], 4)
            self.assertEqual(runtime["family_id"], "entity_action_bc")
            self.assertEqual(runtime["policy_vocab_path"], str((repo_path / "artifacts/policy_vocab.json").resolve()))
            self.assertAlmostEqual(runtime["switch_logit_bias"], 0.45)

    def test_load_entity_invariance_runtime_builds_invariance_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            metadata_path = self.write_metadata(
                repo_path,
                {
                    "family_id": "entity_invariance_aux",
                    "family_version": 1,
                    "family_name": "entity_invariance_aux_v1",
                    "model_release_id": "entity_invariance_aux_test",
                    "model_name": "entity_invariance_aux_test",
                    "policy_model_path": "artifacts/entity_policy.keras",
                    "policy_vocab_path": "artifacts/policy_vocab.json",
                    "entity_token_vocab_path": "artifacts/entity_token_vocabs.json",
                    "num_action_classes": 2,
                    "hidden_dim": 8,
                    "depth": 1,
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "token_embed_dim": 4,
                    "latent_dim": 12,
                },
            )

            fake_model = FakeModel()
            captured_kwargs: dict[str, object] = {}

            def fake_builder(**kwargs):
                captured_kwargs.update(kwargs)
                return None, fake_model, None

            with mock.patch.object(
                EntityServerRuntimeImpl,
                "build_entity_action_models",
                side_effect=AssertionError("action builder should not be called"),
            ), mock.patch.object(EntityServerRuntimeImpl, "build_entity_invariance_models", side_effect=fake_builder):
                runtime = load_entity_runtime_artifacts(metadata_path, repo_path=repo_path)

            self.assertEqual(runtime["kind"], "entity")
            self.assertEqual(runtime["input_mode"], "entity_invariance")
            self.assertEqual(runtime["model_id"], "entity_invariance_aux_test")
            self.assertEqual(fake_model.loaded_weights, (repo_path / "artifacts/entity_policy.keras").resolve())
            self.assertEqual(captured_kwargs["latent_dim"], 12)
            self.assertEqual(runtime["family_id"], "entity_invariance_aux")
            self.assertAlmostEqual(runtime["switch_logit_bias"], 0.0)


if __name__ == "__main__":
    unittest.main()
