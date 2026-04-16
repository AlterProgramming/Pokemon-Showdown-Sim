from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import EntityServerRuntime
from EntityServerRuntime import load_entity_runtime_artifacts, predict_entity_candidate_logits
from server import EntityServerRuntime as EntityServerRuntimeImpl


class FakeModel:
    def __init__(self) -> None:
        self.loaded_weights: Path | None = None

    def load_weights(self, path: Path) -> None:
        self.loaded_weights = Path(path)

    def predict(self, batched_inputs, verbose: int = 0):  # pragma: no cover - only used if test expands
        return np.asarray([[0.25, 0.75]], dtype=np.float32)


class FakeCallableModel(FakeModel):
    def __call__(self, batched_inputs, training: bool = False):
        self.last_inputs = batched_inputs
        return {
            "policy": np.asarray([[0.1, 0.9, -1e9]], dtype=np.float32),
            "value": np.asarray([[0.7]], dtype=np.float32),
        }


class FakeOnnxSpec:
    def __init__(self, name: str, type_name: str) -> None:
        self.name = name
        self.type = type_name


class FakeOnnxSession:
    def __init__(self) -> None:
        self.last_inputs: dict[str, np.ndarray] | None = None

    def get_inputs(self):
        return [
            FakeOnnxSpec("pokemon_species", "tensor(int32)"),
            FakeOnnxSpec("pokemon_item", "tensor(int32)"),
            FakeOnnxSpec("pokemon_ability", "tensor(int32)"),
            FakeOnnxSpec("pokemon_tera", "tensor(int32)"),
            FakeOnnxSpec("pokemon_status", "tensor(int32)"),
            FakeOnnxSpec("pokemon_side", "tensor(int32)"),
            FakeOnnxSpec("pokemon_slot", "tensor(int32)"),
            FakeOnnxSpec("pokemon_observed_moves", "tensor(int32)"),
            FakeOnnxSpec("pokemon_numeric", "tensor(float)"),
            FakeOnnxSpec("weather", "tensor(int32)"),
            FakeOnnxSpec("global_conditions", "tensor(int32)"),
            FakeOnnxSpec("global_numeric", "tensor(float)"),
            FakeOnnxSpec("candidate_type", "tensor(int32)"),
            FakeOnnxSpec("candidate_move", "tensor(int32)"),
            FakeOnnxSpec("candidate_switch_slot", "tensor(int32)"),
            FakeOnnxSpec("candidate_mask", "tensor(float)"),
        ]

    def get_outputs(self):
        return [FakeOnnxSpec("policy", "tensor(float)"), FakeOnnxSpec("value", "tensor(float)")]

    def run(self, _unused, inputs):
        self.last_inputs = inputs
        return [
            np.asarray([[0.2, 0.8, -1e9, -1e9]], dtype=np.float32),
            np.asarray([[0.65]], dtype=np.float32),
        ]


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

    def test_load_entity_action_v2_runtime_builds_v2_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()
            metadata = {
                "family_id": "entity_action_v2",
                "family_version": 1,
                "family_name": "entity_action_v2",
                "model_release_id": "entity_action_v2_test",
                "model_name": "entity_action_v2_test",
                "policy_model_path": "artifacts/entity_policy.keras",
                "training_model_path": "artifacts/training_model.keras",
                "entity_token_vocab_path": "artifacts/entity_token_vocabs.json",
                "hidden_dim": 8,
                "depth": 1,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "token_embed_dim": 4,
                "predict_value": True,
                "value_weight": 0.25,
                "max_candidates": 6,
            }
            metadata_path = artifacts_dir / "training_metadata_entity.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            (artifacts_dir / "entity_policy.keras").write_text("weights", encoding="utf-8")
            (artifacts_dir / "training_model.keras").write_text("weights", encoding="utf-8")
            token_vocabs = {
                "species": {"<UNK>": 0, "bulbasaur": 1},
                "item": {"<UNK>": 0, "berry": 1},
                "ability": {"<UNK>": 0, "overgrow": 1},
                "tera": {"<UNK>": 0, "grass": 1},
                "status": {"<UNK>": 0, "brn": 1},
                "move": {"<PAD>": 0, "<UNK>": 1, "tackle": 2},
                "weather": {"<UNK>": 0, "raindance": 1},
                "global_condition": {"<UNK>": 0, "trickroom": 1},
            }
            (artifacts_dir / "entity_token_vocabs.json").write_text(json.dumps(token_vocabs), encoding="utf-8")

            fake_training_model = FakeCallableModel()
            fake_policy_model = FakeModel()
            fake_policy_value_model = FakeModel()
            captured_kwargs: dict[str, object] = {}

            def fake_v2_builder(**kwargs):
                captured_kwargs.update(kwargs)
                return fake_training_model, fake_policy_model, fake_policy_value_model

            with mock.patch.object(
                EntityServerRuntimeImpl,
                "build_entity_action_v2_models",
                side_effect=fake_v2_builder,
            ):
                runtime = load_entity_runtime_artifacts(metadata_path, repo_path=repo_path)

            self.assertEqual(runtime["input_mode"], "entity_action_v2")
            self.assertEqual(runtime["family_id"], "entity_action_v2")
            self.assertIsNone(runtime["action_vocab"])
            self.assertEqual(captured_kwargs["max_candidates"], 6)
            self.assertTrue(runtime["has_value_head"])
            self.assertEqual(fake_training_model.loaded_weights, (repo_path / "artifacts/training_model.keras").resolve())

    def test_predict_entity_candidate_logits_encodes_legal_candidates(self) -> None:
        runtime = {
            "input_mode": "entity_action_v2",
            "token_vocabs": {
                "species": {"<PAD>": 0, "<UNK>": 1, "Pikachu": 2, "Bulbasaur": 3, "Squirtle": 4},
                "item": {"<PAD>": 0, "<UNK>": 1},
                "ability": {"<PAD>": 0, "<UNK>": 1},
                "tera": {"<PAD>": 0, "<UNK>": 1},
                "status": {"<PAD>": 0, "<UNK>": 1, "<NONE_STATUS>": 2, "brn": 3},
                "move": {"<PAD>": 0, "<UNK>": 1, "thunderbolt": 2, "quickattack": 3, "surf": 4},
                "weather": {"<PAD>": 0, "<UNK>": 1, "<NONE_WEATHER>": 2, "raindance": 3},
                "global_condition": {"<PAD>": 0, "<UNK>": 1, "trickroom": 2},
            },
            "model": FakeCallableModel(),
            "has_value_head": True,
            "max_candidates": 4,
        }
        battle_state = {
            "turn_index": 3,
            "field": {"weather": "raindance", "global_conditions": ["trickroom"]},
            "p1": {"active_uid": "p1a", "slots": ["p1a", "p1b", None, None, None, None], "side_conditions": {}},
            "p2": {"active_uid": "p2a", "slots": ["p2a", None, None, None, None, None], "side_conditions": {}},
            "mons": {
                "p1a": {"uid": "p1a", "player": "p1", "species": "Pikachu", "hp_frac": 0.75, "status": None, "ability": None, "item": None, "tera_type": None, "terastallized": False, "public_revealed": True, "fainted": False, "boosts": {}, "observed_moves": ["thunderbolt"]},
                "p1b": {"uid": "p1b", "player": "p1", "species": "Bulbasaur", "hp_frac": 1.0, "status": None, "ability": None, "item": None, "tera_type": None, "terastallized": False, "public_revealed": True, "fainted": False, "boosts": {}, "observed_moves": []},
                "p2a": {"uid": "p2a", "player": "p2", "species": "Squirtle", "hp_frac": 1.0, "status": None, "ability": None, "item": None, "tera_type": None, "terastallized": False, "public_revealed": True, "fainted": False, "boosts": {}, "observed_moves": ["surf"]},
            },
        }
        logits, candidate_tokens = predict_entity_candidate_logits(
            runtime,
            battle_state,
            "p1",
            legal_moves=[{"id": "thunderbolt"}, {"id": "quickattack"}],
            legal_switches=[{"slot": 2}],
        )

        self.assertEqual(candidate_tokens[:3], ["move:thunderbolt", "move:quickattack", "switch:2"])
        self.assertEqual(logits.shape, (3,))
        self.assertAlmostEqual(float(runtime["_last_value_estimate"]), 0.7, places=5)

    def test_predict_entity_candidate_logits_uses_onnx_runtime_for_v2(self) -> None:
        fake_onnx = FakeOnnxSession()
        runtime = {
            "input_mode": "entity_action_v2",
            "token_vocabs": {
                "species": {"<PAD>": 0, "<UNK>": 1, "Pikachu": 2, "Bulbasaur": 3, "Squirtle": 4},
                "item": {"<PAD>": 0, "<UNK>": 1},
                "ability": {"<PAD>": 0, "<UNK>": 1},
                "tera": {"<PAD>": 0, "<UNK>": 1},
                "status": {"<PAD>": 0, "<UNK>": 1, "<NONE_STATUS>": 2, "brn": 3},
                "move": {"<PAD>": 0, "<UNK>": 1, "thunderbolt": 2, "quickattack": 3, "surf": 4},
                "weather": {"<PAD>": 0, "<UNK>": 1, "<NONE_WEATHER>": 2, "raindance": 3},
                "global_condition": {"<PAD>": 0, "<UNK>": 1, "trickroom": 2},
            },
            "model": FakeCallableModel(),
            "onnx_session": fake_onnx,
            "has_value_head": True,
            "max_candidates": 4,
        }
        battle_state = {
            "turn_index": 3,
            "field": {"weather": "raindance", "global_conditions": ["trickroom"]},
            "p1": {"active_uid": "p1a", "slots": ["p1a", "p1b", None, None, None, None], "side_conditions": {}},
            "p2": {"active_uid": "p2a", "slots": ["p2a", None, None, None, None, None], "side_conditions": {}},
            "mons": {
                "p1a": {"uid": "p1a", "player": "p1", "species": "Pikachu", "hp_frac": 0.75, "status": None, "ability": None, "item": None, "tera_type": None, "terastallized": False, "public_revealed": True, "fainted": False, "boosts": {}, "observed_moves": ["thunderbolt"]},
                "p1b": {"uid": "p1b", "player": "p1", "species": "Bulbasaur", "hp_frac": 1.0, "status": None, "ability": None, "item": None, "tera_type": None, "terastallized": False, "public_revealed": True, "fainted": False, "boosts": {}, "observed_moves": []},
                "p2a": {"uid": "p2a", "player": "p2", "species": "Squirtle", "hp_frac": 1.0, "status": None, "ability": None, "item": None, "tera_type": None, "terastallized": False, "public_revealed": True, "fainted": False, "boosts": {}, "observed_moves": ["surf"]},
            },
        }

        logits, candidate_tokens = predict_entity_candidate_logits(
            runtime,
            battle_state,
            "p1",
            legal_moves=[{"id": "thunderbolt"}, {"id": "quickattack"}],
            legal_switches=[{"slot": 2}],
        )

        self.assertEqual(candidate_tokens[:3], ["move:thunderbolt", "move:quickattack", "switch:2"])
        self.assertEqual(logits.shape, (4,))
        self.assertAlmostEqual(float(runtime["_last_value_estimate"]), 0.65, places=5)
        self.assertEqual(str(fake_onnx.last_inputs["pokemon_species"].dtype), "int32")
        self.assertEqual(str(fake_onnx.last_inputs["candidate_type"].dtype), "int32")
        self.assertEqual(str(fake_onnx.last_inputs["candidate_mask"].dtype), "float32")


if __name__ == "__main__":
    unittest.main()
