from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ModelRegistry import (
    REGISTRY_FILENAME,
    build_model_registry,
    model_id_from_name,
    parse_model_id_list,
    resolve_artifact_path,
    select_registered_models,
    write_model_registry,
)


class ModelRegistryTests(unittest.TestCase):
    def test_model_id_from_name_handles_numbered_and_large_models(self) -> None:
        self.assertEqual(model_id_from_name("model_4"), "model4")
        self.assertEqual(model_id_from_name("model_2_large"), "model2_large")
        self.assertEqual(model_id_from_name("model"), "model")

    def test_parse_model_id_list_deduplicates_and_preserves_order(self) -> None:
        self.assertEqual(
            parse_model_id_list("model4, model2_large, model4, , model3"),
            ["model4", "model2_large", "model3"],
        )

    def test_select_registered_models_supports_multi_subset(self) -> None:
        registered = {"model2": {}, "model4": {}, "model4_large": {}}
        selected = select_registered_models(
            registered,
            mode="multi",
            requested_model_ids=["model4", "model4_large"],
        )
        self.assertEqual(selected, ["model4", "model4_large"])

    def test_select_registered_models_rejects_subset_for_single_mode(self) -> None:
        registered = {"model2": {}, "model4": {}}
        with self.assertRaises(ValueError):
            select_registered_models(
                registered,
                mode="model4",
                requested_model_ids=["model4"],
            )

    def test_resolve_artifact_path_prefers_repo_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()
            metadata_path = artifacts_dir / "training_metadata_4.json"
            metadata_path.write_text("{}", encoding="utf-8")
            model_path = artifacts_dir / "model_4.keras"
            model_path.write_text("placeholder", encoding="utf-8")

            resolved = resolve_artifact_path(repo_path, metadata_path, "artifacts/model_4.keras")
            self.assertEqual(resolved, model_path.resolve())

    def test_build_and_write_registry_from_training_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()

            (artifacts_dir / "model_1.keras").write_text("m1", encoding="utf-8")
            (artifacts_dir / "action_vocab_1.json").write_text("{}", encoding="utf-8")
            (artifacts_dir / "model_2_large.keras").write_text("m2", encoding="utf-8")
            (artifacts_dir / "action_vocab_2_large.json").write_text("{}", encoding="utf-8")

            metadata_1 = {
                "policy_model_path": "artifacts/model_1.keras",
                "policy_vocab_path": "artifacts/action_vocab_1.json",
                "model_name": "model_1",
                "model_variant": "default",
                "predict_value": False,
                "predict_turn_outcome": False,
                "feature_dim": 582,
                "num_action_classes": 357,
            }
            metadata_2_large = {
                "policy_model_path": "artifacts/model_2_large.keras",
                "policy_vocab_path": "artifacts/action_vocab_2_large.json",
                "model_name": "model_2_large",
                "model_variant": "model_2_large",
                "predict_value": True,
                "predict_turn_outcome": True,
                "feature_dim": 582,
                "num_action_classes": 357,
            }

            (artifacts_dir / "training_metadata_1.json").write_text(
                json.dumps(metadata_1),
                encoding="utf-8",
            )
            (artifacts_dir / "training_metadata_2_large.json").write_text(
                json.dumps(metadata_2_large),
                encoding="utf-8",
            )

            registry = build_model_registry(repo_path)
            self.assertEqual(registry["default_model_id"], "model1")
            self.assertEqual(sorted(registry["models"].keys()), ["model1", "model2_large"])
            self.assertTrue(registry["models"]["model2_large"]["predict_value"])

            registry_path = write_model_registry(repo_path)
            self.assertEqual(registry_path, artifacts_dir / REGISTRY_FILENAME)
            persisted = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(persisted["models"].keys()), ["model1", "model2_large"])


if __name__ == "__main__":
    unittest.main()
