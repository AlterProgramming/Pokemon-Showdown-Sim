from __future__ import annotations

import unittest
from pathlib import Path

from TransferLearning import (
    ReleaseIdentity,
    apply_transfer_metadata,
    build_initialization_source,
    collect_artifact_paths,
    describe_transfer,
    resolve_artifact_path,
    same_family,
    same_generation,
    select_primary_checkpoint_path,
)


class TransferLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target_metadata = {
            "family_id": "entity_action_bc",
            "family_version": 1,
            "model_release_id": "entity_action_bc_v1_20260327_run3",
            "model_name": "entity_action_bc_v1_20260327_run3",
        }

    def test_release_identity_round_trip(self) -> None:
        identity = ReleaseIdentity.from_metadata(self.target_metadata)
        self.assertEqual(identity.family_id, "entity_action_bc")
        self.assertEqual(identity.family_version, 1)
        self.assertEqual(identity.model_release_id, "entity_action_bc_v1_20260327_run3")
        self.assertEqual(identity.to_metadata()["family_version"], 1)

    def test_collect_artifact_paths_ignores_empty_values(self) -> None:
        paths = collect_artifact_paths(
            {
                "policy_model_path": "artifacts/model.keras",
                "training_model_path": "",
                "reward_profile_path": None,
            }
        )
        self.assertEqual(paths, {"policy_model_path": "artifacts/model.keras"})

    def test_resolve_artifact_path_handles_relative_and_absolute_paths(self) -> None:
        base_dir = Path("C:/tmp/project")
        rel = resolve_artifact_path(base_dir, "artifacts/model.keras")
        abs_path = resolve_artifact_path(base_dir, "C:/tmp/model.keras")
        self.assertEqual(rel, Path("C:/tmp/project/artifacts/model.keras"))
        self.assertEqual(abs_path, Path("C:/tmp/model.keras"))

    def test_select_primary_checkpoint_path_prefers_training_model(self) -> None:
        checkpoint = select_primary_checkpoint_path(
            {
                "policy_model_path": "artifacts/policy.keras",
                "training_model_path": "artifacts/training.keras",
            }
        )
        self.assertEqual(checkpoint, Path("artifacts/training.keras"))

    def test_same_family_and_generation(self) -> None:
        source_metadata = {
            "family_id": "entity_action_bc",
            "family_version": 1,
            "model_release_id": "entity_action_bc_v1_20260327_run2",
        }
        self.assertTrue(same_family(self.target_metadata, source_metadata))
        self.assertTrue(same_generation(self.target_metadata, source_metadata))

    def test_build_initialization_source_for_scratch(self) -> None:
        init = build_initialization_source(self.target_metadata)
        self.assertEqual(init["mode"], "scratch")
        self.assertEqual(init["relationship"], "none")
        self.assertIsNone(init["source"])
        self.assertEqual(init["target"]["model_release_id"], self.target_metadata["model_release_id"])

    def test_build_initialization_source_for_same_generation_warm_start(self) -> None:
        source_metadata = {
            "family_id": "entity_action_bc",
            "family_version": 1,
            "model_release_id": "entity_action_bc_v1_20260327_run2",
            "training_model_path": "artifacts/entity_action_bc_v1_20260327_run2.keras",
            "policy_model_path": "artifacts/entity_action_bc_v1_20260327_run2.keras",
        }
        init = build_initialization_source(
            self.target_metadata,
            source_metadata=source_metadata,
        )
        self.assertEqual(init["mode"], "transfer")
        self.assertEqual(init["relationship"], "same_generation_warm_start")
        self.assertEqual(init["source"]["model_release_id"], "entity_action_bc_v1_20260327_run2")
        self.assertEqual(Path(init["checkpoint_path"]), Path("artifacts/entity_action_bc_v1_20260327_run2.keras"))
        self.assertEqual(init["artifact_key"], "training_model_path")

    def test_build_initialization_source_for_cross_family_transfer(self) -> None:
        source_metadata = {
            "family_id": "vector_joint_bc_transition_value",
            "family_version": 1,
            "model_release_id": "model4",
            "policy_value_model_path": "artifacts/policy_value_model_4.keras",
        }
        init = build_initialization_source(
            self.target_metadata,
            source_metadata=source_metadata,
        )
        self.assertEqual(init["mode"], "transfer")
        self.assertEqual(init["relationship"], "cross_family_transfer")
        self.assertEqual(Path(init["checkpoint_path"]), Path("artifacts/policy_value_model_4.keras"))
        self.assertEqual(init["artifact_key"], "policy_value_model_path")

    def test_apply_transfer_metadata_fills_parent_and_initialization_source(self) -> None:
        source_metadata = {
            "family_id": "vector_joint_bc_transition_value",
            "family_version": 1,
            "model_release_id": "model4",
            "policy_model_path": "artifacts/model4.keras",
        }
        enriched = apply_transfer_metadata(
            self.target_metadata,
            source_metadata=source_metadata,
        )
        self.assertEqual(enriched["parent_release_id"], "model4")
        self.assertEqual(enriched["initialization_source"]["mode"], "transfer")
        self.assertIn("model4", describe_transfer(enriched))

    def test_apply_transfer_metadata_uses_explicit_checkpoint(self) -> None:
        enriched = apply_transfer_metadata(
            self.target_metadata,
            explicit_checkpoint_path="checkpoints/warmstart.keras",
        )
        self.assertEqual(enriched["initialization_source"]["mode"], "checkpoint")
        self.assertEqual(Path(enriched["initialization_source"]["checkpoint_path"]), Path("checkpoints/warmstart.keras"))


if __name__ == "__main__":
    unittest.main()
