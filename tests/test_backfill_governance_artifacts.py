from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backfill_governance_artifacts import backfill_repo


class BackfillGovernanceArtifactsTests(unittest.TestCase):
    def test_backfill_repo_creates_manifest_and_evaluation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()

            (artifacts_dir / "model_7.keras").write_text("model", encoding="utf-8")
            (artifacts_dir / "action_vocab_7.json").write_text("{}", encoding="utf-8")
            metadata = {
                "policy_model_path": "artifacts/model_7.keras",
                "policy_vocab_path": "artifacts/action_vocab_7.json",
                "model_name": "model_7",
                "action_space": "joint",
                "include_switches": True,
                "predict_turn_outcome": True,
                "predict_value": False,
                "epochs_completed": 5,
                "train_examples": 100,
                "val_examples": 25,
            }
            metadata_path = artifacts_dir / "training_metadata_7.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            stats = backfill_repo(repo_path)
            self.assertEqual(stats["updated_metadata_files"], 1)
            self.assertEqual(stats["created_evaluation_summaries"], 1)
            self.assertEqual(stats["created_run_manifests"], 1)

            updated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_metadata["family_id"], "vector_joint_bc_transition")
            self.assertEqual(
                Path(updated_metadata["evaluation_summary_path"]).as_posix(),
                "artifacts/evaluation_summary_7.json",
            )
            self.assertEqual(
                Path(updated_metadata["run_manifest_path"]).as_posix(),
                "artifacts/run_manifest_7.json",
            )

            evaluation_summary = json.loads((artifacts_dir / "evaluation_summary_7.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation_summary["status"], "backfilled_without_training_history")
            self.assertEqual(evaluation_summary["epochs_completed"], 5)

            run_manifest = json.loads((artifacts_dir / "run_manifest_7.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["model_release_id"], "model7")
            self.assertEqual(run_manifest["family_id"], "vector_joint_bc_transition")


if __name__ == "__main__":
    unittest.main()
