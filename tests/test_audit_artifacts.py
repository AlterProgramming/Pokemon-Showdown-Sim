from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_artifacts import audit_artifacts


class AuditArtifactsTests(unittest.TestCase):
    def test_audit_artifacts_reports_missing_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            artifacts_dir = repo_path / "artifacts"
            artifacts_dir.mkdir()

            (artifacts_dir / "model_9.keras").write_text("model", encoding="utf-8")
            (artifacts_dir / "action_vocab_9.json").write_text("{}", encoding="utf-8")
            metadata = {
                "policy_model_path": "artifacts/model_9.keras",
                "policy_vocab_path": "artifacts/action_vocab_9.json",
                "model_name": "model_9",
                "action_space": "joint",
                "include_switches": True,
                "predict_value": False,
                "predict_turn_outcome": True,
                "evaluation_summary_path": "artifacts/evaluation_summary_9.json",
                "run_manifest_path": "artifacts/run_manifest_9.json",
            }
            (artifacts_dir / "training_metadata_9.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            report = audit_artifacts(repo_path)
            self.assertEqual(report["release_count"], 1)
            self.assertEqual(report["releases_with_missing_artifacts"], 1)
            self.assertEqual(report["releases_missing_governance_artifacts"], 1)
            self.assertEqual(report["releases_with_run_manifest"], 0)
            self.assertEqual(report["releases_with_evaluation_summary"], 0)
            release = report["releases"][0]
            self.assertEqual(release["family_id"], "vector_joint_bc_transition")
            self.assertIn("evaluation_summary_path", release["missing_artifacts"])
            self.assertIn("run_manifest_path", release["missing_artifacts"])
            self.assertIn("evaluation_summary_path", release["missing_governance_artifacts"])
            self.assertIn("run_manifest_path", release["missing_governance_artifacts"])


if __name__ == "__main__":
    unittest.main()
