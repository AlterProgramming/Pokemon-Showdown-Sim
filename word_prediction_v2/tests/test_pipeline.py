from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from word_prediction_model.pipeline import load_model, train_and_save


class PipelineTests(unittest.TestCase):
    def test_train_and_save_writes_testable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_and_save(
                output_root=Path(tmpdir),
                run_name="test_run",
                repeats_per_entry=12,
                corpus_seed=3,
            )

            run_dir = Path(result["run_dir"])
            self.assertTrue(run_dir.exists())

            metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))
            evaluation = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
            model = load_model(run_dir / "model.json")

            self.assertEqual(metadata["model_family"], "word_prediction_embedding_v1")
            self.assertGreater(metadata["num_examples"], 0)
            self.assertGreaterEqual(evaluation["overall_accuracy"], 0.7)
            self.assertEqual(model.predict(["joy", "smile", "bright"], top_k=1)[0].word, "happy")


if __name__ == "__main__":
    unittest.main()
