from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from word_prediction_model.benchmark import summarize
from word_prediction_model.pipeline import train_and_save


class BenchmarkTests(unittest.TestCase):
    def test_summarize_reports_benchmark_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_and_save(
                output_root=Path(tmpdir),
                run_name="bench_run",
                repeats_per_entry=12,
                corpus_seed=4,
            )
            summary = summarize(Path(result["run_dir"]))

            self.assertEqual(summary["model_family"], "word_prediction_embedding_v1")
            self.assertGreaterEqual(summary["overall_accuracy"], 0.7)
            self.assertGreaterEqual(summary["training_ticks"], 0)
            self.assertIn("num_failures", summary)


if __name__ == "__main__":
    unittest.main()
