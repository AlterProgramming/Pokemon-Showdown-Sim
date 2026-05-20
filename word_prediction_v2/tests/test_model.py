from __future__ import annotations

import unittest

from word_prediction_model.dataset import build_training_corpus
from word_prediction_model.model import TrainingConfig, train_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = build_training_corpus(repeats_per_entry=56, seed=5)
        cls.model = train_model(
            corpus,
            TrainingConfig(
                embedding_dim=24,
                epochs=90,
                learning_rate=0.07,
                alignment_weight=0.4,
                seed=9,
            ),
        )

    def test_prompt_prediction_prefers_expected_word(self) -> None:
        ranked = self.model.predict(["joy", "smile", "bright"], top_k=1)

        self.assertEqual(ranked[0].word, "happy")

    def test_surface_form_misspelling_stays_near_canonical_word(self) -> None:
        ranked = self.model.nearest_words_for_surface_form("happee", top_k=1)

        self.assertEqual(ranked[0].word, "happy")

    def test_batch_prompt_prediction_matches_single_prediction(self) -> None:
        prompts = [
            ["joy", "smile", "bright"],
            ["danger", "safe", "recover"],
            ["pivot", "retreat", "swap"],
        ]

        batch_ranked = self.model.predict_batch(prompts, top_k=3)

        self.assertEqual(len(batch_ranked), len(prompts))
        for prompt, ranked_batch in zip(prompts, batch_ranked):
            ranked_single = self.model.predict(prompt, top_k=3)
            self.assertEqual(
                [item.word for item in ranked_batch],
                [item.word for item in ranked_single],
            )
            for batch_item, single_item in zip(ranked_batch, ranked_single):
                self.assertAlmostEqual(batch_item.score, single_item.score)

    def test_model_stays_under_parameter_budget(self) -> None:
        self.assertLessEqual(self.model.parameter_count, 10_000)

    def test_attention_report_aligns_with_top_predictions(self) -> None:
        report = self.model.prompt_attention_report(["joy", "smile", "bright"], top_k=2)

        self.assertEqual(report["tokens"], ["joy", "smile", "bright"])
        self.assertEqual(report["matrix_shape"], [2, 3])
        self.assertEqual(len(report["rows"]), 2)
        self.assertEqual(report["rows"][0]["word"], "happy")
        self.assertAlmostEqual(sum(report["rows"][0]["token_weights"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
