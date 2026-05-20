from __future__ import annotations

import unittest

from word_prediction_model.dataset import build_training_corpus
from word_prediction_model.text import char_ngrams, generate_typo_variants


class DatasetTests(unittest.TestCase):
    def test_typo_generator_keeps_canonical_and_adds_variants(self) -> None:
        variants = generate_typo_variants("happy")

        self.assertIn("happy", variants)
        self.assertIn("hapy", variants)
        self.assertGreater(len(variants), 5)

    def test_char_ngrams_are_non_empty(self) -> None:
        grams = char_ngrams("forest")

        self.assertTrue(grams)
        self.assertIn("<f", grams)

    def test_corpus_contains_vocabularies_and_examples(self) -> None:
        corpus = build_training_corpus(repeats_per_entry=4, seed=1)

        self.assertTrue(corpus.examples)
        self.assertIn("happy", corpus.label_vocab)
        self.assertIn("joy", corpus.token_vocab)
        self.assertGreater(len(corpus.ngram_vocab), 10)

    def test_bucketed_ngrams_stay_small(self) -> None:
        corpus = build_training_corpus(repeats_per_entry=4, seed=1, ngram_bucket_count=32)

        self.assertEqual(len(corpus.ngram_vocab), 33)


if __name__ == "__main__":
    unittest.main()
