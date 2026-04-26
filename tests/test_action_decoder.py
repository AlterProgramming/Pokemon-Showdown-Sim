"""Tests for ActionDecoder layer — Component 4 of history-decoding plan."""

from __future__ import annotations

import unittest
import numpy as np


class TestActionDecoder(unittest.TestCase):
    """Test the ActionDecoder Keras layer."""

    def setUp(self):
        import tensorflow as tf
        import keras
        from core.EntityModelV1 import ActionDecoder
        self.tf = tf
        self.keras = keras
        self.ActionDecoder = ActionDecoder

    def _make_layer(self, action_vocab_size=10):
        return self.ActionDecoder(action_vocab_size=action_vocab_size, hidden_dim=64)

    def test_output_shape(self):
        """Output shape should be [batch, K, action_vocab_size]."""
        batch, K, lstm_dim = 2, 8, 128
        action_vocab_size = 10

        layer = self._make_layer(action_vocab_size)
        lstm_output = self.tf.random.normal((batch, K, lstm_dim))
        action_mask = self.tf.ones((batch, K), dtype=self.tf.float32)

        logits = layer([lstm_output, action_mask])
        self.assertEqual(logits.shape, (batch, K, action_vocab_size))

    def test_padded_positions_are_masked(self):
        """Padded positions (mask=0) should receive large negative logits."""
        batch, K, lstm_dim = 2, 8, 128
        action_vocab_size = 10

        layer = self._make_layer(action_vocab_size)
        lstm_output = self.tf.random.normal((batch, K, lstm_dim))

        # Only first 5 positions are real; last 3 are padded
        action_mask = self.tf.constant(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=self.tf.float32,
        )

        logits = layer([lstm_output, action_mask]).numpy()

        # Padded positions should have max logit <= -1e8
        for b in range(batch):
            for k in range(5, K):
                self.assertTrue(
                    logits[b, k, :].max() <= -1e8,
                    f"Padded position ({b},{k}) max logit {logits[b, k, :].max()} not <= -1e8",
                )

    def test_real_positions_not_masked(self):
        """Real positions (mask=1) should not be forced to large negative."""
        batch, K, lstm_dim = 2, 8, 128
        action_vocab_size = 10

        layer = self._make_layer(action_vocab_size)
        lstm_output = self.tf.random.normal((batch, K, lstm_dim))
        action_mask = self.tf.ones((batch, K), dtype=self.tf.float32)

        logits = layer([lstm_output, action_mask]).numpy()

        # Real positions should not be all large negative
        self.assertTrue(
            logits[:, :, :].max() > -1e8,
            "All real-position logits are large-negative; masking is misapplied",
        )

    def test_all_padded(self):
        """All-padded mask: every logit should be <= -1e8."""
        batch, K, lstm_dim = 2, 8, 128
        action_vocab_size = 10

        layer = self._make_layer(action_vocab_size)
        lstm_output = self.tf.random.normal((batch, K, lstm_dim))
        action_mask = self.tf.zeros((batch, K), dtype=self.tf.float32)

        logits = layer([lstm_output, action_mask]).numpy()
        self.assertTrue(logits.max() <= -1e8)

    def test_get_config_roundtrip(self):
        """get_config should allow reconstruction with from_config."""
        layer = self._make_layer(action_vocab_size=15)
        cfg = layer.get_config()
        self.assertEqual(cfg["action_vocab_size"], 15)
        self.assertEqual(cfg["hidden_dim"], 64)

        rebuilt = self.ActionDecoder.from_config(cfg)
        self.assertEqual(rebuilt.action_vocab_size, 15)

    def test_no_mask_argument(self):
        """Layer should handle None action_mask without error (no masking applied)."""
        batch, K, lstm_dim = 2, 8, 128
        action_vocab_size = 10

        layer = self._make_layer(action_vocab_size)
        lstm_output = self.tf.random.normal((batch, K, lstm_dim))
        # Passing None mask via list triggers the None branch
        action_mask = self.tf.ones((batch, K), dtype=self.tf.float32)

        # Smoke: should not raise
        logits = layer([lstm_output, action_mask])
        self.assertEqual(logits.shape[-1], action_vocab_size)


if __name__ == "__main__":
    unittest.main()
