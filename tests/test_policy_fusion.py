"""Tests for PastActionContext and policy head fusion — Component 4 of history-decoding plan."""

from __future__ import annotations

import unittest
import numpy as np


class TestPastActionContext(unittest.TestCase):
    """Test the PastActionContext cross-attention layer."""

    def setUp(self):
        import tensorflow as tf
        import keras
        from core.EntityModelV1 import PastActionContext
        self.tf = tf
        self.keras = keras
        self.PastActionContext = PastActionContext

    def test_output_shape(self):
        """Output context should be [batch, attn_dim]."""
        batch, K, action_vocab_size, shared_dim, attn_dim = 2, 8, 10, 64, 32

        layer = self.PastActionContext(attn_dim=attn_dim)
        shared = self.tf.random.normal((batch, shared_dim))
        action_logits = self.tf.random.normal((batch, K, action_vocab_size))
        action_mask = self.tf.ones((batch, K), dtype=self.tf.float32)

        context = layer([shared, action_logits, action_mask])
        self.assertEqual(context.shape, (batch, attn_dim))

    def test_output_shape_varying_batch(self):
        """Shape check with batch=1."""
        layer = self.PastActionContext(attn_dim=16)
        shared = self.tf.random.normal((1, 64))
        action_logits = self.tf.random.normal((1, 8, 10))
        action_mask = self.tf.ones((1, 8), dtype=self.tf.float32)

        context = layer([shared, action_logits, action_mask])
        self.assertEqual(context.shape, (1, 16))

    def test_masked_positions_ignored(self):
        """All-zero mask should not crash; result should still be [batch, attn_dim]."""
        batch, K, action_vocab_size, attn_dim = 2, 8, 10, 32

        layer = self.PastActionContext(attn_dim=attn_dim)
        shared = self.tf.random.normal((batch, 64))
        action_logits = self.tf.random.normal((batch, K, action_vocab_size))
        action_mask = self.tf.zeros((batch, K), dtype=self.tf.float32)

        context = layer([shared, action_logits, action_mask])
        self.assertEqual(context.shape, (batch, attn_dim))

    def test_get_config_roundtrip(self):
        """from_config(get_config()) should reconstruct the layer."""
        layer = self.PastActionContext(attn_dim=48)
        cfg = layer.get_config()
        self.assertEqual(cfg["attn_dim"], 48)

        rebuilt = self.PastActionContext.from_config(cfg)
        self.assertEqual(rebuilt.attn_dim, 48)


class TestPolicyFusion(unittest.TestCase):
    """Test that build_entity_action_models with use_history_decoding produces correct output shapes."""

    def setUp(self):
        import tensorflow as tf
        import keras
        self.tf = tf
        self.keras = keras

    def _build_model(self, use_history_decoding=True):
        from core.EntityModelV1 import build_entity_action_models
        from core.EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES
        from core.StateVectorization import SIDE_CONDITION_ORDER, STAT_ORDER

        vocab_sizes = {
            "species": 100, "item": 50, "ability": 50, "tera": 20,
            "status": 10, "move": 200, "weather": 10, "global_condition": 20,
        }
        return build_entity_action_models(
            vocab_sizes=vocab_sizes,
            num_policy_classes=20,
            hidden_dim=32,
            depth=1,
            dropout=0.0,
            learning_rate=1e-3,
            token_embed_dim=8,
            use_history=True,
            history_vocab_size=500,
            history_lstm_dim=16,
            history_turns=4,
            history_events_per_turn=8,
            use_history_decoding=use_history_decoding,
            action_vocab_size=12 if use_history_decoding else None,
            decoded_action_weight=0.15,
        )

    def test_model_compiles_without_error(self):
        """build_entity_action_models with use_history_decoding=True should not raise."""
        training_model, policy_model, policy_value_model, attn_model = self._build_model()
        self.assertIsNotNone(training_model)

    def test_policy_output_shape(self):
        """Policy logits shape matches num_policy_classes."""
        training_model, policy_model, _, _ = self._build_model()
        # The policy output key is "policy"
        policy_out = training_model.output["policy"]
        # Last dim should be num_policy_classes=20
        self.assertEqual(policy_out.shape[-1], 20)

    def test_loss_dict_includes_decoded_actions(self):
        """Training model's compiled losses should include decoded_actions key."""
        training_model, _, _, _ = self._build_model()
        # Keras stores compiled loss in model.compiled_loss or model.loss
        # Check via model.output keys
        output_keys = list(training_model.output.keys())
        self.assertIn("decoded_actions", output_keys, f"Output keys: {output_keys}")

    def test_decoded_actions_output_shape(self):
        """decoded_actions output should be [batch, history_turns, action_vocab_size]."""
        training_model, _, _, _ = self._build_model()
        decoded_out = training_model.output["decoded_actions"]
        # Shape should be (None, 4, 12) — (batch, history_turns=4, action_vocab_size=12)
        self.assertEqual(decoded_out.shape[1], 4)   # history_turns
        self.assertEqual(decoded_out.shape[2], 12)  # action_vocab_size

    def test_requires_action_vocab_size(self):
        """use_history_decoding=True without action_vocab_size should raise ValueError."""
        from core.EntityModelV1 import build_entity_action_models

        vocab_sizes = {
            "species": 100, "item": 50, "ability": 50, "tera": 20,
            "status": 10, "move": 200, "weather": 10, "global_condition": 20,
        }
        with self.assertRaises(ValueError):
            build_entity_action_models(
                vocab_sizes=vocab_sizes,
                num_policy_classes=20,
                hidden_dim=32,
                depth=1,
                dropout=0.0,
                learning_rate=1e-3,
                use_history=True,
                history_vocab_size=500,
                history_lstm_dim=16,
                history_turns=4,
                history_events_per_turn=8,
                use_history_decoding=True,
                action_vocab_size=None,  # Missing — should raise
            )

    def test_without_history_decoding_no_decoded_actions(self):
        """Without use_history_decoding, no decoded_actions key in outputs."""
        training_model, _, _, _ = self._build_model(use_history_decoding=False)
        # When use_history_decoding=False, there are no auxiliary heads besides policy,
        # so the model might be the simple policy-only form
        if hasattr(training_model, "output") and isinstance(training_model.output, dict):
            self.assertNotIn("decoded_actions", training_model.output)


if __name__ == "__main__":
    unittest.main()
