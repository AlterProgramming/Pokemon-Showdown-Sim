"""Tests for action vocab building and history decoding in EntityTensorization."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

import numpy as np

from core.EntityTensorization import (
    build_action_vocab,
    build_entity_training_bundle,
    vectorize_entity_multitask_dataset,
)
from core.StateVectorization import (
    build_action_context_vocab,
    build_action_vocab as build_policy_action_vocab,
)
from core.TurnEventTokenizer import encode_action_history


class TestBuildActionVocab(unittest.TestCase):
    """Test the build_action_vocab function."""

    def test_empty_examples(self):
        """Empty examples should return vocab with only PAD and UNK."""
        examples = []
        vocab = build_action_vocab(examples)
        self.assertEqual(vocab, {"PAD": 0, "UNK": 1})

    def test_single_action(self):
        """Single action should be assigned ID 2."""
        examples = [
            {
                "past_turn_actions": ["move tackle"],
            }
        ]
        vocab = build_action_vocab(examples)
        self.assertEqual(vocab["PAD"], 0)
        self.assertEqual(vocab["UNK"], 1)
        self.assertEqual(vocab["move tackle"], 2)

    def test_multiple_unique_actions(self):
        """Multiple unique actions should be assigned sequential IDs."""
        examples = [
            {"past_turn_actions": ["move tackle", "move thunderbolt"]},
            {"past_turn_actions": ["switch pikachu"]},
        ]
        vocab = build_action_vocab(examples)
        self.assertEqual(vocab["PAD"], 0)
        self.assertEqual(vocab["UNK"], 1)
        self.assertGreaterEqual(vocab["move tackle"], 2)
        self.assertGreaterEqual(vocab["move thunderbolt"], 2)
        self.assertGreaterEqual(vocab["switch pikachu"], 2)

    def test_duplicate_actions_consolidated(self):
        """Duplicate actions should map to the same ID."""
        examples = [
            {"past_turn_actions": ["move tackle", "move tackle"]},
            {"past_turn_actions": ["move tackle"]},
        ]
        vocab = build_action_vocab(examples)
        self.assertEqual(vocab["move tackle"], 2)
        # All three "move tackle" references should map to the same ID
        for ex in examples:
            for action in ex["past_turn_actions"]:
                self.assertEqual(vocab[action], 2)

    def test_no_past_turn_actions_key(self):
        """Examples without past_turn_actions should not cause errors."""
        examples = [
            {"some_other_key": "value"},
            {"past_turn_actions": []},
        ]
        vocab = build_action_vocab(examples)
        self.assertEqual(vocab, {"PAD": 0, "UNK": 1})


class TestEncodeActionHistory(unittest.TestCase):
    """Test the encode_action_history function from TurnEventTokenizer."""

    def test_empty_actions_pads_correctly(self):
        """Empty action list should produce all-pad IDs and all-zero mask."""
        vocab = {"PAD": 0, "UNK": 1, "move tackle": 2}
        action_ids, action_mask = encode_action_history([], vocab, 4)
        self.assertEqual(len(action_ids), 4)
        self.assertEqual(len(action_mask), 4)
        self.assertTrue(all(aid == 0 for aid in action_ids))
        self.assertTrue(all(m == 0.0 for m in action_mask))

    def test_single_action_left_padded(self):
        """Single action should be left-padded with zeros."""
        vocab = {"PAD": 0, "UNK": 1, "move tackle": 2}
        action_ids, action_mask = encode_action_history(["move tackle"], vocab, 4)
        self.assertEqual(len(action_ids), 4)
        self.assertEqual(len(action_mask), 4)
        # Expected: [0, 0, 0, 2] (left-padded)
        self.assertTrue(all(aid == 0 for aid in action_ids[:-1]))
        self.assertEqual(action_ids[-1], 2)
        # Expected: [0.0, 0.0, 0.0, 1.0] (left-padded)
        self.assertTrue(all(m == 0.0 for m in action_mask[:-1]))
        self.assertEqual(action_mask[-1], 1.0)

    def test_multiple_actions_ordered(self):
        """Multiple actions should be ordered left-to-right."""
        vocab = {"PAD": 0, "UNK": 1, "move tackle": 2, "switch pikachu": 3}
        action_ids, action_mask = encode_action_history(
            ["move tackle", "switch pikachu"], vocab, 4
        )
        self.assertEqual(len(action_ids), 4)
        self.assertEqual(len(action_mask), 4)
        # Expected: [0, 0, 2, 3]
        self.assertEqual(action_ids[0], 0)
        self.assertEqual(action_ids[1], 0)
        self.assertEqual(action_ids[2], 2)
        self.assertEqual(action_ids[3], 3)
        # Expected: [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(action_mask[0], 0.0)
        self.assertEqual(action_mask[1], 0.0)
        self.assertEqual(action_mask[2], 1.0)
        self.assertEqual(action_mask[3], 1.0)

    def test_unknown_action_uses_unk_token(self):
        """Unknown actions should map to UNK ID."""
        vocab = {"PAD": 0, "UNK": 1, "move tackle": 2}
        action_ids, action_mask = encode_action_history(["move unknown"], vocab, 4)
        # Unknown action should map to UNK (1)
        self.assertEqual(action_ids[-1], 1)

    def test_mask_is_float32(self):
        """Mask should be float32 for TensorFlow compatibility."""
        vocab = {"PAD": 0, "UNK": 1, "move tackle": 2}
        action_ids, action_mask = encode_action_history(["move tackle"], vocab, 4)
        # Check that mask values are floats
        self.assertTrue(all(isinstance(m, (float, np.floating)) for m in action_mask))


class TestVectorizeWithHistoryDecoding(unittest.TestCase):
    """Test vectorize_entity_multitask_dataset with history_decoding enabled."""

    def make_minimal_example(self) -> Dict[str, Any]:
        """Create a minimal valid example for vectorization."""
        return {
            "state": {
                "turn_index": 1,
                "field": {
                    "weather": "none",
                    "global_conditions": [],
                },
                "p1": {
                    "active_uid": "p1a",
                    "slots": ["p1a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "p2": {
                    "active_uid": "p2a",
                    "slots": ["p2a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "mons": {
                    "p1a": {
                        "uid": "p1a",
                        "player": "p1",
                        "species": "pikachu",
                        "hp_frac": 1.0,
                        "status": None,
                        "ability": "static",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-tackle"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                    "p2a": {
                        "uid": "p2a",
                        "player": "p2",
                        "species": "charizard",
                        "hp_frac": 1.0,
                        "status": None,
                        "ability": "blaze",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-flamethrower"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                },
            },
            "next_state": {
                "turn_index": 2,
                "field": {
                    "weather": "none",
                    "global_conditions": [],
                },
                "p1": {
                    "active_uid": "p1a",
                    "slots": ["p1a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "p2": {
                    "active_uid": "p2a",
                    "slots": ["p2a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "mons": {
                    "p1a": {
                        "uid": "p1a",
                        "player": "p1",
                        "species": "pikachu",
                        "hp_frac": 0.9,
                        "status": None,
                        "ability": "static",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-tackle"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                    "p2a": {
                        "uid": "p2a",
                        "player": "p2",
                        "species": "charizard",
                        "hp_frac": 0.95,
                        "status": None,
                        "ability": "blaze",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-flamethrower"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                },
            },
            "player": "p1",
            "action": "move tackle",
            "action_token": "move:tackle",
            "opponent_action_token": "move:flamethrower",
            "past_turn_actions": ["move tackle", "switch pikachu"],
            "terminal_result": None,
        }

    def test_action_vocab_none_raises_error(self):
        """include_history_decoding=True without action_vocab should raise ValueError."""
        examples = [self.make_minimal_example()]

        # Build required vocabs
        policy_vocab = build_policy_action_vocab(
            examples, min_count=1, include_switches=True
        )
        action_context_vocab = build_action_context_vocab(examples)
        from core.EntityTensorization import build_entity_token_vocabs

        token_vocabs = build_entity_token_vocabs(examples)

        with self.assertRaises(ValueError) as cm:
            vectorize_entity_multitask_dataset(
                examples,
                policy_vocab=policy_vocab,
                token_vocabs=token_vocabs,
                action_context_vocab=action_context_vocab,
                include_switches=True,
                include_history_decoding=True,
                action_vocab=None,  # Missing
            )
        self.assertIn("action_vocab required", str(cm.exception))

    def test_past_action_ids_shape_correct(self):
        """past_action_ids should have shape [batch, history_turns]."""
        examples = [self.make_minimal_example()]

        # Build all required vocabs
        policy_vocab = build_policy_action_vocab(
            examples, min_count=1, include_switches=True
        )
        action_context_vocab = build_action_context_vocab(examples)
        action_vocab = build_action_vocab(examples)
        from core.EntityTensorization import build_entity_token_vocabs

        token_vocabs = build_entity_token_vocabs(examples)

        history_turns = 8
        X, y = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=policy_vocab,
            token_vocabs=token_vocabs,
            action_context_vocab=action_context_vocab,
            include_switches=True,
            include_history_decoding=True,
            action_vocab=action_vocab,
            history_turns=history_turns,
        )

        # Check that past_action_ids is in X
        self.assertIn("past_action_ids", X)
        self.assertEqual(len(X["past_action_ids"]), len(examples))
        # Each action_ids list should have length history_turns
        for action_ids in X["past_action_ids"]:
            self.assertEqual(len(action_ids), history_turns)

    def test_past_action_mask_dtype_float(self):
        """past_action_mask should be float values for TensorFlow compatibility."""
        examples = [self.make_minimal_example()]

        # Build all required vocabs
        policy_vocab = build_policy_action_vocab(
            examples, min_count=1, include_switches=True
        )
        action_context_vocab = build_action_context_vocab(examples)
        action_vocab = build_action_vocab(examples)
        from core.EntityTensorization import build_entity_token_vocabs

        token_vocabs = build_entity_token_vocabs(examples)

        X, y = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=policy_vocab,
            token_vocabs=token_vocabs,
            action_context_vocab=action_context_vocab,
            include_switches=True,
            include_history_decoding=True,
            action_vocab=action_vocab,
        )

        # Check that past_action_mask is in X
        self.assertIn("past_action_mask", X)
        self.assertEqual(len(X["past_action_mask"]), len(examples))
        # All mask values should be float-like
        for mask in X["past_action_mask"]:
            self.assertTrue(all(isinstance(m, (float, np.floating)) for m in mask))


class TestBuildEntityTrainingBundleWithHistoryDecoding(unittest.TestCase):
    """Test build_entity_training_bundle with history_decoding enabled."""

    def make_minimal_example(self) -> Dict[str, Any]:
        """Create a minimal valid example for bundle building."""
        return {
            "state": {
                "turn_index": 1,
                "field": {
                    "weather": "none",
                    "global_conditions": [],
                },
                "p1": {
                    "active_uid": "p1a",
                    "slots": ["p1a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "p2": {
                    "active_uid": "p2a",
                    "slots": ["p2a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "mons": {
                    "p1a": {
                        "uid": "p1a",
                        "player": "p1",
                        "species": "pikachu",
                        "hp_frac": 1.0,
                        "status": None,
                        "ability": "static",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-tackle"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                    "p2a": {
                        "uid": "p2a",
                        "player": "p2",
                        "species": "charizard",
                        "hp_frac": 1.0,
                        "status": None,
                        "ability": "blaze",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-flamethrower"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                },
            },
            "next_state": {
                "turn_index": 2,
                "field": {
                    "weather": "none",
                    "global_conditions": [],
                },
                "p1": {
                    "active_uid": "p1a",
                    "slots": ["p1a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "p2": {
                    "active_uid": "p2a",
                    "slots": ["p2a", None, None, None, None, None],
                    "side_conditions": {},
                },
                "mons": {
                    "p1a": {
                        "uid": "p1a",
                        "player": "p1",
                        "species": "pikachu",
                        "hp_frac": 0.9,
                        "status": None,
                        "ability": "static",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-tackle"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                    "p2a": {
                        "uid": "p2a",
                        "player": "p2",
                        "species": "charizard",
                        "hp_frac": 0.95,
                        "status": None,
                        "ability": "blaze",
                        "item": None,
                        "tera_type": None,
                        "terastallized": False,
                        "public_revealed": True,
                        "active": True,
                        "fainted": False,
                        "observed_moves": ["move-flamethrower"],
                        "boosts": {
                            "atk": 0,
                            "def": 0,
                            "spa": 0,
                            "spd": 0,
                            "spe": 0,
                        },
                    },
                },
            },
            "player": "p1",
            "action": "move tackle",
            "action_token": "move:tackle",
            "opponent_action_token": "move:flamethrower",
            "past_turn_actions": ["move tackle", "switch pikachu"],
            "terminal_result": None,
        }

    def test_action_vocab_included_when_decoding_enabled(self):
        """action_vocab should be in bundle when include_history_decoding=True."""
        examples = [self.make_minimal_example()]

        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_history_decoding=True,
        )

        self.assertIn("action_vocab", bundle)
        self.assertIsInstance(bundle["action_vocab"], dict)
        self.assertIn("PAD", bundle["action_vocab"])
        self.assertIn("UNK", bundle["action_vocab"])

    def test_action_vocab_empty_when_decoding_disabled(self):
        """action_vocab should be empty when include_history_decoding=False."""
        examples = [self.make_minimal_example()]

        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_history_decoding=False,
        )

        # action_vocab key should still exist but be empty
        self.assertIn("action_vocab", bundle)
        self.assertEqual(bundle["action_vocab"], {})


if __name__ == "__main__":
    unittest.main()
