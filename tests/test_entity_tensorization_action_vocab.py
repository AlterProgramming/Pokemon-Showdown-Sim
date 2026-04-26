"""Tests for EntityTensorization action vocab and encoding."""

import pytest
from core.EntityTensorization import (
    build_action_vocab,
    vectorize_entity_multitask_dataset,
    build_entity_training_bundle,
    ENTITY_INT_INPUT_KEYS,
)


class TestBuildActionVocab:
    """Test action vocab building from examples."""

    def test_build_action_vocab_basic(self):
        """Test basic vocab construction with simple action strings."""
        examples = [
            {"past_turn_actions": ["move tackle", "move thunderbolt"]},
            {"past_turn_actions": ["switch pikachu", "move tackle"]},
            {"past_turn_actions": []},
        ]
        vocab = build_action_vocab(examples)

        # Check structure
        assert isinstance(vocab, dict)
        assert vocab["PAD"] == 0
        assert vocab["UNK"] == 1

        # Check actions are mapped
        assert "move tackle" in vocab
        assert "move thunderbolt" in vocab
        assert "switch pikachu" in vocab

        # Check IDs are sequential starting at 2
        actions = {k: v for k, v in vocab.items() if k not in ["PAD", "UNK"]}
        ids = sorted(actions.values())
        assert ids[0] == 2
        assert ids[-1] == len(vocab) - 1

    def test_build_action_vocab_empty_examples(self):
        """Test vocab building with no actions."""
        examples = [
            {"past_turn_actions": []},
            {},
        ]
        vocab = build_action_vocab(examples)

        # Should still have special tokens
        assert vocab["PAD"] == 0
        assert vocab["UNK"] == 1
        assert len(vocab) == 2

    def test_build_action_vocab_deduplication(self):
        """Test that duplicate actions get same ID."""
        examples = [
            {"past_turn_actions": ["move tackle"]},
            {"past_turn_actions": ["move tackle"]},
            {"past_turn_actions": ["move tackle"]},
        ]
        vocab = build_action_vocab(examples)

        # Should only have one entry for move tackle
        assert sum(1 for k in vocab if k == "move tackle") == 1

    def test_build_action_vocab_in_int_keys(self):
        """Test that past_action_ids is in ENTITY_INT_INPUT_KEYS."""
        assert "past_action_ids" in ENTITY_INT_INPUT_KEYS


class TestActionHistoryEncoding:
    """Test action history encoding in vectorization."""

    def test_vectorize_with_history_decoding_basic(self):
        """Test action history encoding in vectorization."""
        # Minimal mock examples
        examples = [
            {
                "action": "move tackle",
                "action_token": "move tackle",
                "state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "next_state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "player": "p1",
                "terminal_result": None,
                "past_turn_actions": ["move tackle", "move thunderbolt"],
            }
        ]

        # Build minimal vocabs
        action_vocab = build_action_vocab(examples)
        token_vocabs = {
            "species": {"<PAD>": 0, "<UNK>": 1, "pikachu": 2},
            "item": {"<PAD>": 0, "<UNK>": 1, "leftovers": 2},
            "ability": {"<PAD>": 0, "<UNK>": 1, "static": 2},
            "tera": {"<PAD>": 0, "<UNK>": 1, "electric": 2},
            "status": {"<PAD>": 0, "<UNK>": 1, "none": 2},
            "move": {"<PAD>": 0, "<UNK>": 1, "tackle": 2},
            "weather": {"<PAD>": 0, "<UNK>": 1, "none": 2},
            "global_condition": {"<PAD>": 0, "<UNK>": 1},
        }
        policy_vocab = {"<UNK>": 1, "move tackle": 2, "move thunderbolt": 3}

        # Vectorize with history decoding
        X, y = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=policy_vocab,
            token_vocabs=token_vocabs,
            include_switches=True,
            include_history_decoding=True,
            action_vocab=action_vocab,
            history_turns=8,
        )

        # Check that past_action_ids and past_action_mask are populated
        assert "past_action_ids" in X
        assert "past_action_mask" in X
        assert len(X["past_action_ids"]) == 1
        assert len(X["past_action_mask"]) == 1

        # Check shapes and values
        action_ids = X["past_action_ids"][0]
        action_mask = X["past_action_mask"][0]
        assert len(action_ids) == 8  # max_turns (default history_turns)
        assert len(action_mask) == 8

        # Check mask values: should have 0s for padding and 1s for real actions
        assert sum(action_mask) == 2  # 2 real actions
        assert action_mask[-2:] == [1.0, 1.0]  # Last 2 are real (right-aligned)
        assert action_mask[:6] == [0.0] * 6  # First 6 are padding (left-aligned)

    def test_vectorize_without_history_decoding(self):
        """Test that past_action_ids is not created when disabled."""
        examples = [
            {
                "action": "move tackle",
                "action_token": "move tackle",
                "state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "next_state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "player": "p1",
                "terminal_result": None,
                "past_turn_actions": ["move tackle"],
            }
        ]

        action_vocab = build_action_vocab(examples)
        token_vocabs = {
            "species": {"<PAD>": 0, "<UNK>": 1, "pikachu": 2},
            "item": {"<PAD>": 0, "<UNK>": 1, "leftovers": 2},
            "ability": {"<PAD>": 0, "<UNK>": 1, "static": 2},
            "tera": {"<PAD>": 0, "<UNK>": 1, "electric": 2},
            "status": {"<PAD>": 0, "<UNK>": 1, "none": 2},
            "move": {"<PAD>": 0, "<UNK>": 1, "tackle": 2},
            "weather": {"<PAD>": 0, "<UNK>": 1, "none": 2},
            "global_condition": {"<PAD>": 0, "<UNK>": 1},
        }
        policy_vocab = {"<UNK>": 1, "move tackle": 2}

        X, y = vectorize_entity_multitask_dataset(
            examples,
            policy_vocab=policy_vocab,
            token_vocabs=token_vocabs,
            include_switches=True,
            include_history_decoding=False,
            history_turns=8,
        )

        # Should not have past_action_ids
        assert "past_action_ids" not in X
        assert "past_action_mask" not in X

    def test_vectorize_history_decoding_missing_vocab_error(self):
        """Test that missing action_vocab raises ValueError."""
        examples = [
            {
                "action": "move tackle",
                "action_token": "move tackle",
                "state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "next_state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "player": "p1",
                "terminal_result": None,
                "past_turn_actions": ["move tackle"],
            }
        ]

        token_vocabs = {
            "species": {"<PAD>": 0, "<UNK>": 1, "pikachu": 2},
            "item": {"<PAD>": 0, "<UNK>": 1, "leftovers": 2},
            "ability": {"<PAD>": 0, "<UNK>": 1, "static": 2},
            "tera": {"<PAD>": 0, "<UNK>": 1, "electric": 2},
            "status": {"<PAD>": 0, "<UNK>": 1, "none": 2},
            "move": {"<PAD>": 0, "<UNK>": 1, "tackle": 2},
            "weather": {"<PAD>": 0, "<UNK>": 1, "none": 2},
            "global_condition": {"<PAD>": 0, "<UNK>": 1},
        }
        policy_vocab = {"<UNK>": 1, "move tackle": 2}

        with pytest.raises(ValueError, match="action_vocab required"):
            vectorize_entity_multitask_dataset(
                examples,
                policy_vocab=policy_vocab,
                token_vocabs=token_vocabs,
                include_switches=True,
                include_history_decoding=True,
                action_vocab=None,
                history_turns=8,
            )


class TestBuildEntityTrainingBundle:
    """Test bundle building with history decoding."""

    def test_bundle_includes_action_vocab_when_enabled(self):
        """Test that action_vocab is built when include_history_decoding=True."""
        examples = [
            {
                "action": "move tackle",
                "action_token": "move tackle",
                "state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "next_state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "player": "p1",
                "past_turn_actions": ["move tackle"],
            }
        ]

        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_history_decoding=True,
        )

        assert "action_vocab" in bundle
        assert isinstance(bundle["action_vocab"], dict)
        assert "move tackle" in bundle["action_vocab"]

    def test_bundle_excludes_action_vocab_when_disabled(self):
        """Test that action_vocab is empty dict when disabled."""
        examples = [
            {
                "action": "move tackle",
                "action_token": "move tackle",
                "state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "next_state": {
                    "p1": {"side_conditions": {}},
                    "p2": {"side_conditions": {}},
                    "turn_index": 1,
                },
                "player": "p1",
                "past_turn_actions": ["move tackle"],
            }
        ]

        bundle = build_entity_training_bundle(
            examples,
            include_switches=True,
            min_move_count=1,
            include_transition=False,
            include_value=False,
            include_history_decoding=False,
        )

        # Should have empty action_vocab in return
        assert bundle.get("action_vocab", {}) == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
