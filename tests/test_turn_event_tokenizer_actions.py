"""Tests for encode_action_history() function in TurnEventTokenizer."""

import pytest
from core.TurnEventTokenizer import encode_action_history


class TestEncodeActionHistory:
    """Test encode_action_history function."""

    def test_empty_actions_list(self):
        """Empty actions list should produce all PAD and all-zeros mask."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2, "UNK": 4}
        max_turns = 5

        action_ids, action_mask = encode_action_history([], action_vocab, max_turns)

        assert action_ids == [0, 0, 0, 0, 0]
        assert action_mask == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_single_action_left_padded(self):
        """Single action should be left-padded to max_turns."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2}
        max_turns = 5

        action_ids, action_mask = encode_action_history(
            ["move tackle"], action_vocab, max_turns
        )

        assert action_ids == [0, 0, 0, 0, 1]
        assert action_mask == [0.0, 0.0, 0.0, 0.0, 1.0]

    def test_exactly_max_turns_actions(self):
        """Exactly max_turns actions should have no padding."""
        action_vocab = {
            "move tackle": 1,
            "switch pikachu": 2,
            "move thunderbolt": 3,
            "move earthquake": 4,
        }
        max_turns = 4
        actions = ["move tackle", "switch pikachu", "move thunderbolt", "move earthquake"]

        action_ids, action_mask = encode_action_history(actions, action_vocab, max_turns)

        assert action_ids == [1, 2, 3, 4]
        assert action_mask == [1.0, 1.0, 1.0, 1.0]

    def test_more_than_max_turns_clipped(self):
        """More than max_turns actions should be clipped to last K."""
        action_vocab = {
            "move tackle": 1,
            "switch pikachu": 2,
            "move thunderbolt": 3,
            "move earthquake": 4,
            "move hydro pump": 5,
        }
        max_turns = 3
        actions = [
            "move tackle",
            "switch pikachu",
            "move thunderbolt",
            "move earthquake",
            "move hydro pump",
        ]

        action_ids, action_mask = encode_action_history(actions, action_vocab, max_turns)

        # Should keep last 3: thunderbolt, earthquake, hydro pump
        assert action_ids == [3, 4, 5]
        assert action_mask == [1.0, 1.0, 1.0]

    def test_unknown_action_uses_unk_token(self):
        """Unknown actions should use UNK token."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2, "UNK": 4}
        max_turns = 3

        action_ids, action_mask = encode_action_history(
            ["move tackle", "unknown_action", "switch pikachu"],
            action_vocab,
            max_turns,
        )

        assert action_ids == [1, 4, 2]
        assert action_mask == [1.0, 1.0, 1.0]

    def test_unknown_action_fallback_to_pad_when_no_unk(self):
        """If vocab has no UNK, unknown actions should fall back to 0 (PAD)."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2}
        max_turns = 3

        action_ids, action_mask = encode_action_history(
            ["move tackle", "unknown_action", "switch pikachu"],
            action_vocab,
            max_turns,
        )

        assert action_ids == [1, 0, 2]
        assert action_mask == [1.0, 1.0, 1.0]

    def test_mask_shape_matches_max_turns(self):
        """Mask and action_ids should always have length max_turns."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2}
        max_turns = 8

        # Test with various numbers of actions
        for num_actions in [0, 1, 3, 8, 15]:
            actions = ["move tackle"] * num_actions
            action_ids, action_mask = encode_action_history(
                actions, action_vocab, max_turns
            )

            assert len(action_ids) == max_turns
            assert len(action_mask) == max_turns

    def test_mask_correctness_padded_vs_real(self):
        """Padded positions should be 0.0, real should be 1.0."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2}
        max_turns = 8

        action_ids, action_mask = encode_action_history(
            ["move tackle", "switch pikachu", "move tackle"],
            action_vocab,
            max_turns,
        )

        # First 5 should be padded (0.0), last 3 should be real (1.0)
        assert action_mask == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    def test_padding_with_multiple_actions(self):
        """Test left-padding with multiple real actions."""
        action_vocab = {"move tackle": 1, "switch pikachu": 2, "move thunderbolt": 3}
        max_turns = 8
        actions = ["move tackle", "switch pikachu", "move thunderbolt"]

        action_ids, action_mask = encode_action_history(actions, action_vocab, max_turns)

        assert action_ids == [0, 0, 0, 0, 0, 1, 2, 3]
        assert action_mask == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    def test_float32_mask_type(self):
        """Mask values should be float (for TensorFlow compatibility)."""
        action_vocab = {"move tackle": 1}
        max_turns = 2

        _, action_mask = encode_action_history(
            ["move tackle"], action_vocab, max_turns
        )

        for val in action_mask:
            assert isinstance(val, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
