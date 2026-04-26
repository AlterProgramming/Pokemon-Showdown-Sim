"""Tests for BattleStateTracker action capture infrastructure."""

import pytest
import sys
from pathlib import Path
from collections import deque

# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.BattleStateTracker import BattleStateTracker


class TestActionCapture:
    """Test cases for action capture functionality."""

    def test_capture_actions_disabled_by_default(self):
        """When capture_actions=False, actions should not be captured."""
        tracker = BattleStateTracker(history_turns=2, capture_actions=False)

        turn = {
            "turn_number": 1,
            "events": []
        }
        tracker.apply_turn(turn)

        # past_turn_actions should be empty
        assert len(tracker._past_turn_actions) == 0
        assert list(tracker._past_turn_actions) == []

    def test_capture_actions_enabled(self):
        """When capture_actions=True, actions should be captured."""
        tracker = BattleStateTracker(history_turns=2, capture_actions=True)

        # Record an action and apply a turn
        tracker.record_action("move tackle")
        turn = {
            "turn_number": 1,
            "events": []
        }
        tracker.apply_turn(turn)

        # Action should be captured
        assert len(tracker._past_turn_actions) == 1
        assert list(tracker._past_turn_actions) == ["move tackle"]

    def test_unknown_action_when_not_recorded(self):
        """When capture_actions=True but no action recorded, should log 'UNKNOWN'."""
        tracker = BattleStateTracker(history_turns=2, capture_actions=True)

        # Apply a turn without recording an action
        turn = {
            "turn_number": 1,
            "events": []
        }
        tracker.apply_turn(turn)

        # Should record "UNKNOWN"
        assert len(tracker._past_turn_actions) == 1
        assert list(tracker._past_turn_actions) == ["UNKNOWN"]

    def test_multiple_turns_with_actions(self):
        """Test action capture across multiple turns with deque ordering."""
        tracker = BattleStateTracker(history_turns=3, capture_actions=True)

        # Turn 1
        tracker.record_action("move tackle")
        turn1 = {"turn_number": 1, "events": []}
        tracker.apply_turn(turn1)

        # Turn 2
        tracker.record_action("switch:2")
        turn2 = {"turn_number": 2, "events": []}
        tracker.apply_turn(turn2)

        # Turn 3
        tracker.record_action("move earthquake")
        turn3 = {"turn_number": 3, "events": []}
        tracker.apply_turn(turn3)

        # All actions should be in order
        assert list(tracker._past_turn_actions) == ["move tackle", "switch:2", "move earthquake"]

    def test_deque_maxlen_respects_history_turns(self):
        """Test that deque respects maxlen from history_turns."""
        tracker = BattleStateTracker(history_turns=2, capture_actions=True)

        # Turn 1
        tracker.record_action("move tackle")
        turn1 = {"turn_number": 1, "events": []}
        tracker.apply_turn(turn1)

        # Turn 2
        tracker.record_action("move earthquake")
        turn2 = {"turn_number": 2, "events": []}
        tracker.apply_turn(turn2)

        # Turn 3 - should push out turn 1
        tracker.record_action("move surf")
        turn3 = {"turn_number": 3, "events": []}
        tracker.apply_turn(turn3)

        # Only last 2 actions should remain
        assert list(tracker._past_turn_actions) == ["move earthquake", "move surf"]

    def test_reset_clears_actions(self):
        """Test that reset() clears the action deque."""
        tracker = BattleStateTracker(history_turns=2, capture_actions=True)

        # Record and apply a turn
        tracker.record_action("move tackle")
        turn = {"turn_number": 1, "events": []}
        tracker.apply_turn(turn)

        assert len(tracker._past_turn_actions) == 1

        # Reset should clear it
        tracker.reset()
        assert len(tracker._past_turn_actions) == 0

    def test_action_cleared_after_apply_turn(self):
        """Test that _current_turn_action is cleared after apply_turn."""
        tracker = BattleStateTracker(history_turns=1, capture_actions=True)

        tracker.record_action("move tackle")
        assert tracker._current_turn_action == "move tackle"

        turn = {"turn_number": 1, "events": []}
        tracker.apply_turn(turn)

        # Current action should be cleared for next turn
        assert tracker._current_turn_action is None

    def test_record_action_updates_current(self):
        """Test that record_action sets _current_turn_action."""
        tracker = BattleStateTracker(history_turns=1, capture_actions=True)

        assert tracker._current_turn_action is None

        tracker.record_action("move drainpunch")
        assert tracker._current_turn_action == "move drainpunch"

        tracker.record_action("switch:4")
        assert tracker._current_turn_action == "switch:4"

    def test_past_turn_actions_in_iter_turn_examples(self):
        """Test that past_turn_actions appears in iter_turn_examples output."""
        battle = {
            "battle_id": "test-battle",
            "team_revelation": {"teams": {"p1": [], "p2": []}},
            "turns": [
                {"turn_number": 1, "events": []},
                {"turn_number": 2, "events": []},
            ]
        }

        tracker = BattleStateTracker(history_turns=2, capture_actions=True)

        examples = list(tracker.iter_turn_examples(battle, player="p1"))

        # All examples should have past_turn_actions key
        for ex in examples:
            assert "past_turn_actions" in ex
            assert isinstance(ex["past_turn_actions"], list)

    def test_no_capture_has_empty_past_actions(self):
        """When capture_actions=False, past_turn_actions should always be empty."""
        battle = {
            "battle_id": "test-battle",
            "team_revelation": {"teams": {"p1": [], "p2": []}},
            "turns": [
                {"turn_number": 1, "events": []},
                {"turn_number": 2, "events": []},
            ]
        }

        tracker = BattleStateTracker(history_turns=2, capture_actions=False)

        examples = list(tracker.iter_turn_examples(battle, player="p1"))

        for ex in examples:
            assert ex["past_turn_actions"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
