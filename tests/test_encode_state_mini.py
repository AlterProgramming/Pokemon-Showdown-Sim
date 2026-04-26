import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.StateVectorization import encode_state_mini, encode_state_v0, mini_state_dim


def _minimal_state():
    return {
        "p1": {
            "active_uid": "a1",
            "slots": ["a1", "b1", "c1", "d1", "e1", "f1"],
            "side_conditions": {},
        },
        "p2": {
            "active_uid": "a2",
            "slots": ["a2", "b2", "c2", "d2", "e2", "f2"],
            "side_conditions": {},
        },
        "mons": {
            "a1": {"player": "p1"},
            "a2": {"player": "p2"},
        },
        "field": {},
        "turn_index": 3,
    }


def test_mini_state_dim_constant():
    assert mini_state_dim() == 262


def test_mini_dim_matches_constant():
    state = _minimal_state()
    vec = encode_state_mini(state, "p1")
    assert len(vec) == mini_state_dim()


def test_mini_smaller_than_v0():
    state = _minimal_state()
    assert len(encode_state_mini(state, "p1")) < len(encode_state_v0(state, "p1"))


def test_mini_p2_perspective():
    state = _minimal_state()
    vec = encode_state_mini(state, "p2")
    assert len(vec) == 262


def test_mini_invalid_perspective():
    import pytest
    state = _minimal_state()
    with pytest.raises(ValueError):
        encode_state_mini(state, "p3")


def test_mini_all_floats():
    state = _minimal_state()
    vec = encode_state_mini(state, "p1")
    assert all(isinstance(v, float) for v in vec)


def test_history_vectorization_shape():
    import numpy as np
    from core.StateVectorization import vectorize_multitask_dataset
    from core.TurnEventTokenizer import build_sequence_vocab

    # Minimal battle examples with past_turn_events
    examples = [
        {
            "state": _minimal_state(),
            "player": "p1",
            "action": ("move", 1),
            "action_token": "move:1",
            "past_turn_events": [
                [{"type": "start", "player": "p1"}] * 6,
                [{"type": "action", "player": "p1"}] * 6,
            ] * 4,  # 8 turns of ~6 events each
        },
        {
            "state": _minimal_state(),
            "player": "p1",
            "action": ("move", 2),
            "action_token": "move:2",
            "past_turn_events": [
                [{"type": "start", "player": "p1"}] * 6,
            ] * 8,
        },
    ]

    # Build vocabs (required for vectorization)
    policy_vocab = {"move:1": 0, "move:2": 1, "switch:a1": 2}
    sequence_vocab = build_sequence_vocab(examples)

    # Vectorize with history enabled
    X, y = vectorize_multitask_dataset(
        examples,
        policy_vocab,
        include_switches=False,
        use_action_tokens=True,
        include_history=True,
        history_turns=8,
        history_events_per_turn=24,
        sequence_vocab=sequence_vocab,
    )

    # Check shapes
    assert "event_history_tokens" in X, "event_history_tokens not in X"
    assert "event_history_mask" in X, "event_history_mask not in X"

    # Convert to numpy arrays for shape checking
    hist_tokens = np.array(X["event_history_tokens"], dtype=np.int32)
    hist_mask = np.array(X["event_history_mask"], dtype=np.float32)

    assert hist_tokens.shape == (2, 8, 24), \
        f"Expected shape (2, 8, 24), got {hist_tokens.shape}"
    assert hist_mask.shape == (2, 8), \
        f"Expected shape (2, 8), got {hist_mask.shape}"

    # Verify dtypes
    assert hist_tokens.dtype == np.int32, \
        f"event_history_tokens should be int32, got {hist_tokens.dtype}"
    assert hist_mask.dtype == np.float32, \
        f"event_history_mask should be float32, got {hist_mask.dtype}"
