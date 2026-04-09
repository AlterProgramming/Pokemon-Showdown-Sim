from __future__ import annotations

"""Tensorization for the first legal-action-conditioned entity policy family.

This v2 line keeps the existing public entity-state encoding from v1, but changes
the policy target from a global action vocabulary class to an index over the legal
actions available on the current turn.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .EntityActionV1 import build_entity_action_graph
from .EntityTensorization import (
    PAD_TOKEN,
    encode_entity_state,
    entity_tensor_layout,
    vocab_lookup,
)


MAX_LEGAL_ACTIONS = 10
ACTION_TYPE_PAD = 0
ACTION_TYPE_MOVE = 1
ACTION_TYPE_SWITCH = 2
ENTITY_V2_INT_INPUT_KEYS = {
    "candidate_type",
    "candidate_move",
    "candidate_switch_slot",
}


def entity_candidate_layout(*, max_candidates: int = MAX_LEGAL_ACTIONS) -> List[Dict[str, Any]]:
    """Describe the legal-action tensors appended by the v2 policy family."""
    return [
        {
            "name": "candidate_type",
            "shape": [max_candidates],
            "description": "Per-candidate type ids: 0=pad, 1=move, 2=switch.",
        },
        {
            "name": "candidate_move",
            "shape": [max_candidates],
            "description": "Per-candidate move token ids when candidate_type=move, else PAD.",
        },
        {
            "name": "candidate_switch_slot",
            "shape": [max_candidates],
            "description": "Per-candidate self-side slot indices 1-6 when candidate_type=switch, else 0.",
        },
        {
            "name": "candidate_mask",
            "shape": [max_candidates],
            "description": "Float mask: 1 for real legal candidates, 0 for padding.",
        },
    ]


def entity_action_v2_layout(*, max_candidates: int = MAX_LEGAL_ACTIONS) -> List[Dict[str, Any]]:
    """Describe the full tensor contract for v2."""
    return entity_tensor_layout() + entity_candidate_layout(max_candidates=max_candidates)


def _switch_slot_from_candidate(candidate: Dict[str, Any]) -> int:
    token = str(candidate.get("token") or "")
    if token.startswith("switch:"):
        try:
            return int(token.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def encode_entity_candidates(
    state: Dict[str, Any],
    *,
    perspective_player: str,
    token_vocabs: Dict[str, Dict[str, int]],
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
    chosen_action: Tuple[str, str] | None = None,
    chosen_action_token: str | None = None,
    max_candidates: int = MAX_LEGAL_ACTIONS,
) -> Dict[str, Any]:
    """Encode the current turn's legal actions into fixed candidate tensors."""
    graph = build_entity_action_graph(
        state=state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        chosen_action=chosen_action,
        chosen_action_token=chosen_action_token,
    )
    action_candidates = list(graph["action_candidates"])[:max_candidates]

    candidate_type = [ACTION_TYPE_PAD] * max_candidates
    candidate_move = [token_vocabs["move"].get(PAD_TOKEN, 0)] * max_candidates
    candidate_switch_slot = [0] * max_candidates
    candidate_mask = [0.0] * max_candidates
    candidate_tokens = [""] * max_candidates
    chosen_candidate_index = -1

    for idx, candidate in enumerate(action_candidates):
        candidate_mask[idx] = 1.0
        candidate_tokens[idx] = str(candidate.get("token") or "")
        action_type = str(candidate.get("action_type") or "")
        if action_type == "move":
            candidate_type[idx] = ACTION_TYPE_MOVE
            move_id = candidate_tokens[idx].split(":", 1)[1] if ":" in candidate_tokens[idx] else ""
            candidate_move[idx] = vocab_lookup(token_vocabs["move"], move_id)
        elif action_type == "switch":
            candidate_type[idx] = ACTION_TYPE_SWITCH
            candidate_switch_slot[idx] = _switch_slot_from_candidate(candidate)
        if candidate.get("is_chosen"):
            chosen_candidate_index = idx

    if chosen_action_token is not None and chosen_candidate_index < 0:
        raise ValueError(f"Chosen action {chosen_action_token!r} is not present in encoded candidates")

    return {
        "candidate_type": candidate_type,
        "candidate_move": candidate_move,
        "candidate_switch_slot": candidate_switch_slot,
        "candidate_mask": candidate_mask,
        "candidate_tokens": candidate_tokens,
        "chosen_candidate_index": chosen_candidate_index,
    }


def encode_entity_state_with_candidates(
    state: Dict[str, Any],
    *,
    perspective_player: str,
    token_vocabs: Dict[str, Dict[str, int]],
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
    chosen_action: Tuple[str, str] | None = None,
    chosen_action_token: str | None = None,
    max_candidates: int = MAX_LEGAL_ACTIONS,
) -> Dict[str, Any]:
    """Combine v1 state inputs with v2 legal-action candidate inputs."""
    encoded_state = encode_entity_state(
        state,
        perspective_player=perspective_player,
        token_vocabs=token_vocabs,
    )
    encoded_candidates = encode_entity_candidates(
        state,
        perspective_player=perspective_player,
        token_vocabs=token_vocabs,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        chosen_action=chosen_action,
        chosen_action_token=chosen_action_token,
        max_candidates=max_candidates,
    )
    return {
        **encoded_state,
        "candidate_type": encoded_candidates["candidate_type"],
        "candidate_move": encoded_candidates["candidate_move"],
        "candidate_switch_slot": encoded_candidates["candidate_switch_slot"],
        "candidate_mask": encoded_candidates["candidate_mask"],
        "candidate_tokens": encoded_candidates["candidate_tokens"],
        "chosen_candidate_index": encoded_candidates["chosen_candidate_index"],
    }


def to_single_example_entity_v2_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Batch one encoded v2 example so it can be fed to Keras predict()."""
    batched = {
        key: [value]
        for key, value in raw_inputs.items()
        if key not in {"candidate_tokens", "chosen_candidate_index"}
    }
    out: Dict[str, np.ndarray] = {}
    for key, values in batched.items():
        dtype = np.int64 if key in ENTITY_V2_INT_INPUT_KEYS else np.float32
        if key not in ENTITY_V2_INT_INPUT_KEYS and key != "candidate_mask":
            from .EntityTensorization import ENTITY_INT_INPUT_KEYS

            if key in ENTITY_INT_INPUT_KEYS:
                dtype = np.int64
        out[key] = np.asarray(values, dtype=dtype)
    return out


def vectorize_entity_v2_policy_dataset(
    examples: Sequence[Dict[str, Any]],
    *,
    token_vocabs: Dict[str, Dict[str, int]],
    include_value: bool = False,
    max_candidates: int = MAX_LEGAL_ACTIONS,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Convert examples into v2 legal-action-conditioned training tensors."""
    X: Dict[str, List[Any]] = {
        "pokemon_species": [],
        "pokemon_item": [],
        "pokemon_ability": [],
        "pokemon_tera": [],
        "pokemon_status": [],
        "pokemon_side": [],
        "pokemon_slot": [],
        "pokemon_observed_moves": [],
        "pokemon_numeric": [],
        "weather": [],
        "global_conditions": [],
        "global_numeric": [],
        "candidate_type": [],
        "candidate_move": [],
        "candidate_switch_slot": [],
        "candidate_mask": [],
    }
    y_policy: List[int] = []
    y_value: List[float] = []

    for ex in examples:
        action_token = ex.get("chosen_action_token") or ex.get("action_token")
        action = ex.get("action")
        if not action_token:
            continue

        encoded = encode_entity_state_with_candidates(
            ex["state"],
            perspective_player=str(ex["player"]),
            token_vocabs=token_vocabs,
            legal_moves=ex.get("legal_moves"),
            legal_switches=ex.get("legal_switches"),
            chosen_action=action,
            chosen_action_token=str(action_token),
            max_candidates=max_candidates,
        )
        for key in X:
            X[key].append(encoded[key])
        y_policy.append(int(encoded["chosen_candidate_index"]))

        if include_value:
            terminal_result = ex.get("terminal_result")
            if terminal_result is None:
                raise ValueError("include_value requires terminal_result on every example")
            y_value.append(float(terminal_result))

    targets: Dict[str, List[Any]] = {"policy": y_policy}
    if include_value:
        targets["value"] = y_value
    return X, targets


def build_entity_v2_training_bundle(
    examples: Sequence[Dict[str, Any]],
    *,
    max_candidates: int = MAX_LEGAL_ACTIONS,
) -> Dict[str, Any]:
    """Build the minimal release bundle for the candidate-conditioned family."""
    from .EntityTensorization import build_entity_token_vocabs

    return {
        "token_vocabs": build_entity_token_vocabs(examples),
        "entity_tensor_layout": entity_action_v2_layout(max_candidates=max_candidates),
        "max_candidates": int(max_candidates),
    }
