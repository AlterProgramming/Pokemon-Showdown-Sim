from __future__ import annotations

"""Tensorization utilities for entity_invariance_aux_v1.

This family keeps the existing entity-centric public-state contract, then adds:
    - optional one-step history inputs
    - optional identity remapping for placeholder experiments

The goal is to make identity-regime experiments and latent-history scaffolding
trainable without mutating the older entity_action tensors.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .EntityTensorization import (
    ENTITY_INT_INPUT_KEYS,
    MAX_GLOBAL_CONDITIONS,
    MAX_OBSERVED_MOVES,
    PAD_TOKEN,
    UNK_TOKEN,
    build_entity_training_bundle,
    encode_entity_state,
    entity_tensor_layout,
)
from .StateVectorization import ACTION_CONTEXT_NONE, ACTION_CONTEXT_UNK, encode_turn_outcome


IDENTITY_REGIMES = {"real_id", "placeholder_id", "mixed_id"}
RELABEL_TOKEN_FIELDS = ("species", "item", "ability", "tera", "move")
TOKEN_FIELD_TO_INPUT_KEY = {
    "species": "pokemon_species",
    "item": "pokemon_item",
    "ability": "pokemon_ability",
    "tera": "pokemon_tera",
    "move": "pokemon_observed_moves",
}
INVARIANCE_INT_INPUT_KEYS = set(ENTITY_INT_INPUT_KEYS) | {
    f"prev_{key}" for key in ENTITY_INT_INPUT_KEYS if key not in {"my_action", "opp_action"}
}


def invariance_tensor_layout() -> List[Dict[str, Any]]:
    """Describe the current-plus-previous tensor contract for this family."""
    layout = list(entity_tensor_layout())
    previous_layout: List[Dict[str, Any]] = []
    for entry in entity_tensor_layout():
        prefixed = dict(entry)
        prefixed["name"] = f"prev_{entry['name']}"
        prefixed["description"] = f"Previous turn view for {entry['name']}."
        previous_layout.append(prefixed)
    return layout + previous_layout


def build_identity_remap(
    token_vocabs: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Build deterministic placeholder-id remaps while preserving PAD/UNK ids."""
    rng = np.random.default_rng(seed)
    remap: Dict[str, np.ndarray] = {}
    for field in RELABEL_TOKEN_FIELDS:
        vocab = token_vocabs[field]
        ids = np.arange(len(vocab), dtype=np.int64)
        remapped = ids.copy()
        if len(ids) > 2:
            body = ids[2:].copy()
            rng.shuffle(body)
            remapped[2:] = body
        remap[field] = remapped
    return remap


def _apply_array_remap(values: List[int], remap: np.ndarray) -> List[int]:
    return [int(remap[int(value)]) if 0 <= int(value) < len(remap) else int(value) for value in values]


def apply_identity_remap_to_encoded(
    encoded: Mapping[str, Any],
    remap: Mapping[str, np.ndarray] | None,
) -> Dict[str, Any]:
    """Return a copy of one encoded example under an optional placeholder remap."""
    if remap is None:
        return {key: value for key, value in encoded.items()}

    out = {key: value for key, value in encoded.items()}
    for field in ("species", "item", "ability", "tera"):
        key = TOKEN_FIELD_TO_INPUT_KEY[field]
        out[key] = _apply_array_remap(list(out[key]), remap[field])

    out["pokemon_observed_moves"] = [
        _apply_array_remap(list(move_ids), remap["move"])
        for move_ids in out["pokemon_observed_moves"]
    ]
    return out


def zero_like_encoded_state(encoded: Mapping[str, Any]) -> Dict[str, Any]:
    """Make an all-zero placeholder for missing previous-turn context."""
    zeros: Dict[str, Any] = {}
    for key, value in encoded.items():
        if isinstance(value, list):
            zeros[key] = np.zeros_like(np.asarray(value)).tolist()
        else:
            zeros[key] = value
    return zeros


def _previous_example_lookup(examples: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str, int], Dict[str, Any]]:
    lookup: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for ex in examples:
        battle_id = str(ex.get("battle_id") or "")
        player = str(ex.get("player") or "")
        turn_number = int(ex.get("turn_number") or 0)
        lookup[(battle_id, player, turn_number)] = ex
    return lookup


def to_numpy_invariance_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert current-plus-history entity inputs into numpy arrays."""
    out: Dict[str, np.ndarray] = {}
    for key, values in raw_inputs.items():
        dtype = np.int64 if key in INVARIANCE_INT_INPUT_KEYS else np.float32
        out[key] = np.asarray(values, dtype=dtype)
    return out


def concat_invariance_batches(*batches: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Concatenate multiple named input batches along the example axis."""
    if not batches:
        return {}
    keys = list(batches[0].keys())
    return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in keys}


def concat_target_batches(*batches: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Concatenate multiple target batches along the example axis."""
    if not batches:
        return {}
    keys = list(batches[0].keys())
    return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in keys}


def vectorize_entity_invariance_dataset(
    examples: Sequence[Dict[str, Any]],
    *,
    policy_vocab: Dict[str, int],
    token_vocabs: Dict[str, Dict[str, int]],
    action_context_vocab: Dict[str, int] | None = None,
    include_switches: bool,
    include_transition: bool = False,
    include_value: bool = False,
    include_history: bool = True,
    identity_regime: str = "real_id",
    placeholder_seed: int = 0,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Vectorize one dataset split for entity_invariance_aux_v1."""
    if identity_regime not in IDENTITY_REGIMES - {"mixed_id"}:
        raise ValueError(f"identity_regime must be one of real_id or placeholder_id, got {identity_regime!r}")

    remap = None
    if identity_regime == "placeholder_id":
        remap = build_identity_remap(token_vocabs, seed=placeholder_seed)

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
    }
    if include_history:
        for key in list(X.keys()):
            X[f"prev_{key}"] = []

    y_policy: List[int] = []
    y_transition: List[List[float]] = []
    y_value: List[float] = []
    X_my_action: List[int] = []
    X_opp_action: List[int] = []

    policy_unk = policy_vocab.get(UNK_TOKEN, 1)
    ctx_none = action_context_vocab.get(ACTION_CONTEXT_NONE, 0) if action_context_vocab else 0
    ctx_unk = action_context_vocab.get(ACTION_CONTEXT_UNK, ctx_none) if action_context_vocab else ctx_none
    previous_lookup = _previous_example_lookup(examples)

    for ex in examples:
        action = ex.get("action")
        action_token = ex.get("action_token")
        if action is None or not action_token:
            continue
        if (not include_switches) and str(action_token).startswith("switch:"):
            continue

        current_encoded = encode_entity_state(
            ex["state"],
            perspective_player=str(ex["player"]),
            token_vocabs=token_vocabs,
        )
        current_encoded = apply_identity_remap_to_encoded(current_encoded, remap)
        for key, value in current_encoded.items():
            X[key].append(value)

        if include_history:
            battle_id = str(ex.get("battle_id") or "")
            player = str(ex.get("player") or "")
            turn_number = int(ex.get("turn_number") or 0)
            prev_example = previous_lookup.get((battle_id, player, turn_number - 1))
            if prev_example is None:
                previous_encoded = zero_like_encoded_state(current_encoded)
            else:
                previous_encoded = encode_entity_state(
                    prev_example["state"],
                    perspective_player=player,
                    token_vocabs=token_vocabs,
                )
                previous_encoded = apply_identity_remap_to_encoded(previous_encoded, remap)
            for key, value in previous_encoded.items():
                X[f"prev_{key}"].append(value)

        y_policy.append(int(policy_vocab.get(str(action_token), policy_unk)))

        if include_transition:
            if action_context_vocab is None:
                raise ValueError("include_transition requires action_context_vocab")
            X_my_action.append(int(action_context_vocab.get(str(action_token), ctx_unk)))
            opp_action_token = ex.get("opponent_action_token")
            if opp_action_token is None:
                X_opp_action.append(int(ctx_none))
            else:
                X_opp_action.append(int(action_context_vocab.get(str(opp_action_token), ctx_unk)))
            y_transition.append(
                encode_turn_outcome(
                    ex["state"],
                    ex["next_state"],
                    str(ex["player"]),
                )
            )

        if include_value:
            terminal_result = ex.get("terminal_result")
            if terminal_result is None:
                raise ValueError("include_value requires terminal_result on every example")
            y_value.append(float(terminal_result))

    if include_transition:
        X["my_action"] = X_my_action
        X["opp_action"] = X_opp_action

    targets: Dict[str, List[Any]] = {"policy": y_policy}
    if include_transition:
        targets["transition"] = y_transition
    if include_value:
        targets["value"] = y_value
    return X, targets
