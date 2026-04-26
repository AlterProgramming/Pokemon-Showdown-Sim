from __future__ import annotations

"""Tensorization for the first entity-centric training family.

This file sits between the human-readable entity graph and the Keras model.
Its responsibilities are:
    - build token vocabularies for learned identities
    - keep the explicit public-state slice small and stable
    - emit tensors for policy / transition / value multitask training

It is intentionally conservative. We are not hand-expanding mechanics here; we
are preparing minimal symbolic and numeric inputs so the model can learn
structure from data.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .EntityActionV1 import build_entity_state_view
from .StateVectorization import (
    ACTION_CONTEXT_NONE,
    ACTION_CONTEXT_UNK,
    GLOBAL_CONDITION_ORDER,
    SIDE_CONDITION_ORDER,
    STAT_ORDER,
    build_action_context_vocab,
    build_action_vocab,
    encode_turn_outcome,
)
from .TurnEventTokenizer import build_sequence_vocab, encode_turn_event_sequence, encode_action_history
from .StateVectorization import build_action_vocab as build_policy_vocab


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_OBSERVED_MOVES = 4
MAX_GLOBAL_CONDITIONS = len(GLOBAL_CONDITION_ORDER)
ENTITY_INT_INPUT_KEYS = {
    "pokemon_species",
    "pokemon_item",
    "pokemon_ability",
    "pokemon_tera",
    "pokemon_status",
    "pokemon_side",
    "pokemon_slot",
    "pokemon_observed_moves",
    "weather",
    "global_conditions",
    "my_action",
    "opp_action",
    "event_history_tokens",
    "past_action_ids",
}


def make_token_vocab(tokens: Iterable[str]) -> Dict[str, int]:
    # Reserve 0 for padding and 1 for unknowns so embedding layers can mask consistently.
    ordered = [PAD_TOKEN, UNK_TOKEN]
    seen = set(ordered)
    for token in sorted(str(token) for token in tokens if str(token)):
        if token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return {token: idx for idx, token in enumerate(ordered)}


def vocab_lookup(vocab: Dict[str, int], token: str) -> int:
    """Resolve a token id, falling back to the shared unknown id when needed."""
    return int(vocab.get(str(token), vocab.get(UNK_TOKEN, 1)))


def entity_tensor_layout() -> List[Dict[str, Any]]:
    """Describe the tensor contract saved into metadata for reproducibility."""
    return [
        {
            "name": "pokemon_species",
            "shape": [12],
            "description": "Per-slot species token ids for self slots 1-6 followed by opponent slots 1-6.",
        },
        {
            "name": "pokemon_item",
            "shape": [12],
            "description": "Per-slot item token ids.",
        },
        {
            "name": "pokemon_ability",
            "shape": [12],
            "description": "Per-slot ability token ids.",
        },
        {
            "name": "pokemon_tera",
            "shape": [12],
            "description": "Per-slot tera-type token ids.",
        },
        {
            "name": "pokemon_status",
            "shape": [12],
            "description": "Per-slot status token ids.",
        },
        {
            "name": "pokemon_side",
            "shape": [12],
            "description": "Fixed side ids: self slots then opponent slots.",
        },
        {
            "name": "pokemon_slot",
            "shape": [12],
            "description": "Fixed slot indices 1-6 repeated for both sides.",
        },
        {
            "name": "pokemon_observed_moves",
            "shape": [12, MAX_OBSERVED_MOVES],
            "description": "Per-slot observed move token ids, padded to a fixed width.",
        },
        {
            "name": "pokemon_numeric",
            "shape": [12, 6 + len(STAT_ORDER)],
            "description": "Per-slot explicit numeric public state: hp, reveal/activity/faint/tera flags, and boosts.",
        },
        {
            "name": "weather",
            "shape": [1],
            "description": "Global weather token id.",
        },
        {
            "name": "global_conditions",
            "shape": [MAX_GLOBAL_CONDITIONS],
            "description": "Observed global conditions as token ids, padded to a fixed width.",
        },
        {
            "name": "global_numeric",
            "shape": [1 + 2 * len(SIDE_CONDITION_ORDER)],
            "description": "Normalized turn index plus explicit self/opponent side condition levels.",
        },
    ]


def build_entity_token_vocabs(examples: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Collect the symbolic vocabularies that this family learns through embeddings."""
    tokens_by_field: Dict[str, set[str]] = {
        "species": set(),
        "item": set(),
        "ability": set(),
        "tera": set(),
        "status": set(),
        "move": set(),
        "weather": set(),
        "global_condition": set(),
    }

    for ex in examples:
        state_view = build_entity_state_view(
            state=ex["state"],
            perspective_player=str(ex["player"]),
        )
        for entity in state_view["pokemon_entities"]:
            token_inputs = entity["token_inputs"]
            tokens_by_field["species"].add(str(token_inputs["species"]))
            tokens_by_field["item"].add(str(token_inputs["item"]))
            tokens_by_field["ability"].add(str(token_inputs["ability"]))
            tokens_by_field["tera"].add(str(token_inputs["tera_type"]))
            tokens_by_field["status"].add(str(token_inputs["status"]))
            for move_id in token_inputs.get("observed_moves", []) or []:
                tokens_by_field["move"].add(str(move_id))

        global_tokens = state_view["global_entity"]["token_inputs"]
        tokens_by_field["weather"].add(str(global_tokens["weather"]))
        for cond in global_tokens.get("global_conditions", []) or []:
            tokens_by_field["global_condition"].add(str(cond))

    return {field: make_token_vocab(tokens) for field, tokens in tokens_by_field.items()}


def build_action_vocab(examples: List[dict]) -> Dict[str, int]:
    """
    Build vocabulary of action tokens from all examples.

    Returns:
        {action_string: action_id, ...}
        Special tokens: 0 = PAD, 1 = UNK
    """
    vocab = {
        "PAD": 0,
        "UNK": 1,
    }
    next_id = 2

    for ex in examples:
        for action in ex.get("past_turn_actions", []):
            if action not in vocab:
                vocab[action] = next_id
                next_id += 1

    return vocab


def _pad_or_trim_ids(
    values: Sequence[str],
    *,
    vocab: Dict[str, int],
    width: int,
) -> List[int]:
    """Pad token id sequences to a fixed width so batch shapes stay stable."""
    ids = [vocab_lookup(vocab, value) for value in list(values)[:width]]
    if len(ids) < width:
        ids.extend([vocab[PAD_TOKEN]] * (width - len(ids)))
    return ids


def _normalize_side_conditions(side_conditions: Dict[str, Any]) -> List[float]:
    """Normalize public side-condition levels into a compact numeric slice."""
    feats: List[float] = []
    for cond in SIDE_CONDITION_ORDER:
        raw = float((side_conditions or {}).get(cond, 0))
        cap = 3.0 if cond == "spikes" else 2.0 if cond == "toxicspikes" else 1.0
        feats.append(min(raw, cap) / cap)
    return feats


def encode_entity_state(
    state: Dict[str, Any],
    *,
    perspective_player: str,
    token_vocabs: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    state_view = build_entity_state_view(state=state, perspective_player=perspective_player)
    return encode_entity_state_from_view(
        state_view,
        state=state,
        perspective_player=perspective_player,
        token_vocabs=token_vocabs,
    )


def encode_entity_state_from_view(
    state_view: Dict[str, Any],
    *,
    state: Dict[str, Any],
    perspective_player: str,
    token_vocabs: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    # The entity view gives us the minimal symbolic state; tensorization turns that into
    # learned token ids plus a small explicit numeric slice for public battle state.
    pokemon_entities = list(state_view["pokemon_entities"])
    global_entity = dict(state_view["global_entity"])

    pokemon_species: List[int] = []
    pokemon_item: List[int] = []
    pokemon_ability: List[int] = []
    pokemon_tera: List[int] = []
    pokemon_status: List[int] = []
    pokemon_side: List[int] = []
    pokemon_slot: List[int] = []
    pokemon_observed_moves: List[List[int]] = []
    pokemon_numeric: List[List[float]] = []

    for entity in pokemon_entities:
        token_inputs = entity["token_inputs"]
        state_features = entity["state_features"]
        pokemon_species.append(vocab_lookup(token_vocabs["species"], token_inputs["species"]))
        pokemon_item.append(vocab_lookup(token_vocabs["item"], token_inputs["item"]))
        pokemon_ability.append(vocab_lookup(token_vocabs["ability"], token_inputs["ability"]))
        pokemon_tera.append(vocab_lookup(token_vocabs["tera"], token_inputs["tera_type"]))
        pokemon_status.append(vocab_lookup(token_vocabs["status"], token_inputs["status"]))
        pokemon_side.append(1 if entity["side"] == "self" else 2)
        pokemon_slot.append(int(entity["slot_index"]))
        pokemon_observed_moves.append(
            _pad_or_trim_ids(
                token_inputs.get("observed_moves", []) or [],
                vocab=token_vocabs["move"],
                width=MAX_OBSERVED_MOVES,
            )
        )
        pokemon_numeric.append(
            [
                # Keep only directly observed public state explicit. Species/item/ability/etc.
                # are learned through their token embeddings rather than mechanic tables.
                float(state_features["hp_frac"]),
                float(state_features["hp_known"]),
                float(state_features["public_revealed"]),
                float(state_features["active"]),
                float(state_features["fainted"]),
                float(state_features["terastallized"]),
            ]
            + [float(state_features["boosts"].get(stat, 0.0)) / 6.0 for stat in STAT_ORDER]
        )

    global_tokens = global_entity["token_inputs"]
    weather = [vocab_lookup(token_vocabs["weather"], global_tokens["weather"])]
    global_conditions = _pad_or_trim_ids(
        global_tokens.get("global_conditions", []) or [],
        vocab=token_vocabs["global_condition"],
        width=MAX_GLOBAL_CONDITIONS,
    )
    global_numeric = [min(float(state.get("turn_index", 0)), 50.0) / 50.0]
    global_numeric += _normalize_side_conditions((state.get(perspective_player, {}) or {}).get("side_conditions", {}) or {})
    other_player = "p2" if perspective_player == "p1" else "p1"
    global_numeric += _normalize_side_conditions((state.get(other_player, {}) or {}).get("side_conditions", {}) or {})

    return {
        "pokemon_species": pokemon_species,
        "pokemon_item": pokemon_item,
        "pokemon_ability": pokemon_ability,
        "pokemon_tera": pokemon_tera,
        "pokemon_status": pokemon_status,
        "pokemon_side": pokemon_side,
        "pokemon_slot": pokemon_slot,
        "pokemon_observed_moves": pokemon_observed_moves,
        "pokemon_numeric": pokemon_numeric,
        "weather": weather,
        "global_conditions": global_conditions,
        "global_numeric": global_numeric,
    }


def to_numpy_entity_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert named entity inputs into numpy arrays with the expected dtypes.

    This helper is shared by both training and live inference so the entity family
    keeps one consistent input contract.
    """
    out: Dict[str, np.ndarray] = {}
    for key, values in raw_inputs.items():
        dtype = np.int32 if key in ENTITY_INT_INPUT_KEYS else np.float32
        out[key] = np.asarray(values, dtype=dtype)
    return out


def to_single_example_entity_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Batch one encoded entity example so it can be fed to Keras predict()."""
    return to_numpy_entity_inputs({key: [value] for key, value in raw_inputs.items()})


def vectorize_entity_multitask_dataset(
    examples: Sequence[Dict[str, Any]],
    *,
    policy_vocab: Dict[str, int],
    token_vocabs: Dict[str, Dict[str, int]],
    action_context_vocab: Dict[str, int] | None = None,
    include_switches: bool,
    include_transition: bool = False,
    include_value: bool = False,
    include_sequence: bool = False,
    sequence_vocab: Dict[str, int] | None = None,
    max_seq_len: int = 32,
    include_history: bool = False,
    history_turns: int = 8,
    history_events_per_turn: int = 24,
    include_history_decoding: bool = False,
    action_vocab: Dict[str, int] | None = None,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Convert entity examples into raw tensor lists for supervised multitask training.

    The current entity family still learns from logged human actions, so this
    function stays battle-log friendly:
        - policy target = chosen action token
        - transition target = public next-state summary
        - value target = terminal result for the acting player
        - sequence target = turn event token sequence (when include_sequence=True)
    """
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
    if include_history_decoding:
        X["past_action_ids"] = []
        X["past_action_mask"] = []
    y_policy: List[int] = []
    y_transition: List[List[float]] = []
    y_value: List[float] = []
    y_sequence: List[List[int]] = []
    X_my_action: List[int] = []
    X_opp_action: List[int] = []

    policy_unk = policy_vocab.get(UNK_TOKEN, 1)
    ctx_none = action_context_vocab.get(ACTION_CONTEXT_NONE, 0) if action_context_vocab else 0
    ctx_unk = action_context_vocab.get(ACTION_CONTEXT_UNK, ctx_none) if action_context_vocab else ctx_none

    if include_sequence and sequence_vocab is None:
        raise ValueError("include_sequence requires sequence_vocab")

    if include_history and sequence_vocab is None:
        raise ValueError("include_history=True requires sequence_vocab to be provided.")

    X_history_tokens: list = []
    X_history_mask: list = []

    for ex in examples:
        action = ex.get("action")
        action_token = ex.get("action_token")
        if action is None or not action_token:
            continue
        if (not include_switches) and str(action_token).startswith("switch:"):
            continue

        encoded = encode_entity_state(
            ex["state"],
            perspective_player=str(ex["player"]),
            token_vocabs=token_vocabs,
        )
        for key, value in encoded.items():
            X[key].append(value)

        y_policy.append(int(policy_vocab.get(str(action_token), policy_unk)))

        if include_transition or include_sequence:
            if action_context_vocab is None:
                raise ValueError("include_transition and include_sequence both require action_context_vocab")
            # Action context is used by both the transition head and the sequence head.
            X_my_action.append(int(action_context_vocab.get(str(action_token), ctx_unk)))
            opp_action_token = ex.get("opponent_action_token")
            if opp_action_token is None:
                X_opp_action.append(int(ctx_none))
            else:
                X_opp_action.append(int(action_context_vocab.get(str(opp_action_token), ctx_unk)))

        if include_transition:
            # Transition prediction is conditioned on both sides' chosen actions so the
            # auxiliary head can learn short-horizon battle mechanics from public logs.
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
            # Value stays aligned to terminal outcome probability, not shaped return.
            y_value.append(float(terminal_result))

        if include_sequence:
            turn_events = ex.get("turn_events_v1") or []
            y_sequence.append(
                encode_turn_event_sequence(turn_events, sequence_vocab, max_seq_len)
            )

        if include_history:
            from .TurnEventTokenizer import encode_event_history
            hist_tokens, hist_mask = encode_event_history(
                ex.get("past_turn_events") or [],
                sequence_vocab,
                history_turns,
                history_events_per_turn,
            )
            X_history_tokens.append(hist_tokens)
            X_history_mask.append(hist_mask)

        if include_history_decoding:
            if action_vocab is None:
                raise ValueError("action_vocab required when include_history_decoding=True")

            action_ids, action_mask = encode_action_history(
                ex.get("past_turn_actions", []),
                action_vocab,
                history_turns,
            )
            X["past_action_ids"].append(action_ids)
            X["past_action_mask"].append(action_mask)

    if include_history:
        X["event_history_tokens"] = X_history_tokens
        X["event_history_mask"] = X_history_mask

    if include_transition or include_sequence:
        X["my_action"] = X_my_action
        X["opp_action"] = X_opp_action

    # Validation guard for history decoding
    if include_history_decoding:
        if not action_vocab or len(action_vocab) == 0:
            raise ValueError(
                "Cannot vectorize with history decoding: action_vocab is empty or None"
            )

    targets: Dict[str, List[Any]] = {"policy": y_policy}
    if include_transition:
        targets["transition"] = y_transition
    if include_value:
        targets["value"] = y_value
    if include_sequence:
        targets["sequence"] = y_sequence
    return X, targets


def build_entity_training_bundle(
    examples: Sequence[Dict[str, Any]],
    *,
    include_switches: bool,
    min_move_count: int,
    include_transition: bool,
    include_value: bool,
    include_sequence: bool = False,
    max_seq_len: int = 32,
    include_history: bool = False,
    history_turns: int = 8,
    history_events_per_turn: int = 24,
    include_history_decoding: bool = False,
) -> Dict[str, Any]:
    """Build the vocabulary bundle that defines a concrete entity-family release."""
    # Policy vocab: uses StateVectorization's build_action_vocab with frequency filtering
    policy_vocab = build_policy_vocab(
        examples,
        min_count=min_move_count,
        include_switches=include_switches,
    )
    # action_context_vocab is needed by both the transition head and the sequence head.
    need_action_context = include_transition or include_sequence
    action_context_vocab = build_action_context_vocab(examples) if need_action_context else None
    token_vocabs = build_entity_token_vocabs(examples)
    bundle: Dict[str, Any] = {
        "policy_vocab": policy_vocab,
        "action_context_vocab": action_context_vocab,
        "token_vocabs": token_vocabs,
        "entity_tensor_layout": entity_tensor_layout(),
    }
    need_sequence_vocab = include_sequence or include_history or include_history_decoding
    if need_sequence_vocab:
        bundle["sequence_vocab"] = build_sequence_vocab(examples)

    # Build action vocab for history decoding: simple token→id mapping without filtering
    if include_history_decoding:
        bundle["action_vocab"] = build_action_vocab(examples)

    if include_history:
        bundle["history_turns"] = history_turns
        bundle["history_events_per_turn"] = history_events_per_turn

    return {
        **bundle,
        "action_vocab": bundle.get("action_vocab", {}),
    }
