"""Tokenizer for TurnEventV1 sequences.

Converts lists of TurnEventV1 event dicts into integer token sequences
suitable for training an autoregressive decoder head.  Each event dict
is mapped to a single composite string key that encodes the event type
and its most important payload fields.  A vocabulary built from training
data assigns each unique key a stable integer ID.
"""

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ---------------------------------------------------------------------------
# Composite key generation
# ---------------------------------------------------------------------------


def event_to_composite_key(event_dict: dict) -> str:
    """Convert a single TurnEventV1 dict into a deterministic composite key.

    The key encodes the event type followed by colon-separated payload
    fields that are most relevant for each event family.  Missing fields
    fall back to empty strings (or 0 for numeric fields) via ``.get()``.

    Parameters
    ----------
    event_dict : dict
        A dict as produced by ``TurnEventV1.to_dict()``.

    Returns
    -------
    str
        A composite string token such as ``"move:p2:thunderbolt:p1"``.
    """
    etype = event_dict.get("event_type", "")

    if etype == "move":
        actor = event_dict.get("actor_side", "")
        move = event_dict.get("move_id", "")
        target = event_dict.get("target_side", "")
        return f"move:{actor}:{move}:{target}"

    if etype == "switch":
        actor = event_dict.get("actor_side", "")
        species = event_dict.get("species_id", "")
        slot = event_dict.get("slot_index", 0)
        return f"switch:{actor}:{species}:{slot}"

    if etype == "damage":
        target = event_dict.get("target_side", "")
        hp_bin = event_dict.get("hp_delta_bin", 0)
        return f"damage:{target}:hp_bin_{hp_bin}"

    if etype == "heal":
        target = event_dict.get("target_side", "")
        hp_bin = event_dict.get("hp_delta_bin", 0)
        return f"heal:{target}:hp_bin_{hp_bin}"

    if etype == "status_start":
        target = event_dict.get("target_side", "")
        status = event_dict.get("status", "")
        return f"status_start:{target}:{status}"

    if etype == "status_end":
        target = event_dict.get("target_side", "")
        status = event_dict.get("status", "")
        return f"status_end:{target}:{status}"

    if etype == "boost":
        target = event_dict.get("target_side", "")
        stat = event_dict.get("boost_stat", "")
        delta = event_dict.get("boost_delta", 0)
        return f"boost:{target}:{stat}:{delta}"

    if etype == "unboost":
        target = event_dict.get("target_side", "")
        stat = event_dict.get("boost_stat", "")
        delta = event_dict.get("boost_delta", 0)
        return f"unboost:{target}:{stat}:{delta}"

    if etype == "faint":
        target = event_dict.get("target_side", "")
        return f"faint:{target}"

    if etype == "weather":
        weather = event_dict.get("weather", "")
        return f"weather:{weather}"

    if etype == "field":
        terrain = event_dict.get("terrain", "")
        removal = "end" if event_dict.get("is_removal", False) else "start"
        return f"field:{terrain}:{removal}"

    if etype == "side_condition":
        actor = event_dict.get("actor_side", "")
        cond = event_dict.get("side_condition", "")
        removal = "end" if event_dict.get("is_removal", False) else "start"
        return f"side_condition:{actor}:{cond}:{removal}"

    if etype == "forme_change":
        target = event_dict.get("target_side", "")
        kind = event_dict.get("forme_change_kind", "")
        # "species" changes use species_id; "tera" uses status field
        if kind == "species":
            detail = event_dict.get("species_id", "")
        else:
            detail = event_dict.get("status", "")
        return f"forme_change:{target}:{kind}:{detail}"

    if etype == "turn_end":
        return "turn_end"

    # Fallback for unknown event types — still deterministic.
    return etype


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------


def build_sequence_vocab(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build a token vocabulary from a collection of training examples.

    Each example is expected to contain a ``"turn_events_v1"`` key whose
    value is a list of event dicts.  All unique composite keys are
    collected, sorted alphabetically, and assigned contiguous integer IDs
    starting after the special tokens (ID 4 onward).

    Parameters
    ----------
    examples : list of dict
        Training examples, each containing ``"turn_events_v1"``.

    Returns
    -------
    dict mapping str to int
        The complete vocabulary including special tokens.
    """
    unique_keys: set = set()

    for ex in examples:
        events = ex.get("turn_events_v1", [])
        for ev in events:
            key = event_to_composite_key(ev)
            unique_keys.add(key)

    vocab: Dict[str, int] = {}
    for idx, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = idx

    for idx, key in enumerate(sorted(unique_keys)):
        vocab[key] = idx + len(SPECIAL_TOKENS)

    return vocab


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_turn_event_sequence(
    events: List[dict],
    vocab: Dict[str, int],
    max_len: int,
) -> List[int]:
    """Encode a list of event dicts into a padded integer token sequence.

    The returned list always has exactly *max_len* elements.  A BOS token
    is prepended and an EOS token is appended.  If the sequence (with
    BOS/EOS) exceeds *max_len*, it is truncated so that EOS is always the
    last non-pad token.  Shorter sequences are right-padded with PAD_ID.

    Parameters
    ----------
    events : list of dict
        Event dicts as produced by ``TurnEventV1.to_dict()``.
    vocab : dict
        Token vocabulary as returned by ``build_sequence_vocab()``.
    max_len : int
        Target sequence length (including BOS, EOS, and padding).

    Returns
    -------
    list of int
        Integer token IDs of length *max_len*.
    """
    token_ids = [BOS_ID]
    for ev in events:
        key = event_to_composite_key(ev)
        token_ids.append(vocab.get(key, UNK_ID))
    token_ids.append(EOS_ID)

    if len(token_ids) > max_len:
        token_ids = token_ids[: max_len - 1]
        token_ids.append(EOS_ID)

    # Pad to max_len.
    token_ids.extend([PAD_ID] * (max_len - len(token_ids)))

    return token_ids


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_turn_event_sequence(
    token_ids: List[int],
    vocab: Dict[str, int],
) -> List[str]:
    """Decode integer token IDs back to human-readable token strings.

    Decoding stops at the first EOS or PAD token (neither is included in
    the output).  Unknown IDs are rendered as ``"<UNK>"``.

    Parameters
    ----------
    token_ids : list of int
        Integer token sequence.
    vocab : dict
        Token vocabulary (str -> int).

    Returns
    -------
    list of str
        Token strings up to (but not including) EOS/PAD.
    """
    reverse: Dict[int, str] = {v: k for k, v in vocab.items()}

    result: List[str] = []
    for tid in token_ids:
        if tid == EOS_ID or tid == PAD_ID:
            break
        result.append(reverse.get(tid, UNK_TOKEN))

    return result


# ---------------------------------------------------------------------------
# History encoding
# ---------------------------------------------------------------------------


def encode_event_history(
    past_events_list: List[List[dict]],
    vocab: Dict[str, int],
    max_turns: int,
    max_events_per_turn: int,
) -> Tuple[List[List[int]], List[float]]:
    """Encode a history of past turn event lists into a fixed-shape matrix.

    Parameters
    ----------
    past_events_list : list of list of dict
        Up to max_turns turn-event lists (oldest first). Each inner list
        contains event dicts as produced by TurnEventV1.to_dict().
    vocab : dict
        Token vocabulary from build_sequence_vocab().
    max_turns : int
        K — the temporal dimension (number of history rows).
    max_events_per_turn : int
        E — maximum tokens per row (event dimension).

    Returns
    -------
    tokens : List[List[int]]
        Shape [max_turns, max_events_per_turn]. Left-padded with PAD rows
        so real turns are right-aligned (index K-num_real .. K-1).
    mask : List[float]
        Shape [max_turns]. 1.0 for real turns, 0.0 for pad rows.
    """
    # Take only the most recent max_turns entries
    clipped = list(past_events_list[-max_turns:]) if len(past_events_list) > max_turns else list(past_events_list)
    num_real = len(clipped)
    num_pad = max_turns - num_real

    tokens: List[List[int]] = []
    mask: List[float] = []

    # Left-pad with all-PAD rows
    pad_row = [PAD_ID] * max_events_per_turn
    for _ in range(num_pad):
        tokens.append(list(pad_row))
        mask.append(0.0)

    # Encode real turns
    for turn_events in clipped:
        row = encode_turn_event_sequence(turn_events, vocab, max_events_per_turn)
        tokens.append(row)
        mask.append(1.0)

    return tokens, mask


def encode_action_history(
    past_actions: List[str],
    action_vocab: Dict[str, int],
    max_turns: int,
) -> Tuple[List[int], List[float]]:
    """Encode past action tokens to IDs and left-pad to max_turns.

    Parameters
    ----------
    past_actions : list of str
        List of action strings, e.g., ["move tackle", "switch pikachu", ...].
    action_vocab : dict
        Mapping from action string to integer ID.
    max_turns : int
        K — pad to this length.

    Returns
    -------
    action_ids : list of int
        Shape [max_turns]. Token IDs, left-padded with 0 (PAD).
    action_mask : list of float
        Shape [max_turns]. 1.0 for real turns, 0.0 for padded.
    """
    # Tokenize each action using vocab.get with UNK fallback
    action_ids = [action_vocab.get(act, action_vocab.get("UNK", 0)) for act in past_actions]

    # Left-pad with zeros to length max_turns
    padded_ids = [0] * (max_turns - len(action_ids)) + action_ids
    padded_ids = padded_ids[-max_turns:]  # Clip to max_turns if too long

    # Create mask: 1.0 for real, 0.0 for padded
    num_real = min(len(action_ids), max_turns)
    num_padded = max_turns - num_real
    mask = [0.0] * num_padded + [1.0] * num_real

    return padded_ids, mask
