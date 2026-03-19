from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from BattleStateTracker import BattleStateTracker, STAT_ORDER
from StaticDex import StaticDex


STATUS_ORDER: List[str] = ["brn", "par", "psn", "tox", "slp", "frz"]
WEATHER_ORDER: List[str] = ["raindance", "sunnyday", "sandstorm", "snow"]
GLOBAL_CONDITION_ORDER: List[str] = [
    "electricterrain",
    "grassyterrain",
    "mistyterrain",
    "psychicterrain",
    "trickroom",
]
SIDE_CONDITION_ORDER: List[str] = [
    "stealthrock",
    "stickyweb",
    "spikes",
    "toxicspikes",
    "reflect",
    "lightscreen",
    "auroraveil",
    "tailwind",
]
SIDE_CONDITION_CAPS: Dict[str, float] = {
    "spikes": 3.0,
    "toxicspikes": 2.0,
}
ACTION_CONTEXT_NONE = "<NONE>"
ACTION_CONTEXT_UNK = "<UNK>"


def one_hot(value: Optional[str], choices: List[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def safe_hp_frac(mon: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns (hp_frac_value, hp_known_flag)
    """
    hf = mon.get("hp_frac")
    if hf is None:
        return (0.0, 0.0)
    return (float(hf), 1.0)


def stable_hash(s: str) -> int:
    # Deterministic (unlike Python's built-in hash, which is salted per run)
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def hashed_move_bag(move_ids: List[str], dim: int = 64) -> List[float]:
    v = [0.0] * dim
    for move_id in move_ids:
        j = stable_hash(move_id) % dim
        v[j] = 1.0
    return v


def hashed_species(species: Optional[str], dim: int = 32) -> List[float]:
    if species is None:
        return [0.0] * dim
    j = stable_hash(species) % dim
    v = [0.0] * dim
    v[j] = 1.0
    return v


def mon_visible_to_player(mon: Optional[Dict[str, Any]], perspective_player: str) -> bool:
    if mon is None:
        return False
    return mon.get("player") == perspective_player or bool(mon.get("public_revealed"))


def visible_species(mon: Optional[Dict[str, Any]], perspective_player: str) -> Optional[str]:
    if not mon_visible_to_player(mon, perspective_player):
        return None
    return mon.get("species")


def visible_status(mon: Optional[Dict[str, Any]], perspective_player: str) -> Optional[str]:
    if not mon_visible_to_player(mon, perspective_player):
        return None
    return mon.get("status")


def visible_observed_moves(mon: Optional[Dict[str, Any]], perspective_player: str) -> List[str]:
    if not mon_visible_to_player(mon, perspective_player):
        return []
    return mon.get("observed_moves", []) or []


def mon_features(
    mon: Optional[Dict[str, Any]],
    perspective_player: str,
    move_hash_dim: int = 64,
    species_hash_dim: int = 32,
) -> List[float]:
    if mon is None:
        base = [0.0] * 7 + [0.0] * len(STAT_ORDER) + [0.0] * len(STATUS_ORDER)
        return base + [0.0] * move_hash_dim + [0.0] * species_hash_dim

    hp_frac_val, hp_known = safe_hp_frac(mon)
    fainted = 1.0 if mon.get("fainted") else 0.0
    visible_flag = 1.0 if mon_visible_to_player(mon, perspective_player) else 0.0
    terastallized = 1.0 if mon.get("terastallized") else 0.0
    ability_known = 1.0 if mon.get("ability") else 0.0
    item_known = 1.0 if mon.get("item") else 0.0

    boosts = mon.get("boosts", {}) or {}
    boost_vec = [float(boosts.get(k, 0)) for k in STAT_ORDER]

    status_vec = one_hot(visible_status(mon, perspective_player), STATUS_ORDER)
    obs_vec = hashed_move_bag(visible_observed_moves(mon, perspective_player), move_hash_dim)
    species_vec = hashed_species(visible_species(mon, perspective_player), species_hash_dim)

    return (
        [
            hp_frac_val,
            hp_known,
            fainted,
            visible_flag,
            terastallized,
            ability_known,
            item_known,
        ]
        + boost_vec
        + status_vec
        + obs_vec
        + species_vec
    )


def bench_slot_features(
    mon: Optional[Dict[str, Any]],
    perspective_player: str,
    species_hash_dim: int = 16,
) -> List[float]:
    """
    Bench representation:
      [revealed, fainted, hp_known, hp_frac, terastallized] +
      [status one-hot] +
      [species hash if visible]
    """
    if mon is None:
        return [0.0] * (5 + len(STATUS_ORDER) + species_hash_dim)

    hp_frac_val, hp_known = safe_hp_frac(mon)
    revealed = 1.0 if mon_visible_to_player(mon, perspective_player) else 0.0
    fainted = 1.0 if mon.get("fainted") else 0.0
    terastallized = 1.0 if mon.get("terastallized") else 0.0
    status_vec = one_hot(visible_status(mon, perspective_player), STATUS_ORDER)
    species_vec = hashed_species(visible_species(mon, perspective_player), species_hash_dim)
    return [revealed, fainted, hp_known, hp_frac_val, terastallized] + status_vec + species_vec


def field_features(state: Dict[str, Any]) -> List[float]:
    field = state.get("field", {}) or {}
    weather_vec = one_hot(field.get("weather"), WEATHER_ORDER)
    global_conditions = set(field.get("global_conditions", []) or [])
    global_vec = [1.0 if cond in global_conditions else 0.0 for cond in GLOBAL_CONDITION_ORDER]
    return weather_vec + global_vec


def side_condition_features(side_state: Dict[str, Any]) -> List[float]:
    conditions = side_state.get("side_conditions", {}) or {}
    feats: List[float] = []
    for cond in SIDE_CONDITION_ORDER:
        cap = SIDE_CONDITION_CAPS.get(cond, 1.0)
        feats.append(min(float(conditions.get(cond, 0)), cap) / cap)
    return feats


def active_feature_dim(move_hash_dim: int = 64, species_hash_dim: int = 32) -> int:
    return 7 + len(STAT_ORDER) + len(STATUS_ORDER) + move_hash_dim + species_hash_dim


def bench_feature_dim(species_hash_dim: int = 16) -> int:
    return 5 + len(STATUS_ORDER) + species_hash_dim


def field_feature_dim() -> int:
    return len(WEATHER_ORDER) + len(GLOBAL_CONDITION_ORDER)


def state_vector_layout() -> List[Dict[str, Any]]:
    active_dim = active_feature_dim()
    bench_dim = bench_feature_dim()
    return [
        {
            "name": "my_active",
            "size": active_dim,
            "description": "My active mon: hp, knowledge flags, boosts, status, tera/item/ability flags, revealed moves hash, visible species hash.",
        },
        {
            "name": "opponent_active",
            "size": active_dim,
            "description": "Opponent active mon with the same feature template, masked to public information.",
        },
        {
            "name": "my_bench",
            "size": 6 * bench_dim,
            "description": "Six bench slots: revealed flag, hp/fainted/tera, bench status, visible species hash.",
        },
        {
            "name": "opponent_bench",
            "size": 6 * bench_dim,
            "description": "Opponent bench slots with the same bench template, masked to public information.",
        },
        {
            "name": "field",
            "size": field_feature_dim(),
            "description": "Weather plus global field conditions such as terrain or Trick Room.",
        },
        {
            "name": "my_side_conditions",
            "size": len(SIDE_CONDITION_ORDER),
            "description": "My side's public side conditions such as rocks, spikes, screens, veil, and Tailwind.",
        },
        {
            "name": "opponent_side_conditions",
            "size": len(SIDE_CONDITION_ORDER),
            "description": "Opponent side's public side conditions.",
        },
        {
            "name": "turn_index",
            "size": 1,
            "description": "Normalized capped turn number.",
        },
    ]


def state_vector_dim() -> int:
    return sum(int(block["size"]) for block in state_vector_layout())


def normalize_boosts(boosts: Optional[Dict[str, Any]]) -> List[float]:
    boost_map = boosts or {}
    return [float(boost_map.get(stat, 0)) / 6.0 for stat in STAT_ORDER]


def tracked_active_outcome_features(
    mon: Optional[Dict[str, Any]],
    perspective_player: str,
) -> List[float]:
    if mon is None:
        return [0.0] * (4 + len(STATUS_ORDER) + len(STAT_ORDER))

    hp_frac_val, hp_known = safe_hp_frac(mon)
    fainted = 1.0 if mon.get("fainted") else 0.0
    terastallized = 1.0 if mon.get("terastallized") else 0.0
    status_vec = one_hot(visible_status(mon, perspective_player), STATUS_ORDER)
    boost_vec = normalize_boosts(mon.get("boosts"))
    return [hp_frac_val, hp_known, fainted, terastallized] + status_vec + boost_vec


def post_active_outcome_features(
    mon: Optional[Dict[str, Any]],
    perspective_player: str,
    species_hash_dim: int = 16,
) -> List[float]:
    return tracked_active_outcome_features(mon, perspective_player) + hashed_species(
        visible_species(mon, perspective_player),
        species_hash_dim,
    )


def turn_outcome_layout() -> List[Dict[str, Any]]:
    tracked_dim = 4 + len(STATUS_ORDER) + len(STAT_ORDER)
    post_active_dim = tracked_dim + 16
    return [
        {
            "name": "my_pre_active_after",
            "size": tracked_dim,
            "description": "End-of-turn summary for my active that started the turn.",
        },
        {
            "name": "opponent_pre_active_after",
            "size": tracked_dim,
            "description": "End-of-turn summary for the opponent active that started the turn.",
        },
        {
            "name": "my_post_active",
            "size": post_active_dim,
            "description": "End-of-turn active on my side, including visible species hash.",
        },
        {
            "name": "opponent_post_active",
            "size": post_active_dim,
            "description": "End-of-turn active on the opponent side, including visible species hash.",
        },
        {
            "name": "active_changed_flags",
            "size": 2,
            "description": "Whether either side ended the turn with a different active.",
        },
        {
            "name": "field_after",
            "size": field_feature_dim(),
            "description": "Weather and global field conditions after the turn resolves.",
        },
        {
            "name": "my_side_conditions_after",
            "size": len(SIDE_CONDITION_ORDER),
            "description": "My side conditions after the turn.",
        },
        {
            "name": "opponent_side_conditions_after",
            "size": len(SIDE_CONDITION_ORDER),
            "description": "Opponent side conditions after the turn.",
        },
    ]


def turn_outcome_dim() -> int:
    return sum(int(block["size"]) for block in turn_outcome_layout())


def encode_turn_outcome(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    perspective_player: str,
) -> List[float]:
    if perspective_player not in ("p1", "p2"):
        raise ValueError("perspective_player must be 'p1' or 'p2'")

    other = "p2" if perspective_player == "p1" else "p1"
    mons_after = state_after["mons"]

    my_pre_uid = state_before[perspective_player]["active_uid"]
    opp_pre_uid = state_before[other]["active_uid"]
    my_post_uid = state_after[perspective_player]["active_uid"]
    opp_post_uid = state_after[other]["active_uid"]

    my_pre_after = mons_after.get(my_pre_uid) if my_pre_uid else None
    opp_pre_after = mons_after.get(opp_pre_uid) if opp_pre_uid else None
    my_post_active = mons_after.get(my_post_uid) if my_post_uid else None
    opp_post_active = mons_after.get(opp_post_uid) if opp_post_uid else None

    vec: List[float] = []
    vec += tracked_active_outcome_features(my_pre_after, perspective_player)
    vec += tracked_active_outcome_features(opp_pre_after, perspective_player)
    vec += post_active_outcome_features(my_post_active, perspective_player)
    vec += post_active_outcome_features(opp_post_active, perspective_player)
    vec += [
        1.0 if my_post_uid != my_pre_uid else 0.0,
        1.0 if opp_post_uid != opp_pre_uid else 0.0,
    ]
    vec += field_features(state_after)
    vec += side_condition_features(state_after[perspective_player])
    vec += side_condition_features(state_after[other])
    return vec


def encode_state_v0(state: Dict[str, Any], perspective_player: str) -> List[float]:
    """
    Fixed-length vector from snapshot "state", from view of perspective_player ("p1" or "p2").
    """
    if perspective_player not in ("p1", "p2"):
        raise ValueError("perspective_player must be 'p1' or 'p2'")

    other = "p2" if perspective_player == "p1" else "p1"

    mons = state["mons"]
    my_active_uid = state[perspective_player]["active_uid"]
    opp_active_uid = state[other]["active_uid"]

    my_active = mons.get(my_active_uid) if my_active_uid else None
    opp_active = mons.get(opp_active_uid) if opp_active_uid else None

    vec: List[float] = []
    vec += mon_features(my_active, perspective_player)
    vec += mon_features(opp_active, perspective_player)

    my_slots = state[perspective_player]["slots"]
    opp_slots = state[other]["slots"]

    for uid in my_slots:
        vec += bench_slot_features(mons.get(uid) if uid else None, perspective_player)

    for uid in opp_slots:
        vec += bench_slot_features(mons.get(uid) if uid else None, perspective_player)

    vec += field_features(state)
    vec += side_condition_features(state[perspective_player])
    vec += side_condition_features(state[other])

    t = float(state.get("turn_index", 0))
    vec.append(min(t, 50.0) / 50.0)

    return vec


def encode_state_with_static(
    state: Dict[str, Any],
    perspective_player: str,
    dex: StaticDex,
) -> Tuple[List[float], Tuple[int, int, int, int, int, int]]:
    """
    Returns:
      x_num: numeric vector (encode_state_v0 + base stats for both actives)
      cats: (my_species, op_species, my_t1, my_t2, op_t1, op_t2)
    """
    x_num = encode_state_v0(state, perspective_player)

    other = "p2" if perspective_player == "p1" else "p1"
    mons = state["mons"]

    my_uid = state[perspective_player]["active_uid"]
    op_uid = state[other]["active_uid"]

    my_mon = mons.get(my_uid) if my_uid else None
    op_mon = mons.get(op_uid) if op_uid else None

    my_species_name = visible_species(my_mon, perspective_player)
    op_species_name = visible_species(op_mon, perspective_player)

    my_sp, my_t1, my_t2, my_stats = dex.lookup(my_species_name)
    op_sp, op_t1, op_t2, op_stats = dex.lookup(op_species_name)

    x_num = x_num + my_stats + op_stats

    cats = (my_sp, op_sp, my_t1, my_t2, op_t1, op_t2)
    return x_num, cats


def vectorize_dataset_static(
    examples: List[Dict[str, Any]],
    move_vocab: Dict[str, int],
    dex: StaticDex,
) -> Tuple[Dict[str, List[Any]], List[int]]:
    """
    Returns:
      X: dict of inputs that matches the embedding model's Input layer names
      y: sparse class ids
    """
    X_num: List[List[float]] = []
    X_my_sp: List[int] = []
    X_op_sp: List[int] = []
    X_my_t1: List[int] = []
    X_my_t2: List[int] = []
    X_op_t1: List[int] = []
    X_op_t2: List[int] = []

    y: List[int] = []
    unk = move_vocab.get("<UNK>", 0)

    for ex in examples:
        player = ex["player"]
        x_num, cats = encode_state_with_static(ex["state"], player, dex)
        my_sp, op_sp, my_t1, my_t2, op_t1, op_t2 = cats

        X_num.append(x_num)
        X_my_sp.append(int(my_sp))
        X_op_sp.append(int(op_sp))
        X_my_t1.append(int(my_t1))
        X_my_t2.append(int(my_t2))
        X_op_t1.append(int(op_t1))
        X_op_t2.append(int(op_t2))

        _, move_id = ex["action"]
        y.append(int(move_vocab.get(move_id, unk)))

    X = {
        "num": X_num,
        "my_species": X_my_sp,
        "op_species": X_op_sp,
        "my_t1": X_my_t1,
        "my_t2": X_my_t2,
        "op_t1": X_op_t1,
        "op_t2": X_op_t2,
    }

    return X, y


def iter_turn_examples_both_players(
    tracker: BattleStateTracker,
    battle: Dict[str, Any],
    include_switches: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Yields examples for BOTH p1 and p2.
    Each example has a 'player' field indicating perspective.
    """
    for player in ("p1", "p2"):
        for ex in tracker.iter_turn_examples(
            battle,
            player=player,
            include_switches=include_switches,
        ):
            yield {
                "battle_id": ex["battle_id"],
                "turn_number": ex["turn_number"],
                "player": player,
                "state": ex["state"],
                "next_state": ex["next_state"],
                "action": ex["action"],
                "action_token": ex["action_token"],
                "opponent_action": ex["opponent_action"],
                "opponent_action_token": ex["opponent_action_token"],
            }


def build_move_vocab(examples: List[Dict[str, Any]], min_count: int = 1) -> Dict[str, int]:
    """
    Move vocab from dataset examples.

    Reserve 0 for <UNK> to handle unseen moves at inference time.
    """
    counts: Dict[str, int] = {}
    for ex in examples:
        kind, move_id = ex["action"]
        if kind != "move":
            continue
        counts[move_id] = counts.get(move_id, 0) + 1

    vocab: Dict[str, int] = {"<UNK>": 0}
    for move_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= min_count:
            vocab[move_id] = len(vocab)
    return vocab


def build_action_vocab(
    examples: List[Dict[str, Any]],
    min_count: int = 1,
    include_switches: bool = True,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        action_token = ex.get("action_token")
        if not action_token:
            continue
        if not include_switches and action_token.startswith("switch:"):
            continue
        counts[action_token] = counts.get(action_token, 0) + 1

    vocab: Dict[str, int] = {"<UNK>": 0}
    if include_switches:
        for slot in range(1, 7):
            token = f"switch:{slot}"
            vocab[token] = len(vocab)

    for action_token, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < min_count:
            continue
        if action_token not in vocab:
            vocab[action_token] = len(vocab)
    return vocab


def build_action_context_vocab(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {
        ACTION_CONTEXT_NONE: 0,
        ACTION_CONTEXT_UNK: 1,
    }
    for ex in examples:
        for token in (ex.get("action_token"), ex.get("opponent_action_token")):
            if not token or token in vocab:
                continue
            vocab[token] = len(vocab)
    return vocab


def vectorize_dataset(
    examples: List[Dict[str, Any]],
    move_vocab: Dict[str, int],
    state_encoder: Callable[[Dict[str, Any], str], List[float]] = encode_state_v0,
) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []

    unk = move_vocab.get("<UNK>", 0)
    for ex in examples:
        player = ex["player"]
        vec = state_encoder(ex["state"], perspective_player=player)
        _, move_id = ex["action"]
        label = move_vocab.get(move_id, unk)
        X.append(vec)
        y.append(label)
    return X, y


def vectorize_action_dataset(
    examples: List[Dict[str, Any]],
    action_vocab: Dict[str, int],
    include_switches: bool = True,
    state_encoder: Callable[[Dict[str, Any], str], List[float]] = encode_state_v0,
) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []

    unk = action_vocab.get("<UNK>", 0)
    for ex in examples:
        action_token = ex.get("action_token")
        if not action_token:
            continue
        if not include_switches and action_token.startswith("switch:"):
            continue

        player = ex["player"]
        vec = state_encoder(ex["state"], perspective_player=player)
        label = action_vocab.get(action_token, unk)
        X.append(vec)
        y.append(label)
    return X, y


def vectorize_action_transition_dataset(
    examples: List[Dict[str, Any]],
    action_vocab: Dict[str, int],
    action_context_vocab: Dict[str, int],
    include_switches: bool = True,
    state_encoder: Callable[[Dict[str, Any], str], List[float]] = encode_state_v0,
    outcome_encoder: Callable[[Dict[str, Any], Dict[str, Any], str], List[float]] = encode_turn_outcome,
) -> Tuple[Dict[str, List[List[float]] | List[int]], List[int], List[List[float]]]:
    X_state: List[List[float]] = []
    X_my_action: List[int] = []
    X_opp_action: List[int] = []
    y_policy: List[int] = []
    y_transition: List[List[float]] = []

    action_unk = action_vocab.get("<UNK>", 0)
    ctx_none = action_context_vocab.get(ACTION_CONTEXT_NONE, 0)
    ctx_unk = action_context_vocab.get(ACTION_CONTEXT_UNK, ctx_none)

    for ex in examples:
        action_token = ex.get("action_token")
        if not action_token:
            continue
        if not include_switches and action_token.startswith("switch:"):
            continue

        player = ex["player"]
        opp_action_token = ex.get("opponent_action_token")

        X_state.append(state_encoder(ex["state"], perspective_player=player))
        X_my_action.append(action_context_vocab.get(action_token, ctx_unk))
        if opp_action_token is None:
            X_opp_action.append(ctx_none)
        else:
            X_opp_action.append(action_context_vocab.get(opp_action_token, ctx_unk))

        y_policy.append(action_vocab.get(action_token, action_unk))
        y_transition.append(outcome_encoder(ex["state"], ex["next_state"], player))

    X = {
        "state": X_state,
        "my_action": X_my_action,
        "opp_action": X_opp_action,
    }
    return X, y_policy, y_transition
