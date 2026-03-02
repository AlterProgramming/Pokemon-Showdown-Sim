from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Iterator

from BattleStateTracker import STAT_ORDER, BattleStateTracker

from StaticDex import StaticDex

STATUS_ORDER: List[str] = ["brn", "par", "psn", "tox", "slp", "frz"]


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

def hashed_species(species: Optional[str], dim: int = 32) -> List[float]:
    if species is None:
        return [0.0] * dim
    j = stable_hash(species) % dim
    v = [0.0] * dim
    v[j] = 1.0
    return v

# def mon_features(mon: Optional[Dict[str, Any]]) -> List[float]:
#     """
#     Features for an active mon.
#     If mon is None -> all zeros with known flags = 0.
#     Layout:
#       [hp_frac, hp_known, fainted] +
#       [boosts in STAT_ORDER] +
#       [status one-hot in STATUS_ORDER]
#     """
#     if mon is None:
#         return [0.0, 0.0, 0.0] + [0.0] * len(STAT_ORDER) + [0.0] * len(STATUS_ORDER)

#     hp_frac_val, hp_known = safe_hp_frac(mon)
#     fainted = 1.0 if mon.get("fainted") else 0.0

#     boosts = mon.get("boosts", {}) or {}
#     boost_vec = [float(boosts.get(k, 0)) for k in STAT_ORDER]

#     status = mon.get("status")
#     status_vec = one_hot(status, STATUS_ORDER)

#     return [hp_frac_val, hp_known, fainted] + boost_vec + status_vec
# def mon_features(mon: Optional[Dict[str, Any]], move_hash_dim: int = 64) -> List[float]:
#     # [hp_frac, hp_known, fainted] + boosts + status + observed_moves_hash
#     if mon is None:
#         base = [0.0, 0.0, 0.0] + [0.0] * len(STAT_ORDER) + [0.0] * len(STATUS_ORDER)
#         return base + [0.0] * move_hash_dim

#     hp_frac_val, hp_known = safe_hp_frac(mon)
#     fainted = 1.0 if mon.get("fainted") else 0.0

#     boosts = mon.get("boosts", {}) or {}
#     boost_vec = [float(boosts.get(k, 0)) for k in STAT_ORDER]

#     status = mon.get("status")
#     status_vec = one_hot(status, STATUS_ORDER)

#     obs = mon.get("observed_moves", []) or []
#     obs_vec = hashed_move_bag(obs, dim=move_hash_dim)

#     return [hp_frac_val, hp_known, fainted] + boost_vec + status_vec + obs_vec

def mon_features(
    mon: Optional[Dict[str, Any]],
    move_hash_dim: int = 64,
    species_hash_dim: int = 32,
) -> List[float]:
    if mon is None:
        base = [0.0, 0.0, 0.0] \
             + [0.0] * len(STAT_ORDER) \
             + [0.0] * len(STATUS_ORDER)
        return base + [0.0] * move_hash_dim + [0.0] * species_hash_dim

    hp_frac_val, hp_known = safe_hp_frac(mon)
    fainted = 1.0 if mon.get("fainted") else 0.0

    boosts = mon.get("boosts", {}) or {}
    boost_vec = [float(boosts.get(k, 0)) for k in STAT_ORDER]

    status_vec = one_hot(mon.get("status"), STATUS_ORDER)

    obs_vec = hashed_move_bag(mon.get("observed_moves", []), move_hash_dim)
    species_vec = hashed_species(mon.get("species"), species_hash_dim)

    return (
        [hp_frac_val, hp_known, fainted]
        + boost_vec
        + status_vec
        + obs_vec
        + species_vec
    )

def bench_slot_features(mon: Optional[Dict[str, Any]]) -> List[float]:
    """
    Tiny bench representation: [fainted, hp_known, hp_frac]
    """
    if mon is None:
        return [0.0, 0.0, 0.0]
    hp_frac_val, hp_known = safe_hp_frac(mon)
    fainted = 1.0 if mon.get("fainted") else 0.0
    return [fainted, hp_known, hp_frac_val]


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
    vec += mon_features(my_active)
    vec += mon_features(opp_active)

    # Bench slots (6) for each side based on stable slot list
    my_slots = state[perspective_player]["slots"]
    opp_slots = state[other]["slots"]

    for uid in my_slots:
        vec += bench_slot_features(mons.get(uid) if uid else None)

    for uid in opp_slots:
        vec += bench_slot_features(mons.get(uid) if uid else None)

    # Simple capped turn signal
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
    x_num = encode_state_v0(state, perspective_player)  # your existing numeric vector

    other = "p2" if perspective_player == "p1" else "p1"
    mons = state["mons"]

    my_uid = state[perspective_player]["active_uid"]
    op_uid = state[other]["active_uid"]

    my_species_name = mons.get(my_uid, {}).get("species") if my_uid else None
    op_species_name = mons.get(op_uid, {}).get("species") if op_uid else None

    my_sp, my_t1, my_t2, my_stats = dex.lookup(my_species_name)
    op_sp, op_t1, op_t2, op_stats = dex.lookup(op_species_name)

    # Append base stats to numeric vector (simple + effective)
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
) -> Iterator[Dict[str, Any]]:
    """
    Yields move-only examples for BOTH p1 and p2.
    Each example has a 'player' field indicating perspective.
    """
    for player in ("p1", "p2"):
        for ex in tracker.iter_turn_examples(battle, player=player, include_switches=False):
            yield {
                "battle_id": ex["battle_id"],
                "turn_number": ex["turn_number"],
                "player": player,
                "state": ex["state"],
                "action": ex["action"],  # ("move", move_id)
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
    for move_id, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= min_count:
            vocab[move_id] = len(vocab)
    return vocab

def stable_hash(s: str) -> int:
    # Deterministic (unlike Python's built-in hash, which is salted per run)
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def hashed_move_bag(move_ids: List[str], dim: int = 64) -> List[float]:
    v = [0.0] * dim
    for m in move_ids:
        j = stable_hash(m) % dim
        v[j] = 1.0
    return v


def vectorize_dataset(
    examples: List[Dict[str, Any]],
    move_vocab: Dict[str, int],
) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []

    unk = move_vocab.get("<UNK>", 0)
    for ex in examples:
        player = ex["player"]
        vec = encode_state_v0(ex["state"], perspective_player=player)
        _, move_id = ex["action"]
        label = move_vocab.get(move_id, unk)
        X.append(vec)
        y.append(label)

    return X, y
