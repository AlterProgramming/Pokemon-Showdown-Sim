from __future__ import annotations

"""Entity-centric view builders for the first graph-ready policy family.

This module does not train anything by itself. Its job is to convert the raw
battle tracker state into a stable, human-readable entity view that later code
can tensorize, visualize, or eventually feed to a graph-style model.

The key design choice in this family is:
    - explicit public battle state stays numeric
    - stable symbolic identity stays as tokens
    - hidden or unknown information becomes explicit unknown tokens

That gives us a graph-ready representation without hand-authoring mechanic
tables such as move power, type matchups, or item effect databases.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .BattleStateTracker import STAT_ORDER


STATUS_NONE_TOKEN = "<NONE_STATUS>"
WEATHER_NONE_TOKEN = "<NONE_WEATHER>"
UNK_SPECIES_TOKEN = "<UNK_SPECIES>"
UNK_ITEM_TOKEN = "<UNK_ITEM>"
UNK_ABILITY_TOKEN = "<UNK_ABILITY>"
UNK_TERA_TOKEN = "<UNK_TERA>"


def other_player(player: str) -> str:
    """Return the opposing player id for a two-player battle state."""
    if player == "p1":
        return "p2"
    if player == "p2":
        return "p1"
    raise ValueError("player must be 'p1' or 'p2'")


def _safe_hp(mon: Optional[dict[str, Any]]) -> tuple[float, float]:
    """Return normalized HP plus a flag for whether HP is currently known.

    We use a pair rather than a single number because "unknown HP" and
    "known HP of zero" are meaningfully different situations.
    """
    if mon is None:
        return 0.0, 0.0
    hp_frac = mon.get("hp_frac")
    if hp_frac is None:
        return 0.0, 0.0
    return float(hp_frac), 1.0


def _token_or_unknown(value: Optional[str], unknown_token: str) -> str:
    """Normalize missing symbolic identities into explicit unknown tokens."""
    if value is None or value == "":
        return unknown_token
    return str(value)


def _normalize_action_token(action: Tuple[str, str], *, switch_slot: int | None = None) -> str:
    """Translate tracker-style actions into the shared action-token format."""
    kind, value = action
    if kind == "move":
        return f"move:{value}"
    if kind == "switch":
        if switch_slot is None:
            return f"switch:{value}"
        return f"switch:{switch_slot}"
    return f"{kind}:{value}"


def _fallback_move_candidates(
    active_mon: Optional[dict[str, Any]],
    chosen_action: Optional[Tuple[str, str]],
) -> list[str]:
    """Recover move candidates from observed moves when no legal request exists.

    Offline battle logs do not include the full simulator request object, so this
    fallback approximates the legal move set from public observations. We also
    force the chosen move into the candidate list so supervised training does not
    silently drop the recorded action.
    """
    move_ids = list((active_mon or {}).get("observed_moves", []) or [])
    if chosen_action is not None and chosen_action[0] == "move":
        chosen_move = str(chosen_action[1])
        if chosen_move not in move_ids:
            move_ids.append(chosen_move)
    return sorted(dict.fromkeys(str(move_id) for move_id in move_ids))


def _fallback_switch_slots(
    state: dict[str, Any],
    *,
    player: str,
    chosen_action: Optional[Tuple[str, str]] = None,
    chosen_action_token: Optional[str] = None,
) -> list[int]:
    """Recover plausible switch targets from side slots when request data is absent.

    Like move fallback, we force the recorded chosen switch into the candidate
    list so offline replay training does not fail when legality reconstruction
    misses the logged switch target.
    """
    side = state.get(player, {}) or {}
    active_uid = side.get("active_uid")
    mons = state.get("mons", {}) or {}
    slots = side.get("slots", []) or []
    switch_slots: list[int] = []
    for idx, uid in enumerate(slots, start=1):
        if uid is None or uid == active_uid:
            continue
        mon = mons.get(uid)
        if mon is not None and mon.get("fainted"):
            continue
        switch_slots.append(idx)

    chosen_slot: int | None = None
    if chosen_action is not None and chosen_action[0] == "switch":
        chosen_uid = str(chosen_action[1])
        for idx, uid in enumerate(slots, start=1):
            if uid == chosen_uid:
                chosen_slot = idx
                break
    if chosen_slot is None and chosen_action_token and str(chosen_action_token).startswith("switch:"):
        try:
            chosen_slot = int(str(chosen_action_token).split(":", 1)[1])
        except ValueError:
            chosen_slot = None

    if chosen_slot is not None and chosen_slot not in switch_slots:
        switch_slots.append(chosen_slot)
    return sorted(dict.fromkeys(int(slot) for slot in switch_slots if int(slot) > 0))


def _build_pokemon_entity(
    *,
    state: dict[str, Any],
    player: str,
    perspective_player: str,
    slot_index: int,
) -> dict[str, Any]:
    """Build one Pokemon-slot entity from the public tracker state.

    Each slot exists whether or not it is fully revealed. That matters because the
    next family wants these same slots to later carry belief state instead of
    disappearing when information is hidden.
    """
    side = state.get(player, {}) or {}
    mons = state.get("mons", {}) or {}
    slot_uids = side.get("slots", []) or []
    uid = slot_uids[slot_index - 1] if 0 <= slot_index - 1 < len(slot_uids) else None
    mon = mons.get(uid) if uid is not None else None
    hp_frac, hp_known = _safe_hp(mon)
    boosts = (mon or {}).get("boosts", {}) or {}
    active_uid = side.get("active_uid")
    perspective_side = "self" if player == perspective_player else "opponent"
    observed_moves = list((mon or {}).get("observed_moves", []) or [])

    return {
        "id": f"pokemon:{perspective_side}:slot{slot_index}",
        "entity_type": "pokemon",
        "side": perspective_side,
        "player": player,
        "slot_index": slot_index,
        "uid": uid,
        "token_inputs": {
            # These symbolic identities are intentionally left as tokens. The model
            # will learn what they mean through embeddings and auxiliary pressure.
            "species": _token_or_unknown((mon or {}).get("species"), UNK_SPECIES_TOKEN),
            "item": _token_or_unknown((mon or {}).get("item"), UNK_ITEM_TOKEN),
            "ability": _token_or_unknown((mon or {}).get("ability"), UNK_ABILITY_TOKEN),
            "tera_type": _token_or_unknown((mon or {}).get("tera_type"), UNK_TERA_TOKEN),
            "status": _token_or_unknown((mon or {}).get("status"), STATUS_NONE_TOKEN),
            "observed_moves": [str(move_id) for move_id in observed_moves],
        },
        "state_features": {
            # This explicit slice is only what is publicly true right now, not a
            # mechanic expansion of what the Pokemon can do in theory.
            "hp_frac": hp_frac,
            "hp_known": hp_known,
            "public_revealed": 1.0 if (mon or {}).get("public_revealed") else 0.0,
            "active": 1.0 if uid is not None and uid == active_uid else 0.0,
            "fainted": 1.0 if (mon or {}).get("fainted") else 0.0,
            "terastallized": 1.0 if (mon or {}).get("terastallized") else 0.0,
            "boosts": {stat: float(boosts.get(stat, 0)) for stat in STAT_ORDER},
        },
        "display": {
            "title": (mon or {}).get("species") or f"{perspective_side} slot {slot_index}",
            "subtitle": f"{perspective_side} slot {slot_index}",
        },
    }


def _build_global_entity(state: dict[str, Any]) -> dict[str, Any]:
    """Build the shared battle-context entity.

    The global node carries board-wide state such as weather and side conditions so
    the Pokemon-slot entities do not need to duplicate that information.
    """
    field = state.get("field", {}) or {}
    p1 = state.get("p1", {}) or {}
    p2 = state.get("p2", {}) or {}
    return {
        "id": "global:battle",
        "entity_type": "global",
        "token_inputs": {
            "weather": _token_or_unknown(field.get("weather"), WEATHER_NONE_TOKEN),
            "global_conditions": [str(cond) for cond in sorted(field.get("global_conditions", []) or [])],
            "my_side_conditions": [],
            "opponent_side_conditions": [],
        },
        "state_features": {
            "turn_index": float(state.get("turn_index", 0)),
            "p1_side_conditions": {str(key): float(value) for key, value in (p1.get("side_conditions", {}) or {}).items()},
            "p2_side_conditions": {str(key): float(value) for key, value in (p2.get("side_conditions", {}) or {}).items()},
        },
        "display": {
            "title": "Battle Context",
            "subtitle": "Global state",
        },
    }


def _build_move_candidate_entities(
    *,
    move_ids: Sequence[str],
    chosen_action_token: Optional[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-turn move entities plus their action-candidate metadata.

    The important shift here is that moves are concrete candidates in the current
    state, not just entries in a global class vocabulary.
    """
    entities: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for idx, move_id in enumerate(move_ids, start=1):
        action_token = f"move:{move_id}"
        entity_id = f"move_candidate:{idx}"
        entities.append(
            {
                "id": entity_id,
                "entity_type": "move_candidate",
                "token_inputs": {
                    "move": str(move_id),
                },
                "state_features": {
                    "candidate_index": float(idx),
                    "is_chosen": 1.0 if chosen_action_token == action_token else 0.0,
                },
                "display": {
                    "title": str(move_id),
                    "subtitle": f"move {idx}",
                },
            }
        )
        candidates.append(
            {
                "action_id": action_token,
                "action_type": "move",
                "target_entity_id": entity_id,
                "token": action_token,
                "is_legal": True,
                "is_chosen": chosen_action_token == action_token,
            }
        )
    return entities, candidates


def _build_switch_action_candidates(
    *,
    state: dict[str, Any],
    perspective_player: str,
    switch_slots: Sequence[int],
    chosen_action_token: Optional[str],
) -> list[dict[str, Any]]:
    """Build switch candidates by pointing actions at existing self-side slot entities."""
    side_label = "self"
    side = state.get(perspective_player, {}) or {}
    slot_uids = side.get("slots", []) or []
    mons = state.get("mons", {}) or {}
    candidates: list[dict[str, Any]] = []
    for slot_index in switch_slots:
        uid = slot_uids[slot_index - 1] if 0 <= slot_index - 1 < len(slot_uids) else None
        mon = mons.get(uid) if uid is not None else None
        action_token = f"switch:{slot_index}"
        candidates.append(
            {
                "action_id": action_token,
                "action_type": "switch",
                "target_entity_id": f"pokemon:{side_label}:slot{slot_index}",
                "token": action_token,
                "is_legal": True,
                "is_chosen": chosen_action_token == action_token,
                "display_name": (mon or {}).get("species") or f"slot {slot_index}",
            }
        )
    return candidates


def build_entity_state_view(
    *,
    state: dict[str, Any],
    perspective_player: str,
) -> dict[str, Any]:
    """Return the minimal entity-centric state without per-turn action candidates.

    Other modules use this shared view for:
        - tensorization
        - static previews
        - future belief-aware upgrades
    """
    other = other_player(perspective_player)
    pokemon_entities = [
        _build_pokemon_entity(
            state=state,
            player=perspective_player,
            perspective_player=perspective_player,
            slot_index=slot_index,
        )
        for slot_index in range(1, 7)
    ] + [
        _build_pokemon_entity(
            state=state,
            player=other,
            perspective_player=perspective_player,
            slot_index=slot_index,
        )
        for slot_index in range(1, 7)
    ]

    global_entity = _build_global_entity(state)
    return {
        "perspective_player": perspective_player,
        "pokemon_entities": pokemon_entities,
        "global_entity": global_entity,
    }


def build_entity_action_graph(
    *,
    state: dict[str, Any],
    perspective_player: str,
    legal_moves: Optional[Sequence[dict[str, Any]]] = None,
    legal_switches: Optional[Sequence[dict[str, Any]]] = None,
    chosen_action: Optional[Tuple[str, str]] = None,
    chosen_action_token: Optional[str] = None,
) -> dict[str, Any]:
    """Build the full graph-style training/preview view for one decision point.

    This includes:
        - stable entities
        - lightweight relational edges
        - concrete move/switch action candidates

    In v1 this is mostly a contract and observability tool. The trainable model
    still uses a simplified pooled encoder until we move to true action-wise legal
    scoring backed by simulator request objects.
    """
    if chosen_action_token is None and chosen_action is not None:
        if chosen_action[0] == "switch":
            chosen_uid = chosen_action[1]
            chosen_slot = None
            for idx, uid in enumerate((state.get(perspective_player, {}) or {}).get("slots", []) or [], start=1):
                if uid == chosen_uid:
                    chosen_slot = idx
                    break
            chosen_action_token = _normalize_action_token(chosen_action, switch_slot=chosen_slot)
        else:
            chosen_action_token = _normalize_action_token(chosen_action)

    state_view = build_entity_state_view(
        state=state,
        perspective_player=perspective_player,
    )
    pokemon_entities = list(state_view["pokemon_entities"])
    global_entity = dict(state_view["global_entity"])
    my_active_entity_id = next(
        (entity["id"] for entity in pokemon_entities if entity["side"] == "self" and entity["state_features"]["active"] > 0.5),
        "pokemon:self:slot1",
    )
    opp_active_entity_id = next(
        (entity["id"] for entity in pokemon_entities if entity["side"] == "opponent" and entity["state_features"]["active"] > 0.5),
        "pokemon:opponent:slot1",
    )

    if legal_moves is not None:
        # Preferred path once simulator requests are available: score the actual
        # legal move list for this turn.
        move_ids = [
            str(entry.get("id") or entry.get("move") or "")
            for entry in legal_moves
            if str(entry.get("id") or entry.get("move") or "").strip()
        ]
    else:
        # Offline logs only let us approximate legality from observed moves.
        my_active_uid = (state.get(perspective_player, {}) or {}).get("active_uid")
        my_active_mon = (state.get("mons", {}) or {}).get(my_active_uid) if my_active_uid else None
        move_ids = _fallback_move_candidates(my_active_mon, chosen_action)
    move_entities, move_action_candidates = _build_move_candidate_entities(
        move_ids=move_ids,
        chosen_action_token=chosen_action_token,
    )

    if legal_switches is not None:
        switch_slots = [
            int(entry["slot"])
            for entry in legal_switches
            if isinstance(entry, dict) and entry.get("slot") is not None
        ]
    else:
        switch_slots = _fallback_switch_slots(
            state,
            player=perspective_player,
            chosen_action=chosen_action,
            chosen_action_token=chosen_action_token,
        )
    switch_action_candidates = _build_switch_action_candidates(
        state=state,
        perspective_player=perspective_player,
        switch_slots=switch_slots,
        chosen_action_token=chosen_action_token,
    )

    edges: list[dict[str, Any]] = []
    for entity in pokemon_entities:
        # The global context is allowed to influence every slot.
        edges.append(
            {
                "source": global_entity["id"],
                "target": entity["id"],
                "edge_type": "global_context",
            }
        )

    edges.append(
        {
            "source": my_active_entity_id,
            "target": opp_active_entity_id,
            "edge_type": "active_matchup",
        }
    )

    for entity in pokemon_entities:
        # Bench-to-active edges let the model eventually reason about switch targets
        # as concrete team members rather than as a generic "switch" action.
        if entity["side"] == "self" and entity["id"] != my_active_entity_id:
            edges.append(
                {
                    "source": entity["id"],
                    "target": my_active_entity_id,
                    "edge_type": "bench_to_active",
                }
            )
        if entity["side"] == "opponent" and entity["id"] != opp_active_entity_id:
            edges.append(
                {
                    "source": entity["id"],
                    "target": opp_active_entity_id,
                    "edge_type": "bench_to_active",
                }
            )

    for move_entity in move_entities:
        # Legal move candidates are connected to both actives because the quality of
        # a move depends on the current matchup, not just on move identity alone.
        edges.append(
            {
                "source": move_entity["id"],
                "target": my_active_entity_id,
                "edge_type": "move_to_self_active",
            }
        )
        edges.append(
            {
                "source": move_entity["id"],
                "target": opp_active_entity_id,
                "edge_type": "move_to_opp_active",
            }
        )

    action_candidates = move_action_candidates + switch_action_candidates
    graph = {
        "family_id": "entity_action_bc",
        "family_version": 1,
        "state_schema_version": "entity_action_v1",
        "perspective_player": perspective_player,
        "entities": [global_entity] + pokemon_entities + move_entities,
        "edges": edges,
        "action_candidates": action_candidates,
        "heads": [
            {
                "head_id": "policy",
                "head_type": "action_scoring",
                "description": "Scores concrete legal move and switch candidates.",
                "targets": [candidate["action_id"] for candidate in action_candidates],
            },
            {
                "head_id": "transition",
                "head_type": "optional_auxiliary",
                "description": "Optional auxiliary head for next public state summary.",
            },
            {
                "head_id": "value",
                "head_type": "optional_auxiliary",
                "description": "Optional auxiliary head for return-to-go or win-probability style value prediction.",
            },
        ],
        "summary": {
            "pokemon_entity_count": 12,
            "move_candidate_count": len(move_entities),
            "switch_candidate_count": len(switch_action_candidates),
            "edge_count": len(edges),
            "chosen_action_token": chosen_action_token,
        },
    }
    return graph
