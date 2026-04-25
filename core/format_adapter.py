"""Format adapter: per-decision sharded JSONL -> legacy per-game dicts.

The model-league training pipeline writes one JSON record per decision to
``databases/model-league/training/examples/.../*.jsonl``. Each record has:

    {
      "battleId": "...", "modelCheckpointId": "...", "perspectivePlayer": "p1",
      "requestKind": "move", "result": "win"|"loss"|"tie",
      "chosenAction": "move 1", "recordedAt": "2026-04-23T00:36:46.053Z",
      "usedFallback": false, "modelRequest": {..., "battle_state": {...}},
      "modelResponse": {"action_token": "move:earthquake", ...},
      ...
    }

The legacy ``rl_examples_*.jsonl`` format groups decisions per game:

    {
      "game_id": "...", "model_id": "...", "outcome": 1.0,
      "perspective_player": "p1",
      "decisions": [
        {"state_json": {...}, "modelResponse": {"action_token": ...},
         "turn": int, "usedFallback": bool},
        ...
      ],
    }

This module reads sharded JSONL files and emits legacy-shaped game dicts so
the existing RL trainer can consume them with minimal changes.

Quality filters (applied before emission):
    * drop any game whose ``result`` is ``"tie"`` (outcome 0.5)
    * for ``usedFallback: true`` decisions, either drop the decision or drop
      the whole game depending on ``drop_fallback_mode``

This module has NO ML dependencies — stdlib only — so its tests run in <1s.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_VALID_FALLBACK_MODES = ("decision", "game")


def _result_to_outcome(result: str | None) -> float | None:
    """Map the sharded ``result`` field to a scalar outcome.

    Returns ``None`` for unrecognised values so the caller can decide whether
    to drop or propagate.
    """
    if result == "win":
        return 1.0
    if result == "loss":
        return 0.0
    if result == "tie":
        return 0.5
    return None


def _extract_state_json(record: dict) -> Any | None:
    """Return the dense battle-state dict from a sharded record, if present.

    Sharded records store the encoded state inside ``modelRequest.battle_state``
    (which is the shape ``encode_entity_state`` expects). If ``modelRequest``
    is itself the state dict (no ``battle_state`` key) we fall back to it; if
    neither shape works we return ``None``.
    """
    model_request = record.get("modelRequest")
    if not isinstance(model_request, dict):
        return None
    battle_state = model_request.get("battle_state")
    if isinstance(battle_state, dict):
        return battle_state
    # Fallback: modelRequest itself already looks like a state dict
    if {"turn_index", "p1", "p2", "mons"} & set(model_request.keys()):
        return model_request
    return None


def _turn_from_record(record: dict, state_json: Any) -> int | None:
    """Recover a turn index from the record."""
    top = record.get("turn")
    if isinstance(top, int):
        return top
    if isinstance(state_json, dict):
        t = state_json.get("turn_index")
        if isinstance(t, int):
            return t
    return None


def sharded_to_per_game(
    shard_paths: Iterable[Path],
    *,
    drop_fallback_mode: str = "decision",
    base_model_id: str | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Convert sharded per-decision JSONL records into legacy per-game dicts.

    Parameters
    ----------
    shard_paths:
        Absolute paths to ``*.jsonl`` shard files. Each line is one decision.
    drop_fallback_mode:
        ``"decision"`` (default) drops only fallback-flagged decisions.
        ``"game"`` drops the entire game if any decision used the fallback.
    base_model_id:
        Optional filter: keep only records whose ``modelCheckpointId`` equals
        this value. Shards are normally pre-scoped to one checkpoint, but the
        filter is defensive against cross-model pollution.

    Returns
    -------
    games, drop_counts:
        ``games`` is a list of legacy-shaped dicts:
            ``{"game_id", "model_id", "outcome", "perspective_player",
               "decisions": [{"state_json", "modelResponse", "turn",
               "usedFallback"}, ...]}``
        ``drop_counts`` is a dict with keys:
            ``used_fallback`` — decisions dropped (or games lost to fallback
                in ``"game"`` mode, counted as one per dropped decision)
            ``ties`` — games dropped because ``result == "tie"``
            ``records_seen`` — raw record count before any filter (including
                cross-model records rejected by ``base_model_id``); coarse
                upper bound on shard volume
    """
    if drop_fallback_mode not in _VALID_FALLBACK_MODES:
        raise ValueError(
            f"drop_fallback_mode must be one of {_VALID_FALLBACK_MODES}, "
            f"got {drop_fallback_mode!r}"
        )

    # battleId -> list[ (recordedAt, decision_dict, used_fallback, meta) ]
    by_battle: dict[str, list[tuple[str, dict, bool, dict]]] = defaultdict(list)

    # drop_counts.records_seen counts every record the adapter inspected
    # (including the ones filtered out by modelCheckpointId); it is a coarse
    # upper bound on the raw data volume. 'ties' and 'used_fallback' are the
    # adapter-side attrition; the trainer records 'vocab_miss' and its own
    # 'total_seen' post-encoding.
    drop_counts = {"used_fallback": 0, "ties": 0, "records_seen": 0}
    # Track vector-only records so we can emit a loud warning if the adapter
    # produces zero games because battle_state was missing everywhere.
    vector_only_records = 0

    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        if not shard_path.exists():
            continue
        with open(shard_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                drop_counts["records_seen"] += 1

                if base_model_id is not None:
                    if record.get("modelCheckpointId") != base_model_id:
                        continue

                battle_id = record.get("battleId")
                if not battle_id:
                    continue

                state_json = _extract_state_json(record)
                if state_json is None:
                    model_request = record.get("modelRequest")
                    if (
                        isinstance(model_request, dict)
                        and "state_vector" in model_request
                        and "battle_state" not in model_request
                    ):
                        vector_only_records += 1
                    continue
                used_fallback = bool(record.get("usedFallback", False))
                model_response = record.get("modelResponse") or {}
                turn = _turn_from_record(record, state_json)
                recorded_at = record.get("recordedAt") or ""

                decision = {
                    "state_json": state_json,
                    "modelResponse": model_response,
                    "turn": turn,
                    "usedFallback": used_fallback,
                }

                meta = {
                    "modelCheckpointId": record.get("modelCheckpointId"),
                    "perspectivePlayer": record.get("perspectivePlayer", "p1"),
                    "result": record.get("result"),
                }

                by_battle[battle_id].append(
                    (recorded_at, decision, used_fallback, meta)
                )

    games: list[dict] = []

    for battle_id, entries in by_battle.items():
        # ISO-8601 recordedAt sorts lexicographically
        entries.sort(key=lambda row: row[0])

        # Derive outcome from the latest record's result (should be uniform,
        # but use the latest for safety).
        result = None
        for _, _, _, meta in reversed(entries):
            if meta.get("result"):
                result = meta["result"]
                break
        outcome = _result_to_outcome(result)

        if outcome == 0.5:
            # Tie — drop the whole game.
            drop_counts["ties"] += 1
            continue
        if outcome is None:
            # Unknown result — drop silently.
            continue

        # Build per-game metadata from the first entry.
        _, _, _, first_meta = entries[0]
        model_id = first_meta.get("modelCheckpointId")
        perspective_player = first_meta.get("perspectivePlayer", "p1")

        if drop_fallback_mode == "game":
            if any(used for (_, _, used, _) in entries):
                # Count each dropped decision towards used_fallback.
                drop_counts["used_fallback"] += sum(
                    1 for (_, _, used, _) in entries if used
                )
                continue
            kept_decisions = [decision for (_, decision, _, _) in entries]
        else:  # "decision"
            kept_decisions = []
            for (_, decision, used, _) in entries:
                if used:
                    drop_counts["used_fallback"] += 1
                    continue
                kept_decisions.append(decision)

        if not kept_decisions:
            continue

        games.append(
            {
                "game_id": battle_id,
                "model_id": model_id,
                "outcome": outcome,
                "perspective_player": perspective_player,
                "decisions": kept_decisions,
            }
        )

    if not games and vector_only_records > 0:
        # Loud warning: every inspected record had a dense state_vector but no
        # expandable battle_state dict. The adapter cannot reconstruct a state
        # from a vector, so a vector->state path would be needed here.
        import warnings

        warnings.warn(
            f"[format_adapter] Produced 0 games but saw {vector_only_records} "
            "records with 'state_vector' and no 'battle_state'. The adapter "
            "does not reconstruct state from dense vectors; the upstream "
            "training-example writer must emit battle_state dicts."
        )

    return games, drop_counts


__all__ = ["sharded_to_per_game"]
