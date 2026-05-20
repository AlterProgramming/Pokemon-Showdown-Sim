"""Action-level retrieval bias for word_policy_v3_actions.

Pulls top-k similar past games from artifacts/game-embeddings.npy (same corpus
v2 uses) and produces a per-move bias the policy_adapter blends into
_scored_moves_for_predictions_with_state, alongside the existing word-score
and state-score terms.

Hard contracts (identical defensive shape to semantic.py):
    - If artifacts/game-embeddings.npy or .meta.json or artifacts/game-index.json
      is missing, every call returns an empty bias dict.
    - If sentence-transformers is not importable, ditto.
    - If the action loader registers zero games (no source file has semantic
      action_token strings on disk), ditto.
    - If action_retrieval_weight is 0.0, policy_adapter short-circuits before
      calling into this module.

Schema split (2026-05-17): the 808-game corpus is built from 60 self-play
files (action_token = "move:dynamaxcannon", semantic name) plus 1 league-smoke
file (action_chosen = "move:1", slot index). Only the semantic-name format is
usable for cross-game aggregation — slot indices have no meaning outside the
captured turn's legal_moves order. The loader filters to that subset (~388
of 808 games), preserving similarity retrieval over the full corpus while
restricting action aggregation to usable games.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .semantic import _load_corpus, _load_encoder, _state_paragraph


_EXAMPLES_DIR = (
    Path(__file__).resolve().parents[1]
    / "Pokemon-Showdown-Agents-Go-Brrrr"
    / "training"
    / "examples"
)


def _is_semantic_move_token(tok: str) -> bool:
    """True only for `move:<alpha-name>` tokens.

    Rejects:
      - `move:1` (slot-index form captured by league runner — slot order
        is per-turn and has no cross-game meaning).
      - `switch:<anything>` (this slice biases moves only; switches still
        flow through _bench_switch_score on the existing path. Including
        switch tokens here would inflate n_win/n_loss denominators in the
        bias function without contributing to the move-keyed numerator).
    """
    if not isinstance(tok, str) or not tok.startswith("move:"):
        return False
    suffix = tok[5:]
    return bool(suffix) and not suffix.isdigit()


@lru_cache(maxsize=1)
def _load_game_actions() -> Optional[Dict[str, Dict[str, Any]]]:
    """Build {game_id: {actions: [tokens], outcome: WIN/LOSS}} from source files.

    Only registers games whose decisions[] entries contain a semantic
    action token. Returns None if the corpus index cannot load; empty dict
    if no source files yield usable actions.
    """
    corpus = _load_corpus()
    if not corpus:
        return None
    index = corpus["index"]
    by_source: Dict[str, list] = {}
    for row in index:
        sf = row.get("source_file")
        if sf:
            by_source.setdefault(sf, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for sf, rows in by_source.items():
        path = _EXAMPLES_DIR / sf
        if not path.is_file():
            continue
        rows_by_gid = {r.get("game_id"): r for r in rows if r.get("game_id")}
        try:
            with open(path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    gid = rec.get("game_id")
                    if gid not in rows_by_gid:
                        continue
                    decisions = rec.get("decisions") or []
                    actions = []
                    for d in decisions:
                        tok = d.get("action_token") or d.get("action_chosen")
                        if _is_semantic_move_token(tok):
                            actions.append(tok)
                    if actions:
                        out[gid] = {
                            "actions": actions,
                            "outcome": (rows_by_gid[gid].get("outcome") or "").upper(),
                        }
        except Exception:
            continue
    return out


def _normalize_move_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def action_retrieval_bias(
    battle_state: Dict[str, Any],
    perspective_player: str,
    legal_move_names: Sequence[str],
    *,
    top_k: int = 8,
    similarity_floor: float = 0.30,
    boost: float = 0.25,
    penalty: float = -0.1,
) -> Dict[str, float]:
    """Return {move_key: bias} where move_key is `move:<normalized_name>`.

    For each candidate legal move, the bias is `boost * (win_freq - loss_freq)`
    when positive (win_freq > loss_freq among retrieved games) or
    `penalty * (loss_freq - win_freq)` when negative. Frequencies are computed
    per-game-occurrence: a move counts at most once per retrieved game.
    """
    if not legal_move_names:
        return {}
    corpus = _load_corpus()
    if not corpus:
        return {}
    encoder = _load_encoder()
    if encoder is None:
        return {}
    actions_db = _load_game_actions()
    if not actions_db:
        return {}

    paragraph = _state_paragraph(battle_state, perspective_player)
    try:
        import numpy as np
        q = encoder.encode([paragraph], convert_to_numpy=True, normalize_embeddings=True)[0]
    except Exception:
        return {}

    vecs = corpus["vecs"]
    game_ids = corpus["meta"]["game_ids"]
    sims = vecs @ q

    win_game_counts: Counter = Counter()
    loss_game_counts: Counter = Counter()
    n_win = 0
    n_loss = 0

    order = sims.argsort()[::-1]
    for idx in order:
        if sims[idx] < similarity_floor:
            break
        if n_win >= top_k and n_loss >= top_k:
            break
        gid = game_ids[idx] if idx < len(game_ids) else None
        entry = actions_db.get(gid)
        if not entry:
            continue
        outcome = entry["outcome"]
        unique_actions = set(entry["actions"])
        if outcome == "WIN" and n_win < top_k:
            n_win += 1
            for a in unique_actions:
                win_game_counts[a] += 1
        elif outcome == "LOSS" and n_loss < top_k:
            n_loss += 1
            for a in unique_actions:
                loss_game_counts[a] += 1

    if n_win == 0:
        return {}

    bias: Dict[str, float] = {}
    for name in legal_move_names:
        key = "move:" + _normalize_move_name(name)
        win_f = win_game_counts.get(key, 0) / max(n_win, 1)
        loss_f = loss_game_counts.get(key, 0) / max(n_loss, 1) if n_loss > 0 else 0.0
        diff = win_f - loss_f
        if diff > 0:
            bias[key] = boost * min(diff, 1.0)
        elif diff < 0:
            bias[key] = penalty * min(-diff, 1.0)
    return bias


def is_active() -> bool:
    """True only when corpus, encoder, and action db all load. Cheap to call."""
    return (
        _load_corpus() is not None
        and _load_encoder() is not None
        and bool(_load_game_actions())
    )
