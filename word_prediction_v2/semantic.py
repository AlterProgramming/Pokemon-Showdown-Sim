"""Semantic retrieval bias for word_policy_v2_semantic.

Pulls top-k similar past games from artifacts/game-embeddings.npy (built by
ps-game-embed) and produces a per-word bias the policy_adapter blends into
the word-prediction ranking.

Hard contracts:
    - If artifacts/game-embeddings.npy or .meta.json or artifacts/game-index.json
      is missing, every call returns an empty bias dict. The decoder behavior is
      then byte-identical to the baseline.
    - If sentence-transformers is not importable, ditto.
    - If semantic_retrieval_weight is 0.0, the policy_adapter must short-circuit
      before calling into this module.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "alter-programming-game-search" / "artifacts"
_FALLBACK_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


# Map semantic words emitted by the WordEmbeddingModel to the keyword sets
# the threat-aware decoder already uses. A retrieved game "uses" a word if
# any of its unique_moves_normalized lands in the mapped set. Words not in
# the map fall through to the "general attack" pool (any move not in the
# named categories), which captures attack/finish/safeko/revenge/etc.
#
# Kept in sync with policy_adapter.{STATUS,SETUP,SCOUT,RECOVER}_KEYWORDS.
_STATUS = {"toxic", "poisonpowder", "stunspore", "thunderwave", "spore", "hypnosis",
           "sleeppowder", "willowisp", "leechseed", "yawn", "encore"}
_SETUP = {"swordsdance", "nastyplot", "bulkup", "calmmind", "dragondance",
          "quiverdance", "shellsmash", "agility", "curse", "growth", "trailblaze",
          "irondefense", "acidarmor", "cottonguard"}
_SCOUT = {"protect", "detect", "substitute", "uturn", "voltswitch", "flipturn",
          "partingshot", "batonpass"}
_RECOVER = {"recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun",
            "synthesis", "rest"}
_HAZARD_SET = {"stealthrock", "spikes", "stickyweb", "toxicspikes", "stoneaxe", "ceaselessedge"}
_HAZARD_CLEAR = {"rapidspin", "defog", "tidyup", "courtchange"}
_NAMED_CATEGORY_ALL = _STATUS | _SETUP | _SCOUT | _RECOVER | _HAZARD_SET | _HAZARD_CLEAR

WORD_CATEGORY: Dict[str, set] = {
    "status": _STATUS,
    "setup": _SETUP,
    "scout": _SCOUT,
    "recover": _RECOVER,
    "stabilize": _RECOVER | _STATUS,
    "wall": _RECOVER | _SETUP,
    "stall": _RECOVER | _STATUS,
    "hazard": _HAZARD_SET,
    "spike": _HAZARD_SET,
    "clear": _HAZARD_CLEAR,
    "spin": _HAZARD_CLEAR,
    "pivot": _SCOUT,
}


def _resolve_artifact_root() -> Optional[Path]:
    override = os.environ.get("PS_GAME_SEARCH_ARTIFACTS", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.is_dir() else None
    for cand in (_ARTIFACT_ROOT, _FALLBACK_ARTIFACT_ROOT):
        if cand.is_dir():
            return cand
    return None


@lru_cache(maxsize=1)
def _load_corpus() -> Optional[Dict[str, Any]]:
    """Load vectors + meta + index. Returns None if any piece is missing."""
    root = _resolve_artifact_root()
    if not root:
        return None
    vec_path = root / "game-embeddings.npy"
    meta_path = root / "game-embeddings.meta.json"
    index_path = root / "game-index.json"
    if not (vec_path.is_file() and meta_path.is_file() and index_path.is_file()):
        return None
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return None
    import numpy as np
    try:
        vecs = np.load(vec_path)
        meta = json.loads(meta_path.read_text())
        index = json.loads(index_path.read_text())
    except Exception:
        return None
    if not isinstance(index, list) or vecs.shape[0] != len(meta.get("game_ids", [])):
        return None
    index_by_id = {r.get("game_id"): r for r in index if isinstance(r, dict)}
    return {
        "vecs": vecs,
        "meta": meta,
        "index": index,
        "index_by_id": index_by_id,
    }


@lru_cache(maxsize=1)
def _load_encoder():
    """Load the sentence-transformer model. None on any failure."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    corpus = _load_corpus()
    if not corpus:
        return None
    model_name = corpus["meta"].get("model", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _state_paragraph(
    battle_state: Dict[str, Any],
    perspective_player: str,
) -> str:
    """Compress the current battle state to a paragraph for encoding.

    Schema must stay close to ps-game-embed's game_paragraph() shape so the
    query and corpus inhabit the same neighborhood.
    """
    mons = (battle_state or {}).get("mons") or {}
    species: list[str] = []
    moves_norm: set[str] = set()
    abilities: set[str] = set()
    items: set[str] = set()
    for m in mons.values():
        if not isinstance(m, dict):
            continue
        sp = m.get("species")
        if sp:
            species.append(sp)
        ab = m.get("ability")
        if ab:
            abilities.add(ab)
        it = m.get("item")
        if it:
            items.add(it)
        for mv in (m.get("observed_moves") or []):
            n = re.sub(r"[^a-z0-9]", "", str(mv or "").lower())
            if n:
                moves_norm.add(n)
    turn = (battle_state or {}).get("turn_index") or 0
    species_s = ", ".join(species[:8])
    moves_s = ", ".join(sorted(moves_norm)[:10])
    abilities_s = ", ".join(sorted(abilities)[:6])
    items_s = ", ".join(sorted(items)[:6])
    return (
        f"model=? outcome=PROBE turns={turn} faints=?\n"
        f"species={species_s}\n"
        f"moves={moves_s}\n"
        f"abilities={abilities_s} items={items_s}\n"
        f"actions=?\n"
        f"perspective={perspective_player}"
    )


def _norm_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(word or "").lower())


def _word_match_count(moves: Sequence[str], norm_word: str) -> int:
    """Count moves consistent with the semantic word.

    Named-category words (recover/setup/status/scout/etc.) match moves in
    the corresponding keyword set. Unmapped words (attack/finish/preserve/
    safeko/tempo/revenge/wall/...) match the "general attack" pool — moves
    that are NOT in any named category — under the assumption that those
    words drive offensive lines.
    """
    if not moves:
        return 0
    if norm_word in WORD_CATEGORY:
        bucket = WORD_CATEGORY[norm_word]
        return sum(1 for m in moves if m in bucket)
    # General-attack pool: any move not assigned to a named category.
    return sum(1 for m in moves if m not in _NAMED_CATEGORY_ALL)


def semantic_word_bias(
    battle_state: Dict[str, Any],
    perspective_player: str,
    candidate_words: Sequence[str],
    *,
    top_k: int = 8,
    similarity_floor: float = 0.30,
    boost: float = 0.6,
    penalty: float = -0.2,
) -> Dict[str, float]:
    """Return {normalized_word: bias} for each candidate semantic word.

    For each retrieved WIN/LOSS game, count how many of the game's
    unique_moves_normalized are consistent with the candidate word's
    category (see WORD_CATEGORY + _word_match_count). The bias for a
    word is the difference between the average per-game match-count in
    WIN games vs LOSS games, scaled by `boost`. Words with negative
    differential get a softer `penalty`-scaled adjustment.
    """
    if not candidate_words:
        return {}
    corpus = _load_corpus()
    if not corpus:
        return {}
    encoder = _load_encoder()
    if encoder is None:
        return {}

    paragraph = _state_paragraph(battle_state, perspective_player)
    try:
        import numpy as np
        q = encoder.encode([paragraph], convert_to_numpy=True, normalize_embeddings=True)[0]
    except Exception:
        return {}

    vecs = corpus["vecs"]
    game_ids = corpus["meta"]["game_ids"]
    index_by_id = corpus["index_by_id"]
    sims = vecs @ q

    win_games: list[list[str]] = []
    loss_games: list[list[str]] = []

    order = sims.argsort()[::-1]
    for idx in order:
        if sims[idx] < similarity_floor:
            break
        if len(win_games) >= top_k and len(loss_games) >= top_k:
            break
        gid = game_ids[idx] if idx < len(game_ids) else None
        row = index_by_id.get(gid)
        if not row:
            continue
        outcome = (row.get("outcome") or "").upper()
        moves = list(row.get("unique_moves_normalized") or [])
        if outcome == "WIN" and len(win_games) < top_k:
            win_games.append(moves)
        elif outcome == "LOSS" and len(loss_games) < top_k:
            loss_games.append(moves)

    if not win_games:
        return {}

    bias: Dict[str, float] = {}
    for w in candidate_words:
        nw = _norm_word(w)
        if not nw:
            continue
        win_avg = sum(_word_match_count(g, nw) for g in win_games) / max(len(win_games), 1)
        loss_avg = (
            sum(_word_match_count(g, nw) for g in loss_games) / max(len(loss_games), 1)
            if loss_games else 0.0
        )
        diff = win_avg - loss_avg
        if diff > 0:
            bias[nw] = boost * min(diff / 4.0, 1.0)  # normalize: 4 matches ≈ saturated
        elif diff < 0:
            bias[nw] = penalty * min(-diff / 4.0, 1.0)
        else:
            bias[nw] = 0.0
    return bias


def is_active() -> bool:
    """True only if corpus + encoder both load cleanly. Cheap to call repeatedly."""
    return _load_corpus() is not None and _load_encoder() is not None
