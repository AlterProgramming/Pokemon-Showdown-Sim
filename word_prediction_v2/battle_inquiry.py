from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .battle_lexicon import battle_lexicon
from .dataset import build_training_corpus
from .model import Prediction, TrainingConfig, WordEmbeddingModel, train_model
from .pipeline import DEFAULT_ARTIFACT_ROOT, load_model
from .text import normalize_prompt, normalize_token


DEFAULT_CONFIG = TrainingConfig(
    embedding_dim=8,
    epochs=90,
    learning_rate=0.06,
    alignment_weight=0.45,
    seed=17,
)
DEFAULT_BATTLE_RUN_NAME = "battle_inquiry_v3"


@dataclass(frozen=True)
class InquiryAnswer:
    prompt_tokens: tuple[str, ...]
    predictions: tuple[Prediction, ...]
    model_source: str
    attention_report: Dict[str, Any] | None = None


def _empty_boosts() -> Dict[str, int]:
    return {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def sample_battle_state() -> Dict[str, Any]:
    return {
        "p1": {"active_uid": "p1a", "slots": ["p1a", "p1b", None, None, None, None]},
        "p2": {"active_uid": "p2a", "slots": ["p2a", None, None, None, None, None]},
        "mons": {
            "p1a": {
                "uid": "p1a",
                "species": "Pikachu",
                "hp_frac": 0.22,
                "status": "brn",
                "fainted": False,
                "boosts": _empty_boosts(),
            },
            "p1b": {
                "uid": "p1b",
                "species": "Bulbasaur",
                "hp_frac": 0.94,
                "status": None,
                "fainted": False,
                "boosts": _empty_boosts(),
            },
            "p2a": {
                "uid": "p2a",
                "species": "Squirtle",
                "hp_frac": 0.64,
                "status": None,
                "fainted": False,
                "boosts": _empty_boosts(),
            },
        },
    }


def train_battle_model() -> WordEmbeddingModel:
    corpus = build_training_corpus(
        lexicon=battle_lexicon(),
        repeats_per_entry=72,
        seed=19,
        ngram_bucket_count=128,
    )
    return train_model(corpus, DEFAULT_CONFIG)


def _battle_artifact_dir(output_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    return output_root / DEFAULT_BATTLE_RUN_NAME


def _battle_model_path(output_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    return _battle_artifact_dir(output_root) / "model.json"


def save_battle_model(output_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    run_dir = _battle_artifact_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = train_battle_model()
    model_path = _battle_model_path(output_root)
    metadata_path = run_dir / "training_metadata.json"
    model_path.write_text(json.dumps(model.to_serializable(), indent=2, sort_keys=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "model_family": "battle_inquiry_word_model_v1",
                "run_name": DEFAULT_BATTLE_RUN_NAME,
                "embedding_dim": DEFAULT_CONFIG.embedding_dim,
                "max_parameters": DEFAULT_CONFIG.max_parameters,
                "epochs": DEFAULT_CONFIG.epochs,
                "learning_rate": DEFAULT_CONFIG.learning_rate,
                "alignment_weight": DEFAULT_CONFIG.alignment_weight,
                "seed": DEFAULT_CONFIG.seed,
                "parameter_count": model.parameter_count,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model_path


def get_battle_model(output_root: Path = DEFAULT_ARTIFACT_ROOT) -> tuple[WordEmbeddingModel, str]:
    model_path = _battle_model_path(output_root)
    if model_path.exists():
        return load_model(model_path), "cache"
    save_battle_model(output_root)
    return load_model(model_path), "trained"


def _question_tokens(question: str) -> List[str]:
    return list(_question_tokens_cached(question))


@lru_cache(maxsize=256)
def _question_tokens_cached(question: str) -> tuple[str, ...]:
    return tuple(normalize_prompt(question.replace("?", " ").replace(",", " ").split()))


def _extract_payload_context(payload: Dict[str, Any]) -> tuple[Dict[str, Any], str, Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    battle_state = payload.get("battle_state")
    if isinstance(battle_state, dict):
        perspective = payload.get("perspective_player")
        if perspective not in {"p1", "p2"}:
            perspective = "p1"
        legal_moves = payload.get("legal_moves") if isinstance(payload.get("legal_moves"), list) else []
        legal_switches = payload.get("legal_switches") if isinstance(payload.get("legal_switches"), list) else []
        return battle_state, perspective, legal_moves, legal_switches
    return payload, "p1", [], []


def _get_active_mon(battle_state: Dict[str, Any], player: str) -> Dict[str, Any] | None:
    side = battle_state.get(player) or {}
    active_uid = side.get("active_uid")
    mons = battle_state.get("mons") or {}
    if not active_uid:
        return None
    mon = mons.get(active_uid)
    return mon if isinstance(mon, dict) else None


def _count_healthy_bench(battle_state: Dict[str, Any], player: str) -> int:
    side = battle_state.get(player) or {}
    mons = battle_state.get("mons") or {}
    count = 0
    for uid in side.get("slots") or []:
        if not uid or uid == side.get("active_uid"):
            continue
        mon = mons.get(uid) or {}
        if not mon.get("fainted") and float(mon.get("hp_frac") or 0.0) > 0.2:
            count += 1
    return count


def build_battle_prompt_tokens(
    question: str,
    battle_state: Dict[str, Any],
    *,
    perspective_player: str = "p1",
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
) -> List[str]:
    other_player = "p2" if perspective_player == "p1" else "p1"
    my_active = _get_active_mon(battle_state, perspective_player) or {}
    opp_active = _get_active_mon(battle_state, other_player) or {}
    question_tokens = set(_question_tokens(question))

    tokens: List[str] = []
    my_hp = float(my_active.get("hp_frac") or 0.0)
    opp_hp = float(opp_active.get("hp_frac") or 0.0)
    my_status = normalize_token(str(my_active.get("status") or ""))
    bench_count = _count_healthy_bench(battle_state, perspective_player)
    boosts = my_active.get("boosts") or {}
    total_boost = sum(int(boosts.get(stat, 0) or 0) for stat in ("atk", "spa", "spe"))
    legal_move_names = {
        normalize_token(str(move.get("id") or move.get("move") or ""))
        for move in (legal_moves or [])
        if isinstance(move, dict)
    }
    has_priority_hint = any(
        name in {"quickattack", "aquajet", "iceshard", "machpunch", "bulletpunch", "shadowsneak", "suckerpunch", "extremespeed", "fakeout", "accelerock", "firstimpression"}
        for name in legal_move_names
    )

    if {"switch", "swap", "pivot", "retreat", "change"} & question_tokens:
        tokens.extend(["pivot", "retreat", "swap"])
    if {"attack", "move", "damage", "hit", "offense"} & question_tokens:
        tokens.extend(["damage", "offense", "pressure"])
    if {"ko", "knock", "kill", "finish", "lethal"} & question_tokens:
        tokens.extend(["ko", "lethal", "secure"])
    if {"ahead", "winning", "advantage", "lead", "favored", "favour"} & question_tokens:
        tokens.extend(["pressure", "steady", "secure"])
    if {"risk", "risky", "danger", "unsafe", "gamble", "uncertain"} & question_tokens:
        tokens.extend(["guess", "volatile", "danger"])
    safe_ko_question = bool({"knockout", "ko", "knock", "finish", "close", "lethal"} & question_tokens)

    if {"safe", "survive", "recover", "stabilize", "stabilise"} & question_tokens and not safe_ko_question:
        tokens.extend(["safe", "recover", "steady"])
    if {"boost", "setup", "sweep", "snowball"} & question_tokens:
        tokens.extend(["boost", "charge", "snowball"])
    if {"what", "unknown", "scout", "reveal", "info"} & question_tokens:
        tokens.extend(["reveal", "learn", "info"])
    if {"priority", "quick", "first", "speed", "faster", "fast", "outspeed"} & question_tokens:
        tokens.extend(["quick", "first", "cleanup"])
    if {"revenge", "retaliate", "answer", "punish", "revengekill"} & question_tokens:
        tokens.extend(["retaliate", "answer", "punish"])
    if {"safe", "safeko", "certain", "accurate", "reliable", "clean", "knockout"} & question_tokens:
        tokens.extend(["reliable", "accurate", "clean"])
    if {"tempo", "momentum", "pace", "initiative", "flow"} & question_tokens:
        tokens.extend(["momentum", "pace", "flow"])
    if {"sacrifice", "sack", "trade", "fodder", "giveup"} & question_tokens:
        tokens.extend(["trade", "fodder", "spend"])

    if my_hp <= 0.3 and not safe_ko_question:
        tokens.extend(["save", "protect", "recover"])
        if bench_count > 0:
            tokens.extend(["pivot", "escape"])
    if opp_hp <= 0.3:
        tokens.extend(["ko", "end", "pressure"])
    if my_status in {"brn", "psn", "tox", "par", "slp"}:
        tokens.extend(["burn" if my_status == "brn" else "cripple", "recover", "reset"])
    if total_boost >= 2 and my_hp >= 0.45:
        tokens.extend(["boost", "stack", "pressure"])
    if my_hp > opp_hp + 0.25:
        tokens.extend(["secure", "steady", "pressure"])
    elif opp_hp > my_hp + 0.25 and not safe_ko_question:
        tokens.extend(["danger", "safe", "recover"])
    if (legal_switches and len(legal_switches) > 0 and my_hp <= 0.45) or bench_count > 1:
        tokens.extend(["swap", "rotate"])
    if legal_moves and len(legal_moves) > 0 and opp_hp <= 0.5:
        tokens.extend(["strike", "close"])
    if not legal_switches and legal_moves:
        tokens.extend(["strike", "pressure"])
    if has_priority_hint and opp_hp <= 0.4:
        tokens.extend(["quick", "cleanup", "reliable"])
    if has_priority_hint and my_hp <= 0.35 and opp_hp <= 0.45:
        tokens.extend(["retaliate", "answer", "first"])
    if my_hp <= 0.3 and opp_hp <= 0.35:
        tokens.extend(["clean", "certain", "convert"])
    if my_hp < opp_hp and opp_hp <= 0.55:
        tokens.extend(["momentum", "pace", "trade"])

    unique_tokens = []
    seen = set()
    for token in normalize_prompt(tokens):
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    return unique_tokens[:5] or ["info", "test", "steady"]


def answer_battle_inquiry(
    question: str,
    battle_state: Dict[str, Any],
    *,
    perspective_player: str = "p1",
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
    top_k: int = 3,
    model: WordEmbeddingModel | None = None,
    output_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> InquiryAnswer:
    prompt_tokens = build_battle_prompt_tokens(
        question,
        battle_state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
    )
    if model is None:
        trained_model, model_source = get_battle_model(output_root)
    else:
        trained_model, model_source = model, "provided"
    predictions = tuple(trained_model.predict(prompt_tokens, top_k=top_k))
    attention_report = trained_model.prompt_attention_report(prompt_tokens, top_k=top_k)
    return InquiryAnswer(
        prompt_tokens=tuple(prompt_tokens),
        predictions=predictions,
        model_source=model_source,
        attention_report=attention_report,
    )


def _load_battle_state(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return sample_battle_state()
    return json.loads(path.read_text(encoding="utf-8"))


def load_inquiry_context(path: Path | None) -> tuple[Dict[str, Any], str, Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    if path is None:
        return sample_battle_state(), "p1", [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return sample_battle_state(), "p1", [], []
    return _extract_payload_context(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer battle inquiries with the word model.")
    parser.add_argument("--question", required=True, help="Inquiry about the battle state.")
    parser.add_argument("--battle-state-file", type=Path, default=None, help="Optional JSON battle state file or normalized bridge payload.")
    parser.add_argument("--perspective-player", choices=["p1", "p2"], default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    battle_state, inferred_perspective, legal_moves, legal_switches = load_inquiry_context(args.battle_state_file)
    answer = answer_battle_inquiry(
        args.question,
        battle_state,
        perspective_player=args.perspective_player or inferred_perspective,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        top_k=args.top_k,
        output_root=args.output_root,
    )
    print(f"model_source={answer.model_source}")
    print(f"prompt_tokens={list(answer.prompt_tokens)}")
    print(
        "predictions="
        + json.dumps(
            [{"word": item.word, "score": item.score} for item in answer.predictions],
            indent=2,
        )
    )
    print(
        "attention_report="
        + json.dumps(
            answer.attention_report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
