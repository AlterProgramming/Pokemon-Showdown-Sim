from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .battle_inquiry import answer_battle_inquiry, sample_battle_state


def _proxy_question(loss: Dict[str, Any]) -> str:
    tags = set(loss.get("tags") or [])
    if "recovery_loop" in tags or "status_game" in tags:
        return "what is the safe play?"
    if "setup_spiral" in tags:
        return "should I setup here?"
    if "short_tempo_loss" in tags:
        return "should I attack now?"
    if "opponent_setup_pressure" in tags:
        return "is this too risky?"
    return "what is the safe play?"


def _proxy_context(loss: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    state = sample_battle_state()
    my_active = state["mons"]["p1a"]
    opp_active = state["mons"]["p2a"]

    p1_status_events = int(loss.get("p1_status_events") or 0)
    p2_status_events = int(loss.get("p2_status_events") or 0)
    p1_recover_moves = int(loss.get("p1_recover_moves") or 0)
    p1_setup_moves = int(loss.get("p1_setup_moves") or 0)
    p2_recover_moves = int(loss.get("p2_recover_moves") or 0)
    turns = int(loss.get("turns") or 0)
    tags = set(loss.get("tags") or [])

    my_active["hp_frac"] = 0.58
    opp_active["hp_frac"] = 0.62
    legal_moves: List[Dict[str, Any]] = [{"move": "Thunderbolt", "slot": 1}]
    legal_switches: List[Dict[str, Any]] = []

    if p1_status_events > 0 or "status_game" in tags:
        my_active["status"] = "tox"
        my_active["hp_frac"] = 0.33 if p1_recover_moves > 0 else 0.42
        legal_moves.append({"move": "Recover", "slot": 2})
    if p2_status_events > 0:
        legal_moves.append({"move": "Thunder Wave", "slot": len(legal_moves) + 1})
    if p1_setup_moves > 0 or "setup_spiral" in tags:
        my_active["boosts"]["atk"] = min(p1_setup_moves, 3)
        legal_moves.append({"move": "Swords Dance", "slot": len(legal_moves) + 1})
        my_active["hp_frac"] = min(float(my_active["hp_frac"]), 0.48)
    if p2_recover_moves > 0 or "recovery_loop" in tags:
        opp_active["hp_frac"] = max(float(opp_active["hp_frac"]), 0.72)
    if "short_tempo_loss" in tags:
        my_active["hp_frac"] = 0.46
        opp_active["hp_frac"] = 0.56
    if "opponent_setup_pressure" in tags:
        opp_active["boosts"]["atk"] = 2
        opp_active["hp_frac"] = max(float(opp_active["hp_frac"]), 0.68)
        legal_switches = [{"slot": 2, "hp_frac": 0.9}]
    if turns >= 30:
        opp_active["hp_frac"] = max(float(opp_active["hp_frac"]), 0.7)
        my_active["hp_frac"] = min(float(my_active["hp_frac"]), 0.4)

    return state, legal_moves, legal_switches


def analyze_loss_attention(summary_payload: Dict[str, Any], *, top_k: int = 3) -> Dict[str, Any]:
    losses = list(summary_payload.get("losses") or [])
    per_loss: List[Dict[str, Any]] = []
    tag_token_totals: Dict[str, Counter[str]] = defaultdict(Counter)
    primary_word_counts: Counter[str] = Counter()

    for loss in losses:
        question = _proxy_question(loss)
        battle_state, legal_moves, legal_switches = _proxy_context(loss)
        answer = answer_battle_inquiry(
            question,
            battle_state,
            perspective_player="p1",
            legal_moves=legal_moves,
            legal_switches=legal_switches,
            top_k=top_k,
        )
        primary_word = answer.predictions[0].word if answer.predictions else None
        primary_word_counts.update([primary_word] if primary_word else [])
        top_row = (answer.attention_report or {}).get("rows", [{}])[0]
        token_weights = {
            token: float(weight)
            for token, weight in zip(answer.prompt_tokens, top_row.get("token_weights", []))
        }
        for tag in loss.get("tags") or ["untagged"]:
            tag_token_totals[tag].update(token_weights)
        per_loss.append(
            {
                "file": loss.get("file"),
                "tags": list(loss.get("tags") or []),
                "question": question,
                "prompt_tokens": list(answer.prompt_tokens),
                "primary_word": primary_word,
                "attention_row": top_row,
                "token_weights": token_weights,
            }
        )

    tag_focus = {
        tag: [
            {"token": token, "weight": float(weight)}
            for token, weight in counter.most_common(5)
        ]
        for tag, counter in sorted(tag_token_totals.items())
    }
    return {
        "loss_count": len(losses),
        "primary_word_counts": dict(primary_word_counts.most_common()),
        "tag_attention_focus": tag_focus,
        "losses": per_loss,
    }


def analyze_loss_attention_file(summary_path: Path, *, top_k: int = 3) -> Dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result = analyze_loss_attention(payload, top_k=top_k)
    result["summary_file"] = str(summary_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze attention focus across replay-loss summaries.")
    parser.add_argument("summary_json", type=Path, help="Path to a replay-loss summary JSON file.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_loss_attention_file(args.summary_json, top_k=args.top_k)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
