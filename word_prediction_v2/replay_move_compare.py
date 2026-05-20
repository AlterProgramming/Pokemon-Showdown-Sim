from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .policy_adapter import debug_choose_action_from_inquiry
from .replay_reconstruction import reconstruct_replay_file
from .text import normalize_token


def _decoded_action_id(action: Dict[str, Any]) -> tuple[str, str]:
    action_type = str(action.get("type") or "none")
    if action_type == "move":
        best_move = action.get("best_move") or {}
        move_id = str(best_move.get("id") or normalize_token(str(best_move.get("move") or "")))
        return action_type, move_id
    if action_type == "switch":
        slot = action.get("slot")
        return action_type, f"slot:{slot}"
    return action_type, ""


def _serialize_scored_move(row: Dict[str, Any]) -> Dict[str, Any]:
    move = row["move"]
    move_id = str(move.get("id") or normalize_token(str(move.get("move") or "")))
    return {
        "move_id": move_id,
        "move_name": str(move.get("move") or move.get("id") or ""),
        "slot": move.get("slot"),
        "total_score": float(row["total_score"]),
        "word_score": float(row["word_score"]),
        "state_score": float(row["state_score"]),
        "word_contributions": [
            {
                "word": str(item["word"]),
                "prediction_score": float(item["prediction_score"]),
                "contribution": float(item["contribution"]),
            }
            for item in row["word_contributions"]
        ],
    }


def analyze_replay_move_choices(
    replay_path: Path,
    *,
    limit: int | None = None,
    top_k: int = 3,
    top_move_k: int = 5,
) -> Dict[str, Any]:
    final_state, decision_points = reconstruct_replay_file(replay_path)
    rows: List[Dict[str, Any]] = []
    exact_match_count = 0
    word_counter = Counter()

    for index, point in enumerate(decision_points):
        if limit is not None and index >= limit:
            break
        debug = debug_choose_action_from_inquiry(
            point.question,
            point.battle_state,
            perspective_player=point.player,
            legal_moves=list(point.legal_moves),
            legal_switches=list(point.legal_switches),
            top_k=top_k,
        )
        action = debug["action"]
        decoded_action_type, decoded_action_id = _decoded_action_id(action)
        predicted_words = list(debug["predictions"])
        primary_word = str(predicted_words[0]["word"]) if predicted_words else None
        if primary_word:
            word_counter.update([primary_word])

        serialized_moves = [_serialize_scored_move(item) for item in debug["scored_moves"][:top_move_k]]
        actual_move_rank = None
        actual_move_score = None
        for rank, move_row in enumerate(debug["scored_moves"], start=1):
            move_id = str(move_row["move"].get("id") or normalize_token(str(move_row["move"].get("move") or "")))
            if move_id == point.actual_action_id:
                actual_move_rank = rank
                actual_move_score = float(move_row["total_score"])
                break

        matches_actual = decoded_action_type == point.actual_action_type and decoded_action_id == point.actual_action_id
        if matches_actual:
            exact_match_count += 1

        rows.append(
            {
                "turn_index": point.turn_index,
                "question": point.question,
                "actual_action_type": point.actual_action_type,
                "actual_action_id": point.actual_action_id,
                "decoded_action_type": decoded_action_type,
                "decoded_action_id": decoded_action_id,
                "matches_actual": matches_actual,
                "predicted_primary_word": primary_word,
                "predicted_words": predicted_words,
                "prompt_tokens": list(debug["prompt_tokens"]),
                "attention_report": debug["attention_report"],
                "actual_move_rank": actual_move_rank,
                "actual_move_score": actual_move_score,
                "top_scored_moves": serialized_moves,
            }
        )

    compared_count = len(rows)
    return {
        "replay_file": str(replay_path),
        "decision_count": len(decision_points),
        "compared_decision_count": compared_count,
        "exact_match_count": exact_match_count,
        "exact_match_rate": float(exact_match_count / compared_count) if compared_count else 0.0,
        "primary_word_counts": dict(word_counter),
        "final_state_turn": final_state.get("turn_index"),
        "rows": rows,
    }


def aggregate_replay_move_choices(
    replay_paths: Sequence[Path],
    *,
    top_k: int = 3,
    top_move_k: int = 5,
) -> Dict[str, Any]:
    replay_rows: List[Dict[str, Any]] = []
    overall_word_counts = Counter()
    exact_match_total = 0
    compared_total = 0

    for replay_path in replay_paths:
        payload = analyze_replay_move_choices(
            replay_path,
            top_k=top_k,
            top_move_k=top_move_k,
        )
        overall_word_counts.update(payload["primary_word_counts"])
        exact_match_total += int(payload["exact_match_count"])
        compared_total += int(payload["compared_decision_count"])
        replay_rows.append(
            {
                "replay_file": payload["replay_file"],
                "decision_count": payload["decision_count"],
                "compared_decision_count": payload["compared_decision_count"],
                "exact_match_count": payload["exact_match_count"],
                "exact_match_rate": payload["exact_match_rate"],
                "primary_word_counts": payload["primary_word_counts"],
            }
        )

    return {
        "replay_count": len(replay_paths),
        "compared_decision_count": compared_total,
        "exact_match_count": exact_match_total,
        "exact_match_rate": float(exact_match_total / compared_total) if compared_total else 0.0,
        "overall_primary_word_counts": dict(overall_word_counts),
        "replays": replay_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reconstructed replay actions against decoder move choices.")
    parser.add_argument("replay_paths", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-move-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if len(args.replay_paths) == 1:
        payload = analyze_replay_move_choices(
            args.replay_paths[0],
            limit=args.limit,
            top_k=args.top_k,
            top_move_k=args.top_move_k,
        )
    else:
        payload = aggregate_replay_move_choices(
            list(args.replay_paths),
            top_k=args.top_k,
            top_move_k=args.top_move_k,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
