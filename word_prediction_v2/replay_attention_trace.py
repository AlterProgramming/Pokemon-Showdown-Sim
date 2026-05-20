from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .battle_inquiry import answer_battle_inquiry
from .replay_reconstruction import reconstruct_replay_file


def analyze_replay_attention(path: Path, *, limit: int | None = None, top_k: int = 3) -> Dict[str, Any]:
    final_state, decision_points = reconstruct_replay_file(path)
    rows: List[Dict[str, Any]] = []

    for index, point in enumerate(decision_points):
        if limit is not None and index >= limit:
            break
        answer = answer_battle_inquiry(
            point.question,
            point.battle_state,
            perspective_player=point.player,
            legal_moves=list(point.legal_moves),
            legal_switches=list(point.legal_switches),
            top_k=top_k,
        )
        rows.append(
            {
                "turn_index": point.turn_index,
                "question": point.question,
                "actual_action_type": point.actual_action_type,
                "actual_action_id": point.actual_action_id,
                "predicted_words": [
                    {"word": item.word, "score": item.score}
                    for item in answer.predictions
                ],
                "attention_report": answer.attention_report,
            }
        )

    return {
        "replay_file": str(path),
        "decision_count": len(decision_points),
        "traced_decision_count": len(rows),
        "final_state_turn": final_state.get("turn_index"),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace attention over reconstructed replay decision points.")
    parser.add_argument("replay_html", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = analyze_replay_attention(args.replay_html, limit=args.limit, top_k=args.top_k)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
