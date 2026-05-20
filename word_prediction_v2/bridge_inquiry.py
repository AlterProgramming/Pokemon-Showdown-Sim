from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .battle_inquiry import answer_battle_inquiry, load_inquiry_context
from .pipeline import DEFAULT_ARTIFACT_ROOT


def build_bridge_response(
    payload: Dict[str, Any],
    *,
    question: str,
    top_k: int = 3,
    output_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    battle_state, perspective_player, legal_moves, legal_switches = (
        load_inquiry_context(None) if not payload else (
            payload.get("battle_state") if isinstance(payload.get("battle_state"), dict) else payload,
            payload.get("perspective_player") if payload.get("perspective_player") in {"p1", "p2"} else "p1",
            payload.get("legal_moves") if isinstance(payload.get("legal_moves"), list) else [],
            payload.get("legal_switches") if isinstance(payload.get("legal_switches"), list) else [],
        )
    )

    answer = answer_battle_inquiry(
        question,
        battle_state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        top_k=top_k,
        output_root=output_root,
    )
    predictions = [{"word": item.word, "score": item.score} for item in answer.predictions]
    return {
        "question": question,
        "perspective_player": perspective_player,
        "model_source": answer.model_source,
        "prompt_tokens": list(answer.prompt_tokens),
        "primary_word": predictions[0]["word"] if predictions else None,
        "primary_score": predictions[0]["score"] if predictions else None,
        "backup_words": predictions[1:],
        "predictions": predictions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge-facing battle inquiry wrapper.")
    parser.add_argument("--question", required=True, help="Battle inquiry to answer.")
    parser.add_argument("--payload-file", type=Path, required=True, help="Normalized payload JSON file.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.payload_file.read_text(encoding="utf-8"))
    response = build_bridge_response(
        payload,
        question=args.question,
        top_k=args.top_k,
        output_root=args.output_root,
    )
    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
