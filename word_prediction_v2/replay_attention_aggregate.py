from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .replay_attention_trace import analyze_replay_attention


def aggregate_replay_attention(
    replay_paths: List[Path],
    *,
    top_k: int = 3,
    late_turn_threshold: int = 16,
) -> Dict[str, Any]:
    replay_rows: List[Dict[str, Any]] = []
    overall_primary = Counter()
    late_primary = Counter()

    for replay_path in replay_paths:
        payload = analyze_replay_attention(replay_path, top_k=top_k)
        replay_counter = Counter()
        replay_late_counter = Counter()
        for row in payload["rows"]:
            if not row["predicted_words"]:
                continue
            primary = row["predicted_words"][0]["word"]
            replay_counter.update([primary])
            overall_primary.update([primary])
            if int(row["turn_index"]) >= late_turn_threshold:
                replay_late_counter.update([primary])
                late_primary.update([primary])

        replay_rows.append(
            {
                "replay_file": str(replay_path),
                "decision_count": payload["decision_count"],
                "primary_counts": dict(replay_counter),
                "late_primary_counts": dict(replay_late_counter),
            }
        )

    return {
        "replay_count": len(replay_paths),
        "late_turn_threshold": late_turn_threshold,
        "overall_primary_counts": dict(overall_primary),
        "overall_late_primary_counts": dict(late_primary),
        "replays": replay_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate reconstructed attention traces across replay losses.")
    parser.add_argument("replay_paths", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--late-turn-threshold", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = aggregate_replay_attention(
        list(args.replay_paths),
        top_k=args.top_k,
        late_turn_threshold=args.late_turn_threshold,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
