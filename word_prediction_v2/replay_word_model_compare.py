from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .battle_inquiry import answer_battle_inquiry
from .pipeline import DEFAULT_ARTIFACT_ROOT, load_model
from .replay_reconstruction import reconstruct_replay_file


def _load_named_model(run_name: str) -> Any:
    return load_model(DEFAULT_ARTIFACT_ROOT / run_name / "model.json")


def compare_replay_word_models(
    replay_paths: Sequence[Path],
    *,
    old_run_name: str = "battle_inquiry_v2",
    new_run_name: str = "battle_inquiry_v3",
    top_k: int = 3,
    late_turn_threshold: int = 16,
) -> Dict[str, Any]:
    old_model = _load_named_model(old_run_name)
    new_model = _load_named_model(new_run_name)

    overall_old = Counter()
    overall_new = Counter()
    late_old = Counter()
    late_new = Counter()
    changed_pairs = Counter()
    replay_rows: List[Dict[str, Any]] = []

    for replay_path in replay_paths:
        _, decision_points = reconstruct_replay_file(replay_path)
        replay_old = Counter()
        replay_new = Counter()
        replay_changed = Counter()
        changed_rows: List[Dict[str, Any]] = []

        for point in decision_points:
            old_answer = answer_battle_inquiry(
                point.question,
                point.battle_state,
                perspective_player=point.player,
                legal_moves=list(point.legal_moves),
                legal_switches=list(point.legal_switches),
                top_k=top_k,
                model=old_model,
            )
            new_answer = answer_battle_inquiry(
                point.question,
                point.battle_state,
                perspective_player=point.player,
                legal_moves=list(point.legal_moves),
                legal_switches=list(point.legal_switches),
                top_k=top_k,
                model=new_model,
            )
            if not old_answer.predictions or not new_answer.predictions:
                continue
            old_primary = str(old_answer.predictions[0].word)
            new_primary = str(new_answer.predictions[0].word)

            replay_old.update([old_primary])
            replay_new.update([new_primary])
            overall_old.update([old_primary])
            overall_new.update([new_primary])
            if int(point.turn_index) >= late_turn_threshold:
                late_old.update([old_primary])
                late_new.update([new_primary])

            if old_primary != new_primary:
                replay_changed.update([(old_primary, new_primary)])
                changed_pairs.update([(old_primary, new_primary)])
                changed_rows.append(
                    {
                        "turn_index": point.turn_index,
                        "question": point.question,
                        "actual_action_id": point.actual_action_id,
                        "old_primary": old_primary,
                        "new_primary": new_primary,
                        "old_predictions": [
                            {"word": item.word, "score": item.score}
                            for item in old_answer.predictions
                        ],
                        "new_predictions": [
                            {"word": item.word, "score": item.score}
                            for item in new_answer.predictions
                        ],
                    }
                )

        replay_rows.append(
            {
                "replay_file": str(replay_path),
                "decision_count": len(decision_points),
                "old_primary_counts": dict(replay_old),
                "new_primary_counts": dict(replay_new),
                "changed_primary_counts": {
                    f"{old}->{new}": count
                    for (old, new), count in replay_changed.items()
                },
                "changed_rows": changed_rows,
            }
        )

    return {
        "old_run_name": old_run_name,
        "new_run_name": new_run_name,
        "replay_count": len(replay_paths),
        "late_turn_threshold": late_turn_threshold,
        "overall_old_primary_counts": dict(overall_old),
        "overall_new_primary_counts": dict(overall_new),
        "overall_old_late_primary_counts": dict(late_old),
        "overall_new_late_primary_counts": dict(late_new),
        "overall_changed_primary_counts": {
            f"{old}->{new}": count
            for (old, new), count in changed_pairs.items()
        },
        "replays": replay_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reconstructed replay word outputs across two battle word-model runs.")
    parser.add_argument("replay_paths", nargs="+", type=Path)
    parser.add_argument("--old-run-name", default="battle_inquiry_v2")
    parser.add_argument("--new-run-name", default="battle_inquiry_v3")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--late-turn-threshold", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = compare_replay_word_models(
        list(args.replay_paths),
        old_run_name=args.old_run_name,
        new_run_name=args.new_run_name,
        top_k=args.top_k,
        late_turn_threshold=args.late_turn_threshold,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
