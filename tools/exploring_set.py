from __future__ import annotations

import json
from pathlib import Path

from BattleStateTracker import BattleStateTracker
from StateVectorization import build_move_vocab, iter_turn_examples_both_players, vectorize_dataset


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "data" / "gen9randombattle-2390494424.json"
    with path.open("r", encoding="utf-8") as handle:
        battle = json.load(handle)

    tracker = BattleStateTracker()
    examples_both = list(iter_turn_examples_both_players(tracker, battle))
    if not examples_both:
        raise SystemExit("No examples were generated from the sample battle log.")

    print("num examples (both players, move-only):", len(examples_both))
    print("sample:", examples_both[0]["player"], examples_both[0]["turn_number"], examples_both[0]["action"])

    move_vocab = build_move_vocab(examples_both, min_count=1)
    print("num move classes (including <UNK>):", len(move_vocab))
    print("top 10 vocab items:", list(move_vocab.items())[:10])

    X, y = vectorize_dataset(examples_both, move_vocab)
    print("vectorized shapes:", getattr(X, "shape", None), getattr(y, "shape", None))


if __name__ == "__main__":
    main()
