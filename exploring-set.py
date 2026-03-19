from BattleStateTracker import BattleStateTracker
import json
from StateVectorization import iter_turn_examples_both_players
from StateVectorization import build_move_vocab, vectorize_dataset

path = "./gen9randombattle-2390494424.json"
with open(path, "r") as f:
    battle = json.load(f)

tracker = BattleStateTracker()
examples_both = list(iter_turn_examples_both_players(tracker, battle))


print("num examples (both players, move-only):", len(examples_both))
print("sample:", examples_both[0]["player"], examples_both[0]["turn_number"], examples_both[0]["action"])

move_vocab = build_move_vocab(examples_both, min_count=1)
print("num move classes (including <UNK>):", len(move_vocab))
print("top 10 vocab items:", list(move_vocab.items())[:10])

X, y = vectorize_dataset(examples_both, move_vocab)