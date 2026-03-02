import numpy as np
import random
from collections import defaultdict
import json
from BattleStateTracker import BattleStateTracker
from StateVectorization import iter_turn_examples_both_players

def group_split_by_battle_id(examples, val_ratio=0.2, seed=42):
    by_battle = defaultdict(list)
    for i, ex in enumerate(examples):
        by_battle[ex["battle_id"]].append(i)

    battle_ids = list(by_battle.keys())
    rng = random.Random(seed)
    rng.shuffle(battle_ids)

    n_val = max(1, int(len(battle_ids) * val_ratio))
    val_battles = set(battle_ids[:n_val])

    train_idx, val_idx = [], []
    for bid, idxs in by_battle.items():
        (val_idx if bid in val_battles else train_idx).extend(idxs)

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def safe_load_json(p: str):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def ingest_battles_to_examples(
    tracker: BattleStateTracker,
    json_paths,
    max_battles=1000,
    verbose_every=200,
):
    examples = []
    battles_loaded = 0

    for p in json_paths:
        battle = safe_load_json(p)
        if not battle or "turns" not in battle:
            continue

        try:
            # your function that yields move examples for both players
            for ex in iter_turn_examples_both_players(tracker, battle):
                examples.append(ex)
        except Exception:
            # skip broken battle logs
            continue

        battles_loaded += 1
        if verbose_every and battles_loaded % verbose_every == 0:
            print(f"battles_loaded={battles_loaded}  examples={len(examples)}")

        if battles_loaded >= max_battles:
            break

    return examples


