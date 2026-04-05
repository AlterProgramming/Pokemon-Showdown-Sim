# Session 1: TurnEventV1 Extraction

## What Was Built
- `core/TurnEventV1.py` — Structured event schema (14 event types, HP bin quantization, serialization)
- `core/BattleStateTracker.py` — Modified to emit `turn_events_v1` in `iter_turn_examples()` output
- `tests/test_turn_event_v1.py` — Schema and integration tests

## How to Validate
```
python3 -m pytest tests/test_turn_event_v1.py -v
python3 -m pytest tests/test_battle_state_tracker.py -v  # regression check
```

## Design Decisions
- Events are captured inside `_apply_*` handlers, in the order they appear in the turn's event list
- HP deltas use 5% bins (range -20 to +20) for discretization
- TURN_END sentinel marks the end of every turn's event sequence
- `_backfill_visible_actives()` does NOT emit events (runs before apply_turn)
- Side normalization uses raw p1/p2 from UIDs (not perspective-relative)

## What's Next (Session 2)
1. `core/TurnEventTokenizer.py` — Convert TurnEventV1 lists to integer token sequences (BOS/EOS/PAD)
2. Extend `StateVectorization.py` with `vectorize_sequence_targets()`
3. Add `--predict-turn-sequence` autoregressive decoder head in `train_policy.py`

## What's Next (Session 3)
1. TypeScript `TurnEventAccumulator` in `pokemon-showdown-model-feature` repo
2. Browser-side decision-boundary gating (one request per turn)
3. Completeness oracle against flat vector layout
