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

---

# Session 2: TurnEvent Tokenization + Sequence Head

## What Was Built
- `core/TurnEventTokenizer.py` — Composite-key tokenizer converting TurnEventV1 event dicts to integer token sequences with BOS/EOS/PAD
- `core/StateVectorization.py` — Extended `vectorize_multitask_dataset()` with `include_sequence`, `sequence_vocab`, `max_seq_len` parameters
- `train_policy.py` — New `--predict-turn-sequence` flag adds LSTM sequence decoder head conditioned on state + my_action + opp_action. Shared action embeddings with transition head when both active.
- `tests/test_turn_event_tokenizer.py` — 19 tests covering composite keys, vocab building, encode/decode, and vectorization integration

## How to Validate
```
python3 -m pytest tests/test_turn_event_tokenizer.py -v
python3 -m pytest tests/test_turn_event_v1.py -v  # Session 1 regression check
```

## New CLI Flags
- `--predict-turn-sequence` — Enable auxiliary sequence head (default: off)
- `--sequence-weight 0.1` — Loss weight for sequence head
- `--sequence-hidden-dim 128` — LSTM hidden dimension
- `--max-seq-len 32` — Maximum padded token sequence length

## Design Decisions
- **Composite token per event**: Each TurnEventV1 becomes one string token (e.g. `"move:p2:thunderbolt:p1"`), keeping sequence length = number of events per turn (typically 3-15)
- **Non-autoregressive LSTM**: Context is repeated across timesteps via RepeatVector; each position is predicted independently. True autoregressive decoding deferred to future session.
- **Shared action embeddings**: When both `--predict-turn-outcome` and `--predict-turn-sequence` are active, they share the same action embedding layer
- **Vocab built from training data**: `build_sequence_vocab()` scans examples and assigns sorted IDs. Saved as `sequence_vocab.json` artifact.
- **Truncation preserves EOS**: If events exceed max_seq_len, sequence is truncated but EOS token is always the last non-PAD token

## What's Next (Session 3)
1. Phase 2: TypeScript `TurnEventAccumulator` in `pokemon-showdown-model-feature` repo
2. Browser-side decision-boundary gating (one request per turn)
3. Completeness oracle against flat vector layout
4. True autoregressive decoding (teacher-forced at train, beam/greedy at inference)
