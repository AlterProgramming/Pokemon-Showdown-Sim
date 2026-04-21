"""Profile the history encoder data pipeline.

Exercises BattleStateTracker rolling buffer, TurnEventTokenizer.encode_event_history,
and EntityTensorization history vectorization at scale to find bottlenecks before
committing to a full Colab training run.

Usage:
    scalene --cpu --memory scripts/profile_history_pipeline.py
    # or via /profile-training:
    /profile-training scripts/profile_history_pipeline.py
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.TurnEventTokenizer import (
    build_sequence_vocab, encode_event_history, encode_turn_event_sequence,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
)

# ---------------------------------------------------------------------------
# Synthetic event generator
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    {"event_type": "move",   "actor_side": "p1", "move_id": "tackle",    "target_side": "p2"},
    {"event_type": "damage", "target_side": "p2", "hp_delta_bin": -3},
    {"event_type": "switch", "actor_side": "p1", "species_id": "pikachu", "slot_index": 2},
    {"event_type": "faint",  "target_side": "p2"},
    {"event_type": "status_start", "target_side": "p1", "status": "brn"},
    {"event_type": "boost",  "target_side": "p2", "boost_stat": "atk", "boost_delta": 1},
    {"event_type": "weather", "weather": "raindance"},
    {"event_type": "field",  "terrain": "electricterrain", "is_removal": False},
    {"event_type": "side_condition", "actor_side": "p1", "side_condition": "stealthrock", "is_removal": False},
    {"event_type": "turn_end"},
]

def make_synthetic_turn(n_events=6, rng=None):
    rng = rng or random
    return [dict(rng.choice(EVENT_TYPES)) for _ in range(n_events)] + [{"event_type": "turn_end"}]

def make_synthetic_past_turns(n_turns, events_per_turn=6, rng=None):
    return [make_synthetic_turn(events_per_turn, rng) for _ in range(n_turns)]

# ---------------------------------------------------------------------------
# Build a vocabulary from synthetic examples
# ---------------------------------------------------------------------------
N_VOCAB_EXAMPLES = 2000
MAX_TURNS        = 8
MAX_EVENTS       = 24
N_ENCODE_CALLS   = 50_000

rng = random.Random(42)

print(f"Building vocab from {N_VOCAB_EXAMPLES} synthetic examples...")
t0 = time.perf_counter()

vocab_examples = [
    {"turn_events_v1": make_synthetic_turn(rng.randint(2, 10), rng)}
    for _ in range(N_VOCAB_EXAMPLES)
]
vocab = build_sequence_vocab(vocab_examples)
vocab_time = time.perf_counter() - t0
print(f"  vocab size={len(vocab)}  time={vocab_time*1000:.1f}ms")

# ---------------------------------------------------------------------------
# Benchmark encode_turn_event_sequence (single-turn encoding)
# ---------------------------------------------------------------------------
print(f"\nBenchmarking encode_turn_event_sequence x{N_ENCODE_CALLS}...")
t0 = time.perf_counter()
for _ in range(N_ENCODE_CALLS):
    turn_ev = make_synthetic_turn(rng.randint(2, 8), rng)
    encode_turn_event_sequence(turn_ev, vocab, max_len=MAX_EVENTS)
seq_time = time.perf_counter() - t0
print(f"  total={seq_time*1000:.1f}ms  per_call={seq_time/N_ENCODE_CALLS*1e6:.1f}µs")

# ---------------------------------------------------------------------------
# Benchmark encode_event_history (history matrix encoding)
# ---------------------------------------------------------------------------
N_HISTORY_CALLS = 20_000

print(f"\nBenchmarking encode_event_history x{N_HISTORY_CALLS} (K={MAX_TURNS}, E={MAX_EVENTS})...")
t0 = time.perf_counter()
for i in range(N_HISTORY_CALLS):
    n_turns = rng.randint(0, MAX_TURNS)
    past = make_synthetic_past_turns(n_turns, events_per_turn=rng.randint(2, 8), rng=rng)
    tokens, mask = encode_event_history(past, vocab, MAX_TURNS, MAX_EVENTS)
history_time = time.perf_counter() - t0
print(f"  total={history_time*1000:.1f}ms  per_call={history_time/N_HISTORY_CALLS*1e6:.1f}µs")

# ---------------------------------------------------------------------------
# Benchmark BattleStateTracker deque append (rolling buffer)
# ---------------------------------------------------------------------------
print(f"\nBenchmarking deque append (rolling buffer) x{N_ENCODE_CALLS}...")
from collections import deque
buf = deque(maxlen=MAX_TURNS)
t0 = time.perf_counter()
for _ in range(N_ENCODE_CALLS):
    turn_ev = make_synthetic_turn(rng.randint(2, 8), rng)
    buf.append([ev for ev in turn_ev])
deque_time = time.perf_counter() - t0
print(f"  total={deque_time*1000:.1f}ms  per_call={deque_time/N_ENCODE_CALLS*1e6:.1f}µs")

# ---------------------------------------------------------------------------
# Simulate a full dataset vectorization pass (N battles × T turns × history)
# ---------------------------------------------------------------------------
N_BATTLES    = 200
TURNS_PER_BATTLE = 20

print(f"\nSimulating full dataset pass: {N_BATTLES} battles × {TURNS_PER_BATTLE} turns × history encode...")
t0 = time.perf_counter()
total_examples = 0
for b in range(N_BATTLES):
    hist_buf = deque(maxlen=MAX_TURNS)
    for t in range(TURNS_PER_BATTLE):
        # Simulate iter_turn_examples: snapshot, then encode, then push
        past = list(hist_buf)
        tokens, mask = encode_event_history(past, vocab, MAX_TURNS, MAX_EVENTS)
        turn_ev = make_synthetic_turn(rng.randint(2, 8), rng)
        hist_buf.append(turn_ev)
        total_examples += 1
full_time = time.perf_counter() - t0
print(f"  total={full_time*1000:.1f}ms  examples={total_examples:,}")
print(f"  per_example={full_time/total_examples*1e6:.1f}µs")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("HISTORY PIPELINE COST SUMMARY")
print("=" * 60)
print(f"  encode_turn_event_sequence : {seq_time/N_ENCODE_CALLS*1e6:.1f} µs/call")
print(f"  encode_event_history       : {history_time/N_HISTORY_CALLS*1e6:.1f} µs/call  (K={MAX_TURNS}, E={MAX_EVENTS})")
print(f"  deque append               : {deque_time/N_ENCODE_CALLS*1e6:.1f} µs/call")
print(f"  full dataset pass          : {full_time/total_examples*1e6:.1f} µs/example")
print()
print(f"  At 5000 battles × ~20 turns = ~100k examples:")
est_total_ms = (full_time / total_examples) * 100_000 * 1000
print(f"  Estimated data-loading cost: {est_total_ms:.0f} ms  ({est_total_ms/1000:.1f}s)")
print("=" * 60)
