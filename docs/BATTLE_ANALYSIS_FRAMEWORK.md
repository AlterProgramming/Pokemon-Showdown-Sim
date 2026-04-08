# Battle Analysis Framework

**Version:** 1.0
**Date:** 2026-04-07
**Purpose:** Reference documentation for battle reconstruction, event sequencing, and statistical analysis in the Pokemon Showdown AI training pipeline.

---

## Overview

Battle analysis is the bridge between raw game events and model training/evaluation. This framework describes:

- **Battle Archive Structure**: How battles are organized and stored after a training/evaluation run
- **Event Indexing**: How within-turn events map to turns and sequential time
- **Battle Reconstruction**: How to rebuild canonical battle state from event logs
- **Statistical Analysis**: How to compute win rates, move accuracy, and tactical metrics
- **Integration Points**: Where battle analysis connects to model training and diagnostics

The framework is **event-centric**: all battle data flows through the `TurnEventV1` schema (defined in `core/TurnEventV1.py`), ensuring consistency across training, serving, and analysis.

---

## Part 1: Battle Archive Structure

### Directory Layout

```
runs/
├── {run_id}/
│   ├── run_metadata.json          # Run configuration and summary
│   ├── event_index.csv            # Turn/step → event mapping
│   ├── battle_logs/
│   │   ├── battle_{game_id}.json  # Single battle events and state snapshots
│   │   ├── battle_{game_id}.json
│   │   └── ...
│   └── diagnostics/
│       ├── win_rate_summary.json
│       ├── move_accuracy.csv
│       ├── tactical_errors.json
│       └── state_reconstruction_failures.json
```

### run_id Naming Convention

`{model_name}_{timestamp}_{tag}` — e.g., `entity_action_bc_v1_20260407_session3`

Where:
- `model_name`: Model family/variant being evaluated
- `timestamp`: ISO 8601 date (YYYYMMDD) or datetime (YYYYMMDDTHHmmss)
- `tag`: Optional session/run identifier for disambiguation

### run_metadata.json

Top-level summary of a training/evaluation run. Required fields:

```json
{
  "run_id": "entity_action_bc_v1_20260407_session3",
  "model_release_id": "entity_action_bc_v1",
  "model_path": "artifacts/entity_action_bc_v1_20260327_run2.keras",
  "policy_vocab_path": "artifacts/entity_action_bc_v1_20260327_run2.policy_vocab.json",
  "run_type": "evaluation_vs_baseline",
  "baseline_model_id": "entity_action_bc_v1_20260327_run2",
  "num_battles": 100,
  "battles_won": 67,
  "win_rate": 0.67,
  "created_timestamp": "2026-04-07T15:30:00Z",
  "completed_timestamp": "2026-04-07T16:45:30Z",
  "battle_format": "gen9randombattle",
  "evaluation_config": {
    "num_workers": 4,
    "max_turns_per_battle": 200,
    "seed": 42
  },
  "high_level_stats": {
    "avg_turns_per_battle": 45.3,
    "mean_turn_outcome_accuracy": 0.82,
    "mean_sequence_accuracy": 0.76,
    "move_accuracy": 0.81,
    "switch_accuracy": 0.64
  }
}
```

**Key Metrics:**
- `win_rate`: P(model wins) over all battles
- `mean_turn_outcome_accuracy`: Fraction of turn transition predictions that match actual next state
- `mean_sequence_accuracy`: Token-level accuracy on turn_events_v1 auxiliary head
- `move_accuracy`: Fraction of turns where model chose a legal move that was used
- `switch_accuracy`: Fraction of switch turns where model chose a legal switch that was used

---

## Part 2: Event Index (event_index.csv)

Maps turn/step boundaries to event records in the battle log files.

### Format

```csv
battle_id,turn_number,step_in_turn,first_event_idx,last_event_idx,num_events,game_state_hash
game_001,1,1,0,5,6,abc123def
game_001,1,2,6,12,7,def456ghi
game_001,2,1,13,18,6,ghi789jkl
...
game_002,1,1,0,4,5,jkl012mno
...
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `battle_id` | str | Unique identifier for the battle (e.g., `game_001`, `pokemonshowdown_abc123`) |
| `turn_number` | int | Battle turn (1-indexed). One turn = both players' actions |
| `step_in_turn` | int | Substep within the turn (1 or 2). Step 1 = P1's action resolution; Step 2 = P2's action resolution |
| `first_event_idx` | int | Index of first event in `battle_logs/{battle_id}.json` events array (0-indexed) |
| `last_event_idx` | int | Index of last event (inclusive) |
| `num_events` | int | Total events in this substep = `last_event_idx - first_event_idx + 1` |
| `game_state_hash` | str | Optional SHA-256 hash of canonical state after this step (for integrity verification) |

### Purpose

- **Fast lookups**: Find all events for a specific turn without scanning entire battle log
- **Sequential reconstruction**: Replay battles in turn/step order
- **Event boundaries**: Identify which events belong together (e.g., all events from P1's move resolution)

---

## Part 3: Battle Logs (battle_{game_id}.json)

Per-battle events and state snapshots.

### Structure

```json
{
  "metadata": {
    "battle_id": "game_001",
    "p1_name": "model",
    "p2_name": "random",
    "p1_won": true,
    "duration_turns": 45,
    "created_timestamp": "2026-04-07T15:30:15Z",
    "format": "gen9randombattle",
    "seed": 12345
  },
  "initial_state": {
    "turn_index": 0,
    "field": {...},
    "p1": {...},
    "p2": {...},
    "mons": {...}
  },
  "events": [
    {
      "event_type": "move",
      "actor_side": "p1",
      "target_side": "p2",
      "move_id": "tackle"
    },
    {
      "event_type": "damage",
      "target_side": "p2",
      "hp_delta_bin": -5
    },
    {
      "event_type": "turn_end"
    },
    ...
  ],
  "decision_points": [
    {
      "turn": 1,
      "step": 1,
      "player": "p1",
      "state_before": {...},
      "action_chosen": "move:tackle",
      "action_index": 0,
      "legal_actions": ["move:tackle", "move:body-slam", "switch:pikachu"],
      "model_logits": [0.8, 0.15, 0.05],
      "move_outcome": "success",
      "state_after": {...}
    },
    ...
  ]
}
```

### Key Sections

#### `metadata`
Run context for this battle.

#### `initial_state`
Canonical battle state at turn 0. See `BattleStateTracker` for full schema (in `core/BattleStateTracker.py`).

#### `events`
Ordered array of `TurnEventV1` objects (as dicts). Each event represents one atomic change during the battle.

**Event Types** (from `core/TurnEventV1.py`):
- `move`: Actor used a move against target
- `switch`: Actor switched to new Pokémon
- `damage` / `heal`: Target's HP changed
- `boost` / `unboost`: Target's stat stage changed
- `status_start` / `status_end`: Status condition applied/removed
- `faint`: Target fainted
- `weather`: Global weather changed
- `field`: Terrain changed
- `side_condition`: Reflect/Light Screen/entry hazards/etc changed
- `forme_change`: Pokémon changed form (Tera/evolution)
- `turn_end`: Turn completed

#### `decision_points`
Model's action choices and outcomes. One per turn/player pair.

**Fields:**
- `turn`, `step`, `player`: Timing of the decision
- `state_before`: Canonical state when the player must choose an action
- `action_chosen`: Token representation of chosen action (e.g., `move:tackle`)
- `action_index`: Index into `legal_actions` array
- `legal_actions`: All valid actions available at this decision point
- `model_logits`: Softmax-normalized logits from policy head (sum ≈ 1.0)
- `move_outcome`: Result of executing the action—one of:
  - `success`: Action executed as expected
  - `no_effect`: Move hit but dealt no damage (immunity, status already applied, etc.)
  - `illegal`: Action was not in legal set (infrastructure error)
  - `setup_wasted`: Status/setup move used against immune/already-affected target
  - `switch_forced`: Actor was forced to switch (active Pokémon fainted)
  - `partial_trap_trapped`: Actor in partial trap, couldn't switch
- `state_after`: Canonical state after this player's action

---

## Part 4: Run Metadata Structure

### Metadata Inheritance Chain

```
run_metadata.json                          [Run-level summary]
    ↓
battle_logs/battle_{id}.json:metadata      [Per-battle summary]
    ↓
battle_logs/battle_{id}.json:decision_points  [Per-decision detail]
    ↓
event_index.csv + battle_logs:events       [Per-event atomics]
```

### Training Metadata vs. Run Metadata

**Training Metadata** (`artifacts/training_metadata_{model_id}.json`):
- Hyperparameters, learning rate, batch size, epochs, etc.
- Model architecture, vocab sizes
- Data source, train/val split
- Produced at **training time** by `train_entity_action.py` or `train_policy.py`

**Run Metadata** (`runs/{run_id}/run_metadata.json`):
- Model release ID and artifact paths (references training metadata)
- Evaluation configuration (num battles, format, workers)
- Aggregate statistics and win rate
- Produced at **evaluation time** by evaluation scripts

### Relationship Example

```python
# Training run (produces artifacts/)
python train_entity_action.py \
  --model-name entity_action_bc_v1 \
  --predict-turn-sequence \
  --output-dir artifacts

# Creates: artifacts/training_metadata_entity_action_bc_v1.json

# Evaluation run (produces runs/{run_id}/)
python evaluate_model.py \
  --model-path artifacts/entity_action_bc_v1.keras \
  --policy-vocab artifacts/entity_action_bc_v1.policy_vocab.json \
  --num-battles 100 \
  --output-run-id entity_action_bc_v1_20260407_session3

# Creates: runs/entity_action_bc_v1_20260407_session3/run_metadata.json
#   (which references the training_metadata_entity_action_bc_v1.json)
```

---

## Part 5: Battle Reconstruction

Battle reconstruction means: given a battle log's events and initial state, rebuild the canonical state at any point in time.

### Reconstruction Algorithm

```
Given:
  - initial_state (turn 0)
  - events array
  - target_turn, target_step

Process:
  1. Load initial_state → current_state
  2. For each event in order:
       - Parse event (type, actor, target, fields)
       - Apply state mutation (BattleStateTracker semantics)
       - If event produces turn_end:
         - Increment turn_number
         - If step == 2, reset step to 1; else step = 2
       - If (turn == target_turn AND step == target_step):
           - Return current_state
  3. Return final state if target beyond battle end
```

### Implementation

The entity and flat families use `BattleStateTracker` (in `core/BattleStateTracker.py`) for canonical state management:

```python
from BattleStateTracker import BattleStateTracker

tracker = BattleStateTracker()
tracker.initialize(initial_state)

for event_dict in events:
    tracker.apply_event(event_dict)  # Mutates tracker.state
    if tracker.state["turn_index"] == target_turn:
        return tracker.state
```

### Reconstruction Integrity Checks

**Check 1: State Hash Consistency**
```python
# event_index.csv provides game_state_hash for each substep
# Verify: hash(reconstructed_state) == game_state_hash

import hashlib
import json

reconstructed = reconstruct_state_at(battle_log, turn, step)
canonical_hash = json.dumps(reconstructed, sort_keys=True)
actual_hash = hashlib.sha256(canonical_hash.encode()).hexdigest()
assert actual_hash == event_index_row["game_state_hash"]
```

**Check 2: Decision Point Validation**
```python
# For each decision_point in battle log:
# Verify that state_before matches reconstructed state at that turn/step

reconstructed_before = reconstruct_state_at(battle_log, dp["turn"], dp["step"])
assert states_equal(reconstructed_before, dp["state_before"])
```

**Check 3: Event Sequence Tokenization**
```python
# For each substep, reconstruct turn_events_v1 from the events
# and verify that encoding matches model's training targets

from core.TurnEventTokenizer import encode_turn_event_sequence

substep_events = [e for e in events if e_index_range_matches(e, turn, step)]
tokens = encode_turn_event_sequence(substep_events, sequence_vocab, max_len=32)
assert tokens == model_input["sequence"]  # From training/inference
```

---

## Part 6: Statistical Analysis

### Win Rate Analysis

**Basic Metric**
```python
win_rate = num_battles_won / num_battles_total
```

**Confidence Intervals** (Wilson score binomial proportion)
```python
# For n trials, k successes
# 95% CI uses z=1.96
ci_lower = (k + z²/2n - z√(k(n-k)/n + z²/4n²)) / (n + z²/n)
ci_upper = (k + z²/2n + z√(k(n-k)/n + z²/4n²)) / (n + z²/n)
```

**Head-to-Head Comparison**
```python
# McNEMAR test: paired comparison of model_A vs model_B
# across same set of battles

from scipy.stats import binom_test
n_ab = num_battles_A_won_B_lost
n_ba = num_battles_B_won_A_lost
p_value = binom_test(min(n_ab, n_ba), n_ab + n_ba, 0.5)
# p < 0.05 suggests significant difference
```

### Action Accuracy Metrics

**Move Accuracy**
```python
# Fraction of turns where chosen action was a legal move

move_turns = sum(1 for dp in decision_points if dp["legal_actions"][0].startswith("move:"))
correct_move_turns = sum(
    1 for dp in decision_points
    if dp["legal_actions"][0].startswith("move:")
    and dp["action_chosen"].startswith("move:")
)
move_accuracy = correct_move_turns / move_turns if move_turns > 0 else 0.0
```

**Switch Accuracy**
```python
# Fraction of forced-switch turns where model chose a valid switch

switch_turns = sum(1 for dp in decision_points if "switch:" in str(dp["legal_actions"]))
correct_switch_turns = sum(
    1 for dp in decision_points
    if "switch:" in str(dp["legal_actions"])
    and dp["action_chosen"].startswith("switch:")
)
switch_accuracy = correct_switch_turns / switch_turns if switch_turns > 0 else 0.0
```

### Tactical Error Classification

**Error Categories:**

1. **Repeated Switch Syndrome**
   ```python
   # Flag: model switches to same Pokémon multiple times in a row
   # within a short window (e.g., 5 turns)

   switches = [dp for dp in decision_points if "switch:" in dp["action_chosen"]]
   for i in range(len(switches) - 2):
       if (switches[i]["action_chosen"] == switches[i+1]["action_chosen"] and
           switches[i+1]["turn"] - switches[i]["turn"] <= 5):
           errors.append({
               "type": "repeated_switch",
               "turn": switches[i]["turn"],
               "species": switches[i]["action_chosen"]
           })
   ```

2. **Status Into Already-Statused Target**
   ```python
   # Flag: model uses status move (e.g., Toxic) against target
   # that already has a status condition

   for dp in decision_points:
       if "status" in dp["move_outcome"] or dp["move_outcome"] == "no_effect":
           if dp["state_before"]["opponent_active"]["status"] is not None:
               errors.append({
                   "type": "status_wasted",
                   "turn": dp["turn"],
                   "move": dp["action_chosen"]
               })
   ```

3. **Non-Super-Effective Move Usage**
   ```python
   # Flag: model repeatedly uses moves that don't hit super-effectively
   # against the opponent's Pokémon

   for dp in decision_points:
       if dp["action_chosen"].startswith("move:"):
           move = dp["action_chosen"][5:]  # Strip "move:" prefix
           target_species = dp["state_before"]["opponent_active"]["species"]
           if not is_super_effective(move, target_species):
               errors.append({
                   "type": "non_se_move",
                   "turn": dp["turn"],
                   "move": move,
                   "target": target_species
               })
   ```

4. **Wasted Setup**
   ```python
   # Flag: model uses setup move (Swords Dance, Dragon Dance, etc.)
   # against opponent with Taunt or when already maxed out

   for dp in decision_points:
       if dp["move_outcome"] == "setup_wasted":
           errors.append({
               "type": "wasted_setup",
               "turn": dp["turn"],
               "move": dp["action_chosen"]
           })
   ```

5. **Three Consecutive Event Failures**
   ```python
   # Flag: model makes 3+ consecutive turns with unexpected outcomes
   # (immunity, blocked by ability, etc.)

   failures = [dp for dp in decision_points if dp["move_outcome"] != "success"]
   for i in range(len(failures) - 2):
       if (failures[i+2]["turn"] - failures[i]["turn"] <= 3 and
           failures[i]["player"] == failures[i+1]["player"]):
           errors.append({
               "type": "consecutive_failures",
               "turn_range": (failures[i]["turn"], failures[i+2]["turn"]),
               "player": failures[i]["player"]
           })
   ```

### Turn Outcome Prediction Accuracy

For models trained with `--predict-turn-outcome`:

```python
# Fraction of model's predicted next-state that matches actual next-state

correct_predictions = sum(
    1 for dp in decision_points
    if np.argmax(dp["transition_logits"]) == encode_turn_outcome(dp["state_before"], dp["state_after"])
)
turn_outcome_accuracy = correct_predictions / len(decision_points)
```

### Sequence Auxiliary Accuracy

For models trained with `--predict-turn-sequence`:

```python
# Token-level accuracy on predicted turn events
# (See TurnEventTokenizer for token definitions)

from core.TurnEventTokenizer import encode_turn_event_sequence

total_tokens = 0
correct_tokens = 0

for battle_log in battle_logs:
    for turn in range(1, battle_log["duration_turns"] + 1):
        events = get_events_for_turn(battle_log, turn)
        predicted_tokens = model_output["sequence"][turn]  # shape: (max_seq_len,)
        actual_tokens = encode_turn_event_sequence(events, sequence_vocab, max_seq_len=32)

        # Count only non-PAD tokens (PAD_ID = 0)
        mask = actual_tokens > 0
        correct_tokens += np.sum(predicted_tokens[mask] == actual_tokens[mask])
        total_tokens += np.sum(mask)

sequence_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
```

---

## Part 7: Integration Points

### With Model Training

**Input**: Training examples → Battle logs (raw events)

```python
# In train_entity_action.py, train_policy.py:
examples = ingest_battles_to_examples(tracker, json_paths, ...)

# Each example carries:
example = {
    "state": {...},                    # Reconstructed via tracker
    "next_state": {...},
    "action_token": "move:tackle",
    "turn_events_v1": [{...}, {...}],  # Events for this turn
    "terminal_result": 1.0 or 0.0,
}
```

**Integration Point**: `BattleStateTracker` emits both state AND `turn_events_v1` in parallel.

### With Model Evaluation

**Output**: Evaluation run → Battle logs (with decision points and outcomes)

```python
# In evaluation script:
for battle in run_battles(model, opponent_model, num_battles=100):
    battle_log = {
        "metadata": {...},
        "initial_state": battle.initial_state,
        "events": battle.turn_events_v1,  # Raw events
        "decision_points": [
            {
                "turn": ...,
                "state_before": ...,
                "action_chosen": model.choose_action(state),
                "move_outcome": evaluate_action(state, action, next_state),
                "state_after": ...,
            }
            for each decision
        ]
    }
    save_battle_log(battle_log)
```

**Integration Point**: Model's action choice + actual state transition = decision point outcome.

### With Diagnostics

**Input**: Battle logs → Diagnostic reports

```
battle_logs/ → win_rate_summary.json
            → move_accuracy.csv
            → tactical_errors.json
            → state_reconstruction_failures.json
            → sequence_accuracy_per_turn.csv
```

**Tools to Build**:
1. `analyze_win_rate.py`: Compute win rate, CI, head-to-head
2. `analyze_action_accuracy.py`: Move/switch accuracy by player, battle phase
3. `analyze_tactical_errors.py`: Classify errors, count by type
4. `validate_reconstruction.py`: Hash check, state consistency, event tokenization

---

## Part 8: Data Quality & Validation

### Invariants

1. **Event Sequence Completeness**
   - Every turn must have a `turn_end` event
   - No events after the final `turn_end` in a complete battle
   - Event ordering is monotonic (turn_numbers increase or stay same within turn)

2. **Decision Point Mapping**
   - Every decision_point must correspond to an entry in event_index.csv
   - State before/after must be consistent with events between boundaries
   - No decision_points with illegal actions

3. **State Consistency**
   - HP fractions always in [0.0, 1.0]
   - Stat stages always in [-6, +6]
   - No Pokémon in two places simultaneously
   - Fainted Pokémon never appear as active

4. **Model Integration**
   - All action tokens in decision_points must be in policy vocab
   - Model logits must sum to ~1.0 (or be int64 token IDs for sequence head)
   - Embedding indices must be within vocab size bounds

### Validation Checklist

When adding a new battle log to a run:

- [ ] `metadata.battle_id` is unique within run
- [ ] `initial_state` passes `BattleStateTracker.validate(initial_state)`
- [ ] All events have required fields for their type
- [ ] Events array contains at least 1 event (non-empty battle)
- [ ] decision_points count matches turn count
- [ ] decision_points are ordered by turn, step, player
- [ ] All decision_point.legal_actions are in policy vocab
- [ ] decision_point.move_outcome is in allowed set
- [ ] state_hash matches `hash(reconstructed_state)` for spot checks

---

## Part 9: Future Extensions

### Planned Additions

1. **Belief State Tracking**: Track player's inferred belief about opponent's team/HP
2. **Effectiveness Matrices**: Pre-computed tables for super-effective/resisted/immune checks
3. **Action Legality Verification**: Flag impossible action sequences (e.g., switching when can't)
4. **Rollout Traces**: Record counterfactual states (what if model chose differently)
5. **Comparative Eval**: Side-by-side traces of model vs. baseline

### Extension Template

To add a new diagnostic:

```python
# analyze_my_metric.py
def analyze_my_metric(battle_logs, run_metadata):
    results = {
        "metric_name": "my_metric",
        "run_id": run_metadata["run_id"],
        "values_by_battle": {},
        "aggregate": {}
    }

    for battle_log in battle_logs:
        value = compute_metric(battle_log)
        results["values_by_battle"][battle_log["metadata"]["battle_id"]] = value

    results["aggregate"] = {
        "mean": np.mean(list(results["values_by_battle"].values())),
        "std": np.std(list(results["values_by_battle"].values())),
        "min": min(results["values_by_battle"].values()),
        "max": max(results["values_by_battle"].values()),
    }

    return results

# In run evaluation harness:
metrics = analyze_my_metric(battle_logs, run_metadata)
save_json(f"runs/{run_id}/diagnostics/my_metric.json", metrics)
```

---

## References

### Core Files

- `core/TurnEventV1.py` — Event schema and tokenization
- `core/TurnEventTokenizer.py` — Event sequence encoding to tokens
- `core/BattleStateTracker.py` — Canonical state management
- `train_entity_action.py`, `train_policy.py` — Training entry points
- `replay_value_trace.py` — Example diagnostic tool

### Related Docs

- `CLAUDE_ENTITY_SEQUENCE_AUX_HANDOFF.md` — Entity auxiliary training spec
- `docs/ENTITY_ARCHITECTURE.md` (if exists) — Entity model structure
- Pokemon Showdown Protocol — Battle event semantics

---

## Quick Start: Analysis Pipeline

```bash
# 1. Run evaluation (produces battle logs)
python evaluate_model.py \
  --model-path artifacts/entity_action_bc_v1.keras \
  --policy-vocab artifacts/entity_action_bc_v1.policy_vocab.json \
  --num-battles 100 \
  --output-run-id entity_action_bc_v1_20260407_session3

# 2. Generate event index
python tools/generate_event_index.py \
  runs/entity_action_bc_v1_20260407_session3/

# 3. Compute diagnostics
python tools/analyze_win_rate.py runs/entity_action_bc_v1_20260407_session3/
python tools/analyze_action_accuracy.py runs/entity_action_bc_v1_20260407_session3/
python tools/analyze_tactical_errors.py runs/entity_action_bc_v1_20260407_session3/

# 4. Validate reconstruction
python tools/validate_reconstruction.py runs/entity_action_bc_v1_20260407_session3/

# 5. View summary
cat runs/entity_action_bc_v1_20260407_session3/run_metadata.json | python -m json.tool
```

---

**Last Updated**: 2026-04-07
**Maintained By**: Entity sequence auxiliary training team
