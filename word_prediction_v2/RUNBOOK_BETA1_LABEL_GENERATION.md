# Runbook: β-1 Offline Label Generation

**Purpose:** Generate threat-awareness and type-effectiveness labels for entity model auxiliary-head training.

**Generator script:** `word_prediction_model/scripts/generate_beta_labels.py`
**Output:** `artifacts/beta_labels_v1/`

---

## Overview

β-1 executes the `word_policy_v1` symbolic decoder offline on the entity training corpus (5000 Kaggle battles) to produce:

1. **Threat scores** — replicating `policy_adapter.py` threat evaluation (per-opponent-active)
2. **Type-effectiveness multipliers** — lookup from `pokemon_type_utils` (per-legal-move)

These labels supervise two new auxiliary heads in entity model β-2 training.

---

## Prerequisites

- Python 3.12
- Dependencies from `Pokemon-Showdown-Agents-Go-Brrrr/` and `word_prediction_model/`
- Kaggle battle dataset: `/Users/AI-CCORE/.cache/kagglehub/datasets/thephilliplin/pokemon-showdown-battles-gen9-randbats/versions/1/` (auto-cached by Kagglehub on first run)

## One-Line Command (5000 battles, no switches)

```bash
cd /Users/AI-CCORE/alter-programming && \
python3.12 word_prediction_model/scripts/generate_beta_labels.py \
  --max-battles 5000 \
  --out-dir artifacts/beta_labels_v1 \
  --include-switches false
```

**Expected output:**
- `artifacts/beta_labels_v1/labels.jsonl` — 177,789 lines (one per training example)
- `artifacts/beta_labels_v1/summary.json` — timing and distribution stats
- **Wall time:** ~50 s (46.4 s load + 4.2 s generate)

---

## Label Format

Each line in `labels.jsonl` is a JSON record:

```json
{
  "battle_id": "gen9randombattle-1234567890",
  "turn_number": 3,
  "player": "p1",
  "my_species": "pikachu",
  "opp_species": "charizard",
  "my_hp_frac": 0.85,
  "n_opp_observed_moves": 3,
  "threat_score": 1.2,
  "action_kind": "move",
  "action_type_eff": 2.0
}
```

### Field Definitions

| Field | Type | Notes |
|-------|------|-------|
| `battle_id` | str | Kaggle replay identifier |
| `turn_number` | int | 1-indexed turn in the replay |
| `player` | str | `"p1"` or `"p2"` |
| `my_species` | str | Active Pokémon species (actor's perspective) |
| `opp_species` | str | Opponent's active Pokémon |
| `my_hp_frac` | float | Actor's active HP as [0, 1]; None if unknown |
| `n_opp_observed_moves` | int | Count of opponent's revealed moves |
| `threat_score` | float | Symbolic decoder's threat evaluation (0 to ~3) |
| `action_kind` | str | `"move"` or `"switch"`; others skipped by default |
| `action_type_eff` | float or null | Type-effectiveness multiplier of chosen move vs. opp types; null for switches |

### Threat Score Computation

Replicates `word_prediction_model/policy_adapter.py:602-623`:

```python
threat_score = max over opp_moves [
    min(opp_move.base_power / 70.0, 2.0)
    + type_effectiveness_bonus(opp_move_type, my_types)
    + priority_bonus(opp_move.priority, my_hp)
]
```

- **Range:** typically 0–3 (clamped at 2.0 power term + ~1.5 type bonus)
- **Interpretation:** higher = more threatening opponent matchup

### Type-Effectiveness Values

Discrete set: `{0.25, 0.5, 1.0, 2.0, 4.0}` (from Pokémon Gen 9 type chart)

- **0.25** — Not very effective (resists, 1 layer)
- **0.5** — Not very effective (resists, 2 layers)
- **1.0** — Neutral effectiveness
- **2.0** — Super-effective (weak to, 1 layer)
- **4.0** — Super-effective (weak to, 2 layers)

---

## Validation

**Summary statistics** (from 5000-battle run):

```json
{
  "examples_loaded": 178271,
  "labels_written": 177789,
  "examples_skipped": 482,
  "load_time_s": 46.4,
  "gen_time_s": 4.2,
  "avg_threat_score": 1.014,
  "action_kind_distribution": {
    "move": 177789,
    "switch": 0
  },
  "action_type_eff_distribution": {
    "1.00": 104297,
    "2.00": 39551,
    "0.50": 22231,
    "0.00": 4025,
    "4.00": 3072,
    "0.25": 1998
  }
}
```

**Sanity checks:**
- ✅ Neutral (1.00) dominates at 58.7% — matches random Pokémon matchup expectation
- ✅ Super-effective (2.00) is second at 22.2% — expected for random-team diversity
- ✅ Avg threat score 1.014 is near neutral — neither team heavily outmatched
- ✅ Skip count 482 is <0.3% — no meaningful data loss

---

## Variants

### With Switches Included

```bash
python3.12 word_prediction_model/scripts/generate_beta_labels.py \
  --max-battles 5000 \
  --out-dir artifacts/beta_labels_v2_with_switches \
  --include-switches true
```

**Difference:** 185,546 labels (vs 177,789) to match the original entity training corpus that used `include_switches=true`.

### Subset Runs (Debugging)

```bash
python3.12 word_prediction_model/scripts/generate_beta_labels.py \
  --max-battles 10 \
  --out-dir /tmp/beta_test
```

Generates ~355 labels in ~0.6 s for fast iteration.

---

## Regeneration and Updates

To regenerate β-1 labels from scratch:

```bash
rm -rf artifacts/beta_labels_v1
python3.12 word_prediction_model/scripts/generate_beta_labels.py \
  --max-battles 5000 \
  --out-dir artifacts/beta_labels_v1
```

**No code changes needed** — the script reads the Kaggle dataset (auto-cached), applies fixed threat/type-eff logic.

To extend to a different dataset:

1. Update `--kaggle-dir` to point to new replay source
2. Re-run with new `--max-battles` count
3. Check summary stats for sanity

---

## Integration with β-2 Training

In `train_entity_action.py` (β-2 phase):

```python
# Load labels
labels_by_key = load_beta_labels(args.beta_labels_jsonl)

# Attach to training examples
attach_auxiliary_labels_to_examples(examples, labels_by_key)

# Vectorize for loss
threat_targets, type_eff_targets = vectorize_auxiliary_labels(
    examples,
    policy_vocab,
    legal_moves_mask
)

# Wire into training loss
model.compile(
    loss={
        "policy": "categorical_crossentropy",
        "threat": "mse",
        "type_eff": "categorical_crossentropy",
    },
    loss_weights={
        "policy": 1.0,
        "threat": args.threat_weight,
        "type_eff": args.type_eff_weight,
    }
)
```

Examples without matching labels (e.g., if using different battle corpus) default to:
- `threat_score = 0.0`
- `action_type_eff = NaN` (masked in loss)

---

## Troubleshooting

### "Kaggle dataset not found"

The script expects the dataset at:
```
/Users/AI-CCORE/.cache/kagglehub/datasets/thephilliplin/pokemon-showdown-battles-gen9-randbats/versions/1/
```

If missing:
```bash
pip install kagglehub
kagglehub dataset download thephilliplin/pokemon-showdown-battles-gen9-randbats
```

### "NotImplementedError: move_metadata(...)"

Move metadata lookups use `word_prediction_model/pokemon_move_metadata.py`. Ensure it's up-to-date with the current Pokémon move database (Gen 9).

### "Threat score all zeros"

If all threat scores compute to 0.0:
- Check that opponent observed moves are populated in the battle tracker
- Verify move metadata is being loaded (see above)
- Run a 10-battle subset with `--max-battles 10` and inspect raw JSON

### "Type-effectiveness has unexpected values"

Verify the type chart in `pokemon_type_utils.type_effectiveness()` matches Gen 9. Run a smoke test:

```python
from word_prediction_model.pokemon_type_utils import type_effectiveness
print(type_effectiveness("fire", ["water"]))  # should be 0.5
print(type_effectiveness("fire", ["grass"]))  # should be 2.0
```

---

## Paper Integration

Paper sections referencing β-1:

- **§3.3 Proposed β:** "Offline label generation (β-1) was executed..." — links to this runbook
- **§4.2 Empirical Anchors → β-1 Results:** Type-effectiveness distribution table, avg threat_score
- **§5.2 Limitations:** "β-1 is executed; β-2 is pending"

---

## Performance Notes

- **Load time:** 46.4 s for 5000 battles (~9.3 ms per replay)
- **Generation time:** 4.2 s for 177,789 examples (~24 µs per label)
- **Bottleneck:** Battle state tracker reconstruction (BattleStateTracker + ingest_battles_to_examples)
- **Scaling:** For 10,000 battles, expect ~100 s total

To speed up: use `--max-battles 1000` for faster iteration; only run full 5000 when committing to training.

