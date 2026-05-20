# Action Space Mismatch — Research Report

**Date:** 2026-05-06
**Status:** Findings confirmed. Two sprints proposed.

---

## Summary

The `entity_action_bc` model family (v1, beta2) trains a 357-class softmax over an action vocabulary where at most **9 classes are ever legal at any given turn**. The vocabulary itself covers only **51% of Gen 9's legal move pool**. These two structural mismatches waste gradient capacity during training and cap the model's reachable ceiling before any architectural or data improvements are applied.

The `entity_action_v2` family already resolves both issues via `legal_candidates` scoring. The path forward is to either adopt v2's architecture for the next bc training run or add a legal-action mask to v1's training loss.

---

## Vocabulary Structure (confirmed)

| Component | Count | Notes |
|---|---|---|
| Named moves in policy vocab | 350 | `move:movename` format, indices 7–356 |
| Switch slots | 6 | `switch:1` through `switch:6`, indices 1–6 |
| `<UNK>` token | 1 | Index 0; OOV fallback |
| **Total policy classes** | **357** | |
| Gen 9 legal moves (simulator) | 685 | Moves where `isNonstandard` is unset |
| Total simulator move DB | 954 | Includes Past (207), Gigantamax (33), LGPE (13), etc. |
| **Missing from training vocab** | **335** | 49% of Gen 9 legal moves are OOV |

Legal actions per turn:

- Move-turn: ≤4 move slots + ≤5 switch targets = **max 9**
- Forced switch: ≤5 switch targets = **max 5**
- `entity_action_v2` hard-codes `MAX_LEGAL_ACTIONS = 10` as a safe upper bound

---

## Root Causes

### Root Cause 1 — Unmasked full-vocab training loss

**File:** `core/EntityModelV1.py:765`

```python
keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

The policy loss is computed over all 357 classes on every training step. No `legal_action_mask` is passed to the loss function. The model receives a single sparse integer target (the chosen action's token ID) and must learn — purely from data distribution — to suppress logits for ~348 classes that are never the target.

This works only if:
1. Every illegal class has zero co-occurrence with every game state across all training examples, and
2. The optimizer can efficiently push down 348 denominator terms per step without compromising gradient flow to the 9 that matter.

Neither condition holds perfectly. The result is diffuse gradient that slows convergence on legally-relevant logits.

**Comparison:** `entity_action_v2` (`core/EntityModelV2.py:155-163`) applies `MaskCandidateLogits` — a hard `-1e9` mask **before softmax** on a candidate set of ≤10 actions. The softmax denominator contains only legal classes. Training gradient is entirely concentrated on legal actions.

### Root Cause 2 — Corpus-bound vocabulary covers only 51% of Gen 9

The policy vocabulary was built from ~5,000 random battle logs. Moves that did not appear in those battles are absent. At inference, any encounter with a missing move maps to `<UNK>` (index 0) — a class the model was never trained to choose.

This means the model cannot distinguish between OOV moves; they all collapse to the same token. In random battles, the 335 missing moves include many that appear regularly in competitive Gen 9 play but were simply absent from the training corpus's random-format distribution.

---

## Inference Masking — What Actually Happens

### v1 (entity_action_bc)

1. Model runs full forward pass → outputs 357 raw logits
2. `softmax(logits)` fires over all 357 classes — probability mass distributes across illegal classes
3. `pick_best_action()` (`core/ActionSelection.py:141–186`) iterates **only** over legal move tokens and switch tokens, picking the highest-probability legal action
4. Illegal logits are never read — but they already absorbed probability mass from the softmax denominator

**Effect:** legal actions receive diluted probability. A move that should receive 40% probability might receive 15% because 348 illegal classes collectively hold the rest. The argmax is still correct (best legal action wins), but the probability calibration is poor, and any sampling or temperature-based selection degrades.

### v2 (entity_action_v2)

1. Legal candidates encoded as `candidate_mask` (1.0 for legal, 0.0 for padding), max 10
2. `MaskCandidateLogits` layer applies `-1e9` to masked slots **before** softmax
3. Softmax fires over ≤10 classes — all probability mass stays within legal actions
4. `select_best_v2_candidate()` (`serve_entity_model_benchmark.py:479–485`) selects from candidate tokens

---

## Conflicts Resolved

| Prior claim | Status | Evidence |
|---|---|---|
| "illegal actions masked to -inf at v1 inference" | **OVERTURNED** | `ActionSelection.py:141–186` — iteration only, no -inf masking |
| "351 moves in training vocab" | **CORRECTED** | 350 named moves + 1 `<UNK>`; `<UNK>` is not a move |
| "training loss is masked to legal actions" | **OVERTURNED** | `EntityModelV1.py:765` — unmasked `SparseCategoricalCrossentropy` |

---

## Impact × Cost Matrix

| Item | Impact | Cost | Classification |
|---|---|---|---|
| Legal-masked cross-entropy in v1 training | High — directly concentrates gradient on legal actions | Medium — add `legal_action_mask` tensor to batch + loss | **Plan sprint** |
| Migrate next bc run to v2 legal_candidates architecture | High — eliminates root cause 1 entirely | Medium — training pipeline already exists in v2 | **Plan sprint** |
| Audit 335 missing moves by frequency in randbat | Medium — scopes how often OOV is actually hit | Low — grep vocab vs simulator move list | **Fix now** |
| Hard-mask v1 inference pre-softmax | Low — inference is functionally correct; this only improves calibration | Low — `ActionSelection.py` edit | **Opportunistic** |
| Expand vocabulary to cover all 685 Gen 9 moves | Low in isolation — OOV moves still need training examples | High — requires retraining from scratch with expanded corpus | **Plan sprint (deferred)** |

---

## Action List

### Immediate

1. **[Fix now]** Audit OOV move frequency:
   ```bash
   # Compare policy vocab move names against Gen 9 simulator move list
   python3 -c "
   import json
   vocab = json.load(open('artifacts/entity_action_bc_v1_beta2_20260501_1550/entity_action_bc_v1_beta2_20260501_1550.policy_vocab.json'))
   vocab_moves = {k.replace('move:','') for k in vocab if k.startswith('move:')}
   print(f'Vocab moves: {len(vocab_moves)}')
   # Cross-reference against sim data to find high-frequency missing moves
   "
   ```
   Then compare against `pokemon-showdown-model-feature/data/moves.ts` to identify the highest-frequency absent moves.

### Sprint A — Legal-masked training loss (v1 path)

2. **[Plan sprint]** Add `legal_action_mask` tensor (shape: `[batch, num_policy_classes]`, dtype float32, 1.0 for legal, 0.0 for illegal) to the tensorization output in `core/EntityTensorization.py`.

3. **[Plan sprint]** Implement masked cross-entropy in `core/EntityModelV1.py`:
   - Replace `SparseCategoricalCrossentropy` with a custom loss that sets illegal class logits to `-1e9` before the cross-entropy computation using the per-example mask.
   - Requires batch to carry `legal_action_mask` as a third element alongside `(x, y_policy)`.

### Sprint B — Migrate to v2 architecture (recommended path)

4. **[Plan sprint]** The v2 `legal_candidates` architecture already solves both root causes:
   - Softmax over ≤10 legal actions → no wasted gradient
   - Candidate tokens are the actual named moves the Pokemon has, not a corpus-constrained vocab
   - A move not in the training corpus still gets scored if it appears as a legal candidate at inference

   Next `entity_action_bc` training run should use `entity_action_v2`'s tensorization and model architecture with the bc training regime (offline imitation learning, return-weighted cross-entropy).

### Opportunistic

5. **[Opportunistic]** Pre-softmax hard masking at v1 inference (`core/ActionSelection.py`): build a `-1e9` logit mask from `legal_moves` + `legal_switches` before `softmax()`, improving probability calibration for any analysis that uses logit values (not just argmax).

---

## Open Questions

1. **How often does OOV actually trigger in practice?** The `<UNK>` token is present but the league benchmarks show 0 fallback move choices. This needs verification — are OOV moves hitting `<UNK>` and coincidentally not being the best legal option, or are they not being encountered at all?

2. **Does v2's legal_candidates approach require the move to appear in training data?** The candidate token is looked up in the entity vocab at inference, not the policy vocab. If the entity vocab also has a 352-move ceiling, OOV moves would still collapse. Needs audit.

3. **Is the 30% win rate of 1550 attributable to the switch-overuse problem (joint action space + +0.2 bias) or to gradient dilution?** The 0547 model (move-only, `action_space: move_only`) reached 1503 ELO without any switch logic. Separating these effects requires a controlled ablation.

---

## Relationship to 0547 Insight

The `entity_action_bc_v1_beta2_20260501_0547` model is labeled `joint-policy` in the league but has `action_space: move_only` and `include_switches: False` in its training metadata. It makes zero voluntary switches yet achieves the highest ELO of any entity model (1503). This demonstrates that the beta2 auxiliary heads (threat awareness, type effectiveness) benefit move selection independently of the joint action space. The joint action space + switch logit bias (+0.2) in 1550 appears to be harmful, amplifying a weak switch signal into overuse (11.87 switches/game).

The two problems interact: gradient dilution (root cause 1) may be worse for the switch head than the move head, producing poorly-calibrated switch logits that the +0.2 bias then amplifies.
