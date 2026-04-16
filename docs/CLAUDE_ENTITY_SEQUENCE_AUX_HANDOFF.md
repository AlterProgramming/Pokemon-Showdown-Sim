# Claude Handoff: Entity Sequence Auxiliary Plan

## Goal

Add the same within-turn event-sequence auxiliary training signal to the entity family so we can run a clean side-by-side comparison against the flat/vector family.

This branch is for the handoff only. Do not implement the browser bridge here first. The primary objective is:

- `entity_action_bc_v1` can train with the same `turn_events_v1` sequence target
- the entity trainer saves the same kind of sequence artifacts and metadata as the flat trainer
- the entity family remains runtime-compatible and comparison-friendly

The secondary objective is only to keep the entity family stream-ready:

- do not use the sequence head at live inference time yet
- do not wire browser fragment accumulation in this branch
- do preserve a clean path from canonical `battle_state` to entity view and future event-aware diagnostics

## Scope

In scope:

- entity training parity with the flat sequence auxiliary work
- entity metadata / artifact parity
- entity tests for tensorization, model construction, and trainer metadata
- preserving the current entity runtime contract

Out of scope:

- browser/model-bridge reconstruction work
- live reranking with the sequence head
- belief-state changes
- `entity_invariance_aux_v1` changes unless needed for shared utility reuse
- full graph/message-passing redesign

## Read These First

Entity-side surfaces:

- [train_entity_action.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/train_entity_action.py)
- [core/EntityTensorization.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/EntityTensorization.py)
- [core/EntityModelV1.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/EntityModelV1.py)
- [core/EntityActionV1.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/EntityActionV1.py)

Flat-side reference implementation to mirror:

- [train_policy.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/train_policy.py)
- [core/TurnEventTokenizer.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/TurnEventTokenizer.py)
- [core/TurnEventV1.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/TurnEventV1.py)

Tests to extend:

- [tests/test_entity_tensorization.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_entity_tensorization.py)
- [tests/test_entity_model_v1.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_entity_model_v1.py)
- [tests/test_train_entity_action.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_train_entity_action.py)
- flat references:
  - [tests/test_turn_event_tokenizer.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_turn_event_tokenizer.py)
  - [tests/test_train_policy.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_train_policy.py)

Useful existing entity artifacts to inspect before choosing a new training release:

- [artifacts/training_metadata_entity_action_bc_v1_20260327_run2.json](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/artifacts/training_metadata_entity_action_bc_v1_20260327_run2.json)
- [artifacts/training_metadata_entity_action_bc_v1_20260403_switchbias_run2.json](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/artifacts/training_metadata_entity_action_bc_v1_20260403_switchbias_run2.json)

## Architecture Requirements

Keep the same high-level pattern as the flat sequence work:

- one policy label per pre-decision state
- `turn_events_v1` stays auxiliary supervision
- sequence target is structured symbolic tokens, not free-form text
- sequence head is action-conditioned on `my_action` and `opp_action`
- runtime still uses only the policy head

For the entity family specifically:

- do not replace the entity contract
- do not flatten the entity tensors back into a vector path
- do not make the sequence head a separate model family
- do make the shared entity trunk carry the new auxiliary pressure

## Implementation Order

### 1. Reuse the Existing Sequence Contract

Do not invent a second event-token format.

Use the same tokenizer utilities and target definition as the flat trainer:

- `--predict-turn-sequence`
- `--sequence-weight`
- `--sequence-hidden-dim`
- `--max-seq-len`
- `sequence_target_definition`
- `sequence_vocab_path`

Match the flat metadata terminology unless there is a strong reason not to.

### 2. Extend Entity Tensorization

In [core/EntityTensorization.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/EntityTensorization.py):

- extend `build_entity_training_bundle(...)` so it can optionally build and carry a sequence vocab
- extend `vectorize_entity_multitask_dataset(...)` so it can optionally emit `targets["sequence"]`
- keep `my_action` and `opp_action` available whenever the sequence head is enabled, even if the transition head is off
- reuse the same `turn_events_v1` source already produced by the tracker
- keep padding/truncation behavior aligned with the flat trainer

Important:

- the entity sequence target should reuse the existing turn-event tokenization path rather than re-encoding directly inside entity tensorization
- if the sequence head is enabled without the transition head, still persist `action_context_vocab` because the sequence head is action-conditioned

### 3. Extend the Entity Model

In [core/EntityModelV1.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/core/EntityModelV1.py):

- add optional sequence-head inputs/outputs to the multitask training model
- keep the policy-only inference artifact unchanged
- keep the optional policy+value artifact unchanged unless additional compile wiring is required

Recommended pattern:

- shared entity trunk
- policy head
- existing optional transition head
- existing optional value head
- new LSTM sequence head conditioned on:
  - shared state trunk
  - embedded `my_action`
  - embedded `opp_action`

Mirror the flat-side fixes:

- mask `PAD` tokens in sequence loss
- mask `PAD` tokens in sequence accuracy
- do not describe the head as autoregressive unless it truly consumes shifted previous tokens

This branch should prefer parity with the current flat implementation over architectural novelty.

### 4. Extend the Entity Trainer and Artifacts

In [train_entity_action.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/train_entity_action.py):

- add CLI flags mirroring the flat trainer:
  - `--predict-turn-sequence`
  - `--sequence-weight`
  - `--sequence-hidden-dim`
  - `--max-seq-len`
- reserve artifact paths for the sequence vocab
- save the sequence vocab artifact
- record sequence metadata fields
- include the sequence objective in `objective_set`
- update the training regime string only if needed for clarity, but do not rename the family

Required metadata fields:

- `sequence_target_definition`
- `sequence_vocab_path`
- `sequence_vocab_size`
- `sequence_hidden_dim`
- `sequence_weight`
- `max_seq_len`

If the sequence head is on and transition is off, still record:

- `action_context_vocab_path`
- `num_action_context_classes`

### 5. Keep Runtime Stable

Do not wire the new sequence head into live serving.

For this branch:

- training artifact may contain the richer multitask model
- policy serving artifact remains the same entity policy artifact
- optional policy+value artifact remains the same runtime-facing auxiliary artifact

The point is to compare training pressure, not to change live decision logic yet.

### 6. Leave a Stream-Ready Note, Not a Full Stream Refactor

The user ultimately wants entity models to observe the game through the same reconstructed canonical state.

Do not implement the browser-side fragment accumulator here.

Do:

- keep the entity family contract centered on canonical `battle_state` / tracker state
- add a small code comment or metadata note where helpful so future bridge work can project the canonical state into the entity view without ambiguity

## Tests To Add Or Update

### Tensorization tests

Update [tests/test_entity_tensorization.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_entity_tensorization.py):

- bundle includes sequence vocab when requested
- vectorization emits `targets["sequence"]` with stable padded length
- action-context tensors exist when sequence is enabled without transition

### Model tests

Update [tests/test_entity_model_v1.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_entity_model_v1.py):

- policy+sequence model fits one batch
- policy+transition+value+sequence model fits one batch
- sequence metrics are present and use masked padding

### Trainer metadata tests

Update [tests/test_train_entity_action.py](C:/Users/jeanj/Documents/School%20-%20Research/CSCI%208590%20Introduction%20to%20Deep%20Learning/Pokemon%20Showdown%20Agent/tests/test_train_entity_action.py):

- metadata records sequence artifact path and task definition
- metadata records action-context vocab path when sequence is enabled

### Optional comparison support

If cheap, add a small test that the entity trainer can build sequence-enabled artifacts without changing the policy-serving artifact contract.

## Acceptance Criteria

This work is done when:

- `entity_action_bc_v1` can train with `--predict-turn-sequence`
- entity tensorization emits a valid sequence target from `turn_events_v1`
- entity metadata and artifacts persist the sequence vocab and task definition
- the entity multitask model masks padded tokens in sequence loss/metrics
- existing entity policy serving artifacts remain valid
- tests cover tensorization, model fit, and metadata

## Non-Goals For This Branch

Do not:

- add browser fragment buffering
- add runtime candidate reranking from the sequence head
- fold this into `entity_invariance_aux_v1`
- change the entity action contract from its current vocab-scoring baseline

## Suggested Follow-Up After This Branch

Once the entity sequence auxiliary run exists, the next comparison step should be:

- train one new entity release against the strongest current entity baseline
- run the same style of replay diagnostics that were added on the flat side
- compare not just win rate, but tactical error rates:
  - repeated switch patterns
  - status-into-already-statused targets
  - not-super-effective move usage
  - same-move spam into the same matchup

That comparison will answer the real question:

- does entity structure make the turn-event auxiliary signal materially more useful than it was in the flat family?
