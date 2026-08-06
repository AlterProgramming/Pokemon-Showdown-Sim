# Model1 Network of Theseus Elman Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reproducible Network of Theseus conversion from the frozen Model1 flat policy into a three-layer Elman recurrent policy and evaluate it through the Pokémon simulator.

**Architecture:** The model repository will train a recurrent sequence policy from Model1's 582-dimensional public state and 357-class joint action vocabulary. Progressive CKA stages will replace dense guide layers with recurrent target layers, followed by supervised fine-tuning. The simulator will send an explicit bounded state history, while the server will flatten and normalize it for the sequence serving artifact.

**Tech Stack:** Python 3.12, TensorFlow/Keras, NumPy, pytest/unittest, TypeScript, Pokémon Showdown simulator, existing Flask model server and model registry.

## Global Constraints

- Model1 remains immutable and benchmarkable.
- The word-policy and entity-centric model families are out of scope.
- Public-information state only; no hidden game mechanics or opponent state may be added.
- New artifacts must have a distinct release ID, parent release ID, family ID, and metadata.
- Battle-level train/validation separation is required.
- The existing Model1 move-only simulator profile remains unchanged.
- Tests must be written and observed failing before production implementation.

### Task 1: Pure NoT contracts and sequence utilities

**Files:**
- Create: `tests/test_not_model1_elman.py`
- Create: `core/NoTModel1Elman.py`

**Interfaces:**
- `progressive_replacement_schedule(depth: int) -> tuple[tuple[int, ...], ...]`
- `pad_state_history(history, sequence_length: int, feature_dim: int) -> tuple[np.ndarray, np.ndarray]`
- `linear_cka(left: np.ndarray, right: np.ndarray) -> float`
- `cka_dissimilarity(left: np.ndarray, right: np.ndarray) -> float`
- `build_elman_policy_model(...)` and `build_serving_model(...)` for Keras callers.

- [x] Write failing tests for progressive prefixes, deterministic repeat-first padding, CKA identity, and invalid history shapes.
- [x] Run the focused tests and confirm the absent-module red state before implementation.
- [x] Implement the pure utilities and the Keras model builders.
- [x] Run the focused tests again and confirm they pass.

### Task 2: Model1 sequence dataset and staged alignment trainer

**Files:**
- Create: `tests/test_train_not_model1_elman.py`
- Create: `train_not_model1_elman.py`
- Modify: `core/NoTModel1Elman.py`

**Interfaces:**
- `build_sequence_dataset(examples, action_vocab, sequence_length, feature_dim=582) -> dict[str, np.ndarray]`
- `build_guide_activation_model(guide_model, sequence_length, depth=3) -> keras.Model`
- `train_alignment_stage(...) -> dict[str, float]`
- `train_target_policy(...) -> dict[str, float]`

- [x] Write failing tests for battle-boundary grouping, preserved action labels, and target metadata defaults.
- [x] Run the focused tests and observe the missing trainer behavior.
- [x] Implement sequence construction, frozen-guide activation extraction, progressive CKA optimization, and final masked policy fine-tuning.
- [x] Add CLI options for guide path, data paths, sequence length, stage epochs, fine-tune epochs, batch size, and output directory.
- [x] Save the serving artifact, training artifact, copied action vocabulary, stage metrics, and recipe metadata.
- [x] Run unit tests and a synthetic/real-data bounded fit.

### Task 3: Model registry and server sequence request support

**Files:**
- Create: `tests/test_not_model_server_sequence.py`
- Modify: `core/ModelRegistry.py`
- Modify: `flask_api_multi.py`
- Modify: `core/ModelWorkers.py` only if the serving wrapper requires a worker contract change.

**Interfaces:**
- Metadata fields: `sequence_model`, `sequence_length`, `base_feature_dim`, `sequence_padding`.
- Request field: `state_history: list[list[float]]`.
- Existing vector requests continue to use `state_vector`.

- [x] Write failing tests for repeat-first padding, 678-to-582 normalization per history element, and legacy request routing.
- [x] Run them and observe failure before changing the server.
- [x] Implement sequence request preparation and flattening for the serving artifact.
- [x] Keep legacy stateless requests unchanged.
- [x] Run focused server and registry tests.

### Task 4: Simulator history transport and target profile

**Files:**
- Create or modify: `sim/tools/rl-model-profiles.ts`
- Modify: `sim/tools/rl-agent.ts`
- Modify: `test/sim/tools/rl-model-profiles.js`
- Create: `test/sim/tools/rl-agent-history.js`

- [x] Write failing tests for the recurrent profile and bounded history payload.
- [x] Run the focused Node tests and observe failure.
- [x] Add the explicit target profile and per-agent history buffer.
- [x] Append current state vectors in request order, cap them at the configured history length, and include `state_history` only for the recurrent profile.
- [x] Preserve Model1's existing move-only alias and all word-policy behavior.
- [x] Run the TypeScript build and focused simulator tests.

### Task 5: Verification and experiment handoff

**Files:**
- Modify: `docs/architecture/KDD_model1_not_elman_20260806.md`
- Add: experiment command and results under the generated artifact directory.

- [x] Run pure utility, trainer, registry, server, and simulator tests.
- [x] Run an end-to-end server request against a saved recurrent artifact.
- [x] Run a bounded real-data training smoke test; the local TensorFlow environment is usable.
- [x] Run a one-game simulator smoke benchmark against the converted target.
- [x] Record actual outcomes, deviations, and changed files in the KDD.
