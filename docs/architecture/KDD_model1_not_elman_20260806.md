# KDD: Model1 to Elman Recurrent Network of Theseus

Date: 2026-08-06

## Part 1: Before State

The current starter release is `model1`, registered from `artifacts/training_metadata_1.json` and `artifacts/model_1.keras`. Its documented public input is a 582-dimensional flat state vector. Its action vocabulary is joint move/switch with 357 classes. The training metadata records a three-layer, width-256, ReLU dense policy with dropout 0.1.

The model registry resolves `model1` as the default vector model and preserves the original artifact paths. The simulator sends a current state vector and legal action candidates to `/predict`; the model server validates the vector, runs a stateless worker inference, and chooses the highest-scoring legal move or switch.

The simulator's historical `model1` profile suppresses voluntary switches, while the artifact itself remains joint-action. This profile behavior must remain unchanged for historical benchmarks.

## Part 2: Problem Statement

We need one reproducible architecture-conversion experiment that tests whether the function learned by Model1 can be carried into a recurrent inference architecture in the Pokémon environment.

The experiment must compare the converted recurrent target against a naive recurrent target trained without representational alignment. It must preserve the public-information boundary, action vocabulary, legality handling, and battle evaluation path. A recurrent target also needs ordered observations from earlier turns; the current request contract contains only the current state vector.

This experiment does not change Model1, the entity-centric family, the word-policy path, battle mechanics, or the historical Model1 benchmark profile.

## Part 3: Proposed Changes

1. Add a `vector_joint_bc_not_elman_v1` training family whose parent release is `model1`.
2. Build a three-layer Elman recurrent target with width 256 and the same 357-class policy head.
3. Use a bounded, explicit sequence of 582-dimensional public state vectors. The simulator will send the sequence in the request; the server will pad/truncate it deterministically before inference.
4. Implement progressive replacement stages: target recurrent layer 1; layers 1-2 jointly; layers 1-3 jointly.
5. Align corresponding hidden activations with linear CKA during replacement. The guide remains frozen. After replacement, fine-tune the complete target on action labels.
6. Save a standalone policy artifact, vocabulary copy, training metadata, stage metrics, and parent-release metadata. The registry will discover it without replacing Model1.
7. Add an explicit simulator profile for the recurrent target. It will preserve Model1's historical move-only deployment behavior unless a benchmark explicitly opts into voluntary switches.

## Part 4: Open Questions

1. What sequence length is useful for this environment? Answer with a fixed-length ablation after the first smoke run; the initial release uses 16 turns.
2. Does recurrence improve behavior beyond the stateless guide? Answer with held-out action agreement, game win rate, legal-action rate, and a history-disabled ablation.
3. Does CKA alignment improve the recurrent target over naive training? Answer with the same target architecture and seed under the two training paths.
4. Does the existing 678-dimensional simulator vector arrive at the server for legacy models? Answer in the request normalizer; the recurrent path must reduce each vector to the Model1 582-dimensional layout using the existing legacy strip rule.
5. Can the local TensorFlow environment run the full training job? Answer with a small synthetic fit and a bounded real-data smoke command before attempting the full dataset.

## Part 5: Invariants

1. `artifacts/model_1.keras` and its metadata are never overwritten.
2. The converted policy consumes only public state vectors and produces the same 357 action-token vocabulary as Model1.
3. Every recurrent request has a deterministic history length, ordering, and padding rule.
4. A request without the recurrent model identifier follows the existing stateless vector path.
5. Model1's existing `move-only` profile continues to suppress voluntary switches.
6. The target registry entry identifies `model1` as its parent and has a distinct release ID and artifact paths.
7. Alignment loss uses the frozen guide's representations and cannot update guide parameters.
8. Train/validation splits occur at battle boundaries, never at individual turns.

## Parts 6-8: Outcomes and Artifact Index

Implemented on 2026-08-06:

- `core/NoTModel1Elman.py` contains the repeat-first history contract, 690/678-to-582 Model1 layout normalization, linear CKA, the three-stage schedule, the Elman target builder, and the flat serving wrapper.
- `train_not_model1_elman.py` loads the frozen Model1 guide, splits examples by battle, aligns recurrent prefixes with CKA, fine-tunes the target with masked policy loss, and emits distinct target metadata.
- `flask_api_multi.py`, `core/ModelRegistry.py`, and `core/ModelWorkers.py` preserve the sequence contract and expose it through `/health` and `/predict` without changing legacy stateless requests.
- The simulator has an explicit `not-elman-policy` profile and sends a capped `state_history` only for that profile. Existing Model1 aliases and the word-policy path remain separate.

Verification evidence:

- 38 focused Python tests passed across the NoT utilities, trainer, sequence server preparation, registry, and worker contract.
- A bounded real-data smoke run on one battle produced 47 examples, completed all three alignment stages, fine-tuned the target, and wrote a temporary serving artifact with input `(None, 1164)` and output `(None, 357)`.
- The actual worker process loaded that artifact and returned `(357,)` logits. A live Flask request containing current 690-element state vectors in `state_history` normalized them and selected the known `move:knockoff` legal action.
- One real simulator random-vs-target battle completed with `not-elman-policy`, zero failed games, and the target winning the recorded game. This is a transport/runtime smoke result, not a quality estimate.
- `npm run build` succeeded. The repository-wide TypeScript check still reports unrelated pre-existing dirty-worktree errors; the new profile behavior was verified with focused Mocha tests.

Artifact and experiment boundary:

- The smoke artifact was written under `/private/tmp/not-elman-smoke.fARKId` and was intentionally not copied over the repository's persistent `artifacts/` directory. Run the documented trainer command with the full battle corpus to produce the actual experiment release.
- This implementation is the CKA-aligned transformation. A naive recurrent control and the paper-style held-out quality comparison remain follow-up experiment runs, not silently inferred from the one-game smoke.
