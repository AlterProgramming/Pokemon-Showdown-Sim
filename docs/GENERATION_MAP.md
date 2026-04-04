# Generation Map

## Purpose

This project is no longer just a trainer for one policy network. It is becoming a research platform for:

- reproducible model families
- observable training and evaluation
- entity-centric battle reasoning
- belief-aware decision making under partial observability

The goal of this document is to keep old generations reproducible while making room for new families that change representation, objectives, and training regime in a controlled way.

## Core Rules

These rules should stay true even as the project changes.

- A saved model is not just a weight file. It belongs to a recipe family.
- Old recipe families must remain retrainable after new ones are introduced.
- New research families should be added, not silently substituted for old ones.
- Every run must record enough metadata to reproduce its behavior, not just reload its weights.
- Observability is a first-class feature, not an optional add-on.
- Multimodal work is out of scope for the current mainline. The near-term target is entity-centric and belief-aware battle modeling.

## Current Baseline Families

These families already exist in practice, even if they were not originally named this way.

### `vector_joint_bc_transition_v1`

This is the current flat-vector joint-action baseline with an auxiliary transition head.

- Representation:
  Public-information flat vector from `StateVectorization.py`
- Action parameterization:
  Joint action vocabulary over moves and switches
- Objectives:
  Policy imitation plus auxiliary turn-outcome prediction
- Typical artifacts:
  `model_2.keras`, `model_3.keras`, `model_2_large.keras`
- Current strengths:
  Stable baseline, easy to serve, already integrated with simulator and model registry
- Current limitations:
  Weak relational reasoning, weak switch-target reasoning, no explicit belief state, weak recovery when drifting off human trajectories

### `vector_joint_bc_transition_value_v1`

This is the current flat-vector baseline with both transition and terminal-outcome value heads.

- Representation:
  Same flat public-information vector
- Action parameterization:
  Joint action vocabulary over moves and switches
- Objectives:
  Policy imitation plus turn-outcome prediction plus terminal win-probability value
- Typical artifacts:
  `model_4.keras`, `model_4_large.keras`
- Current strengths:
  Better observability, value traces, richer auxiliary pressure than policy-only or transition-only
- Current limitations:
  Value is still offline Monte Carlo supervision, not yet policy-improving RL

### `vector_joint_bc_value_v1`

This is the current flat-vector baseline with value but without the transition head.

- Representation:
  Same flat public-information vector
- Action parameterization:
  Joint action vocabulary over moves and switches
- Objectives:
  Policy imitation plus terminal win-probability value
- Typical artifacts:
  `model_4_rewarded.keras`
- Current strengths:
  Useful simpler comparison against transition-plus-value
- Current limitations:
  Same as above, plus less auxiliary state-dynamics pressure than the transition family

## Near-Term Research Families

These are the next families to add. Each one should be a new recipe family, not a mutation of the older vector families.

### `entity_action_bc_v1`

First entity-centric supervised family.

- Main goal:
  Replace the flat encoder with an entity-centric encoder while keeping training mostly behavior-cloning based
- Representation:
  Explicit entities for active Pokemon, bench Pokemon, and global battle context
- Action parameterization:
  Score concrete legal actions instead of only predicting from one global class space
- Objectives:
  Policy imitation only
- Why it exists:
  Establish the entity representation without changing too many things at once
- What it should answer:
  Does object-centric representation alone improve switch-target quality and action grounding?

### `entity_action_aux_v1`

Entity-centric family with auxiliary predictive pressure.

- Main goal:
  Keep entity-centric action scoring, then make the latent state more meaningful through multiple objectives
- Representation:
  Same entity-centric state as `entity_action_bc_v1`
- Action parameterization:
  Action-wise scoring over legal moves and legal switch targets
- Objectives:
  Policy imitation plus auxiliary heads
- Recommended auxiliary heads:
  Next-state summary, damage tendency, speed relation, terminal value
- Why it exists:
  Force the latent entity states to encode deeper battle structure rather than shallow action frequency
- What it should answer:
  Does predictive pressure reduce repetitive nonsense and improve action quality?

### `entity_invariance_aux_v1`

Entity-centric family for identity invariance and latent variance consolidation.

- Main goal:
  Test whether stable identity anchors matter more than semantic names, while adding a learned latent state that can absorb unresolved variance across turns
- Representation:
  Same public entity-centric state as `entity_action_aux_v1`, plus a persistent latent battle state and relabelable entity identities
- Action parameterization:
  Same action-wise scoring over legal actions
- Objectives:
  Policy imitation plus auxiliary heads plus relabel-consistency pressure
- Recommended new pressures:
  Placeholder relabel invariance, latent usefulness for next-state or action-robustness prediction, terminal value, controlled-randomness robustness
- Why it exists:
  Separate "identity consistency" from "name semantics" and give the model a place to store unexplained battle variance
- What it should answer:
  Can the agent keep most of its gain when names are replaced by consistent placeholders, and does a persistent latent improve robustness under stochastic opponents and hidden information?

### `entity_belief_aux_v1`

Entity-centric family with explicit belief-state machinery.

- Main goal:
  Introduce evolving latent beliefs for opponent slots under partial observability
- Representation:
  Known self-side entities plus belief-state opponent-slot entities plus global context
- Action parameterization:
  Same action-wise scoring over legal actions
- Objectives:
  Policy imitation plus auxiliary heads that pressure the belief state to become informative
- Recommended new pressures:
  Opponent move-likelihood prediction, speed-order prediction, next-state prediction, damage expectation, value
- Why it exists:
  The project's main failure mode is not only poor representation, but poor reasoning under hidden information and distribution shift
- What it should answer:
  Does explicit belief modeling improve stability once the model leaves the human-data manifold?

### `entity_belief_selfplay_v1`

Entity-centric belief-aware family with self-generated trajectory correction.

- Main goal:
  Expose the policy to its own induced states instead of only human trajectories
- Representation:
  Same as `entity_belief_aux_v1`
- Action parameterization:
  Same action-wise legal scoring
- Objectives:
  Policy loss plus value plus chosen auxiliary heads, with training data that includes model-generated trajectories
- Why it exists:
  The weird repetitive behavior appears mainly once the model acts on its own and enters states missing from the logged-human distribution
- What it should answer:
  Can the agent learn to recover from its own drift instead of only copying humans?

This family intentionally avoids stronger jargon in day-to-day planning. The important idea is simple:

- train on the model's own states too
- give it signals that repeated bad loops are bad
- keep the transition from offline imitation to self-generated correction explicit and observable

## Family Boundaries

These boundaries are important. They keep the project reproducible.

### Frozen Within A Family

Within one recipe family, these should be treated as invariant unless the family version is incremented.

- information policy
  Example: public-only vector, public-plus-belief, debug/full-state
- state schema
- action parameterization
- objective set
- reward definition
- history usage
- legality handling contract
- evaluation bundle definition

### Allowed To Vary Within A Family

These can vary run to run without creating a new family version.

- hidden width and depth
- dropout
- learning rate
- batch size
- training length
- checkpoint initialization choice
- reward weights, if the reward definition itself is unchanged
- model size variants such as default versus large

### Requires A New Family Version

Any of these changes should produce a new family version.

- flat vector to entity graph
- global classifier to legal-action scoring
- no belief state to belief state
- offline-only data to self-generated trajectory training
- public-only information to any other observability policy
- change in the meaning of the value target
- addition of history when history was previously absent in the family definition

## Required Metadata For Every Run

The current metadata is a strong start, but future runs need a richer contract.

Every run should save:

- `family_id`
- `family_version`
- `model_release_id`
- `parent_release_id`
  Optional, but strongly recommended for transfer learning
- `training_regime`
  Example: offline_bc, offline_bc_aux, offline_bc_value, self_generated_correction
- `information_policy`
  Example: public_only, public_plus_belief
- `state_schema_version`
- `entity_schema_version`
  Optional for vector families, required for entity families
- `history_schema_version`
  Required once history is introduced
- `action_parameterization`
  Example: joint_vocab, legal_action_scoring
- `objective_set`
  Explicit list of heads and targets
- `reward_definition_id`
- `reward_config`
- `value_target_definition`
  Example: terminal_win_probability
- `transition_target_definition`
- `data_source_id`
  Example: Kaggle RandBats dataset slug and any preprocessing filters
- `split_definition`
  How train and validation were split
- `environment_definition`
  Simulator/version assumptions
- `initialization_source`
  Scratch or checkpoint path
- `evaluation_bundle_id`
- `trace_schema_version`
- `registry_visibility`
  Whether the artifact is runnable in the standard simulator flow

## Artifact Classes

Every family should produce artifacts in clearly separated roles.

### Always Save

- trainable model artifact
- serving artifact
- vocabularies or action-index artifacts if applicable
- full run metadata
- evaluation summary
- reward profile if reward traces depend on it

### For Observable Families

- replay trace outputs
- per-turn value traces if value exists
- per-head metric summary
- any auxiliary-head calibration summary

### For Belief-Aware Families

- belief-state trace summary
- optionally, per-slot uncertainty or entropy summaries
- rollout or replay diagnostics showing how latent beliefs changed over turns

## Parallel Workstreams

This work can and should be split into parallel tracks.

### Track A: Housekeeping And Governance

This track keeps the project reproducible while the architecture changes.

- Introduce explicit family metadata
- Backfill current flat-vector models into named families
- Standardize evaluation bundles
- Standardize artifact naming around family and release identity
- Keep model registry compatible with older families
- Add run manifests that explain how a model was trained

Suggested branch:

- `codex/family-governance`

### Track B: Entity-Centric Mainline Research

This track explores the next architecture step without breaking the old stack.

- Build entity-centric state schema
- Add legal-action scoring
- Keep supervision first
- Add auxiliary heads next
- Add belief state after the basic entity stack is stable

Suggested branch:

- `codex/entity-action-v1`

### Track C: Belief-Aware Research

This track should start only after Track B has a stable entity baseline.

- Add opponent-slot latent state
- Add update rules and belief-oriented heads
- Add replay diagnostics for latent belief change

Suggested branch:

- `codex/entity-belief-v1`

## Recommended Immediate Order

The next work should happen in this order.

1. Formalize the family metadata and recipe identity rules.
2. Preserve and document the current vector families as stable baselines.
3. Build the first entity-centric supervised family.
4. Add auxiliary heads to the entity family.
5. Add belief-state machinery only after the entity family is understandable and observable.
6. Only then move into self-generated trajectory correction.

This keeps the project understandable while still moving decisively toward the real target.

## Non-Goals For The Current Mainline

These are explicitly not the next mainline target.

- multimodal training
- image reconstruction
- audio-conditioned ontology learning
- fully latent move or item identity without symbolic anchors
- replacing old vector baselines before the entity family is proven

Those may become real branches later, but they are not the mainline test right now.

## Practical Interpretation

What this document means in plain terms:

- Keep the old vector stack alive and reproducible.
- Stop treating the numeric model releases alone as the identity of a model.
- Add explicit recipe families.
- Move the next serious architecture work toward entities, legal actions, and beliefs.
- Use auxiliary heads to force deeper structure.
- Delay multimodal ambition until the entity-belief line is stable.

That is the path that gives the project both progress and memory.
