# Entity Invariance Aux v1

## Purpose

This document defines a new research family:

- `family_id = entity_invariance_aux`
- `family_version = 1`

This family is the first explicit identity-invariance phase for the project. Its job is to answer a narrow but important question:

- did the entity-centric jump come from meaningful names
- or from stable identity anchors

It also introduces a learned latent battle state whose role is to absorb unresolved variance across turns without forcing a hand-authored label like `uncertainty`.

## Current Scaffold

The first runnable implementation of this family is intentionally narrower than the full research target:

- previous-turn context is currently a one-step `t-1` entity view rather than a long recurrent belief state
- `mixed_id` is implemented as real-id plus placeholder-id training augmentation
- explicit paired relabel-consistency loss is still deferred

That keeps the family trainable now while preserving the intended upgrade path.

## Main Principle

The model should learn from persistent entities, not from the literal semantics of their labels.

That means:

- the same underlying entity may appear under a real name or a placeholder name
- the mapping must remain consistent within a battle or run
- behavior should remain stable if the labels change but the relational structure does not

In plain terms:

- keep identity
- relax semantics
- pressure the model to preserve policy quality anyway

## What This Family Adds

Compared with `entity_action_aux_v1`, this family adds four pieces of pressure:

- relabel invariance over entity identities
- a persistent latent state carried across turns
- controlled stochastic exposure during training or evaluation
- explicit bookkeeping for transfer initialization

This is a genuine family addition because it changes history usage and objective meaning, not just hidden width or training length.

## Representation Contract

The public battle representation remains entity-centric:

- self-side Pokemon slot entities
- opponent-side Pokemon slot entities
- global battle context entity
- legal move entities

The new parts are:

- optional placeholder identity tables for species, items, abilities, tera types, and moves
- a persistent latent battle state `z_t`

The latent state is not required to have a human label. It only needs to improve downstream prediction and action choice.

## Identity Regimes

The family should support three regimes under the same experiment umbrella:

- `real_id`
  Use the ordinary entity identities already present in the logs
- `placeholder_id`
  Replace entity identities with arbitrary but consistent placeholders
- `mixed_id`
  Train on a mixture of real and placeholder identity episodes

Compressed identity experiments are valuable, but they should be treated as later ablations rather than part of the first run matrix.

## Latent State Contract

The persistent latent battle state should be updated every turn from the current entity state and prior latent:

- `h_t = encoder(state_t, z_{t-1})`
- `z_t = latent_update(h_t, z_{t-1})`
- `policy_t = policy_head(h_t, z_t)`

Optional auxiliary heads may also read from `h_t` and `z_t`.

Good latent targets include:

- next public-state summary
- action robustness
- opponent action spread
- terminal value

The family should avoid forcing a single human-defined scalar called `uncertainty` unless later inspection shows that is useful.

## Controlled Randomness

This family should not treat randomness as pure nuisance. A small amount of structured randomness is part of the training pressure.

Good sources:

- random-opponent evaluation
- older-policy evaluation
- top-k or stochastic policy rollouts for self-generated state coverage
- placeholder relabeling

Bad sources:

- breaking within-battle identity consistency
- corrupting causal targets
- randomizing observations in ways impossible under the simulator

## Objectives

The recommended objective bundle is:

- `policy`
  Behavior cloning over legal actions
- `value`
  Terminal win probability
- `transition`
  Next public-state summary or another existing auxiliary target
- `relabel_consistency`
  Keep policy and value stable when identity labels are replaced by consistent placeholders
- `latent_usefulness`
  Require the recurrent latent to improve prediction or action quality

The family does not require all heads in its very first run, but `relabel_consistency` is the defining addition.

## Initial Experiment Sheet

Use a small, explicit run matrix first.

### Run A: Real-Identity Control

- `family_id = entity_invariance_aux`
- `identity_regime = real_id`
- `initialization_source = scratch` or transfer from `entity_action_bc_v1`
- objectives:
  `policy + transition + value`
- purpose:
  establish the control inside the new family and confirm that the latent does not immediately degrade the working entity baseline

### Run B: Placeholder-Identity Test

- `family_id = entity_invariance_aux`
- `identity_regime = placeholder_id`
- same architecture and objective weights as Run A
- placeholders must be consistent within each training example and battle rollout
- purpose:
  test whether most of the gain survives semantic relabeling

### Run C: Mixed-Identity Consolidation

- `family_id = entity_invariance_aux`
- `identity_regime = mixed_id`
- mix real and placeholder identity episodes in the same training run
- purpose:
  encourage the encoder and latent state to converge toward relation-level invariants instead of label-specific shortcuts

## Evaluation Bundle

Each run should be compared on:

- held-out policy metrics
- value calibration
- win rate versus random
- win rate versus older vector baselines
- switch count and repetitive-action diagnostics
- embedding-space plots for entity identities
- gap between real-id and placeholder-id performance

The most important acceptance test is simple:

- placeholder performance should stay close enough to real-id performance that identity consistency clearly matters more than label semantics

## Release Naming

Use explicit release names rather than overwriting earlier families. Good examples:

- `entity_invariance_aux_v1_20260327_run1`
- `entity_invariance_aux_v1_20260327_placeholder_run1`
- `entity_invariance_aux_v1_20260327_mixed_run1`

The metadata for every release should include:

- `family_id`
- `family_version`
- `model_release_id`
- `parent_release_id`
- `initialization_source`
- `history_schema_version`
- `identity_regime`
- `objective_set`

## Transfer Hooks

This family should be transfer-friendly from the start.

The intended transfer modes are:

- within-family warm start
- cross-family warm start from `entity_action_bc_v1`
- future reuse of latent-capable layers across later generations

The project should record both:

- where the initialization came from
- how much of the initialization was actually reused

That makes later generation-to-generation comparisons honest.

## What This Family Should Answer

If this family succeeds, it should answer three questions:

- are stable identities more important than semantic names
- does a persistent latent help the model stay robust in stochastic or hidden-information turns
- does the resulting representation become a better transfer base for later generations
