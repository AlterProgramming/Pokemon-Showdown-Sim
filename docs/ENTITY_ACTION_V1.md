# Entity Action v1

## Purpose

This document defines the first entity-centric model family:

- `family_id = entity_action_bc`
- `family_version = 1`

The goal is to move away from a flat board vector and toward entity-centric action scoring while keeping the design minimal, learnable, and observable.

This family is intentionally not belief-aware yet. It is the graph-ready baseline that comes before explicit latent belief state.

## Main Principle

The model should learn mechanics-rich representations from named entities, not from large hand-authored mechanic tables.

That means:

- If an entity has a stable symbolic identity, we give the model the identity token.
- The meaning of that identity should be learned through embeddings and training objectives.
- We only hand the model the minimum explicit numeric state needed to describe the current public battle state.

In plain terms:

- Give it names.
- Give it public state.
- Let it learn what those names mean.

## What Should Be Learned

These should be learned representations, not manually expanded mechanic feature tables.

- Pokemon species identity
- Move identity
- Item identity
- Ability identity
- Tera type identity
- Status identity
- Weather identity
- Terrain and global condition identity
- Side-condition identity

The model should not be handed explicit move power, move category, move type, item effect tables, or ability effect tables in this family.

Those should emerge through training pressure.

## What Must Still Be Explicit

Some things should remain explicit because they are current observed state, not hidden mechanics.

- HP fraction if known
- HP known flag
- Fainted flag
- Active flag
- Bench slot index
- Side identifier
- Current stat boosts
- Current visible status
- Current revealed side conditions
- Current weather and global field conditions
- Current legal action set
- Current turn index or capped turn count

These are not mechanic tables. They are the public state of the game at decision time.

## Minimum Entity Set

This family should start with the smallest entity set that still supports object-centric reasoning.

### Required Entity Types

- One Pokemon entity for each of your six team slots
- One Pokemon entity for each of the opponent's six team slots
- One global battle-context entity
- One move-candidate entity for each currently legal move

That gives a small, stable graph:

- 12 Pokemon entities
- 1 global entity
- up to 4 legal move entities

Switch candidates do not need separate entities in v1 because each switch target is already one of your Pokemon entities.

## Pokemon Entity Features

Each Pokemon entity should be initialized from a mix of:

- learned identity embeddings
- small explicit public-state features

### Learned Identity Inputs

Each Pokemon entity may receive these token inputs if known:

- species token
- revealed item token
- revealed ability token
- revealed tera type token

If something is not known, use an explicit unknown token rather than a zero vector.

That matters because unknown is information too.

### Explicit State Inputs

Each Pokemon entity should also receive:

- side flag
- slot index
- active flag
- public-revealed flag
- fainted flag
- HP fraction plus HP-known flag
- visible status token
- current stat boosts
- terastallized flag

### Observed Move Information

For v1, do not hand the Pokemon entity a handcrafted move-stat summary.

Instead:

- provide the list or bag of observed move identity tokens
- embed those move tokens
- pool them into an observed-move summary for that Pokemon entity

This keeps move identity learned rather than manually unpacked.

## Move-Candidate Entity Features

Legal move candidates should be entities too, not just indices in a classifier output.

Each legal move-candidate entity should receive:

- move identity token
- slot index among the current legal moves
- disabled or enabled flag if available
- optional current PP bucket only if public and reliably available

The key design is:

- do not score "move class 173" globally
- score this concrete legal move in this concrete state

This is one of the main advantages of moving to entities.

## Global Battle Entity Features

The global entity should carry the current shared board context.

It should include:

- weather token
- terrain and global condition tokens
- my side-condition tokens with counts
- opponent side-condition tokens with counts
- capped or normalized turn index

Again, the conditions themselves should be embedded by identity. The only hand-authored numeric part should be condition count or presence.

## Unknown And Hidden Information

This family is not belief-aware yet, but it should be belief-ready.

That means:

- opponent slots always exist as entities, even when partially unknown
- unrevealed species, item, ability, or tera type use explicit unknown tokens
- opponent slots are never omitted just because information is missing

This is important because `entity_belief_aux_v1` should be able to upgrade these same slot entities later into latent belief carriers.

## Graph Structure

The graph should be relational, but the first version should stay simple.

### Required Edges

- every Pokemon entity connects to the global entity
- the two active Pokemon connect to each other
- each of your bench Pokemon connects to your active Pokemon
- each opponent bench Pokemon connects to the opponent active Pokemon
- each legal move entity connects to your active Pokemon
- each legal move entity connects to the opponent active Pokemon

That is enough to support:

- active matchup reasoning
- switch-target reasoning
- move-target reasoning
- board-context conditioning

It is already much better aligned with the structure of the battle than one flat vector.

## Action Scoring

This family should not use one monolithic action-class head.

Instead:

- each legal move entity gets a score
- each legal switch target Pokemon entity gets a score

Then:

- combine those scores over the legal action set
- apply masking from legality
- train the chosen human action to score highest

This is the main action contract for the family.

It is especially important for switches, because the real question is:

- should I switch into this exact Pokemon?

not:

- should I switch in general?

## Encoder Choice

This family should be graph-ready, but the first encoder does not need to be the most ambitious possible GNN.

Any of these are acceptable as the first implementation:

- true message-passing GNN over the entity graph
- shared per-entity encoder plus a small relational mixing block
- graph-attention style encoder

The requirement is not a specific library or layer type.
The requirement is:

- entity-centric state
- relational mixing
- action-wise scoring

That is the real contract.

## Training Objective

This first family should stay simple.

Primary objective:

- behavior cloning over the chosen human legal action

No belief-state recurrence is required yet.
No multimodal training is required.
No self-play correction is required.

This family exists to establish the entity/action interface first.

## What Not To Include In v1

These are intentionally deferred.

- latent opponent belief updates across turns
- recurrent battle memory beyond the current snapshot
- item nodes separate from Pokemon entities
- ability nodes separate from Pokemon entities
- image or audio modalities
- hand-authored move power or type tables
- hand-authored item effect tables
- hand-authored ability effect tables

Those may come later, but they should not be part of `entity_action_bc_v1`.

## Why This Is The Minimum Useful Version

This is the minimum version that:

- preserves named entity identity
- lets the model learn those identities through embeddings
- treats moves and switch targets as concrete actions
- remains simple enough to compare against current vector baselines

If we make it smaller than this, we lose the main point of the family.
If we make it much larger than this, we blur the experiment.

## Path To The Next Family

This family should lead naturally into `entity_action_aux_v1`.

What changes there:

- keep the same entity contract
- keep the same action-scoring contract
- add auxiliary heads that put more pressure on the learned entity representations

Then after that:

- `entity_belief_aux_v1` adds evolving opponent-slot latent belief state

That sequencing matters.

## Summary

The design rule for `entity_action_bc_v1` is:

- use explicit public state for what is currently true
- use learned embeddings for what an entity is
- do not inject hand-authored mechanic tables
- score concrete legal actions instead of global action classes

That gives the project the first real entity-centric family while staying minimal and interpretable.
