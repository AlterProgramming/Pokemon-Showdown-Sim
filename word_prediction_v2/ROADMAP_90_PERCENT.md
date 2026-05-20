# Roadmap To 90 Percent

## Purpose

This document captures the next ceiling plan after the threat-aware decoder
qualified at `85.50%` over `200` games.

The goal is not to guess our way to `90%`. The goal is to identify the next
sources of remaining loss and turn them into explicit decoder layers.

## Plateau note

The project has now spent multiple clean `1000`-game attempts near the same
ceiling, with a best clean result of `87.00%`. That plateau is important.

It suggests the remaining losses are no longer mostly caused by missing coarse
features. They are increasingly caused by interaction effects between existing
heuristics. In practice:

- broad new logic can easily dilute the current strong baseline
- plausible ideas can still regress the benchmark
- local tuning alone has not been enough to break through

So the roadmap below should be read with one extra constraint:

- new layers must be narrow enough to improve a specific failure class without
  disturbing the rest of the decoder

## Current state

Current qualified profile:

- `171 / 200`
- `85.50%`
- `0` failed games

Current architecture already includes:

- typo-aware semantic controller
- battle inquiry prompt builder
- move metadata scoring
- type-aware scoring
- STAB bonus
- threat-aware scoring from revealed opponent moves
- voluntary switch suppression on move turns
- legality-aware trapped-switch protection

## Why 90 percent is different

Getting from `78%` to `85%` was achievable with richer tactical scoring. Getting
from `85.5%` to `90%` is harder because the remaining losses are likely to be
less about obvious move quality and more about policy regime failures.

That means the next gains probably require:

- better game-phase awareness
- better opponent-role inference
- more selective resource management
- careful reintroduction of only high-value switching

## Phase A: opponent-set inference

Status:

- not yet implemented

Problem:

- the decoder sees revealed moves, but it does not yet convert them into a
  structured model of what the opponent probably is

Desired inference outputs:

- offensive attacker
- setup sweeper
- utility / hazard setter
- recovery wall
- priority threat
- status spreader

Why this matters:

- revealed moves should not only contribute direct threat scores
- they should also shape how greedy or passive our line is allowed to be

Expected use:

- penalize setup into likely setup sweepers
- penalize passive play into known hazard/status loops
- prefer cleaner conversions into fragile offensive targets

## Phase B: endgame and anti-throw mode

Status:

- partially implied by current heuristics
- not yet explicit

Problem:

- a high-performing policy can still leak wins by taking unnecessarily fancy
  lines while already ahead

Desired endgame signals:

- count of healthy mons on each side
- current active HP differential
- whether lethal pressure is available now
- whether the opponent has revealed strong comeback tools

Desired behavior:

- when ahead, reduce greed
- prefer safe KOs over setup
- avoid unnecessary recovery if conversion is available
- avoid hazard/status turns that give the opponent back initiative

Why this matters:

- the final few percentage points often live in conversion discipline

## Phase C: narrow voluntary switching

Status:

- currently disabled on move turns because the old switching policy was weak

Problem:

- suppressing voluntary switching fixed a major weakness
- but it may also leave some free wins on the table in clearly favorable pivot
  spots

Desired policy:

- only allow voluntary switching when all of these are true:
  - opponent revealed threat is high
  - current active does not threaten immediate conversion
  - a bench candidate materially resists the known threat
  - the switch does not walk into obvious hazard or status punishment

Why this matters:

- a fully disabled switch regime is safe
- a narrowly reintroduced switch regime may be stronger

Risk:

- this is the easiest place to reintroduce old failure modes
- it should come after stronger opponent inference, not before

## Phase D: hazard and resource economy

Status:

- under-modeled

Problem:

- the current decoder recognizes hazards mostly as weak local moves
- it does not yet model when hazards are genuinely worth the tempo

Desired features:

- my side hazard burden
- opponent side hazard burden
- whether the opponent is likely to switch or be forced to switch
- whether I am already ahead enough that passive chip wins the game

Desired behavior:

- hazards should only be favored when they improve the expected game trajectory,
  not simply because they are "useful moves"

## Phase E: role-specific closeout logic

Status:

- not yet implemented

Problem:

- all attacking lines are still scored somewhat generically

Desired distinctions:

- safe chip
- immediate lethal
- priority cleanup
- anti-setup denial
- low-risk closer

Why this matters:

- the same word `attack` should not map to the same tactical scoring in every
  board state

## Suggested execution order

Recommended order:

1. explicit endgame / anti-throw mode
2. opponent-set inference
3. role-specific closeout logic
4. narrow voluntary switching reintroduction
5. deeper hazard-resource scoring

Reason:

- endgame conversion is likely the easiest remaining gain
- switching is the most dangerous lever and should remain late

## Benchmark discipline for the 90 percent push

The process should stay strict:

1. adapter and server tests first
2. `50`-game smoke run
3. `200`-game qualification
4. only then run `10000`

Do not trust:

- isolated smoke highs
- one-off lucky `10`-game runs
- subjective “felt stronger” changes

## Practical interpretation

The path to `90%` is no longer about making the word model bigger or smarter in
isolation. It is about making the decoder more regime-aware.

The likely final form is:

- word model decides the coarse intent
- decoder decides the tactical realization
- regime logic decides when the policy is in:
  - neutral play
  - survival play
  - conversion play
  - anti-throw play
  - pivot-required play

That is the real next ceiling.
