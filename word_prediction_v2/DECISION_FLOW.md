# Decision Flow

## Purpose

This document describes the operational spine of the current implementation as a
decision-flow diagram in text form.

## 1. High-level flow

```text
Simulator Request
  ->
Policy Server
  ->
Default Question Routing
  ->
Battle Inquiry Prompt Builder
  ->
Word Embedding Model
  ->
Ranked Battle Words
  ->
Action Decoder
  ->
Legal Move / Switch Choice
  ->
Simulator Command
```

## 2. Expanded flow

```text
1. Pokemon Showdown simulator emits a request
   Inputs:
   - battle_state
   - legal_moves
   - legal_switches
   - active request flags

2. policy_server receives /predict
   Checks:
   - JSON validity
   - battle_state presence
   - perspective player
   - trapped / maybeTrapped flags

3. policy_server chooses a question if none was given
   Examples:
   - "can I knock it out now?"
   - "what is the safe play?"
   - "should I attack now?"

4. choose_action_from_inquiry begins
   Inputs:
   - question
   - battle_state
   - legal actions
   - whether voluntary switching is allowed

5. answer_battle_inquiry builds prompt tokens
   Sources:
   - question words
   - own HP
   - opponent HP
   - status
   - boosts
   - bench availability

6. battle word model predicts ranked semantic control words
   Examples:
   - attack
   - finish
   - switch
   - stabilize
   - status

7. policy_adapter decodes the word distribution
   Paths:
   - voluntary switch path
   - move scoring path
   - fallback switch path
   - none path

8. move scoring combines:
   - word-role priors
   - move metadata
   - HP/state heuristics
   - type effectiveness
   - STAB bonus
   - opponent revealed-move threat
   - legality constraints

9. best legal action is returned as JSON
   Examples:
   - type=move, best_move.slot=2
   - type=switch, slot=4

10. simulator executes the action
```

## 3. File-by-file spine

### Root layer

- [text.py](/Users/AI-CCORE/alter-programming/word_prediction_model/text.py)
- [lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/lexicon.py)
- [dataset.py](/Users/AI-CCORE/alter-programming/word_prediction_model/dataset.py)
- [model.py](/Users/AI-CCORE/alter-programming/word_prediction_model/model.py)
- [pipeline.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pipeline.py)

Job:

- define the embedding space and training machinery

### Battle-language layer

- [battle_lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_lexicon.py)
- [battle_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_inquiry.py)

Job:

- turn battle situations into prompt tokens
- turn prompt tokens into battle words

### Decision layer

- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)
- [pokemon_move_metadata.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pokemon_move_metadata.py)
- [pokemon_type_utils.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pokemon_type_utils.py)

Job:

- turn battle words into scored legal actions

### Serving layer

- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py)

Job:

- expose the policy in the interface expected by the simulator

### Benchmark layer

- `pokemon-showdown-model-feature/dist/sim/examples/statistical-runner.js`

Job:

- generate live game outcomes against the random baseline

## 4. Control points

These are the main points where behavior can be changed.

### Control point A: lexicon definition

If you change:

- canonical words
- descriptors
- typo sets

You change:

- the semantic vocabulary available to the controller

### Control point B: prompt-token builder

If you change:

- which state facts become prompt tokens

You change:

- what the word model is effectively told about the battle

### Control point C: default question routing

If you change:

- which natural-language question is asked for a given state

You change:

- which region of word space the controller is biased toward

### Control point D: action decoder

If you change:

- word-role weights
- move metadata weighting
- HP thresholds
- type bonuses
- opponent threat weighting
- switch thresholds

You change:

- how semantic intent becomes a concrete move or switch

### Control point E: simulator exposure

If you change:

- voluntary switch availability

You change:

- the size of the action space itself

## 5. Safety checkpoints

These are the places where structural safety matters more than strategy.

### Trapped switch protection

Implemented in:

- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py)
- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)

Purpose:

- prevent illegal voluntary switches when the request marks the active Pokemon
  as trapped or maybe trapped

### Legal-action gating

Purpose:

- action choices are always made from the legal moves or legal switches supplied
  by the simulator

### Forced-switch preservation

Purpose:

- suppressing voluntary switching must not remove forced-switch behavior

## 6. Why the current spine works better

The improved benchmark profile comes from three structural choices:

1. the word model is used for semantic compression, not for raw move selection
2. the decoder uses symbolic battle priors, especially type effectiveness and
   move metadata
3. the action space is narrowed where the model is weakest

The current threat-aware variant adds a fourth idea:

4. the decoder scores not only what my move does to the opponent, but also what
   the opponent's revealed move set can do back to me

That gives the system a clean hierarchy:

- semantic control at the top
- tactical scoring in the middle
- legality enforcement at the boundary
- simulator truth at the bottom

## 7. Short version

If the whole system had to be described in one sentence:

```text
The simulator provides a battle state, the server converts it into a small
natural-language inquiry, the word model answers with a compact battle concept,
and the decoder turns that concept into the best legal action using symbolic
battle priors.
```
