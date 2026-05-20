# Word Model Findings

## Current note

For the latest stable benchmark and the most recent decoder experiments, see
[RESEARCH_SNAPSHOT_2026_04_09.md](/Users/AI-CCORE/alter-programming/word_prediction_model/RESEARCH_SNAPSHOT_2026_04_09.md).

Current large-sample baseline:

- `sticky_move_run_003`
- `1768 / 2000`
- `88.40%`

## 1. Trajectory: from word prediction to battle policy

The initial command was much smaller than the current system. The original target
was:

- input: a few words that express an idea
- output: one canonical word
- tolerance: misspellings are not an error case, they are part of the model

That first design became a compact embedding model with two paths:

- a prompt path: descriptor words -> averaged token embedding
- a surface-form path: canonical word or misspelling -> averaged character n-gram embedding

The canonical label embedding sits between them. Training asks both paths to
meet at the same word representation. In practical terms:

- prompts classify into a canonical word
- misspellings align toward the same canonical word

This is the base system implemented in [model.py](/Users/AI-CCORE/alter-programming/word_prediction_model/model.py),
[dataset.py](/Users/AI-CCORE/alter-programming/word_prediction_model/dataset.py),
and [lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/lexicon.py).

The key architectural shift came later: instead of treating the predicted word
as the end product, the word became an intermediate semantic control token.

That changed the project into a layered system:

1. battle state + natural question -> prompt tokens
2. prompt tokens -> predicted battle word
3. predicted battle word + legal actions + battle heuristics -> action choice

This is the current battle stack:

- [battle_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_inquiry.py):
  maps question and battle state into a small prompt token set, then predicts
  words like `attack`, `finish`, `switch`, `stabilize`, `status`, `setup`
- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py):
  translates those words into move or switch scores
- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py):
  exposes the policy to the simulator over HTTP

So the current system is not "a word model that happens to know battles". It is
"a battle policy whose latent control language is a small typo-aware word
space."

## 2. What actually mattered in the evolution

Several changes looked important, but only a few really moved outcomes.

### 2.1 Misspelling handling was a representation choice, not preprocessing

This held up from the first version onward. Misspellings were useful because
they forced surface forms to align with canonical words in embedding space.
That gave the project a coherent center:

- the word is the stable semantic object
- the observed spelling is only one noisy surface

This is why the project could absorb later battle-language phrasing variation
without changing its core shape.

### 2.2 Parameter discipline mattered early

The user requested a hard cap of `10,000` parameters. The model was reshaped to
fit well below that limit by:

- keeping embedding dimensions small
- hashing n-grams into stable buckets
- constraining the battle vocabulary

The important result was not just smaller size. The cap forced the system to
become more symbolic and less brute-force. The current battle model is still a
small controller, not a large learned policy pretending to be one.

### 2.3 Cached inquiry models improved iteration speed

Once battle inquiry outputs were cached, the system stopped paying the cost of
retraining on every question. That mattered because it made the battle policy
fast enough to sit in the simulator loop.

### 2.4 The first real policy failure was legality, not strength

The first large-sample failure was an invalid voluntary switch while trapped.
That was not a weakness in the word model. It was a contract violation between:

- simulator request state
- policy server
- action adapter

The fix was structural:

- read `trapped` / `maybeTrapped` from the active request
- disable voluntary switching when those flags are set

This was a turning point because it separated "policy is weak" from "policy is
not yet safe to benchmark".

### 2.5 The strongest improvement came from simplifying the action space

The biggest empirical jump came when voluntary switches were suppressed on move
turns. That turned the model into a move-first controller with forced switches
still allowed.

Interpretation:

- the word model was significantly better at choosing among legal moves than at
  deciding when to rotate
- random battle play punished weak switching more than weak move choice

This was the first sign that the control space needed to be narrowed for the
model’s current capacity.

### 2.6 Type awareness was the first truly strategic feature

Threshold tuning changed results, but not robustly. The first nontrivial
feature that clearly improved scale behavior was type-aware move scoring.

That addition used local data only:

- species typing from local Pokédex data
- move typing and effectiveness from the local diagnostic/type tooling

Once the policy could distinguish neutral, resisted, immune, super-effective,
and STAB attacks, the benchmark profile changed substantially.

This indicates the word model alone is not enough. The policy needs a symbolic
combat prior to turn its coarse semantic word into a good move choice.

### 2.7 Threat awareness appears to be the next strategic layer

After the type-aware decoder, the next qualification jump came from using
revealed opponent move information as an inbound threat signal.

That changed the decoder from:

- "how good is my current move into this target?"

to:

- "how good is my current move once the opponent's known retaliation is taken
  seriously?"

The significance of this is architectural. The decoder is becoming
bidirectional: it evaluates not just pressure applied, but pressure absorbed.
That is the first real step beyond one-turn local greed.

## 3. Data and dials: how they relate

The system has three different kinds of data, and each is governed by different
dials.

### 3.1 Lexical training data

Source:

- [lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/lexicon.py)
- [battle_lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_lexicon.py)

Role:

- defines the canonical words
- defines descriptor tokens
- defines explicit and generated surface forms

Important dials:

- number of canonical labels
- descriptor diversity per label
- typo generation breadth
- semantic overlap between labels

Effect:

- more lexical diversity increases expressive range
- too much overlap blurs the word space
- too few labels makes the controller too coarse

### 3.2 Corpus-generation data

Source:

- [dataset.py](/Users/AI-CCORE/alter-programming/word_prediction_model/dataset.py)

Role:

- turns the lexicon into actual training examples

Important dials:

- `repeats_per_entry`
- `seed`
- `ngram_bucket_count`

Effect:

- `repeats_per_entry` affects training density and stability
- `seed` affects which prompt subsets and forms are seen together
- `ngram_bucket_count` trades typo resolution against parameter budget

Interpretation:

- this layer controls how often the model rehearses each semantic cluster
- it does not add new meaning; it changes how stably meaning is learned

### 3.3 Model-training data

Source:

- [model.py](/Users/AI-CCORE/alter-programming/word_prediction_model/model.py)

Important dials:

- `embedding_dim`
- `epochs`
- `learning_rate`
- `alignment_weight`
- `l2_weight`
- `max_parameters`

Relationship to behavior:

- `embedding_dim`: controls representational room
  too low compresses labels together, too high wastes budget
- `epochs`: controls how far the model sharpens class boundaries
  too low underfits, too high can over-sharpen a synthetic lexicon
- `learning_rate`: changes whether training settles or jitters
- `alignment_weight`: controls how strongly misspellings are forced toward the
  canonical embedding
- `l2_weight`: keeps embeddings from drifting too far
- `max_parameters`: the outer design boundary; this is the most important dial
  because it constrains all others

### 3.4 Battle-state data

Source:

- simulator request payload
- battle snapshot
- legal move list
- legal switch list

Role:

- this is not training data for the word model
- this is inference-time control data

Important dials:

- which state features are converted into prompt tokens
- which request flags are enforced structurally
- how legal actions are scored after the word prediction

This is where the project stopped being "just an embedding model". The battle
policy quality now depends heavily on how state is compressed into prompt tokens
and how the predicted word is decoded back into concrete actions.

## 4. The current wall: why progress slowed near 87 percent

The project is now in a different phase from the one that produced the early
large jumps.

Up through the high-70s and low-80s, progress mostly came from structural
repairs:

- legality fixes
- type-aware decoding
- revealed-threat awareness
- voluntary-switch suppression
- stronger closeout heuristics

Those changes worked because the policy still had obvious missing parts. Each
repair added a capability that simply did not exist before.

Near `87%`, the shape of failure changed. The policy no longer looks broadly
uninformed. It looks locally wrong. That matters because local wrongness is
harder to improve than global incompleteness.

Recent experiments made this visible:

- coefficient retuning was valid, but it moved the clean `1000`-game benchmark
  from `87.00%` down to `86.30%`
- broad opponent-role modeling was plausible, but it degraded the benchmark to
  `84.60%`
- a narrower estimated-lethal layer was cleaner, but still only reached
  `86.70%`

Interpretation:

- the policy is no longer bottlenecked by coarse missing knowledge
- it is increasingly bottlenecked by interaction effects between heuristics

That is the current wall. The remaining losses are less about basic battle
ignorance and more about:

- timing errors
- over- or under-commitment
- regime mismatch
- heuristics that help one slice of states but hurt another

This is also why new ideas now often interfere with old ones. Once a policy is
already strong, new logic cannot be judged only by whether it sounds correct.
It has to coexist with the rest of the decoder without diluting existing
strength.

So the key discovery of the current plateau is not merely that progress became
harder. It is that the project crossed from construction into refinement.

Earlier phase:

- discovering what needed to exist

Current phase:

- discovering what can coexist without degrading the rest of the policy

That is a more difficult phase, but it is also evidence that the architecture
is real enough to have a genuine ceiling rather than just obvious missing
pieces.

## 5. Policy dials

Source:

- [battle_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_inquiry.py)
- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)
- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py)

Important dials:

- prompt-token construction rules
- default battle question routing
- voluntary switch allowance
- move metadata weighting
- HP thresholds for recover / finish / switch
- type-effectiveness bonuses and penalties
- STAB bonus
- opponent revealed-move threat weighting

These dials are where most benchmark lift came from. The lexical model gives a
semantic prior; the policy dials decide whether that prior becomes a legal,
useful battle action.

## 6. The implementation spine

The easiest way to reason about the current system is as a spanning tree with a
single trunk and several branches.

### 6.1 Root: word-space learning

Root files:

- [text.py](/Users/AI-CCORE/alter-programming/word_prediction_model/text.py)
- [lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/lexicon.py)
- [dataset.py](/Users/AI-CCORE/alter-programming/word_prediction_model/dataset.py)
- [model.py](/Users/AI-CCORE/alter-programming/word_prediction_model/model.py)
- [pipeline.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pipeline.py)

Responsibility:

- define the canonical words
- generate prompt and typo examples
- train the embedding model
- serialize and evaluate runs

This is the trunk. Everything else depends on it.

### 4.2 First branch: battle-language specialization

Branch files:

- [battle_lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_lexicon.py)
- [battle_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_inquiry.py)

Responsibility:

- replace generic words with battle-control words
- map battle questions and state to prompt tokens
- produce a ranked word distribution instead of a final action

This branch specializes the general word model without changing its core
training algorithm.

### 4.3 Second branch: interface wrappers

Branch files:

- [bridge_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/bridge_inquiry.py)
- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py)

Responsibility:

- make the inquiry model callable from other systems
- expose structured outputs
- accept simulator payloads

This branch does not decide battle logic by itself. It packages the model for
consumption.

### 4.4 Third branch: action realization

Branch files:

- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)
- [pokemon_type_utils.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pokemon_type_utils.py)

Responsibility:

- take ranked word outputs
- score legal moves and switches
- enforce legality-sensitive constraints
- inject local symbolic priors like typing and STAB

This branch is currently the most performance-sensitive part of the system.
It is where the semantic controller becomes a game-playing policy.

### 4.5 Fourth branch: simulator benchmark loop

External integration point:

- `pokemon-showdown-model-feature/dist/sim/examples/statistical-runner.js`

Role:

- produces real battle outcomes
- is the only arbiter of whether policy changes matter

This branch is not inside the word-model package, but it is the system’s
ground-truth evaluation path.

## 5. Current findings from benchmarking

The benchmark story is now clear enough to state directly.

1. Small smoke runs were directionally useful but too noisy to trust.
2. The trapped-switch legality fix was required before any larger benchmark was
   meaningful.
3. Suppressing voluntary move-turn switches improved outcomes.
4. Type-aware move scoring produced the first major robust jump.

Current validated path:

- profile: joint-policy endpoint
- voluntary switches: disabled on move turns
- forced switches: still allowed
- move scoring: word-model prior + state heuristics + type effectiveness + STAB

Qualification result before the full-scale run:

- `159 / 200` wins
- `79.5%` win rate
- `0` failed games

Live scale run:

- `10000`-game benchmark is in progress
- log: [word_policy_10000.log](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/benchmarks/word_policy_10000.log)

## 6. Main interpretation

The project did not become strong by making the embedding model much more
complex. It became strong by putting the small embedding model in the right
place in the decision stack.

That placement is:

- use the word model for coarse semantic control
- use symbolic local battle knowledge for fine action scoring
- constrain the action space where the model is weakest

So the current lesson is not "the word model learned to play Pokemon by itself."
The real lesson is:

"A small typo-aware word embedding can act as a compact policy language, but it
needs a carefully engineered decoder to become a strong battle agent."

## 7. Next documentation directions

The next useful documents would be:

- a benchmark chronology with exact policy changes and win rates per phase
- a parameter-and-dials table for reproducibility
- a decision-flow diagram from simulator payload to final move choice

## 8. Ceiling lens against the actual random baseline

The ceiling question only makes sense if "random" is interpreted correctly.

As documented in
[RANDOM_BASELINE_CEILING.md](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/docs/RANDOM_BASELINE_CEILING.md),
the local benchmark opponent is not random over the whole action space. In the
benchmark harness it effectively:

- samples legal moves randomly
- samples forced switches randomly
- does not voluntarily switch

That means the opponent is strategically weaker than "full random actions"
would suggest. It also means the ceiling should be very high. A strong policy
should beat this baseline most of the time simply by:

- ranking moves better than chance
- avoiding unnecessary switches
- converting winning positions instead of looping

The more important question is whether the current wall near `88%` is already
close to the irreducible floor.

Right now, the evidence says no.

The remaining losses still look partly strategic:

- `sample_200_status_patch` losses average `11.69` total switches and
  `30.97` turns
- `sample_100_status_loop_focus` is dominated by `status_game` and
  `recovery_loop`
- many saved losses still contain repeated setup or recovery investment against
  an opponent that is not even repositioning voluntarily

At the same time, the semantic layer is usually not collapsing. In the replay
comparison aggregate
[status_loop_focus_replay_move_compare_aggregate.json](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/loss_replays/status_loop_focus_replay_move_compare_aggregate.json),
the decoder matches replayed concrete moves on `180 / 188 = 95.74%` of
reconstructed decisions, and the primary words are still mostly `attack` and
`finish`.

So the current best interpretation is:

- the project is on a real local plateau
- but it is not yet at the true ceiling against this random baseline
- there is still recoverable loss mass
- that recoverable mass is now concentrated in a small number of pathological
  regimes rather than broad tactical incompetence

This is why progress has become expensive without being impossible. The agent is
already good enough to exploit the baseline broadly. The remaining gains are no
longer about general competence. They are about eliminating a narrow residue of
strategic leakage before variance becomes the dominant source of loss.
