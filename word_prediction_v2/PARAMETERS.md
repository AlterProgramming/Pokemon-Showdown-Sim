# Parameters And Dials

## Purpose

This document is a compact map of the knobs that shape the word-model system.
The point is not to list every constant. The point is to show which dials belong
to which layer and what they actually influence.

## 1. Core model dials

Source:

- [model.py](/Users/AI-CCORE/alter-programming/word_prediction_model/model.py)

### `embedding_dim`

Meaning:

- dimensionality of token, label, and n-gram embeddings

Effect:

- higher values increase representational room
- lower values force compression

Tradeoff:

- larger values improve separation up to a point
- but they also consume the parameter budget immediately

Current battle setting:

- `8`

### `epochs`

Meaning:

- number of passes over the synthetic corpus

Effect:

- more epochs sharpen the mapping
- too few underfit the lexicon

Current battle setting:

- `90`

### `learning_rate`

Meaning:

- step size for prompt classification and alignment updates

Effect:

- controls stability vs speed of convergence

Current battle setting:

- `0.06`

### `alignment_weight`

Meaning:

- how strongly misspelled surface forms are pulled toward canonical embeddings

Effect:

- higher values improve typo robustness
- too high can distort label geometry

Current battle setting:

- `0.45`

### `l2_weight`

Meaning:

- regularization on learned embeddings

Effect:

- limits uncontrolled drift

Default:

- `1e-4`

### `max_parameters`

Meaning:

- hard design constraint on total trainable parameters

Effect:

- constrains every architectural choice

Current cap:

- `10000`

Current battle model footprint at the time of capping:

- token vocab: `51`
- label vocab: `11`
- n-gram vocab: `129`
- parameter count: `1528`

## 2. Corpus dials

Source:

- [dataset.py](/Users/AI-CCORE/alter-programming/word_prediction_model/dataset.py)

### `repeats_per_entry`

Meaning:

- how many synthetic examples are generated per lexicon entry

Effect:

- increases rehearsal density
- stabilizes training on a tiny symbolic lexicon

Typical values:

- generic runs: `48` to `72`
- battle inquiry: `72`

### `seed` / `corpus_seed`

Meaning:

- controls synthetic prompt composition and example ordering

Effect:

- changes exact training mixtures
- useful for repeatability

### `ngram_bucket_count`

Meaning:

- number of stable hashed buckets for character n-grams

Effect:

- controls typo-surface resolution under a fixed parameter budget

Tradeoff:

- more buckets separate surface forms better
- fewer buckets save parameters

Battle setting:

- `128`

## 3. Lexicon dials

Sources:

- [lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/lexicon.py)
- [battle_lexicon.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_lexicon.py)

### Canonical vocabulary size

Meaning:

- how many distinct concepts the controller can express

Effect:

- larger vocab increases expressiveness
- too large can blur a small model

Battle vocabulary currently centers on words like:

- `attack`
- `finish`
- `switch`
- `stabilize`
- `preserve`
- `setup`
- `status`
- `risk`
- `scout`
- `wall`

### Descriptor overlap

Meaning:

- how much neighboring canonical words share prompt descriptors

Effect:

- more overlap creates softer concept boundaries
- less overlap creates sharper but more brittle classes

### Surface-form expansion

Meaning:

- explicit misspellings plus generated typo variants

Effect:

- governs how typo-aware the word space is

## 4. Battle prompt-construction dials

Source:

- [battle_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_inquiry.py)

These dials are not numeric arguments. They are rule choices.

### Question-token mapping

Examples:

- `switch` questions -> `pivot`, `retreat`, `swap`
- `attack` questions -> `damage`, `offense`, `pressure`
- `finish` questions -> `ko`, `lethal`, `secure`

Effect:

- determines which semantic cluster the inquiry enters

### State-to-token mapping

Examples:

- low own HP -> `save`, `protect`, `recover`
- low opponent HP -> `ko`, `end`, `pressure`
- strong boost state -> `boost`, `stack`, `pressure`

Effect:

- injects battle facts into the word model without retraining the model

### Prompt truncation

Current behavior:

- final prompt token list is deduplicated and truncated to `5`

Effect:

- keeps inference cheap
- forces prompt construction to choose only the strongest clues

## 5. Action decoder dials

Source:

- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)

This layer decides how predicted words become real actions.

### Word-to-move-role scores

Examples:

- `finish` rewards direct attacks and punishes setup/status/recover
- `status` rewards status moves
- `stabilize` rewards recovery

Effect:

- translates semantic intent into move-role preferences

### State heuristics

Examples:

- low HP increases recovery weight
- low opponent HP increases attacking weight
- setup and status are heavily restricted unless the state is unusually clean

Effect:

- prevents semantically plausible but strategically poor moves

### Type-aware bonuses

Source:

- [pokemon_type_utils.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pokemon_type_utils.py)

Effect:

- boosts super-effective attacks
- penalizes resisted or immune attacks
- adds a small STAB bonus

This is currently the most important strategic dial added after the original
heuristic policy.

### Switch thresholds

Current behavior:

- voluntary switch only if the predicted word is `switch` and HP is in a very
  low range
- no voluntary switching if the request says the active Pokemon is trapped

Effect:

- prevents the model from over-rotating
- enforces request legality

## 6. Serving and benchmark dials

Sources:

- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py)
- simulator benchmark runner

### Default question routing

Examples:

- low opponent HP -> `"can I knock it out now?"`
- low own HP with recovery -> `"what is the safe play?"`
- otherwise -> `"should I attack now?"`

Effect:

- heavily shapes which word the model predicts on live battle turns

### `RL_ALLOW_VOLUNTARY_SWITCHES`

Meaning:

- simulator-level control over whether move-turn voluntary switches are exposed

Observed effect:

- this had a major impact on battle performance

Current strong profile:

- `false`

### `TOTAL_GAMES`, `CONCURRENCY`, `BATTLE_TIMEOUT_MS`

Effect:

- benchmark scale and speed
- not policy quality directly

## 7. Which dials actually mattered most

From the benchmark history so far, the highest-impact dials were:

1. `RL_ALLOW_VOLUNTARY_SWITCHES`
2. type-aware move scoring
3. battle question routing
4. HP-based recovery / finish thresholds
5. parameter budget discipline

The lower-impact dials, relative to battle results, were:

1. small threshold tweaks without new information
2. minor changes to setup/status enthusiasm
3. smoke-run-only optimization

## 8. Reproducible strong profile

Current strongest validated profile before the 10,000-game run:

- battle model config:
  - `embedding_dim=8`
  - `epochs=90`
  - `learning_rate=0.06`
  - `alignment_weight=0.45`
  - `seed=17`
  - `max_parameters=10000`
- battle corpus:
  - `repeats_per_entry=72`
  - `seed=19`
  - `ngram_bucket_count=128`
- action profile:
  - voluntary switches disabled on move turns
  - forced switches enabled
  - type-aware move scoring enabled
  - STAB bonus enabled
  - low-HP recovery preserved

Qualification result for that profile:

- `159 / 200`
- `79.5%`

Scale benchmark result for the same profile:

- `7802 / 10000`
- `78.02%`
- `0` failed games
