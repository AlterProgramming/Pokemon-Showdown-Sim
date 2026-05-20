# Benchmark Chronology

## Purpose

This document records the main benchmark phases of the word-model battle policy.
It is not a perfect lab notebook for every smoke run. It is the shortest useful
history of what changed, what broke, and what actually moved win rate.

## Tick Standard

- `1 tick = 5 seconds`

## Current reference point

The latest stable large-sample profile is documented separately in
[RESEARCH_SNAPSHOT_2026_04_09.md](/Users/AI-CCORE/alter-programming/word_prediction_model/RESEARCH_SNAPSHOT_2026_04_09.md).

Current best large-sample result:

- `sticky_move_run_003`
- `1768 / 2000`
- `88.40%`

## Phase 0: lexical prototype only

System shape:

- prompt words -> canonical word
- misspelled surface forms -> canonical word

Battle status:

- no battle loop integration
- no games played

Interpretation:

- the model existed as a semantic predictor, not yet as a policy

## Phase 1: battle inquiry model

System shape:

- battle question + battle state -> battle word
- examples: `switch`, `finish`, `risk`, `preserve`

Battle status:

- inquiry responses worked
- still not choosing actual simulator actions

Interpretation:

- this phase proved the word model could serve as a battle-language controller
- it still did not prove battle strength

## Phase 2: first live battle wiring

System shape:

- ranked battle words decoded into moves and switches
- local policy server exposed over HTTP to the simulator

Early smoke benchmark:

- `1 / 10`
- `10.00%` win rate

Observed issue:

- excessive switching
- weak move quality

Interpretation:

- the stack was connected, but the decoder was not yet strong enough

## Phase 3: reduced switching and better question routing

Key changes:

- stricter voluntary switch thresholds
- improved default question routing:
  - `finish` for low opponent HP
  - `setup` for healthy setup windows
  - `status` when status spread was available

Smoke result:

- `3 / 10`
- `30.00%` win rate

Interpretation:

- switching discipline mattered
- the model could clear weak-policy failure modes, but was still not strong

## Phase 4: weighted move scoring

Key changes:

- move choice used the full ranked word output instead of only the top word

Smoke result:

- `3 / 10`
- `30.00%` win rate

Interpretation:

- using more of the word distribution helped stability somewhat
- it did not produce a major performance jump by itself

## Phase 5: state-aware move scoring

Key changes:

- move scoring became aware of:
  - own HP
  - opponent HP
  - opponent status
  - boost level
  - move role keywords

Smoke result:

- `6 / 10`
- `60.00%` win rate

Important warning:

- this was still a `10`-game smoke sample
- it showed upside, not robust performance

Interpretation:

- move quality was becoming more important than switch behavior

## Phase 6: legality failure discovered at larger scale

Qualification attempt:

- `200`-game benchmark

Failure:

- invalid voluntary switch while trapped

Root cause:

- the policy could emit a switch even when the request said the active Pokemon
  was trapped

Fix:

- enforce request-level `trapped` / `maybeTrapped`
- disable voluntary switching in those cases

Interpretation:

- this was a safety and contract bug, not a strategic one
- no larger benchmark was trustworthy until this was fixed

## Phase 7: post-fix mixed-policy qualification

After the trapped-switch fix and further heuristic tuning:

`50`-game result:

- `22 / 50`
- `44.00%`

Tuned retry:

- `23 / 50`
- `46.00%`

`200`-game result:

- `75 / 200`
- `37.50%`

Interpretation:

- the mixed move-and-switch policy was stable
- it was not robustly winning
- the earlier smoke highs were largely variance

## Phase 8: voluntary switches suppressed on move turns

Key change:

- `RL_ALLOW_VOLUNTARY_SWITCHES=false`

Effect:

- the policy still handled forced switches
- but it no longer tried to rotate opportunistically during move turns

Initial smoke result:

- `26 / 50`
- `52.00%`

Later `200`-game result before type-aware scoring:

- `84 / 200`
- `42.00%`

Interpretation:

- suppressing voluntary switching clearly helped
- but by itself it was not enough to create a strong large-sample policy

## Phase 9: attack-first simplification

Key change:

- further reduced setup and status enthusiasm
- pushed the policy toward direct attacking lines

Smoke result:

- `22 / 50`
- `44.00%`

Interpretation:

- blind simplification was not enough
- the decoder needed more information, not just harsher thresholds

## Phase 10: robust benchmark profile

Key result:

- `7802 / 10000`
- `78.02%`
- `0` failed games

Interpretation:

- the architecture was no longer merely promising
- it was stable at scale

## Phase 11: threat-aware large-sample improvement

Key result:

- `8319 / 10000`
- `83.19%`
- `0` failed games

Interpretation:

- revealed opponent move information produced a real large-sample gain
- the policy became meaningfully stronger without losing benchmark stability

## Phase 12: current 87 percent wall

Best clean `1000`-game result so far:

- `870 / 1000`
- `87.00%`
- `0` failed games

What happened next:

- retuning attempts were valid but regressed to `86.30%`
- broad opponent-role modeling regressed to `84.60%`
- a narrower estimated-lethal layer reached `86.70%`

Interpretation:

- the project crossed from missing-feature gains into interaction-limited gains
- the remaining work is less about adding broad capability and more about
  targeted regime-specific refinement

## Phase 10: type-aware move scoring

Key changes:

- local species typing lookup
- local move typing lookup
- type-effectiveness bonus / penalty
- STAB bonus

Data sources:

- local Pokédex data
- local move/type diagnostic tables

Smoke result:

- `31 / 50`
- `62.00%`

Qualification result:

- `159 / 200`
- `79.50%`

Failure count:

- `0`

Interpretation:

- this was the first change that looked like a real strategic upgrade
- the policy stopped being mostly lexical-plus-thresholds and became lexical
  plus symbolic battle knowledge

## Phase 11: scale run

Scale benchmark:

- `10000` games
- move-turn voluntary switches disabled
- type-aware move scoring enabled
- final result: `7802 / 10000`
- final win rate: `78.02%`
- failed games: `0`
- timed out games: `0`
- elapsed wall time: `20.38 min`
- throughput: `490.73 games/min`

Live log:

- [word_policy_10000.log](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/benchmarks/word_policy_10000.log)

Why this profile was chosen:

- it had the best large-sample qualification result so far
- it had `0` failed games
- it had a strong enough win rate to justify scale validation

Interpretation:

- the scale run confirmed the qualification result instead of collapsing under
  larger sample size
- the profile is now benchmarked, not just promising

## Main lessons from the chronology

1. Small smoke runs were useful for direction but unreliable for final judgment.
2. Safety and legality had to be fixed before strength mattered.
3. Voluntary switching was a major weakness for this model family.
4. The large performance jump came from symbolic battle priors, especially type
   awareness.
5. The word model works best as a semantic controller, not as the sole source
   of battle intelligence.
