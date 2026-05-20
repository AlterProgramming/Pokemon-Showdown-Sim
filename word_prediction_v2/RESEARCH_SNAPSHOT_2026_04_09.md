# Research Snapshot 2026-04-09

## Current stable benchmark

The strongest currently validated profile is the sticky-move branch:

- recorded run: `sticky_move_run_003`
- result: `1768 / 2000`
- win rate: `88.40%`
- failed games: `0`
- timed out games: `0`

Primary artifact:

- [benchmark.log](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/recorded_runs/sticky_move_run_003/benchmark.log)

This is the best current large-sample result that survived beyond a smoke run.

## What improved recently

The biggest recent gain came from a narrow move-decoder change, not a new
semantic model:

- stronger penalty for sticky repeated attack reuse
- extra demotion for repeated `Knock Off` after item removal
- no reopening of broad voluntary switching

Large-sample effect:

- previous replay-fix branch: `870 / 1000 = 87.00%`
- sticky-move branch: `883 / 1000 = 88.30%`
- sticky-move branch at larger scale: `1768 / 2000 = 88.40%`

The most important validation is that the gain held when scaled from `1000` to
`2000` games.

## What the loss analysis now says

The largest surviving failure buckets are still move-turn problems rather than
switching or hazard logic.

From [graph_summary.json](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/recorded_runs/sticky_move_run_003/graph_summary.json):

- `attack_edge = 5232`
- `opponent_farming = 2292`
- `low_hp_active = 1176`
- `closeout_window = 902`

From [broad_loss_taxonomy.json](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/recorded_runs/sticky_move_run_003/broad_loss_taxonomy.json):

- `repeat_loop = 907`
- `generic_attack_loop = 262`
- `priority_choice = 320`
- `item_progress_choice = 248`
- `frequent_low_hp_state = 232`
- `recover_choice = 58`
- `hazard_choice = 17`
- `setup_choice = 7`

Interpretation:

- hazards are not the main ceiling
- setup is not the main ceiling
- broad switching is not the main ceiling
- the remaining wall is mostly in attack-mode conversion and endgame handling

## Sticky-loop evidence

The sticky-move work targeted a real repeated-family problem.

Comparing the stronger sticky branch against the earlier replay-fix branch:

- `earthquake -> earthquake`: down by `91`
- `shadowball -> shadowball`: down by `84`
- `playrough -> playrough`: down by `63`
- `icebeam -> icebeam`: down by `60`
- `drainpunch -> drainpunch`: down by `50`

The targeted move family improved, and the aggregate win rate moved up with it.

Important remaining repeat families:

- `earthquake`
- `knockoff`
- `thunderbolt`
- `closecombat`
- `bravebird`

So the recent gain was real, but it did not eliminate the broader repeat-loop
class.

## Priority closeout findings

The next hypothesis was that the policy underused priority in low-HP closeout
states.

That insight was directionally correct.

From [priority_closeout_summary.json](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/recorded_runs/sticky_move_run_003/priority_closeout_summary.json)
on the sticky-move branch:

- priority available in closeout: `325`
- priority used in closeout: `178`
- priority not used in closeout: `147`

Frequent non-priority choices instead:

- `closecombat`
- `knockoff`
- `earthquake`
- `bravebird`
- `wavecrash`

So the symptom exists.

## What failed

Two different closeout patches were tested after that diagnosis.

### Broad priority-closeout patch

This broadly pushed the policy toward priority when a priority move existed and
the target was already low.

Results:

- `priority_closeout_run_002`: `425 / 500 = 85.00%`

Conclusion:

- it fixed some intended cases
- but it degraded general move quality too much

### Narrow risky-closeout patch

This only demoted risky non-priority finish attempts:

- recoil or crash-style lines
- defense-dropping lines like `Close Combat`
- self-drop lines or inaccurate finish attempts

It looked better on smaller samples:

- `priority_closeout_run_004`: `448 / 500 = 89.60%`

But failed at scale:

- `priority_closeout_run_005`: `869 / 1000 = 86.90%`

Primary artifact:

- [comparison_vs_sticky_move_run_002.json](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/recorded_runs/priority_closeout_run_005/comparison_vs_sticky_move_run_002.json)

This comparison shows the important tradeoff:

- priority closeout usage improved materially
- but the branch reopened broader attack-loop and farming failures

The key lesson is:

- the priority insight is correct
- the implementation still overcorrects and harms the broader decoder

## Current best interpretation

The policy is now strong enough that improvements mostly come from narrow
decoder adjustments, but those changes can still interact badly.

Current evidence supports:

- keep the sticky-move branch as the best benchmarked profile
- treat low-HP/priority handling as a real but unresolved secondary ceiling
- prefer narrow, survivable decoder fixes over broad semantic or switching
  rewrites

## Recommended next direction

The next likely gain is not:

- hazard logic
- setup logic
- broad voluntary switching

It is more likely one of:

- stricter repeated `Knock Off` / `Thunderbolt` timing
- better distinction between safe strong finish and risky strong finish
- narrower low-HP conversion logic that does not distort general attack choice

That should be pursued from the sticky-move baseline, not from the failed broad
priority branch.
