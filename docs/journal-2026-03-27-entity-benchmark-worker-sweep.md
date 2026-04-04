# Journal Note: Entity Benchmark Bring-Up, Worker Sweep, and Stall Diagnosis

Date: 2026-03-27

## Purpose

This note records the first end-to-end benchmark bring-up for `entity_action_bc_v1_20260327_run2`, along with the later worker-tuned sweep against `model1` through `model4`.

The main goals were:

- verify that the new entity model could be served and benchmarked through the simulator
- compare its early strength against the older vector-family baselines
- tune worker count and runner concurrency for a fairer throughput setting
- identify whether timeout failures were caused by battle loops, inference stalls, or benchmark misconfiguration

This note is intended to be reusable context for future project exploration threads.

## Primary Artifacts

- Entity model run:
  - `artifacts/training_metadata_entity_action_bc_v1_20260327_run2.json`
- First random benchmark:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/random20-20260327-133905`
- First head-to-head vs `model4`:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/vs-model4-20-20260327-135148`
- Worker-tuned sweep:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648`
- Sweep summary files:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/sweep-summary.csv`
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/sweep-summary.json`

## Benchmark Configuration

The final sweep used the following settings:

- entity server vs baseline server
- minimum `2` baseline workers
- runner concurrency `2`
- battle timeout `120000 ms` (`2.00 min`)
- no replay capture for the sweep
- baseline worker profile kept model-specific, but not inflated beyond what the simulator could actually feed

This matters because earlier problems were partly benchmark-launch problems rather than model problems. Once the background entity server launcher was fixed, the simulator genuinely ran two battles in parallel and the comparison became much more trustworthy.

## Pre-Sweep Snapshot

Before the full sweep, two short checks gave useful context:

### 1. Entity model vs random, 20 games

Observed result:

- `7` wins, `13` losses
- win rate `35.00%`
- avg model request latency `420.15 ms`
- avg RL switches per game `13.70`

Interpretation:

- the entity model was operational, but still over-switching and not yet broadly strong
- this was a useful warning against assuming that better training metrics alone implied better universal play

### 2. Entity model vs `model4`, 20 games

Observed result:

- `18` completed games
- `17` wins for entity model
- `1` win for `model4`
- `2` timed out games
- completed-game win rate `94.44%`

Interpretation:

- the entity model could already exploit `model4` very effectively
- however, the timeout count meant this result should be treated as promising rather than final

## Worker-Tuned Sweep Results

The final sweep compared the entity model against `model1` to `model4` under the same concurrency and timeout settings.

### Summary Table

| Target | Completed | Entity Wins | Baseline Wins | Timed Out | Win Rate | Avg Turns | Avg Request Latency | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model1 | 17 | 14 | 3 | 3 | 82.35% | 37.47 | 479.29 ms | 1.27 games/min |
| model2 | 20 | 16 | 4 | 0 | 80.00% | 38.60 | 338.65 ms | 5.74 games/min |
| model3 | 20 | 15 | 5 | 0 | 75.00% | 45.15 | 203.70 ms | 7.20 games/min |
| model4 | 18 | 12 | 6 | 2 | 66.67% | 44.06 | 243.41 ms | 2.74 games/min |

### Switching Observations

The entity model remained switch-heavy, but still usually switched less than the older baselines:

- vs `model1`: entity `11.41` switches/game, baseline `20.71`
- vs `model2`: entity `11.60`, baseline `16.40`
- vs `model3`: entity `17.15`, baseline `24.90`
- vs `model4`: entity `14.67`, baseline `17.61`

Interpretation:

- the entity model has not solved looping or excess switching yet
- but it is already less extreme than several baseline models
- this likely contributes to its strong head-to-head performance even before belief-aware modeling is added

## Stall Diagnosis

The sweep showed two different failure modes, and they should not be treated as one problem.

### Case A: `model1` shows a real server-side inference stall

Evidence:

- `model1` request timeout setting at startup was `15.0s`
- nevertheless, one logged request completed after `208695.50 ms`
- file:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/model1/model1.server.out.log`
- exact lines:
  - line `4098`: `[predict-complete] ... service_ms=208695.50`
  - line `4099`: `[predict] ... elapsed_ms=208695.93`

Supporting evidence from the access log:

- file:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/model1/model1.server.err.log`
- the largest gap between successful `POST /predict` entries was `208s`
- specifically from `15:57:52` to `16:01:20`

Supporting evidence from the runner:

- file:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/model1/runner.log`
- the last live battle sat at `running=1` from about `9.76 min` until the runner jumped to `13.41 min`, then reported a timeout for game `20`

Interpretation:

- this is not just "a battle happened to be long"
- this is a real long-tail inference stall on the baseline `model1` server side
- the stall consumed more than three minutes on one request, even though nominal timeout settings were much lower

### Case B: `model4` timeouts look like long battles, not inference freezes

Evidence:

- file:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/model4/model4.server.out.log`
- the worst server-side request was only `618.63 ms`
- there were no multi-minute service times in that log

Supporting evidence from the runner:

- file:
  - `benchmark_runs/entity_action_bc_v1_20260327_run2/worker-tuned-sweep-20260327-154648/model4/runner.log`
- summary reported:
  - `3621` model requests
  - `3621 succeeded, 0 failed`
  - max request latency `922 ms`

Interpretation:

- `model4` timeout battles appear to be long or looping matchups under the strict two-minute battle cap
- they do not show the same inference-freeze signature as `model1`

## Why `model1` Can Exceed the Configured Request Timeout

There is an important implementation detail in `ModelWorkers.py`.

The supervisor path does not simply fail fast on timeout. Instead:

- `predict_with_metadata(...)` polls for the request timeout
- if the worker does not answer in time, it calls `_restart_locked("timeout")`
- `_restart_locked(...)` stops the worker and synchronously starts a replacement worker in the same request path

This means a timed-out request can still block much longer than the nominal request timeout, because restart and boot happen inline.

That alone does not fully explain a `208s` request, but it does explain why timeout behavior feels much stickier than the configuration suggests. The current design couples:

- request timeout handling
- worker teardown
- worker restart
- worker startup wait

all on the hot request path.

## Amortised Analysis

The most useful way to read this sweep is not only by win rate, but by amortised cost per completed battle and per turn.

The values below use the runner summaries:

- combined model requests
- cumulative model request time
- completed games
- total turns

### Amortised Cost Table

| Target | Requests per Completed Battle | Combined Inference Time per Completed Battle | Combined Inference Time per Turn |
| --- | ---: | ---: | ---: |
| model1 | 158.41 | 75.92 s | 2026.06 ms |
| model2 | 84.80 | 28.71 s | 743.78 ms |
| model3 | 100.05 | 20.37 s | 451.16 ms |
| model4 | 201.17 | 48.97 s | 1111.48 ms |

### Interpretation of the Amortised View

#### `model3` was the cleanest matchup

- best throughput: `7.20 games/min`
- lowest average request latency: `203.70 ms`
- lowest combined inference burden per turn: `451.16 ms`

This is the best current benchmark target if the goal is to compare behavior quality without severe infrastructure distortion.

#### `model2` was also stable

- no timeouts
- good throughput: `5.74 games/min`
- moderate inference burden per turn: `743.78 ms`

This is also a good target for near-term behavior iteration.

#### `model4` is expensive because of battle dynamics, not obvious worker freezing

- `201.17` model requests per completed battle is the largest of the sweep
- combined inference burden per turn is more than `1.1s`
- request success stayed perfect

This suggests the cost is being driven by long interaction sequences, frequent decisions, and matchup dynamics rather than outright inference failure.

#### `model1` is the only matchup where the amortised cost is obviously polluted by a real stall

- `75.92s` of combined inference time per completed battle is far too high for a matchup with only `37.47` turns on average
- combined inference burden per turn rises to about `2.03s`
- this is inconsistent with the ordinary few-hundred-millisecond request pattern elsewhere in the same run

In other words:

- `model1` is not just slower
- one or more stall events are dominating the amortised cost

## Main Conclusions

1. The entity benchmark path is operational.

The new family can be served, benchmarked, and compared against the older vector baselines without relying on the old flat-vector inference path.

2. The entity model is already competitive against the older baselines.

It beat every baseline in the sweep, with the strongest clean results against `model2` and `model3`.

3. The timeout story is not uniform.

- `model1` shows a real long-tail inference stall
- `model4` mostly shows long battles under the two-minute battle cap

4. The current worker timeout design is still too sticky.

Even when timeout logic exists, synchronous restart on the request path makes failures last much longer than they should.

5. `model2` and `model3` are the best short-term evaluation partners.

They give cleaner signal about model quality because they do not confound the comparison with heavy stall behavior.

## Recommended Follow-Up

Near-term follow-up should separate infrastructure repair from agent-improvement work:

- infrastructure:
  - make worker timeout handling fail fast instead of restarting inline
  - rerun `model1` after that change to see whether the 208s freeze disappears
- behavior:
  - continue using `model2` and `model3` for cleaner head-to-head evaluation
  - inspect `model4` timeouts as battle-loop cases rather than server-freeze cases

## Short Reusable Summary

`entity_action_bc_v1_20260327_run2` successfully entered benchmark play and beat `model1` through `model4` in a worker-tuned sweep, but the timeout behavior split into two different classes. Against `model1`, one baseline worker genuinely stalled for about `208.7s`, which inflated amortised cost and slowed the entire run. Against `model4`, timeouts looked like long or looping battles rather than frozen inference. The cleanest current evaluation targets are `model2` and `model3`, while the main infrastructure fix is to decouple timeout failure from synchronous worker restart in the baseline server path.
