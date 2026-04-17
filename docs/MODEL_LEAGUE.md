# Model League

This file is the canonical landing page for cross-model league results.

Use these paths:

- Latest machine-readable snapshot: [model_league_latest.json](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/docs/model_league_latest.json)
- Historical snapshots: `docs/model_league_*.json`

## Current Validated League

Date:
- `2026-04-17`

Run shape:
- `6` models
- `100` games per pairing
- `1500` total games
- `0` failures
- concurrency `10`
- wall time `390.685s`

Pool:
- `word_policy_v1`
- `entity_action_bc_v1_20260408_0428`
- `model1`
- `model2`
- `model4`
- `model5`

## Current Ranking

1. `word_policy_v1` `1751` Elo, `424-76`
2. `entity_action_bc_v1_20260408_0428` `1521` Elo, `271-229`
3. `model5` `1437` Elo, `205-295`
4. `model2` `1434` Elo, `203-297`
5. `model4` `1430` Elo, `200-300`
6. `model1` `1427` Elo, `197-303`

## Notes

- This is the current clean league baseline after stabilizing the shared vector-model serving path.
- The earlier `1000`-games-per-pair long run is not the canonical result because it accumulated serving failures before the stabilization fixes.
- `word_policy_v1` is served from the local path in `pokemon-showdown-model-feature`; the other models come from `Pokemon-Showdown-Sim`.

## Update Rule

When a league run is considered valid:

1. Copy the raw result JSON into a dated `docs/model_league_*.json` snapshot.
2. Update [model_league_latest.json](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/docs/model_league_latest.json) to match the newest validated run.
3. Refresh this file's summary if the pool, ranking, or benchmark shape changes.
