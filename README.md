# Pokemon-Showdown-Sim
A simulator to test model capabilities in real environment

## Repository Layout

The repo root now stays focused on runnable training and serving entrypoints.
Supporting material is grouped into folders:

- `core/` for shared training, state-tracking, reward, and serving helpers
- `tools/` for standalone analysis utilities
- `docs/` for notes that used to live in the root
- `notebooks/` for exploratory notebooks
- `data/` for raw reference JSON inputs
- `artifacts/legacy/` for the older root-level model blobs kept for history

Root-level modules such as `BattleStateTracker.py` and `ModelRegistry.py` remain
as compatibility shims so existing imports and scripts keep working while the
real implementations live under `core/`.

## Entity Action Baseline

The first entity-centric behavior-cloning family now lives in:

- `docs/ENTITY_ACTION_V1.md`
- `train_entity_action.py`
- `core/EntityActionV1.py`
- `core/EntityTensorization.py`
- `core/EntityModelV1.py`

The next identity-invariance phase now lives in:

- `docs/GENERATION_MAP.md`
- `docs/ENTITY_INVARIANCE_AUX_V1.md`
- `train_entity_invariance.py`
- `core/EntityInvarianceTensorization.py`
- `core/EntityInvarianceModelV1.py`

## Simulator Repo

The Pokemon Showdown simulator repo used alongside this project lives at:

`C:\Users\jeanj\Documents\Programming\pokemon-showdown`

Relevant simulator-side files we have been integrating with:

- `C:\Users\jeanj\Documents\Programming\pokemon-showdown\sim\tools\rl-model-client.ts`
- `C:\Users\jeanj\Documents\Programming\pokemon-showdown\sim\tools\rl-agent.ts`
- `C:\Users\jeanj\Documents\Programming\pokemon-showdown\sim\examples\model-vs-model-runner.ts`
- `C:\Users\jeanj\Documents\Programming\pokemon-showdown\dist\sim\examples\statistical-runner.js`
