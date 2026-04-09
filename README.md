# Pokemon-Showdown-Sim
A simulator to test model capabilities in real environment

## Roadmap

The current model-family and training-generation roadmap lives in:

`docs/GENERATION_MAP.md`

Artifact governance and completeness can be audited with:

`tools/audit_artifacts.py`

Historical runs can be backfilled with manifests and evaluation summaries using:

`tools/backfill_governance_artifacts.py`

The concrete first entity-centric family contract lives in:

`docs/ENTITY_ACTION_V1.md`

The identity-invariance follow-on family contract lives in:

`docs/ENTITY_INVARIANCE_AUX_V1.md`

The runnable trainer and launcher for that family live in:

- `train_entity_invariance.py`
- `launchers/start_entity_invariance_training.ps1`

## Repository Layout

The repo is grouped to keep the root focused on runnable code:

- `docs/` holds the roadmap, family contracts, benchmark notes, and the run journal.
- `notebooks/` holds exploratory notebooks.
- `core/` holds the shared battle-state, tensorization, model, registry, and training helpers.
- `server/` holds the server runtime and benchmark implementations.
- `tools/` holds audits, replay utilities, and visualizations.
- `launchers/` holds PowerShell entrypoints.
- `data/` holds raw JSON inputs used by exploratory scripts.
- `artifacts/legacy/` holds the older root-level model blobs.
- `artifacts/`, `logs/`, `benchmark_runs/`, and `out/` hold generated outputs.
- Root-level `*.py` files are mostly runnable entrypoints plus thin compatibility shims for the modules that now live under `core/`, `server/`, and `tools/`.

If you want the doc index, start with `docs/README.md`.
If you want the shared Python module map, start with `core/README.md`.

## Simulator Repo

This repo is meant to be used alongside a sibling `pokemon-showdown` checkout.
Recommended layout:

```text
pokemon-workspace/
  Pokemon-Showdown-Sim/
  pokemon-showdown/
```

If you keep the two repos somewhere else, set the simulator path explicitly in
your own shell scripts or environment, for example with `POKEMON_SHOWDOWN_REPO`.

Relevant simulator-side files we have been integrating with:

- `sim/tools/rl-model-client.ts`
- `sim/tools/rl-agent.ts`
- `sim/tools/runner.ts`
- `sim/tools/model-league-runner.ts`
- `sim/examples/model-vs-model-runner.ts`
- `sim/examples/statistical-runner.ts`

The built JavaScript runner artifact for statistical runs lives at:

- `dist/sim/examples/statistical-runner.js`

## Model Server

`flask_api_multi.py` is the unified model-server entrypoint. It now routes both flat
`state_vector` payloads and entity `battle_state` payloads on the same port, so the
request shape no longer forces a port switch.

The single-model entity benchmark server lives in `server/serve_entity_model_benchmark.py`.

For the Intel Mac handoff path, use the containerized serving flow in:

- `docs/INTEL_MAC_CONTAINER_RUNBOOK.md`

## Benchmarking

For model-vs-model runs, use the simulator-side wrapper in the sibling repo:

- `scripts/benchmark-model-vs-model.ps1`

The recorded replay grid is driven by `-ReplayGrid`, and replay capture is enabled with
`-ReplayCaptureMode` plus `-ReplayCaptureCount`. This example assumes:

- your current shell has `POKEMON_SHOWDOWN_REPO` set to the simulator checkout, or
- you are running it from within the sibling `pokemon-showdown` repo

It uses the runbook's `model2` vs `model4` pairing and prefers port `5009` if it is free:

```powershell
$showdownRepo = if ($env:POKEMON_SHOWDOWN_REPO) {
    $env:POKEMON_SHOWDOWN_REPO
} else {
    (Resolve-Path "..\pokemon-showdown").Path
}
Set-Location $showdownRepo

$port = 5009
while (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
    $port++
}
Write-Host "Using server port $port"

& .\scripts\benchmark-model-vs-model.ps1 `
    -ServerHost '127.0.0.1' `
    -ServerPort $port `
    -ServerModelIDs 'model2,model4' `
    -TotalGames 10 `
    -Concurrency 2 `
    -WorkersPerModel 2 `
    -ReplayCaptureMode all `
    -ReplayCaptureCount 10 `
    -ReplayOutputDir 'logs\replays' `
    -ReplayGrid `
    -ModelAName 'Model2' `
    -ModelAID 'model2' `
    -ModelAProfile 'joint-policy' `
    -ModelBName 'Model4' `
    -ModelBID 'model4' `
    -ModelBProfile 'joint-policy-value'
```

The matching runbook is here:

- `BENCHMARK_RUNBOOK.md`

## Desktop Utilities

There are also a few Windows PowerShell utilities under `scripts/` for desktop-only workflows:

- `scripts/get_codex_session_id.ps1`: prints the active Codex desktop conversation ID from the newest local app log
- `scripts/broadcast_ble_value.ps1`: advertises a compact value over Bluetooth Low Energy, including the current Codex session ID
- `scripts/scan_ble_value.ps1`: listens for the matching BLE broadcast format on another Windows device

Broadcast the current Codex session ID for two minutes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\broadcast_ble_value.ps1 -UseCodexSessionId
```

Listen for it on another Windows device:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\scan_ble_value.ps1 -FirstOnly
```

These utilities use BLE advertising rather than Bluetooth pairing. The payload is intentionally small, so text broadcasts must stay short.

## Local Setup

These launchers are portable if you either keep standard sibling repos or set a
few environment variables yourself:

- `POKEMON_SHOWDOWN_REPO`: path to the simulator repo if it is not a sibling folder
- `PS_AGENT_PYTHON`: path to the Python interpreter or venv to use for this repo
- `PS_AGENT_DATA`: optional path to a local battle-log dataset; if omitted, the
  trainers can fall back to their dataset-download logic

Do not rely on any old machine-local `deepLearning` folder name. Each user
should create their own virtual environment for this repo.

Example on Windows PowerShell:

```powershell
Set-Location .\Pokemon-Showdown-Sim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then either:

- set `PS_AGENT_PYTHON` to `.\.venv\Scripts\python.exe`, or
- let the launchers auto-detect `.\.venv\Scripts\python.exe`

If your environment lives somewhere else, point `PS_AGENT_PYTHON` at that
interpreter explicitly.
