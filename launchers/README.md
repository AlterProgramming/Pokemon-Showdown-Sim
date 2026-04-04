# Launchers

This folder groups the PowerShell launchers used to start training and server
processes from Windows.

These scripts avoid machine-specific paths. They resolve the repo root from the
launcher location and support optional environment overrides:

- `PS_AGENT_PYTHON`: Python interpreter to use
- `PS_AGENT_DATA`: local dataset directory or battle-log path
- `POKEMON_SHOWDOWN_REPO`: simulator repo path when a sibling `pokemon-showdown`
  checkout is not used

- `start_entity_training.ps1`
- `start_entity_invariance_training.ps1`
- `start_entity_benchmark_server.ps1`: starts the unified model server on the
  default `5000` port.
- `run_entity_benchmark_server.ps1`: foreground wrapper for the unified model
  server, also defaulting to `5000`.
