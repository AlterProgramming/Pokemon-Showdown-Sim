# Launchers

This folder groups the PowerShell launchers used to start training and server
processes from Windows.

- `start_entity_training.ps1`
- `start_entity_invariance_training.ps1`
- `start_entity_benchmark_server.ps1`: starts the unified model server on the
  default `5000` port.
- `run_entity_benchmark_server.ps1`: foreground wrapper for the unified model
  server, also defaulting to `5000`.
