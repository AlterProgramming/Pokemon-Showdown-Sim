# Intel Mac Container Runbook

This is the supported low-friction path for running the Python model server alongside `pokemon-showdown` on an Intel Mac.

## Why this path

- The simulator can run natively on macOS.
- The Python server depends on `tensorflow==2.20.0`.
- For the Intel Mac handoff, the stable path is to keep this repo in a Linux-compatible container or VM and expose the same local HTTP contract back to the simulator.

## Pinned serving stack

- Python `3.12.x`
- Flask `3.1.3`
- Keras `3.13.2`
- TensorFlow `2.20.0`

These are pinned in `requirements-serving.txt`.

## Build the image

From this repo:

```bash
docker build -t ps-agent-inference .
```

The image intentionally excludes `artifacts/`, `logs/`, `benchmark_runs/`, and other local outputs. Bring model artifacts in at runtime with a volume mount.

## Required artifact bundle

At minimum, the mounted `artifacts/` directory should contain:

- runnable model files such as `model_4.keras` or `entity_action_bc_v1_*.keras`
- matching vocab files such as `action_vocab_*.json` or `*.policy_vocab.json`
- matching metadata files such as `training_metadata_*.json`
- any auxiliary files referenced by the metadata, such as `policy_value_model_*.keras`, token vocab files, or reward profiles
- `model_registry.json` if you already generated one

Notes:

- `flask_api_multi.py` can rebuild the registry from training metadata, so `model_registry.json` is recommended but not strictly required.
- Keep artifact paths inside metadata relative to the repo, for example `artifacts/model_4.keras`, so the same bundle works on a different machine.

## Start the container

Mount the host artifact directory into `/app/artifacts` and publish the model server on `127.0.0.1:5000`:

```bash
docker run --rm \
  -p 127.0.0.1:5000:5000 \
  -v /absolute/path/to/Pokemon\ Showdown\ Agent/artifacts:/app/artifacts:ro \
  ps-agent-inference
```

To load a subset of models:

```bash
docker run --rm \
  -p 127.0.0.1:5000:5000 \
  -v /absolute/path/to/Pokemon\ Showdown\ Agent/artifacts:/app/artifacts:ro \
  ps-agent-inference \
  --mode multi \
  --model-ids model2,model4
```

## Health and predict checks

Health:

```bash
curl http://127.0.0.1:5000/health
```

Vector predict smoke test:

```bash
python - <<'PY'
import json
import urllib.request

payload = {
    "model_id": "model2",
    "state_vector": [0.0] * 582,
}
req = urllib.request.Request(
    "http://127.0.0.1:5000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=10) as response:
    print(response.read().decode("utf-8"))
PY
```

## Contract back to pokemon-showdown

The simulator expects:

- `GET /health`
- `POST /predict`

Default local endpoint:

- `http://127.0.0.1:5000/predict`

This endpoint is consumed by:

- `node pokemon-showdown browser-model-bridge --model-endpoint http://127.0.0.1:5000/predict`
- the direct benchmark commands in `pokemon-showdown/INTEL_MAC_HANDOFF.md`
- local `model-league` configs pointing at `http://127.0.0.1:5000/predict`
