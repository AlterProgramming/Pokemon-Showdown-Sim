# Transport Upgrade Plan

## Current seam

The current word-policy request path is:

```text
RLAgentAI.receiveRequest
  -> buildModelState()
  -> RLModelClient.query()
  -> fetch(http://127.0.0.1:5010/predict)
  -> Flask/gunicorn policy_server
  -> choose_action_from_inquiry(...)
```

Relevant files:

- [rl-agent.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-agent.ts)
- [rl-model-client.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-model-client.ts)
- [policy_server.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_server.py)
- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)

Observed behavior from local benchmarks:

- server-side uncached compute is small, roughly sub-millisecond
- end-to-end model request latency is much larger, roughly 16-22 ms
- multi-worker gunicorn helped, but transport overhead still dominates

That means the best remaining speed wins come from removing or shrinking the process boundary.

## Option A: in-process policy inside the simulator process

### Goal

Eliminate HTTP entirely by making `RLModelClient` call a local policy implementation without `fetch`.

### Best-fit shape

Port the word-policy serving contract into Node so the simulator process can produce the same response shape locally:

```text
query(modelData)
  -> LocalWordPolicyAdapter.query(modelData)
  -> choose action
  -> return same JSON object shape used today
```

### Files to add/change

- [rl-model-client.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-model-client.ts)
  - split transport from client logic
  - support `transport: "http" | "local" | "ipc"`
- `pokemon-showdown-model-feature/sim/tools/rl-model-local.ts`
  - new local provider implementing the current response contract
- [rl-agent.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-agent.ts)
  - instantiate local transport when configured
- [battle_inquiry.py](/Users/AI-CCORE/alter-programming/word_prediction_model/battle_inquiry.py)
- [policy_adapter.py](/Users/AI-CCORE/alter-programming/word_prediction_model/policy_adapter.py)
  - these define behavior that would need a TypeScript port, or a narrower reimplementation of the same logic

### What has to move into Node

At minimum:

- default question selection
- prompt token builder
- word ranking path
- action decoder
- legality-preserving move/switch response shaping

### Practical implementation strategies

#### A1. Full TypeScript port

Reimplement the word model and adapter logic in TypeScript.

Pros:

- maximum speed
- no Python runtime needed
- no protocol overhead

Cons:

- highest implementation risk
- behavior drift versus Python unless parity tests are added

Estimated gain:

- likely `1.5x` to `3x` over current HTTP path
- rough throughput target: about `600` to `1000` games/min if the remaining simulator cost does not dominate

#### A2. Embedded Python interpreter from Node

Call Python directly from the Node process through an embedded runtime or native bridge.

Pros:

- avoids HTTP
- less logic duplication

Cons:

- operationally messy
- hard to debug and package
- not attractive compared with a persistent child process

Estimated gain:

- potentially strong, but not recommended for maintainability

### Recommendation on Option A

Only do this if the word-policy path is meant to stay long term and justify a TypeScript port. It has the best upside, but it is not the lowest-risk path.

## Option B: persistent local IPC worker

### Goal

Keep Python as the source of truth, but replace per-request HTTP with a long-lived local subprocess speaking over stdio or a local socket.

### Best-fit shape

```text
RLAgentAI.receiveRequest
  -> RLModelClient.query()
  -> IPCWordPolicyClient.send(modelData)
  -> long-lived Python worker
  -> choose_action_from_inquiry(...)
  -> send JSON response back over IPC
```

### Files to add/change

- [rl-model-client.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-model-client.ts)
  - turn into a transport wrapper
- `pokemon-showdown-model-feature/sim/tools/rl-model-ipc-client.ts`
  - maintains a single child process
  - assigns request ids
  - multiplexes concurrent requests
- [rl-agent.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-agent.ts)
  - allow `RL_MODEL_TRANSPORT=ipc`
- `word_prediction_model/ipc_policy_worker.py`
  - reads newline-delimited JSON from stdin
  - returns newline-delimited JSON on stdout
  - loads model once
  - reuses current Python adapter code directly

### Protocol

Use newline-delimited JSON first. It is simple and good enough.

Request:

```json
{"id":"abc123","type":"predict","payload":{...current modelData...}}
```

Response:

```json
{"id":"abc123","ok":true,"result":{...current /predict response...}}
```

Error:

```json
{"id":"abc123","ok":false,"error":"..."}
```

### Why this fits the current codebase

- the simulator already has one client abstraction point: [rl-model-client.ts](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/sim/tools/rl-model-client.ts)
- Python already has a persistent worker pattern in [ModelWorkers.py](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/core/ModelWorkers.py)
- the word-policy code already cleanly returns JSON-serializable responses

### Design details

#### B1. One shared worker per Node process

Start one Python child when `RLAgentAI` is created and reuse it for all requests in that Node process.

This is the simplest cut and should already remove:

- HTTP parsing
- Flask/gunicorn overhead
- repeated TCP stack work

#### B2. Small worker pool

If a single worker becomes a bottleneck, allow `RL_MODEL_IPC_WORKERS=N` and round-robin requests across a fixed child-process pool.

This mirrors the gunicorn worker idea, but with much lower framing overhead.

### Estimated gain

- likely `1.2x` to `2x` over current HTTP path
- rough throughput target: about `500` to `800` games/min

This is the best speed-to-risk tradeoff.

## Option C: different protocol, same architecture

Examples:

- gRPC
- raw TCP
- WebSocket
- Unix domain sockets behind an HTTP-like service

Estimated gain:

- likely only `1.1x` to `1.4x`

Reason:

- these still keep request serialization and the cross-process service boundary
- they help less than a direct local worker

## Recommended order

1. Build Option B first: persistent local IPC worker.
2. Measure end-to-end games/min against current 8-worker gunicorn.
3. Only consider Option A if IPC still leaves too much overhead.

## Minimal implementation plan for Option B

### Phase 1

- add `ipc_policy_worker.py` in the word model repo
- add `RL_MODEL_TRANSPORT` support in the Node client
- support `http` and `ipc` with the same returned action shape
- add a benchmark flag to select transport without changing agent logic

### Phase 2

- add request multiplexing with request ids
- add health ping and automatic child restart
- add structured stderr logging from the worker

### Phase 3

- add a small worker pool
- compare `1`, `2`, `4` IPC workers

## Verification plan

### Functional

- parity test: same request payload through HTTP and IPC returns the same action type and slot
- failure test: worker restart on broken pipe
- concurrency test: multiple simultaneous requests resolve to matching ids

### Performance

- direct microbenchmark on `RLModelClient.query`
- 100-game benchmark
- 200-game benchmark
- compare:
  - games/min
  - avg model request latency
  - avg RL decision time

## Final recommendation

If the target is maximum speed with reasonable implementation risk, use a persistent IPC worker next.

If the target is absolute minimum latency and the policy is stable enough to justify a port, move the policy in-process in TypeScript later.
