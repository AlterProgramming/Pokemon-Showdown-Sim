# Handoff

## State

- Repo: `Pokemon Showdown Agent`
- Branch: `main`
- Remote: `origin/main`
- Current local work from the old sequence branch was carried onto `main` and conflict-resolved there.
- Narrow validation already passed on this state:
  - `.\.venv\Scripts\python.exe -m pytest tests/test_model_registry.py tests/test_sequence_planning.py tests/test_entity_server_runtime.py`

## What Changed

- Added a lightweight sequence-planning helper at `core/SequencePlanning.py`.
- Wired auxiliary sequence/value reranking into `server/serve_entity_model_benchmark.py`.
- Extended `server/EntityServerRuntime.py` and tests to expose auxiliary sequence outputs for rerank/inspect flows.
- Reconciled `flask_api_multi.py` so newer request metrics and the browser-bridge additions both survive on `main`.
- Carried forward the related PowerShell bridge/broadcast script changes in `scripts/`.

## Current Rerank Path

1. `core/BattleStateTracker.py`
   - `iter_turn_examples()` emits `turn_events_v1`, `action_token`, and `opponent_action_token`.
2. `core/EntityTensorization.py`
   - vectorization keeps a fixed-size public state and encodes `turn_events_v1` as auxiliary sequence targets.
3. `core/EntityModelV1.py`
   - the sequence head is action-conditioned and non-autoregressive: repeated context plus LSTM over fixed length.
4. `server/EntityServerRuntime.py`
   - decodes auxiliary outputs for candidate actions.
5. `server/serve_entity_model_benchmark.py`
   - `_select_best_action_with_auxiliary()` blends base policy logit, centered value, sequence score, and voluntary-switch penalty.

## Research Takeaway

Question:
`To what extent can selective retention of historical battle events compensate for the information loss of fixed-size state representations in sequential decision-making under memory constraints?`

Current repo answer:

- It compensates partially, not fully.
- The retained event history helps only when the predicted event tokens expose public short-horizon tactical consequences that the fixed state alone misses.
- The compensation ceiling is low because the retained history is heavily compressed:
  - fixed 12-slot public state
  - max 4 observed moves per slot
  - fixed candidate caps
  - sequence scoring only looks at a sparse subset of public outcomes such as `damage`, `heal`, `status`, `boost`, `unboost`, and `faint`
- The current sequence head is not a real long-memory planner. It predicts a compact public event trace from one state/action context, then rerank uses a heuristic score over that trace.
- Practically, this means selective retention is useful as a tactical biasing signal, but it does not recover the strategic information lost by the fixed-size representation.

## Important Caveat

- `docs/CLAUDE_ENTITY_SEQUENCE_AUX_HANDOFF.md` originally scoped the sequence head as training-only and explicitly said not to wire runtime reranking in that branch.
- The current working state does wire runtime reranking into the benchmark server, so future work should treat that as an intentional experimental divergence from the earlier handoff, not as a contradiction to ignore.

## Shared Conversation Context

- I was able to fetch the shared page and confirm the share title: `Innovative Sequence Benchmarking`.
- The share page did not expose plain conversation text directly in a simple fetch, so this handoff relies primarily on repo evidence rather than a verbatim reconstruction of that thread.

## Next Good Steps

- Run a controlled rerank comparison on `main` with fresh logs and replay exports.
- Measure how often auxiliary sequence scores are non-zero on candidate sets, especially on switch-heavy turns.
- Compare three modes:
  - base policy only
  - policy plus value
  - policy plus value plus sequence rerank
- Separate short-horizon tactical improvements from true long-memory gains; the current implementation mostly tests the former.
