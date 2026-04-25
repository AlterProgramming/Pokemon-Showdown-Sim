from __future__ import annotations

import argparse
import atexit
from collections import deque
import logging
from datetime import datetime, timezone
import sys
import threading
import time
from threading import Lock
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one or more Pokemon Showdown policy models.")
    parser.add_argument("--repo-path", help="Training repo root containing artifacts and ActionLegality.py")
    parser.add_argument(
        "--mode",
        default="multi",
        help="Model ID to serve or 'multi' to load all discovered policy models.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=15.0,
        help="Hard timeout for a single model inference request before the worker is restarted.",
    )
    parser.add_argument(
        "--worker-max-requests",
        type=int,
        default=5000,
        help="Gracefully recycle a model worker after this many completed predictions. Set 0 to disable.",
    )
    parser.add_argument(
        "--worker-max-age-seconds",
        type=float,
        default=3600.0,
        help="Gracefully recycle a model worker after this many seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--disable-worker-recycling",
        action="store_true",
        help="Disable request-count and age-based worker recycling entirely.",
    )
    parser.add_argument(
        "--workers-per-model",
        type=int,
        default=1,
        help="Default number of inference worker processes to launch per loaded model.",
    )
    parser.add_argument(
        "--model-worker-overrides",
        default="",
        help="Per-model worker counts, for example 'model4=4,model2_large=2'.",
    )
    parser.add_argument(
        "--model-ids",
        default="",
        help="Comma-separated model ids to load when --mode multi is used, for example 'model4,model4_large'.",
    )
    parser.add_argument(
        "--worker-startup-timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum time to wait for an inference worker process to load its model and report ready.",
    )
    parser.add_argument(
        "--worker-bootstrap-timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum time to wait for a spawned worker process to boot and signal that it started running.",
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=0.0,
        help="Optional micro-batch collection window per model request. Set 0 to disable waiting.",
    )
    parser.add_argument(
        "--batch-max-size",
        type=int,
        default=1,
        help="Maximum micro-batch size per model worker call. Set 1 to disable batching.",
    )
    parser.add_argument(
        "--request-log-every",
        type=int,
        default=0,
        help="Log every Nth request. Set 0 to disable periodic request logs.",
    )
    parser.add_argument(
        "--request-log-slower-than-ms",
        type=float,
        default=1000.0,
        help="Always log requests slower than this threshold in milliseconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable Werkzeug access logging for every HTTP request.",
    )
    return parser.parse_args()


ARGS = parse_args()
REPO_PATH = Path(ARGS.repo_path).resolve() if ARGS.repo_path else Path(__file__).resolve().parent
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from ActionLegality import filter_legal_revive_targets, filter_legal_switches  # noqa: E402
from ModelRegistry import build_model_registry, parse_model_id_list, select_registered_models  # noqa: E402
from ModelWorkers import (  # noqa: E402
    ModelWorkerPool,
    load_runtime_artifacts,
    parse_worker_count_overrides,
    softmax,
)
from core.StateVectorization import opponent_team_composition_features  # noqa: E402


def vocab_uses_action_tokens(action_vocab: dict[str, int]) -> bool:
    return any(
        token.startswith(("move:", "switch:"))
        for token in action_vocab
        if token != "<UNK>"
    )


def normalize_move_name(move_name: str) -> str:
    return move_name.lower().replace(" ", "")


def build_move_tokens(action_vocab: dict[str, int], move_name: str) -> list[str]:
    normalized = normalize_move_name(move_name)
    if vocab_uses_action_tokens(action_vocab):
        return [f"move:{normalized}"]
    return [normalized]


def build_switch_tokens(action_vocab: dict[str, int], slot: int) -> list[str]:
    if vocab_uses_action_tokens(action_vocab):
        return [f"switch:{slot}"]
    return []


class InferenceBatcher:
    def __init__(self, worker_pool: ModelWorkerPool, *, batch_window_ms: float, batch_max_size: int) -> None:
        self.worker_pool = worker_pool
        self.batch_window_ms = max(0.0, float(batch_window_ms))
        self.batch_max_size = max(1, int(batch_max_size))
        self._condition = threading.Condition()
        self._closed = False
        self._pending: deque[dict[str, Any]] = deque()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{worker_pool.model_id}_batcher")
        self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def predict_with_metadata(self, state_vector: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
        record = {
            "state_vector": state_vector,
            "event": threading.Event(),
            "result": None,
            "error": None,
            "metadata": None,
        }
        with self._condition:
            if self._closed:
                raise RuntimeError("inference batcher is closed")
            self._pending.append(record)
            self._condition.notify()
        record["event"].wait()
        if record["error"] is not None:
            raise record["error"]
        return record["result"], record["metadata"]

    def _collect_batch_locked(self) -> list[dict[str, Any]]:
        while not self._pending and not self._closed:
            self._condition.wait()
        if self._closed and not self._pending:
            return []
        batch = [self._pending.popleft()]
        if self.batch_max_size <= 1:
            return batch
        deadline = time.perf_counter() + (self.batch_window_ms / 1000.0)
        while len(batch) < self.batch_max_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if not self._pending:
                self._condition.wait(timeout=remaining)
                continue
            batch.append(self._pending.popleft())
        while self._pending and len(batch) < self.batch_max_size:
            batch.append(self._pending.popleft())
        return batch

    def _run(self) -> None:
        while True:
            with self._condition:
                batch = self._collect_batch_locked()
            if not batch:
                return
            try:
                if len(batch) == 1:
                    logits, metadata = self.worker_pool.predict_with_metadata(batch[0]["state_vector"])
                    batch[0]["result"] = logits
                    batch[0]["metadata"] = metadata
                else:
                    logits_batch, metadata = self.worker_pool.predict_batch_with_metadata(
                        [record["state_vector"] for record in batch]
                    )
                    for index, record in enumerate(batch):
                        record["result"] = np.asarray(logits_batch[index], dtype=np.float32)
                        record["metadata"] = dict(metadata)
            except Exception as error:
                for record in batch:
                    record["error"] = error
            finally:
                for record in batch:
                    record["event"].set()


def load_models(mode: str) -> tuple[dict[str, dict[str, Any]], str]:
    registry = build_model_registry(REPO_PATH)
    registered_models = registry["models"]
    if not registered_models:
        raise FileNotFoundError(f"No runnable policy models found under {REPO_PATH / 'artifacts'}")

    model_ids = select_registered_models(
        registered_models,
        mode=mode,
        requested_model_ids=parse_model_id_list(ARGS.model_ids),
    )

    artifacts = {
        model_id: load_runtime_artifacts(REPO_PATH, registered_models[model_id])
        for model_id in model_ids
    }
    worker_overrides = parse_worker_count_overrides(ARGS.model_worker_overrides)
    model_count = len(model_ids)
    for model_id, artifact in artifacts.items():
        configured_worker_count = worker_overrides.get(model_id, ARGS.workers_per_model)
        worker_count = resolve_worker_count(
            configured_count=configured_worker_count,
            mode=mode,
            model_count=model_count,
        )
        max_requests_before_recycle, max_worker_age_seconds = resolve_recycle_settings(
            worker_count=worker_count,
            mode=mode,
            model_count=model_count,
            max_requests=ARGS.worker_max_requests,
            max_age_seconds=ARGS.worker_max_age_seconds,
        )
        worker_pool = ModelWorkerPool(
            repo_path=REPO_PATH,
            model_entry=registered_models[model_id],
            worker_count=worker_count,
            request_timeout_seconds=ARGS.request_timeout_seconds,
            max_requests_before_recycle=max_requests_before_recycle,
            max_worker_age_seconds=max_worker_age_seconds,
            bootstrap_timeout_seconds=ARGS.worker_bootstrap_timeout_seconds,
            startup_timeout_seconds=ARGS.worker_startup_timeout_seconds,
        )
        worker_pool.start()
        artifact["worker_pool"] = worker_pool
        artifact["configured_worker_count"] = int(configured_worker_count)
        artifact["effective_worker_count"] = int(worker_count)
        artifact["max_requests_before_recycle"] = int(max_requests_before_recycle)
        artifact["max_worker_age_seconds"] = float(max_worker_age_seconds)
        if ARGS.batch_max_size > 1:
            artifact["batcher"] = InferenceBatcher(
                worker_pool,
                batch_window_ms=ARGS.batch_window_ms,
                batch_max_size=ARGS.batch_max_size,
            )

    default_model_id = (
        str(registry["default_model_id"])
        if registry.get("default_model_id") in artifacts
        else model_ids[0]
    )
    return artifacts, default_model_id

MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {}
DEFAULT_MODEL_ID = ""
APP = Flask(__name__)
REQUEST_LOG_LOCK = threading.Lock()
REQUEST_COUNTER = 0
REQUEST_METRICS_LOCK = threading.Lock()
MODEL_REQUEST_METRICS: dict[str, dict[str, Any]] = {}


def recommended_worker_floor(*, mode: str, model_count: int) -> int:
    return 2 if mode == "multi" and model_count > 1 else 1


def resolve_worker_count(*, configured_count: int, mode: str, model_count: int) -> int:
    return max(int(configured_count), recommended_worker_floor(mode=mode, model_count=model_count))


def resolve_recycle_settings(
    *,
    worker_count: int,
    mode: str,
    model_count: int,
    max_requests: int,
    max_age_seconds: float,
) -> tuple[int, float]:
    if ARGS.disable_worker_recycling:
        return 0, 0.0
    if worker_count < 2 and mode == "multi" and model_count > 1:
        return 0, 0.0
    return int(max_requests), float(max_age_seconds)


def empty_request_metrics() -> dict[str, Any]:
    return {
        "request_count": 0,
        "success_count": 0,
        "error_count": 0,
        "move_count": 0,
        "switch_count": 0,
        "revive_count": 0,
        "none_count": 0,
        "avg_observed_batch_size": 0.0,
        "avg_queue_wait_ms": 0.0,
        "avg_service_ms": 0.0,
        "avg_worker_total_ms": 0.0,
        "avg_elapsed_ms": 0.0,
        "max_observed_batch_size": 0,
        "max_queue_wait_ms": 0.0,
        "max_service_ms": 0.0,
        "max_worker_total_ms": 0.0,
        "max_elapsed_ms": 0.0,
        "_total_batch_size": 0.0,
        "_total_queue_wait_ms": 0.0,
        "_total_service_ms": 0.0,
        "_total_worker_total_ms": 0.0,
        "_total_elapsed_ms": 0.0,
    }


def initialize_request_metrics() -> None:
    with REQUEST_METRICS_LOCK:
        MODEL_REQUEST_METRICS.clear()
        for model_id in MODEL_ARTIFACTS:
            MODEL_REQUEST_METRICS[model_id] = empty_request_metrics()


def record_request_metric(
    *,
    model_id: str,
    elapsed_ms: float,
    queue_wait_ms: float | None = None,
    service_ms: float | None = None,
    worker_total_ms: float | None = None,
    batch_size: int = 1,
    result_type: str | None = None,
    success: bool,
) -> None:
    with REQUEST_METRICS_LOCK:
        metrics = MODEL_REQUEST_METRICS.setdefault(model_id, empty_request_metrics())
        metrics["request_count"] += 1
        if success:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1

        if result_type == "move":
            metrics["move_count"] += 1
        elif result_type == "switch":
            metrics["switch_count"] += 1
        elif result_type == "revive":
            metrics["revive_count"] += 1
        elif result_type == "none":
            metrics["none_count"] += 1

        observed_batch_size = max(1, int(batch_size))
        metrics["_total_batch_size"] += float(observed_batch_size)
        metrics["max_observed_batch_size"] = max(metrics["max_observed_batch_size"], observed_batch_size)
        metrics["_total_elapsed_ms"] += float(elapsed_ms)
        metrics["max_elapsed_ms"] = max(metrics["max_elapsed_ms"], float(elapsed_ms))

        if queue_wait_ms is not None:
            metrics["_total_queue_wait_ms"] += float(queue_wait_ms)
            metrics["max_queue_wait_ms"] = max(metrics["max_queue_wait_ms"], float(queue_wait_ms))
        if service_ms is not None:
            metrics["_total_service_ms"] += float(service_ms)
            metrics["max_service_ms"] = max(metrics["max_service_ms"], float(service_ms))
        if worker_total_ms is not None:
            metrics["_total_worker_total_ms"] += float(worker_total_ms)
            metrics["max_worker_total_ms"] = max(metrics["max_worker_total_ms"], float(worker_total_ms))

        request_count = float(metrics["request_count"])
        metrics["avg_observed_batch_size"] = metrics["_total_batch_size"] / request_count
        metrics["avg_elapsed_ms"] = metrics["_total_elapsed_ms"] / request_count
        metrics["avg_queue_wait_ms"] = metrics["_total_queue_wait_ms"] / request_count
        metrics["avg_service_ms"] = metrics["_total_service_ms"] / request_count
        metrics["avg_worker_total_ms"] = metrics["_total_worker_total_ms"] / request_count


def snapshot_request_metrics() -> dict[str, dict[str, Any]]:
    with REQUEST_METRICS_LOCK:
        result: dict[str, dict[str, Any]] = {}
        for model_id, metrics in MODEL_REQUEST_METRICS.items():
            public_metrics = {
                key: value
                for key, value in metrics.items()
                if not key.startswith("_")
            }
            result[model_id] = public_metrics
        return result


def should_log_request(elapsed_ms: float) -> bool:
    if ARGS.request_log_slower_than_ms > 0 and elapsed_ms >= float(ARGS.request_log_slower_than_ms):
        return True
    if ARGS.request_log_every <= 0:
        return False
    global REQUEST_COUNTER
    with REQUEST_LOG_LOCK:
        REQUEST_COUNTER += 1
        return REQUEST_COUNTER % int(ARGS.request_log_every) == 0

BRIDGE_LOCK = Lock()
BRIDGE_MESSAGES = deque(maxlen=1000)
BRIDGE_NEXT_ID = 1


def bridge_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bridge_coerce_prediction_score(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def bridge_append_message(payload: dict[str, Any]) -> dict[str, Any]:
    global BRIDGE_NEXT_ID

    with BRIDGE_LOCK:
        record = {
            "id": BRIDGE_NEXT_ID,
            "timestamp_utc": bridge_now_iso(),
            "source": str(payload.get("source") or "codex"),
            "target": str(payload.get("target") or "browser"),
            "kind": str(payload.get("kind") or "text"),
            "session_id": payload.get("session_id") or payload.get("SessionId"),
            "value": payload.get("value") or payload.get("Value"),
            "value_kind": payload.get("value_kind") or payload.get("ValueKind"),
            "payload_hex": payload.get("payload_hex") or payload.get("PayloadHex"),
            "prediction_score": bridge_coerce_prediction_score(
                payload.get("prediction_score") if "prediction_score" in payload else payload.get("PredictionScore")
            ),
            "note": payload.get("note") or payload.get("Note"),
        }
        BRIDGE_MESSAGES.append(record)
        BRIDGE_NEXT_ID += 1

    return record


def bridge_messages_after(after_id: int, limit: int) -> list[dict[str, Any]]:
    with BRIDGE_LOCK:
        messages = [message for message in BRIDGE_MESSAGES if int(message["id"]) > after_id]
        if limit > 0:
            messages = messages[:limit]
        return messages


BRIDGE_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Codex Browser Bridge</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f3ec;
      --panel: #fffaf1;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --border: #d6cbb9;
      --shadow: rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 28%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 32px;
      letter-spacing: -0.03em;
    }
    .sub {
      color: var(--muted);
      margin-top: 6px;
    }
    .status {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.12);
      color: var(--accent);
      font-weight: 600;
      font-size: 14px;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px var(--shadow);
      padding: 18px;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
    }
    input, textarea, button {
      font: inherit;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    .row {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #e5e7eb;
      color: #111827;
    }
    .feed {
      display: grid;
      gap: 12px;
    }
    .message {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
      padding: 12px;
    }
    .message pre {
      margin: 8px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      header { align-items: start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Codex Browser Bridge</h1>
        <div class="sub">Bidirectional localhost message bus for Codex and browser sessions.</div>
      </div>
      <div class="status" id="status">Connecting...</div>
    </header>
    <div class="grid">
      <section class="card">
        <h2 style="margin-top:0">Send a message</h2>
        <label for="source">Source</label>
        <input id="source" value="browser" />
        <label for="target">Target</label>
        <input id="target" value="codex" />
        <label for="sessionId">Session ID</label>
        <input id="sessionId" placeholder="optional session id" />
        <label for="value">Value</label>
        <textarea id="value" placeholder="Enter text or a Codex session payload"></textarea>
        <label for="predictionScore">Prediction score</label>
        <input id="predictionScore" value="1.0" />
        <label for="kind">Kind</label>
        <input id="kind" value="text" />
        <div class="row">
          <button id="sendBtn">Send</button>
          <button class="secondary" id="clearBtn" type="button">Clear</button>
        </div>
      </section>
      <section class="card">
        <div class="row" style="justify-content:space-between; align-items:center; margin-top:0">
          <h2 style="margin:0">Recent messages</h2>
          <button class="secondary" id="refreshBtn" type="button">Refresh</button>
        </div>
        <div class="feed" id="feed" style="margin-top:16px"></div>
      </section>
    </div>
  </div>
  <script>
    const feedEl = document.getElementById("feed");
    const statusEl = document.getElementById("status");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const sourceEl = document.getElementById("source");
    const targetEl = document.getElementById("target");
    const sessionIdEl = document.getElementById("sessionId");
    const valueEl = document.getElementById("value");
    const predictionScoreEl = document.getElementById("predictionScore");
    const kindEl = document.getElementById("kind");
    let lastSeenId = 0;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function renderMessage(message) {
      const div = document.createElement("div");
      div.className = "message";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `#${message.id} ${message.timestamp_utc} ${message.source} -> ${message.target} ${message.kind} score=${message.prediction_score}`;
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(message, null, 2);
      div.appendChild(meta);
      div.appendChild(pre);
      return div;
    }

    function upsertMessages(messages) {
      if (!messages.length) {
        return;
      }
      for (const message of messages) {
        if (message.id > lastSeenId) {
          lastSeenId = message.id;
        }
        feedEl.prepend(renderMessage(message));
      }
    }

    async function fetchMessages() {
      const response = await fetch(`/bridge/messages?after_id=${lastSeenId}`);
      if (!response.ok) {
        throw new Error(`Fetch failed: ${response.status}`);
      }
      const payload = await response.json();
      upsertMessages(payload.messages || []);
      setStatus(`Connected. Latest id ${payload.latest_id || 0}`);
    }

    async function sendMessage() {
      const body = {
        source: sourceEl.value || "browser",
        target: targetEl.value || "codex",
        session_id: sessionIdEl.value || "",
        value: valueEl.value || "",
        prediction_score: Number(predictionScoreEl.value || "1"),
        kind: kindEl.value || "text",
      };
      const response = await fetch("/bridge/messages", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Send failed: ${response.status}`);
      }
      const message = await response.json();
      upsertMessages([message]);
      valueEl.value = "";
    }

    sendBtn.addEventListener("click", async () => {
      sendBtn.disabled = true;
      try {
        await sendMessage();
      } catch (error) {
        setStatus(String(error));
      } finally {
        sendBtn.disabled = false;
      }
    });

    clearBtn.addEventListener("click", () => {
      feedEl.innerHTML = "";
      lastSeenId = 0;
    });

    refreshBtn.addEventListener("click", async () => {
      try {
        await fetchMessages();
      } catch (error) {
        setStatus(String(error));
      }
    });

    async function poll() {
      try {
        await fetchMessages();
      } catch (error) {
        setStatus(String(error));
      }
    }

    poll();
    setInterval(poll, 1000);
  </script>
</body>
</html>
"""


def shutdown_workers() -> None:
    for artifact in MODEL_ARTIFACTS.values():
        batcher = artifact.get("batcher")
        if batcher is not None:
            batcher.close()
        worker_pool = artifact.get("worker_pool")
        if worker_pool is not None:
            worker_pool.close()


def log_loaded_models() -> None:
    print(f"[startup] mode={ARGS.mode}")
    print(f"[startup] default_model_id={DEFAULT_MODEL_ID}")
    print(f"[startup] request_timeout_seconds={ARGS.request_timeout_seconds}")
    print(f"[startup] worker_max_requests={ARGS.worker_max_requests}")
    print(f"[startup] worker_max_age_seconds={ARGS.worker_max_age_seconds}")
    print(f"[startup] disable_worker_recycling={ARGS.disable_worker_recycling}")
    print(f"[startup] workers_per_model={ARGS.workers_per_model}")
    print(f"[startup] request_log_every={ARGS.request_log_every}")
    print(f"[startup] request_log_slower_than_ms={ARGS.request_log_slower_than_ms}")
    print(f"[startup] model_worker_overrides={ARGS.model_worker_overrides or '(none)'}")
    print(f"[startup] worker_bootstrap_timeout_seconds={ARGS.worker_bootstrap_timeout_seconds}")
    print(f"[startup] worker_startup_timeout_seconds={ARGS.worker_startup_timeout_seconds}")
    print(f"[startup] batch_window_ms={ARGS.batch_window_ms}")
    print(f"[startup] batch_max_size={ARGS.batch_max_size}")
    print(f"[startup] requested_model_ids={ARGS.model_ids or '(auto)'}")
    print(f"[startup] supported_model_ids={', '.join(sorted(MODEL_ARTIFACTS.keys()))}")
    for model_id in sorted(MODEL_ARTIFACTS.keys()):
        artifacts = MODEL_ARTIFACTS[model_id]
        worker_health = artifacts["worker_pool"].health()
        worker_pids = ",".join(
            str(worker_entry["pid"])
            for worker_entry in worker_health["workers"]
            if worker_entry.get("pid") is not None
        )
        print(
            "[startup] "
            f"model_id={model_id} "
            f"model_path={artifacts['model_path']} "
            f"vocab_path={artifacts['vocab_path']} "
            f"configured_worker_count={artifacts['configured_worker_count']} "
            f"worker_count={worker_health['worker_count']} "
            f"effective_worker_max_requests={artifacts['max_requests_before_recycle']} "
            f"effective_worker_max_age_seconds={artifacts['max_worker_age_seconds']} "
            f"worker_pids={worker_pids or '(starting)'}"
        )
        for worker_entry in worker_health["workers"]:
            timings = worker_entry.get("last_startup_timings") or {}
            if timings:
                print(
                    "[startup-worker] "
                    f"model_id={model_id} "
                    f"worker_index={worker_entry['worker_index']} "
                    f"bootstrap_s={timings.get('process_bootstrap_seconds', 0.0):.2f} "
                    f"tf_import_s={timings.get('tensorflow_import_seconds', 0.0):.2f} "
                    f"model_load_s={timings.get('model_load_seconds', 0.0):.2f} "
                    f"ready_s={timings.get('worker_ready_seconds', 0.0):.2f} "
                    f"total_s={timings.get('total_startup_seconds', 0.0):.2f}"
                )


def choose_model_artifacts(request_data: dict[str, Any]) -> dict[str, Any]:
    requested_model_id = request_data.get("model_id") or DEFAULT_MODEL_ID
    artifacts = MODEL_ARTIFACTS.get(str(requested_model_id))
    if artifacts is None:
        raise KeyError(
            f"Unsupported model_id '{requested_model_id}'. "
            f"Supported values: {', '.join(sorted(MODEL_ARTIFACTS))}"
        )
    return artifacts


def predict_logits(
    model_artifacts: dict[str, Any],
    state_vector: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    batcher = model_artifacts.get("batcher")
    if batcher is not None:
        return batcher.predict_with_metadata(state_vector)
    worker_pool = model_artifacts["worker_pool"]
    if hasattr(worker_pool, "predict_with_metadata"):
        return worker_pool.predict_with_metadata(state_vector)
    return worker_pool.predict(state_vector), {}


def pick_best_slot_target(
    model_artifacts: dict[str, Any],
    logits: np.ndarray,
    slot_targets: list[dict[str, Any]],
    target_type: str,
) -> tuple[dict[str, Any] | None, float | None]:
    probs = softmax(logits)
    action_vocab = model_artifacts["action_vocab"]
    best_target = None
    best_prob = -1.0

    for target in slot_targets:
        slot = target.get("slot")
        if slot is None:
            continue
        for token in build_switch_tokens(action_vocab, int(slot)):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            probability = float(probs[idx])
            if probability > best_prob:
                best_prob = probability
                best_target = {"type": target_type, "payload": target, "token": token}

    if best_target is not None:
        return best_target, best_prob

    if not slot_targets:
        return None, None

    return {"type": target_type, "payload": slot_targets[0], "token": None}, None


def pick_best_action(
    model_artifacts: dict[str, Any],
    logits: np.ndarray,
    legal_moves: list[dict[str, Any]],
    legal_switches: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    probs = softmax(logits)
    action_vocab = model_artifacts["action_vocab"]
    best_action = None
    best_prob = -1.0

    for move in legal_moves:
        move_name = str(move.get("move") or move.get("id") or "")
        for token in build_move_tokens(action_vocab, move_name):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            probability = float(probs[idx])
            if probability > best_prob:
                best_prob = probability
                best_action = {"type": "move", "payload": move, "token": token}

    for switch in legal_switches:
        slot = switch.get("slot")
        if slot is None:
            continue
        for token in build_switch_tokens(action_vocab, int(slot)):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            probability = float(probs[idx])
            if probability > best_prob:
                best_prob = probability
                best_action = {"type": "switch", "payload": switch, "token": token}

    return best_action, best_prob


def validate_state_vector(model_artifacts: dict[str, Any], state_vector: Any) -> list[float]:
    if state_vector is None:
        raise ValueError("missing state_vector")
    if not isinstance(state_vector, list):
        raise ValueError("state_vector must be a list")

    expected_input_dim = model_artifacts["expected_input_dim"]
    if expected_input_dim is not None and len(state_vector) != expected_input_dim:
        raise ValueError(
            f"state_vector has length {len(state_vector)} but model '{model_artifacts['model_id']}' "
            f"expects {expected_input_dim}."
        )
    return state_vector


def build_response_context(model_artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model_artifacts["model_id"],
        "supported_model_ids": sorted(MODEL_ARTIFACTS.keys()),
    }


@APP.route("/bridge", methods=["GET"])
def bridge_page():
    return BRIDGE_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


@APP.route("/bridge/messages", methods=["GET", "POST"])
def bridge_messages():
    if request.method == "POST":
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(error="invalid JSON"), 400
        record = bridge_append_message(data)
        return jsonify(record)

    after_id_raw = request.args.get("after_id", "0")
    limit_raw = request.args.get("limit", "100")
    try:
        after_id = max(int(after_id_raw), 0)
        limit = max(int(limit_raw), 0)
    except ValueError:
        return jsonify(error="after_id and limit must be integers"), 400

    messages = bridge_messages_after(after_id, limit)
    latest_id = messages[-1]["id"] if messages else max(BRIDGE_NEXT_ID - 1, 0)
    return jsonify(messages=messages, latest_id=latest_id)


@APP.route("/health", methods=["GET"])
def health():
    worker_health = {
        model_id: artifacts["worker_pool"].health()
        for model_id, artifacts in MODEL_ARTIFACTS.items()
    }
    request_metrics = snapshot_request_metrics()
    overall_status = "ok" if all(entry["alive"] for entry in worker_health.values()) else "degraded"
    return jsonify(
        status=overall_status,
        mode=ARGS.mode,
        default_model_id=DEFAULT_MODEL_ID,
        supported_model_ids=sorted(MODEL_ARTIFACTS.keys()),
        worker_health=worker_health,
        request_metrics=request_metrics,
    )


def enrich_state_vector_with_team(state_vector, observed_opponent_team, perspective_player):
    """
    Augment state vector with opponent team composition if team info is provided.

    Args:
        state_vector: 582-dim base vector (numpy array or list)
        observed_opponent_team: list of {species: "..."} dicts or None
        perspective_player: "p1" or "p2"

    Returns:
        Augmented vector (678 dims if team provided, else 582 dims)
    """
    if not observed_opponent_team:
        return state_vector

    # Convert observed team to list of dicts expected by opponent_team_composition_features
    team_list = []
    for entry in observed_opponent_team:
        if isinstance(entry, dict):
            species = entry.get("species", "")
        else:
            species = ""
        mon_dict = {"species": species}
        team_list.append(mon_dict)

    # Ensure we have exactly 6 slots (pad if needed)
    while len(team_list) < 6:
        team_list.append({"species": ""})
    team_list = team_list[:6]

    # Compute 96-dim opponent team composition features
    features = opponent_team_composition_features(team_list, perspective_player, species_hash_dim=16)

    # Concatenate to base vector
    if isinstance(state_vector, list):
        return state_vector + features
    else:
        # If numpy array, convert to list, concatenate, convert back
        return list(state_vector) + features


@APP.route("/predict", methods=["POST"])
def predict():
    model_artifacts: dict[str, Any] | None = None
    started_at: float | None = None
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(error="invalid JSON"), 400

        model_artifacts = choose_model_artifacts(data)
        context = build_response_context(model_artifacts)
        state_vector = validate_state_vector(model_artifacts, data.get("state_vector"))

        legal_moves = data.get("legal_moves") or []
        revive_targets, revive_reason = filter_legal_revive_targets(
            data,
            data.get("legal_revives", []) or data.get("legal_switches", []) or [],
        )
        legal_switches, switch_reason = filter_legal_switches(data, data.get("legal_switches", []) or [])

        # Extract opponent team and perspective player from request
        observed_opponent_team = data.get("observed_opponent_team")
        perspective_player = data.get("perspective_player", "p1")

        # Enrich state vector with opponent team features if available
        state_vector = enrich_state_vector_with_team(state_vector, observed_opponent_team, perspective_player)

        started_at = time.perf_counter()
        logits, worker_metrics = predict_logits(model_artifacts, state_vector)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        worker_index = worker_metrics.get("worker_index")
        worker_pid = worker_metrics.get("worker_pid")
        queue_wait_ms = worker_metrics.get("queue_wait_ms")
        service_ms = worker_metrics.get("service_ms")
        batch_size = worker_metrics.get("batch_size", 1)
        if should_log_request(elapsed_ms):
            print(
                "[predict] "
                f"model_id={model_artifacts['model_id']} "
                f"worker_index={worker_index} "
                f"worker_pid={worker_pid} "
                f"batch_size={batch_size} "
                f"state_len={len(state_vector)} "
                f"queue_wait_ms={0.0 if queue_wait_ms is None else float(queue_wait_ms):.2f} "
                f"service_ms={0.0 if service_ms is None else float(service_ms):.2f} "
                f"elapsed_ms={elapsed_ms:.2f}"
            )

        if revive_targets:
            best_revive, best_prob = pick_best_slot_target(model_artifacts, logits, revive_targets, "revive")
            if best_revive is None:
                record_request_metric(
                    model_id=model_artifacts["model_id"],
                    elapsed_ms=elapsed_ms,
                    queue_wait_ms=queue_wait_ms,
                    service_ms=service_ms,
                    worker_total_ms=worker_metrics.get("total_ms"),
                    batch_size=batch_size,
                    result_type="none",
                    success=True,
                )
                return jsonify(
                    type="none",
                    note="revive target request detected but no legal revive targets were available",
                    revive_reason=revive_reason,
                    **context,
                )
            record_request_metric(
                model_id=model_artifacts["model_id"],
                elapsed_ms=elapsed_ms,
                queue_wait_ms=queue_wait_ms,
                service_ms=service_ms,
                worker_total_ms=worker_metrics.get("total_ms"),
                batch_size=batch_size,
                result_type="revive",
                success=True,
            )
            return jsonify(
                best_revive=best_revive["payload"],
                slot=best_revive["payload"].get("slot"),
                type="revive",
                probability=best_prob,
                action_token=best_revive["token"],
                revive_reason=revive_reason,
                **context,
            )

        best_action, best_prob = pick_best_action(model_artifacts, logits, legal_moves, legal_switches)
        if best_action is None:
            record_request_metric(
                model_id=model_artifacts["model_id"],
                elapsed_ms=elapsed_ms,
                queue_wait_ms=queue_wait_ms,
                service_ms=service_ms,
                worker_total_ms=worker_metrics.get("total_ms"),
                batch_size=batch_size,
                result_type="none",
                success=True,
            )
            return jsonify(
                best_action=None,
                type="none",
                note="no legal actions found in vocabulary",
                legal_moves=legal_moves,
                legal_switches=legal_switches,
                switch_reason=switch_reason,
                **context,
            )

        if best_action["type"] == "move":
            record_request_metric(
                model_id=model_artifacts["model_id"],
                elapsed_ms=elapsed_ms,
                queue_wait_ms=queue_wait_ms,
                service_ms=service_ms,
                worker_total_ms=worker_metrics.get("total_ms"),
                batch_size=batch_size,
                result_type="move",
                success=True,
            )
            return jsonify(
                best_move=best_action["payload"],
                type="move",
                probability=best_prob,
                action_token=best_action["token"],
                **context,
            )

        record_request_metric(
            model_id=model_artifacts["model_id"],
            elapsed_ms=elapsed_ms,
            queue_wait_ms=queue_wait_ms,
            service_ms=service_ms,
            worker_total_ms=worker_metrics.get("total_ms"),
            batch_size=batch_size,
            result_type="switch",
            success=True,
        )
        return jsonify(
            best_switch=best_action["payload"],
            slot=best_action["payload"].get("slot"),
            type="switch",
            probability=best_prob,
            action_token=best_action["token"],
            switch_reason=switch_reason,
            **context,
        )
    except KeyError as error:
        return jsonify(error=str(error), supported_model_ids=sorted(MODEL_ARTIFACTS.keys())), 400
    except ValueError as error:
        return jsonify(error=str(error), supported_model_ids=sorted(MODEL_ARTIFACTS.keys())), 400
    except TimeoutError as error:
        return jsonify(
            error=str(error),
            retryable=True,
            supported_model_ids=sorted(MODEL_ARTIFACTS.keys()),
        ), 503
    except Exception as error:
        if model_artifacts is not None:
            elapsed_ms = 0.0 if started_at is None else (time.perf_counter() - started_at) * 1000.0
            record_request_metric(
                model_id=model_artifacts["model_id"],
                elapsed_ms=elapsed_ms,
                result_type="error",
                success=False,
            )
        import traceback

        traceback.print_exc()
        return jsonify(error=f"Server error: {error}"), 500


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    MODEL_ARTIFACTS, DEFAULT_MODEL_ID = load_models(ARGS.mode)
    initialize_request_metrics()
    atexit.register(shutdown_workers)
    log_loaded_models()
    if not ARGS.access_log:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
    APP.run(host=ARGS.host, port=ARGS.port, debug=False, use_reloader=False, threaded=True)
