from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from threading import Condition, Lock, Thread

from flask import Flask, jsonify, request
from werkzeug.serving import run_simple

from .battle_inquiry import get_battle_model
from .battle_questions import _allow_voluntary_switches, default_question_for_state
from .policy_adapter import choose_action_from_inquiry, choose_actions_from_inquiry_batch


app = Flask(__name__)
_MODEL_LOCK = Lock()
_CACHED_MODEL = None
_MICRO_BATCHER = None
_RESPONSE_CACHE = None
_SERVER_STATS = None
MICRO_BATCH_WINDOW_MS = float(os.environ.get("WORD_POLICY_MICRO_BATCH_WINDOW_MS", "2.0"))
MICRO_BATCH_MAX_SIZE = int(os.environ.get("WORD_POLICY_MICRO_BATCH_MAX_SIZE", "32"))
RESPONSE_CACHE_MAX_SIZE = int(os.environ.get("WORD_POLICY_RESPONSE_CACHE_MAX_SIZE", "4096"))
SERVER_HOST = os.environ.get("WORD_POLICY_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("WORD_POLICY_PORT", "5010"))
SERVER_WORKERS = int(os.environ.get("WORD_POLICY_WORKERS", "1"))
STATS_DIR = Path(os.environ.get("WORD_POLICY_STATS_DIR", "/tmp/word_policy_server_stats"))
STATS_FLUSH_EVERY = int(os.environ.get("WORD_POLICY_STATS_FLUSH_EVERY", "100"))

def _get_server_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        with _MODEL_LOCK:
            if _CACHED_MODEL is None:
                _CACHED_MODEL, _ = get_battle_model()
    return _CACHED_MODEL


class _ResponseCache:
    def __init__(self, *, max_size: int) -> None:
        self.max_size = max(1, max_size)
        self._lock = Lock()
        self._entries: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> dict | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return dict(value)

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._entries[key] = dict(value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _get_response_cache() -> _ResponseCache:
    global _RESPONSE_CACHE
    if _RESPONSE_CACHE is None:
        with _MODEL_LOCK:
            if _RESPONSE_CACHE is None:
                _RESPONSE_CACHE = _ResponseCache(max_size=RESPONSE_CACHE_MAX_SIZE)
    return _RESPONSE_CACHE


def _stats_file_path() -> Path:
    return STATS_DIR / f"word_policy_{SERVER_PORT}_{os.getpid()}.json"


def _load_stats_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _empty_stats_payload() -> dict:
    return {
        "request_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "micro_batches": 0,
        "micro_batch_requests": 0,
        "uncached_compute_seconds": 0.0,
    }


def _summarize_stats_payload(payload: dict) -> dict:
    request_count = int(payload.get("request_count", 0) or 0)
    cache_hits = int(payload.get("cache_hits", 0) or 0)
    cache_misses = int(payload.get("cache_misses", 0) or 0)
    micro_batches = int(payload.get("micro_batches", 0) or 0)
    micro_batch_requests = int(payload.get("micro_batch_requests", 0) or 0)
    uncached_compute_seconds = float(payload.get("uncached_compute_seconds", 0.0) or 0.0)
    hit_rate = (cache_hits / request_count) if request_count else 0.0
    avg_batch_size = (micro_batch_requests / micro_batches) if micro_batches else 0.0
    avg_uncached_compute_ms = (
        (uncached_compute_seconds * 1000.0) / micro_batch_requests
        if micro_batch_requests
        else 0.0
    )
    return {
        "request_count": request_count,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": hit_rate,
        "micro_batches": micro_batches,
        "micro_batch_requests": micro_batch_requests,
        "avg_micro_batch_size": avg_batch_size,
        "avg_uncached_compute_ms": avg_uncached_compute_ms,
    }


def _aggregate_stats_payloads() -> dict:
    aggregate = _empty_stats_payload()
    if not STATS_DIR.exists():
        return aggregate
    for path in STATS_DIR.glob(f"word_policy_{SERVER_PORT}_*.json"):
        payload = _load_stats_payload(path)
        aggregate["request_count"] += int(payload.get("request_count", 0) or 0)
        aggregate["cache_hits"] += int(payload.get("cache_hits", 0) or 0)
        aggregate["cache_misses"] += int(payload.get("cache_misses", 0) or 0)
        aggregate["micro_batches"] += int(payload.get("micro_batches", 0) or 0)
        aggregate["micro_batch_requests"] += int(payload.get("micro_batch_requests", 0) or 0)
        aggregate["uncached_compute_seconds"] += float(payload.get("uncached_compute_seconds", 0.0) or 0.0)
    return aggregate


class _ServerStats:
    def __init__(self) -> None:
        self._lock = Lock()
        self._payload = _empty_stats_payload()
        self._dirty_updates = 0
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        self._flush()

    def _flush(self) -> None:
        _stats_file_path().write_text(
            json.dumps(self._payload, sort_keys=True),
            encoding="utf-8",
        )
        self._dirty_updates = 0

    def _maybe_flush(self) -> None:
        if STATS_FLUSH_EVERY <= 1:
            self._flush()
            return
        self._dirty_updates += 1
        if self._dirty_updates >= STATS_FLUSH_EVERY:
            self._flush()

    def record_cache_hit(self) -> None:
        with self._lock:
            self._payload["request_count"] += 1
            self._payload["cache_hits"] += 1
            self._maybe_flush()

    def record_cache_miss(self) -> None:
        with self._lock:
            self._payload["request_count"] += 1
            self._payload["cache_misses"] += 1
            self._maybe_flush()

    def record_batch(self, batch_size: int, compute_seconds: float) -> None:
        with self._lock:
            self._payload["micro_batches"] += 1
            self._payload["micro_batch_requests"] += batch_size
            self._payload["uncached_compute_seconds"] += compute_seconds
            self._maybe_flush()

    def snapshot(self) -> dict:
        with self._lock:
            return _summarize_stats_payload(dict(self._payload))

    def force_flush(self) -> None:
        with self._lock:
            self._flush()

    def clear(self) -> None:
        with self._lock:
            self._payload = _empty_stats_payload()
            self._flush()


def _get_server_stats() -> _ServerStats:
    global _SERVER_STATS
    if _SERVER_STATS is None:
        with _MODEL_LOCK:
            if _SERVER_STATS is None:
                _SERVER_STATS = _ServerStats()
    return _SERVER_STATS


class _PredictMicroBatcher:
    def __init__(self, *, batch_window_ms: float, max_batch_size: int) -> None:
        self.batch_window_s = max(0.0, batch_window_ms) / 1000.0
        self.max_batch_size = max(1, max_batch_size)
        self._cv = Condition()
        self._pending: list[tuple[dict, Future]] = []
        self._worker = Thread(target=self._run, name="word-policy-micro-batcher", daemon=True)
        self._worker.start()

    def submit(self, parsed_request: dict) -> dict:
        cache_key = _cache_key_for_request(parsed_request)
        cached_result = _get_response_cache().get(cache_key)
        if cached_result is not None:
            _get_server_stats().record_cache_hit()
            return cached_result
        _get_server_stats().record_cache_miss()
        future: Future = Future()
        with self._cv:
            self._pending.append((parsed_request, future))
            self._cv.notify()
        result = future.result()
        _get_response_cache().set(cache_key, result)
        return result

    def _collect_batch(self) -> list[tuple[dict, Future]]:
        with self._cv:
            while not self._pending:
                self._cv.wait()
            if self.batch_window_s > 0.0 and len(self._pending) < self.max_batch_size:
                deadline = time.perf_counter() + self.batch_window_s
                while len(self._pending) < self.max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    self._cv.wait(timeout=remaining)
                    if not self._pending:
                        break
            batch = self._pending[: self.max_batch_size]
            del self._pending[: len(batch)]
            return batch

    def _run(self) -> None:
        while True:
            batch = self._collect_batch()
            requests_only = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            try:
                start = time.perf_counter()
                if len(requests_only) == 1:
                    request_data = requests_only[0]
                    results = [
                        choose_action_from_inquiry(
                            request_data["question"],
                            request_data["battle_state"],
                            perspective_player=request_data["perspective_player"],
                            legal_moves=request_data["legal_moves"],
                            legal_switches=request_data["legal_switches"],
                            allow_voluntary_switches=request_data["allow_voluntary_switches"],
                            active_payload=request_data["active"],
                            model=_get_server_model(),
                        )
                    ]
                else:
                    results = choose_actions_from_inquiry_batch(
                        requests_only,
                        model=_get_server_model(),
                    )
                compute_seconds = time.perf_counter() - start
                _get_server_stats().record_batch(len(requests_only), compute_seconds)
                for future, result in zip(futures, results):
                    future.set_result(result)
            except Exception as error:  # pragma: no cover - defensive server path
                for future in futures:
                    future.set_exception(error)


def _get_micro_batcher() -> _PredictMicroBatcher:
    global _MICRO_BATCHER
    if _MICRO_BATCHER is None:
        with _MODEL_LOCK:
            if _MICRO_BATCHER is None:
                _MICRO_BATCHER = _PredictMicroBatcher(
                    batch_window_ms=MICRO_BATCH_WINDOW_MS,
                    max_batch_size=MICRO_BATCH_MAX_SIZE,
                )
    return _MICRO_BATCHER


def _parse_predict_request(data: dict) -> tuple[dict | None, tuple[object, int] | None]:
    battle_state = data.get("battle_state")
    if not isinstance(battle_state, dict):
        return None, (jsonify(error="missing battle_state"), 400)

    perspective_player = data.get("perspective_player")
    if perspective_player not in {"p1", "p2"}:
        perspective_player = "p1"

    return (
        {
            "battle_state": battle_state,
            "perspective_player": perspective_player,
            "question": str(data.get("question") or default_question_for_state(data)),
            "legal_moves": data.get("legal_moves") or [],
            "legal_switches": data.get("legal_switches") or [],
            "allow_voluntary_switches": _allow_voluntary_switches(data),
            "active": data.get("active") or [],
        },
        None,
    )


def _cache_key_for_request(parsed_request: dict) -> str:
    normalized = {
        "battle_state": parsed_request["battle_state"],
        "perspective_player": parsed_request["perspective_player"],
        "question": parsed_request["question"],
        "legal_moves": parsed_request["legal_moves"],
        "legal_switches": parsed_request["legal_switches"],
        "allow_voluntary_switches": parsed_request["allow_voluntary_switches"],
        "active": parsed_request["active"],
    }
    return repr(normalized)


def handle_predict_request(data: dict) -> tuple[dict | None, str | None]:
    parsed_request, error = _parse_predict_request(data)
    if error is not None:
        response, _status = error
        body = response.get_json()
        return None, str(body.get("error") or "invalid request")
    result = _get_micro_batcher().submit(parsed_request)
    return result, None


@app.route("/health", methods=["GET"])
def health():
    _get_server_stats().force_flush()
    aggregate_payload = _aggregate_stats_payloads()
    return jsonify(
        status="ok",
        model="word_policy_v1",
        cache_max_size=RESPONSE_CACHE_MAX_SIZE,
        micro_batch_window_ms=MICRO_BATCH_WINDOW_MS,
        micro_batch_max_size=MICRO_BATCH_MAX_SIZE,
        stats_flush_every=STATS_FLUSH_EVERY,
        worker_pid=os.getpid(),
        worker_stats=_get_server_stats().snapshot(),
        stats=_summarize_stats_payload(aggregate_payload),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="invalid JSON"), 400

    result, error = handle_predict_request(data)
    if error is not None:
        return jsonify(error=error), 400
    return jsonify(result)


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="invalid JSON"), 400

    requests_payload = data.get("requests")
    if not isinstance(requests_payload, list):
        return jsonify(error="missing requests"), 400

    parsed_requests = []
    for index, item in enumerate(requests_payload):
        if not isinstance(item, dict):
            return jsonify(error=f"invalid request at index {index}"), 400
        parsed_request, error = _parse_predict_request(item)
        if error is not None:
            response, status = error
            body = response.get_json()
            return jsonify(error=body.get("error"), index=index), status
        parsed_requests.append(parsed_request)

    results = choose_actions_from_inquiry_batch(
        parsed_requests,
        model=_get_server_model(),
    )
    return jsonify(results=results)


def main() -> None:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    if SERVER_WORKERS > 1:
        run_simple(
            SERVER_HOST,
            SERVER_PORT,
            app,
            use_reloader=False,
            use_debugger=False,
            threaded=False,
            processes=SERVER_WORKERS,
        )
        return
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
