from __future__ import annotations

from collections import deque
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ModelRegistry import resolve_artifact_path


MODEL_WORKER_DEBUG = os.environ.get("PS_MODEL_WORKER_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def parse_worker_count_overrides(raw_value: str | None) -> dict[str, int]:
    if raw_value is None or not raw_value.strip():
        return {}

    overrides: dict[str, int] = {}
    for raw_entry in raw_value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "model worker overrides must use the format 'model4=2,model2_large=1'"
            )
        model_id, raw_count = entry.split("=", 1)
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("model worker override is missing a model id")
        try:
            worker_count = int(raw_count.strip())
        except ValueError as error:
            raise ValueError(f"invalid worker count for model '{model_id}': {raw_count!r}") from error
        if worker_count < 1:
            raise ValueError(f"worker count for model '{model_id}' must be at least 1")
        overrides[model_id] = worker_count
    return overrides


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(logits)
    return exp / denom


def safe_connection_send(connection: Any, payload: dict[str, Any]) -> bool:
    try:
        connection.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError, KeyboardInterrupt):
        return False


def debug_log(message: str) -> None:
    if MODEL_WORKER_DEBUG:
        print(message)


def load_runtime_artifacts(repo_path: Path, model_entry: dict[str, Any]) -> dict[str, Any]:
    model_id = str(model_entry["model_id"])
    metadata_path = repo_path / str(model_entry["metadata_path"])
    model_path = resolve_artifact_path(repo_path, metadata_path, str(model_entry["policy_model_path"]))
    vocab_path = resolve_artifact_path(repo_path, metadata_path, str(model_entry["policy_vocab_path"]))

    with open(vocab_path, "r", encoding="utf-8") as handle:
        action_vocab = json.load(handle)

    expected_input_dim = model_entry.get("feature_dim")
    if expected_input_dim is not None:
        expected_input_dim = int(expected_input_dim)

    return {
        "model_id": model_id,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "expected_input_dim": expected_input_dim,
        "action_vocab": action_vocab,
        "metadata_path": str(metadata_path),
    }


def inference_worker_main(
    repo_path_str: str,
    model_entry: dict[str, Any],
    connection: Any,
) -> None:
    worker_entry_started = time.perf_counter()
    repo_path = Path(repo_path_str).resolve()
    model_id = str(model_entry["model_id"])
    if not safe_connection_send(
        connection,
        {
            "type": "worker_bootstrapped",
            "pid": os.getpid(),
            "model_id": model_id,
        }
    ):
        return
    try:
        tensorflow_import_started = time.perf_counter()
        import tensorflow as tf
        import keras
        keras.config.enable_unsafe_deserialization()
        tensorflow_import_seconds = time.perf_counter() - tensorflow_import_started

        metadata_path = repo_path / str(model_entry["metadata_path"])
        model_path = resolve_artifact_path(repo_path, metadata_path, str(model_entry["policy_model_path"]))
        model_load_started = time.perf_counter()
        model = tf.keras.models.load_model(model_path)
        model_load_seconds = time.perf_counter() - model_load_started
        expected_input_dim = model_entry.get("feature_dim")
        if expected_input_dim is None and getattr(model, "input_shape", None):
            shape = model.input_shape
            if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] is not None:
                expected_input_dim = int(shape[-1])
        if expected_input_dim is not None and int(expected_input_dim) > 0:
            warmup = np.zeros((1, int(expected_input_dim)), dtype=np.float32)
            _ = model(warmup, training=False)
        if not safe_connection_send(
            connection,
            {
                "type": "worker_ready",
                "pid": os.getpid(),
                "model_id": model_id,
                "expected_input_dim": expected_input_dim,
                "timings": {
                    "tensorflow_import_seconds": tensorflow_import_seconds,
                    "model_load_seconds": model_load_seconds,
                    "worker_ready_seconds": time.perf_counter() - worker_entry_started,
                },
            }
        ):
            return
    except Exception as error:  # pragma: no cover - depends on TensorFlow/runtime
        safe_connection_send(
            connection,
            {
                "type": "worker_failed",
                "pid": os.getpid(),
                "model_id": model_id,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "timings": {
                    "worker_ready_seconds": time.perf_counter() - worker_entry_started,
                },
            }
        )
        return

    while True:
        try:
            message = connection.recv()
        except (BrokenPipeError, EOFError, OSError, KeyboardInterrupt):
            return
        if not isinstance(message, dict):
            continue

        message_type = message.get("type")
        if message_type == "shutdown":
            return
        if message_type != "predict":
            continue

        request_id = str(message.get("request_id") or "")
        state_vector = message.get("state_vector")
        try:
            arr = np.asarray(state_vector, dtype=np.float32)[None, :]
            raw_output = model(arr, training=False)
            if hasattr(raw_output, "numpy"):
                raw_output = raw_output.numpy()
            logits = np.asarray(raw_output[0], dtype=np.float32)
            if not safe_connection_send(
                connection,
                {
                    "type": "prediction",
                    "request_id": request_id,
                    "pid": os.getpid(),
                    "model_id": model_id,
                    "logits": logits,
                }
            ):
                return
        except Exception as error:  # pragma: no cover - depends on TensorFlow/runtime
            if not safe_connection_send(
                connection,
                {
                    "type": "prediction_error",
                    "request_id": request_id,
                    "pid": os.getpid(),
                    "model_id": model_id,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            ):
                return


class ModelWorkerSupervisor:
    def __init__(
        self,
        *,
        repo_path: Path,
        model_entry: dict[str, Any],
        worker_index: int = 0,
        request_timeout_seconds: float,
        max_requests_before_recycle: int,
        max_worker_age_seconds: float,
        bootstrap_timeout_seconds: float = 30.0,
        startup_timeout_seconds: float | None = None,
        worker_target: Callable[..., None] = inference_worker_main,
    ) -> None:
        import multiprocessing as mp

        self.repo_path = repo_path.resolve()
        self.model_entry = dict(model_entry)
        self.model_id = str(model_entry["model_id"])
        self.worker_index = int(worker_index)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_requests_before_recycle = int(max_requests_before_recycle)
        self.max_worker_age_seconds = float(max_worker_age_seconds)
        self.bootstrap_timeout_seconds = float(bootstrap_timeout_seconds)
        self.startup_timeout_seconds = float(
            startup_timeout_seconds
            if startup_timeout_seconds is not None
            else max(30.0, self.request_timeout_seconds)
        )
        self.worker_target = worker_target
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._parent_conn: Any = None
        self._process: Any = None

        self.restart_count = 0
        self.total_completed_requests = 0
        self.requests_since_start = 0
        self.last_success_time: float | None = None
        self.last_start_time: float | None = None
        self.last_request_time: float | None = None
        self.last_error: str | None = None
        self.last_restart_reason: str | None = None
        self.last_startup_timings: dict[str, float] | None = None

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def close(self) -> None:
        with self._lock:
            self._stop_locked(graceful=True)

    def ensure_running(self) -> None:
        with self._lock:
            self._ensure_running_locked()

    def predict(self, state_vector: list[float]) -> np.ndarray:
        logits, _ = self.predict_with_metadata(state_vector)
        return logits

    def predict_with_metadata(self, state_vector: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
        with self._lock:
            self._ensure_running_locked()
            request_id = uuid.uuid4().hex
            self.last_request_time = time.time()
            service_started = time.perf_counter()
            self._parent_conn.send(
                {
                    "type": "predict",
                    "request_id": request_id,
                    "state_vector": list(state_vector),
                }
            )
            if not self._parent_conn.poll(timeout=self.request_timeout_seconds):
                self.last_error = (
                    f"prediction timed out after {self.request_timeout_seconds:.2f}s for {self.model_id}"
                )
                self._restart_locked("timeout")
                raise TimeoutError(self.last_error)
            message = self._parent_conn.recv()

            if message.get("type") == "prediction_error":
                self.last_error = str(message.get("error") or "worker prediction error")
                raise RuntimeError(self.last_error)
            if message.get("type") != "prediction" or message.get("request_id") != request_id:
                self.last_error = f"unexpected worker response for {self.model_id}: {message!r}"
                self._restart_locked("protocol_error")
                raise RuntimeError(self.last_error)

            logits = np.asarray(message["logits"], dtype=np.float32)
            service_ms = (time.perf_counter() - service_started) * 1000.0
            self.total_completed_requests += 1
            self.requests_since_start += 1
            self.last_success_time = time.time()
            recycle_reason = self._recycle_reason_locked()
            if recycle_reason is not None:
                self._restart_locked(recycle_reason, graceful=True)
            return logits, {
                "worker_index": self.worker_index,
                "worker_pid": message.get("pid"),
                "service_ms": service_ms,
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            alive = bool(self._process is not None and self._process.is_alive())
            return {
                "model_id": self.model_id,
                "worker_index": self.worker_index,
                "pid": None if self._process is None else self._process.pid,
                "alive": alive,
                "restart_count": self.restart_count,
                "total_completed_requests": self.total_completed_requests,
                "requests_since_start": self.requests_since_start,
                "last_success_time": self.last_success_time,
                "last_start_time": self.last_start_time,
                "last_request_time": self.last_request_time,
                "last_error": self.last_error,
                "last_restart_reason": self.last_restart_reason,
                "last_startup_timings": self.last_startup_timings,
                "bootstrap_timeout_seconds": self.bootstrap_timeout_seconds,
                "request_timeout_seconds": self.request_timeout_seconds,
                "max_requests_before_recycle": self.max_requests_before_recycle,
                "max_worker_age_seconds": self.max_worker_age_seconds,
            }

    def _ensure_running_locked(self) -> None:
        if self._process is None or not self._process.is_alive():
            self._restart_locked("not_running", graceful=False)

    def _start_locked(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        startup_started = time.perf_counter()
        self._parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._process = self._ctx.Process(
            target=self.worker_target,
            args=(str(self.repo_path), self.model_entry, child_conn),
            daemon=True,
            name=f"{self.model_id}_worker_{self.worker_index}",
        )
        self._process.start()
        child_conn.close()
        self.last_start_time = time.time()
        self.requests_since_start = 0
        self.last_startup_timings = None

        if not self._parent_conn.poll(timeout=self.bootstrap_timeout_seconds):
            self.last_error = (
                f"worker bootstrap timed out after {self.bootstrap_timeout_seconds:.2f}s "
                f"for {self.model_id}"
            )
            self._stop_locked(graceful=False)
            raise TimeoutError(self.last_error)
        bootstrap_message = self._parent_conn.recv()

        bootstrap_elapsed = time.perf_counter() - startup_started
        if bootstrap_message.get("type") == "worker_ready":
            message = bootstrap_message
        else:
            if bootstrap_message.get("type") == "worker_failed":
                self.last_error = str(bootstrap_message.get("error") or "worker startup failed")
                self._stop_locked(graceful=False)
                raise RuntimeError(f"{self.last_error}\n{bootstrap_message.get('traceback', '')}".rstrip())
            if bootstrap_message.get("type") != "worker_bootstrapped":
                self.last_error = (
                    f"unexpected worker bootstrap response for {self.model_id}: {bootstrap_message!r}"
                )
                self._stop_locked(graceful=False)
                raise RuntimeError(self.last_error)

            if not self._parent_conn.poll(timeout=self.startup_timeout_seconds):
                self.last_error = (
                    f"worker ready timed out after {self.startup_timeout_seconds:.2f}s for {self.model_id} "
                    f"after bootstrap completed in {bootstrap_elapsed:.2f}s"
                )
                self._stop_locked(graceful=False)
                raise TimeoutError(self.last_error)
            message = self._parent_conn.recv()

        if message.get("type") == "worker_failed":
            self.last_error = str(message.get("error") or "worker startup failed")
            self._stop_locked(graceful=False)
            raise RuntimeError(f"{self.last_error}\n{message.get('traceback', '')}".rstrip())
        if message.get("type") != "worker_ready":
            self.last_error = f"unexpected worker startup response for {self.model_id}: {message!r}"
            self._stop_locked(graceful=False)
            raise RuntimeError(self.last_error)

        worker_timings = dict(message.get("timings") or {})
        self.last_startup_timings = {
            "process_bootstrap_seconds": bootstrap_elapsed,
            "tensorflow_import_seconds": float(worker_timings.get("tensorflow_import_seconds", 0.0)),
            "model_load_seconds": float(worker_timings.get("model_load_seconds", 0.0)),
            "worker_ready_seconds": float(worker_timings.get("worker_ready_seconds", 0.0)),
            "total_startup_seconds": time.perf_counter() - startup_started,
        }
        expected_input_dim = message.get("expected_input_dim")
        if expected_input_dim is not None:
            self.model_entry["feature_dim"] = int(expected_input_dim)

    def _stop_locked(self, *, graceful: bool) -> None:
        process = self._process
        parent_conn = self._parent_conn
        self._process = None
        self._parent_conn = None

        if process is not None:
            try:
                if graceful and parent_conn is not None:
                    parent_conn.send({"type": "shutdown"})
                    process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
            finally:
                if process.is_alive():  # pragma: no cover - platform/runtime dependent
                    process.kill()
                    process.join(timeout=2.0)

        if parent_conn is not None:
            try:
                parent_conn.close()
            except Exception:
                pass

    def _restart_locked(self, reason: str, *, graceful: bool = False) -> None:
        had_process = self._process is not None
        self.last_restart_reason = reason
        self._stop_locked(graceful=graceful)
        if had_process:
            self.restart_count += 1
        self._start_locked()

    def _recycle_reason_locked(self) -> str | None:
        if self.max_requests_before_recycle > 0 and self.requests_since_start >= self.max_requests_before_recycle:
            return "max_requests"
        if self.max_worker_age_seconds > 0 and self.last_start_time is not None:
            if (time.time() - self.last_start_time) >= self.max_worker_age_seconds:
                return "max_age"
        return None


class ModelWorkerPool:
    def __init__(
        self,
        *,
        repo_path: Path,
        model_entry: dict[str, Any],
        worker_count: int,
        request_timeout_seconds: float,
        max_requests_before_recycle: int,
        max_worker_age_seconds: float,
        bootstrap_timeout_seconds: float = 30.0,
        startup_timeout_seconds: float | None = None,
        worker_target: Callable[..., None] = inference_worker_main,
        supervisor_factory: Callable[[int], Any] | None = None,
    ) -> None:
        if int(worker_count) < 1:
            raise ValueError("worker_count must be at least 1")

        self.repo_path = repo_path.resolve()
        self.model_entry = dict(model_entry)
        self.model_id = str(model_entry["model_id"])
        self.worker_count = int(worker_count)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_requests_before_recycle = int(max_requests_before_recycle)
        self.max_worker_age_seconds = float(max_worker_age_seconds)
        self.bootstrap_timeout_seconds = float(bootstrap_timeout_seconds)
        self.startup_timeout_seconds = startup_timeout_seconds
        self.worker_target = worker_target
        self._condition = threading.Condition()
        self._idle_worker_indices = deque(range(self.worker_count))
        self.waiting_requests = 0
        self.total_assigned_requests = 0

        if supervisor_factory is None:
            self._supervisors = [
                ModelWorkerSupervisor(
                    repo_path=self.repo_path,
                    model_entry=self.model_entry,
                    worker_index=worker_index,
                    request_timeout_seconds=self.request_timeout_seconds,
                    max_requests_before_recycle=self.max_requests_before_recycle,
                    max_worker_age_seconds=self.max_worker_age_seconds,
                    bootstrap_timeout_seconds=self.bootstrap_timeout_seconds,
                    startup_timeout_seconds=self.startup_timeout_seconds,
                    worker_target=self.worker_target,
                )
                for worker_index in range(self.worker_count)
            ]
        else:
            self._supervisors = [supervisor_factory(worker_index) for worker_index in range(self.worker_count)]

    def start(self) -> None:
        for supervisor in self._supervisors:
            supervisor.start()

    def close(self) -> None:
        for supervisor in self._supervisors:
            supervisor.close()

    def predict(self, state_vector: list[float]) -> np.ndarray:
        logits, _ = self.predict_with_metadata(state_vector)
        return logits

    def predict_with_metadata(self, state_vector: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
        request_started = time.perf_counter()
        with self._condition:
            while not self._idle_worker_indices:
                self.waiting_requests += 1
                try:
                    self._condition.wait()
                finally:
                    self.waiting_requests -= 1
            worker_index = self._idle_worker_indices.popleft()
            self.total_assigned_requests += 1
            waiting_requests = self.waiting_requests
            queue_wait_ms = (time.perf_counter() - request_started) * 1000.0

        supervisor = self._supervisors[worker_index]
        worker_health = supervisor.health()
        logged_worker_pid = worker_health.get("pid")
        debug_log(
            "[predict-dispatch] "
            f"model_id={self.model_id} "
            f"worker_index={worker_index} "
            f"worker_pid={logged_worker_pid} "
            f"waiting_requests={waiting_requests} "
            f"queue_wait_ms={queue_wait_ms:.2f} "
            f"total_assigned_requests={self.total_assigned_requests}"
        )
        try:
            service_started = time.perf_counter()
            if hasattr(supervisor, "predict_with_metadata"):
                logits, supervisor_metadata = supervisor.predict_with_metadata(state_vector)
            else:
                logits = supervisor.predict(state_vector)
                supervisor_metadata = {}
            service_ms = float(
                supervisor_metadata.get("service_ms", (time.perf_counter() - service_started) * 1000.0)
            )
            logged_worker_pid = supervisor_metadata.get("worker_pid", logged_worker_pid)
            total_ms = queue_wait_ms + service_ms
            debug_log(
                "[predict-complete] "
                f"model_id={self.model_id} "
                f"worker_index={worker_index} "
                f"worker_pid={logged_worker_pid} "
                f"queue_wait_ms={queue_wait_ms:.2f} "
                f"service_ms={service_ms:.2f} "
                f"total_ms={total_ms:.2f}"
            )
            return logits, {
                "worker_index": worker_index,
                "worker_pid": logged_worker_pid,
                "queue_wait_ms": queue_wait_ms,
                "service_ms": service_ms,
                "total_ms": total_ms,
                "waiting_requests_at_dispatch": waiting_requests,
            }
        except Exception as error:
            service_ms = (time.perf_counter() - request_started) * 1000.0 - queue_wait_ms
            total_ms = queue_wait_ms + service_ms
            debug_log(
                "[predict-error] "
                f"model_id={self.model_id} "
                f"worker_index={worker_index} "
                f"worker_pid={logged_worker_pid} "
                f"queue_wait_ms={queue_wait_ms:.2f} "
                f"service_ms={service_ms:.2f} "
                f"total_ms={total_ms:.2f} "
                f"error={error}"
            )
            raise
        finally:
            with self._condition:
                self._idle_worker_indices.append(worker_index)
                self._condition.notify()

    def health(self) -> dict[str, Any]:
        with self._condition:
            idle_workers = len(self._idle_worker_indices)
            waiting_requests = self.waiting_requests
        worker_health = [supervisor.health() for supervisor in self._supervisors]
        return {
            "model_id": self.model_id,
            "worker_count": self.worker_count,
            "busy_workers": self.worker_count - idle_workers,
            "idle_workers": idle_workers,
            "waiting_requests": waiting_requests,
            "total_assigned_requests": self.total_assigned_requests,
            "alive": all(entry.get("alive") for entry in worker_health),
            "workers": worker_health,
        }
