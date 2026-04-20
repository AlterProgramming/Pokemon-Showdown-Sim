from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from ModelWorkers import (
    ModelWorkerPool,
    ModelWorkerSupervisor,
    decode_float32_payload,
    encode_float32_payload,
    parse_worker_count_overrides,
    softmax,
)


ROOT = Path(__file__).resolve().parents[1]


def fake_worker_main(repo_path_str: str, model_entry: dict, connection) -> None:
    connection.send(
        {
            "type": "worker_bootstrapped",
            "pid": os.getpid(),
            "model_id": model_entry["model_id"],
        }
    )
    connection.send(
        {
            "type": "worker_ready",
            "pid": os.getpid(),
            "model_id": model_entry["model_id"],
            "expected_input_dim": 3,
            "timings": {
                "tensorflow_import_seconds": 0.01,
                "model_load_seconds": 0.02,
                "worker_ready_seconds": 0.03,
            },
        }
    )
    while True:
        message = connection.recv()
        if message.get("type") == "shutdown":
            return
        if message.get("type") != "predict":
            continue

        request_id = message["request_id"]
        if "state_vector_bytes" in message:
            state_vector = decode_float32_payload(
                message["state_vector_bytes"],
                tuple(message.get("state_vector_shape") or ()),
            ).tolist()
        else:
            state_vector = list(message["state_vector"])
        if state_vector and state_vector[0] == -999.0:
            time.sleep(1.0)
            continue

        logits_bytes, logits_shape = encode_float32_payload(state_vector)

        connection.send(
            {
                "type": "prediction",
                "request_id": request_id,
                "pid": os.getpid(),
                "model_id": model_entry["model_id"],
                "logits_bytes": logits_bytes,
                "logits_shape": logits_shape,
            }
        )


class FakeSupervisor:
    def __init__(self, worker_index: int, delay_seconds: float = 0.0) -> None:
        self.worker_index = worker_index
        self.delay_seconds = delay_seconds
        self.started = False
        self.closed = False
        self.predict_calls = 0

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def predict(self, state_vector: list[float]) -> np.ndarray:
        self.predict_calls += 1
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return np.asarray([float(self.worker_index)], dtype=np.float32)

    def predict_batch(self, state_vectors: list[list[float]]) -> np.ndarray:
        self.predict_calls += len(state_vectors)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return np.asarray(
            [[float(self.worker_index)] for _ in state_vectors],
            dtype=np.float32,
        )

    def predict_with_metadata(self, state_vector: list[float]) -> tuple[np.ndarray, dict[str, object]]:
        started = time.perf_counter()
        logits = self.predict(state_vector)
        return logits, {
            "worker_index": self.worker_index,
            "worker_pid": 1000 + self.worker_index,
            "service_ms": (time.perf_counter() - started) * 1000.0,
        }

    def predict_batch_with_metadata(self, state_vectors: list[list[float]]) -> tuple[np.ndarray, dict[str, object]]:
        started = time.perf_counter()
        logits = self.predict_batch(state_vectors)
        return logits, {
            "worker_index": self.worker_index,
            "worker_pid": 1000 + self.worker_index,
            "service_ms": (time.perf_counter() - started) * 1000.0,
            "batch_size": len(state_vectors),
        }

    def health(self) -> dict[str, object]:
        return {
            "model_id": "fake",
            "worker_index": self.worker_index,
            "pid": None,
            "alive": self.started and not self.closed,
            "restart_count": 0,
            "total_completed_requests": self.predict_calls,
            "requests_since_start": self.predict_calls,
            "last_success_time": None,
            "last_start_time": None,
            "last_request_time": None,
            "last_error": None,
            "last_restart_reason": None,
            "last_startup_timings": None,
            "bootstrap_timeout_seconds": 0.0,
            "request_timeout_seconds": 0.0,
            "max_requests_before_recycle": 0,
            "max_worker_age_seconds": 0.0,
        }


class ModelWorkerSupervisorTests(unittest.TestCase):
    def make_supervisor(
        self,
        *,
        request_timeout_seconds: float = 0.25,
        max_requests_before_recycle: int = 0,
        max_worker_age_seconds: float = 0.0,
    ) -> ModelWorkerSupervisor:
        return ModelWorkerSupervisor(
            repo_path=ROOT,
            model_entry={"model_id": "fake", "feature_dim": 3},
            request_timeout_seconds=request_timeout_seconds,
            max_requests_before_recycle=max_requests_before_recycle,
            max_worker_age_seconds=max_worker_age_seconds,
            startup_timeout_seconds=1.0,
            worker_target=fake_worker_main,
        )

    def test_softmax_normalizes_logits(self) -> None:
        probs = softmax(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        self.assertGreater(float(probs[2]), float(probs[1]))

    def test_float32_payload_round_trip(self) -> None:
        values = [1.5, 2.5, 3.5]
        payload, shape = encode_float32_payload(values)
        decoded = decode_float32_payload(payload, shape)
        np.testing.assert_allclose(decoded, np.asarray(values, dtype=np.float32))

    def start_or_skip(self, supervisor: ModelWorkerSupervisor) -> None:
        try:
            supervisor.start()
        except PermissionError as error:
            self.skipTest(f"multiprocessing IPC is not permitted in this test environment: {error}")

    def test_timeout_restarts_worker_and_allows_future_predictions(self) -> None:
        supervisor = self.make_supervisor(request_timeout_seconds=0.1)
        self.start_or_skip(supervisor)
        self.addCleanup(supervisor.close)

        with self.assertRaises(TimeoutError):
            supervisor.predict([-999.0, 0.0, 0.0])

        health_after_timeout = supervisor.health()
        self.assertTrue(health_after_timeout["alive"])
        self.assertEqual(health_after_timeout["restart_count"], 1)
        self.assertIsNotNone(health_after_timeout["last_startup_timings"])

        logits = supervisor.predict([1.0, 2.0, 3.0])
        np.testing.assert_allclose(logits, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    def test_max_requests_triggers_graceful_recycle(self) -> None:
        supervisor = self.make_supervisor(max_requests_before_recycle=2)
        self.start_or_skip(supervisor)
        self.addCleanup(supervisor.close)

        first_logits = supervisor.predict([1.0, 0.0, 0.0])
        np.testing.assert_allclose(first_logits, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        health_before_recycle = supervisor.health()
        self.assertEqual(health_before_recycle["restart_count"], 0)
        self.assertEqual(health_before_recycle["requests_since_start"], 1)

        second_logits = supervisor.predict([2.0, 0.0, 0.0])
        np.testing.assert_allclose(second_logits, np.asarray([2.0, 0.0, 0.0], dtype=np.float32))
        health_after_recycle = supervisor.health()
        self.assertTrue(health_after_recycle["alive"])
        self.assertEqual(health_after_recycle["restart_count"], 1)
        self.assertEqual(health_after_recycle["requests_since_start"], 0)
        self.assertEqual(health_after_recycle["total_completed_requests"], 2)


class ModelWorkerPoolTests(unittest.TestCase):
    def test_parse_worker_count_overrides(self) -> None:
        overrides = parse_worker_count_overrides("model4=4, model2_large=2")
        self.assertEqual(overrides, {"model4": 4, "model2_large": 2})

    def test_pool_round_robins_idle_workers(self) -> None:
        pool = ModelWorkerPool(
            repo_path=ROOT,
            model_entry={"model_id": "fake", "feature_dim": 1},
            worker_count=2,
            request_timeout_seconds=0.1,
            max_requests_before_recycle=0,
            max_worker_age_seconds=0.0,
            supervisor_factory=lambda worker_index: FakeSupervisor(worker_index),
        )
        pool.start()
        self.addCleanup(pool.close)

        first = pool.predict([1.0])
        second = pool.predict([1.0])
        np.testing.assert_allclose(first, np.asarray([0.0], dtype=np.float32))
        np.testing.assert_allclose(second, np.asarray([1.0], dtype=np.float32))

    def test_pool_predict_with_metadata_reports_queue_and_service_time(self) -> None:
        pool = ModelWorkerPool(
            repo_path=ROOT,
            model_entry={"model_id": "fake", "feature_dim": 1},
            worker_count=1,
            request_timeout_seconds=0.1,
            max_requests_before_recycle=0,
            max_worker_age_seconds=0.0,
            supervisor_factory=lambda worker_index: FakeSupervisor(worker_index, delay_seconds=0.01),
        )
        pool.start()
        self.addCleanup(pool.close)

        logits, metadata = pool.predict_with_metadata([1.0])
        np.testing.assert_allclose(logits, np.asarray([0.0], dtype=np.float32))
        self.assertEqual(metadata["worker_index"], 0)
        self.assertEqual(metadata["worker_pid"], 1000)
        self.assertGreaterEqual(float(metadata["queue_wait_ms"]), 0.0)
        self.assertGreater(float(metadata["service_ms"]), 0.0)
        self.assertGreaterEqual(float(metadata["total_ms"]), float(metadata["service_ms"]))
        health = pool.health()
        self.assertEqual(health["request_metrics"]["prediction_calls"], 1)
        self.assertEqual(health["request_metrics"]["batch_prediction_calls"], 0)
        self.assertGreaterEqual(float(health["request_metrics"]["avg_total_ms"]), float(metadata["service_ms"]))

    def test_pool_reports_waiting_requests_when_all_workers_busy(self) -> None:
        pool = ModelWorkerPool(
            repo_path=ROOT,
            model_entry={"model_id": "fake", "feature_dim": 1},
            worker_count=1,
            request_timeout_seconds=0.1,
            max_requests_before_recycle=0,
            max_worker_age_seconds=0.0,
            supervisor_factory=lambda worker_index: FakeSupervisor(worker_index, delay_seconds=0.2),
        )
        pool.start()
        self.addCleanup(pool.close)

        started = threading.Event()
        finished = threading.Event()

        def run_predict() -> None:
            started.set()
            pool.predict([1.0])
            finished.set()

        first = threading.Thread(target=run_predict)
        second = threading.Thread(target=run_predict)
        first.start()
        started.wait(timeout=1.0)
        time.sleep(0.05)
        second.start()
        time.sleep(0.05)

        health = pool.health()
        self.assertEqual(health["worker_count"], 1)
        self.assertGreaterEqual(health["busy_workers"], 1)
        self.assertGreaterEqual(health["waiting_requests"], 1)

        first.join(timeout=1.0)
        second.join(timeout=1.0)
        self.assertTrue(finished.is_set())

    def test_pool_predict_batch_with_metadata_reports_batch_size(self) -> None:
        pool = ModelWorkerPool(
            repo_path=ROOT,
            model_entry={"model_id": "fake", "feature_dim": 1},
            worker_count=1,
            request_timeout_seconds=0.1,
            max_requests_before_recycle=0,
            max_worker_age_seconds=0.0,
            supervisor_factory=lambda worker_index: FakeSupervisor(worker_index),
        )
        pool.start()
        self.addCleanup(pool.close)

        logits, metadata = pool.predict_batch_with_metadata([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(logits, np.asarray([[0.0], [0.0], [0.0]], dtype=np.float32))
        self.assertEqual(metadata["worker_index"], 0)
        self.assertEqual(metadata["worker_pid"], 1000)
        self.assertEqual(metadata["batch_size"], 3)
        health = pool.health()
        self.assertEqual(health["request_metrics"]["prediction_calls"], 1)
        self.assertEqual(health["request_metrics"]["batch_prediction_calls"], 1)
        self.assertEqual(health["total_assigned_requests"], 3)
        self.assertEqual(health["request_metrics"]["avg_batch_size"], 3.0)


if __name__ == "__main__":
    unittest.main()
