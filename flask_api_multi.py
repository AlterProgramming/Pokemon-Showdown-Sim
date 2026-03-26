from __future__ import annotations

import argparse
import atexit
import sys
import time
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
    for model_id, artifact in artifacts.items():
        worker_count = worker_overrides.get(model_id, ARGS.workers_per_model)
        worker_pool = ModelWorkerPool(
            repo_path=REPO_PATH,
            model_entry=registered_models[model_id],
            worker_count=worker_count,
            request_timeout_seconds=ARGS.request_timeout_seconds,
            max_requests_before_recycle=ARGS.worker_max_requests,
            max_worker_age_seconds=ARGS.worker_max_age_seconds,
            bootstrap_timeout_seconds=ARGS.worker_bootstrap_timeout_seconds,
            startup_timeout_seconds=ARGS.worker_startup_timeout_seconds,
        )
        worker_pool.start()
        artifact["worker_pool"] = worker_pool

    default_model_id = (
        str(registry["default_model_id"])
        if registry.get("default_model_id") in artifacts
        else model_ids[0]
    )
    return artifacts, default_model_id

MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {}
DEFAULT_MODEL_ID = ""
APP = Flask(__name__)


def shutdown_workers() -> None:
    for artifact in MODEL_ARTIFACTS.values():
        worker_pool = artifact.get("worker_pool")
        if worker_pool is not None:
            worker_pool.close()


def log_loaded_models() -> None:
    print(f"[startup] mode={ARGS.mode}")
    print(f"[startup] default_model_id={DEFAULT_MODEL_ID}")
    print(f"[startup] request_timeout_seconds={ARGS.request_timeout_seconds}")
    print(f"[startup] worker_max_requests={ARGS.worker_max_requests}")
    print(f"[startup] worker_max_age_seconds={ARGS.worker_max_age_seconds}")
    print(f"[startup] workers_per_model={ARGS.workers_per_model}")
    print(f"[startup] model_worker_overrides={ARGS.model_worker_overrides or '(none)'}")
    print(f"[startup] worker_bootstrap_timeout_seconds={ARGS.worker_bootstrap_timeout_seconds}")
    print(f"[startup] worker_startup_timeout_seconds={ARGS.worker_startup_timeout_seconds}")
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
            f"worker_count={worker_health['worker_count']} "
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


@APP.route("/health", methods=["GET"])
def health():
    worker_health = {
        model_id: artifacts["worker_pool"].health()
        for model_id, artifacts in MODEL_ARTIFACTS.items()
    }
    overall_status = "ok" if all(entry["alive"] for entry in worker_health.values()) else "degraded"
    return jsonify(
        status=overall_status,
        mode=ARGS.mode,
        default_model_id=DEFAULT_MODEL_ID,
        supported_model_ids=sorted(MODEL_ARTIFACTS.keys()),
        worker_health=worker_health,
    )


@APP.route("/predict", methods=["POST"])
def predict():
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

        started_at = time.perf_counter()
        logits, worker_metrics = predict_logits(model_artifacts, state_vector)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        worker_index = worker_metrics.get("worker_index")
        worker_pid = worker_metrics.get("worker_pid")
        queue_wait_ms = worker_metrics.get("queue_wait_ms")
        service_ms = worker_metrics.get("service_ms")
        print(
            "[predict] "
            f"model_id={model_artifacts['model_id']} "
            f"worker_index={worker_index} "
            f"worker_pid={worker_pid} "
            f"state_len={len(state_vector)} "
            f"queue_wait_ms={0.0 if queue_wait_ms is None else float(queue_wait_ms):.2f} "
            f"service_ms={0.0 if service_ms is None else float(service_ms):.2f} "
            f"elapsed_ms={elapsed_ms:.2f}"
        )

        if revive_targets:
            best_revive, best_prob = pick_best_slot_target(model_artifacts, logits, revive_targets, "revive")
            if best_revive is None:
                return jsonify(
                    type="none",
                    note="revive target request detected but no legal revive targets were available",
                    revive_reason=revive_reason,
                    **context,
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
            return jsonify(
                best_move=best_action["payload"],
                type="move",
                probability=best_prob,
                action_token=best_action["token"],
                **context,
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
        import traceback

        traceback.print_exc()
        return jsonify(error=f"Server error: {error}"), 500


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    MODEL_ARTIFACTS, DEFAULT_MODEL_ID = load_models(ARGS.mode)
    atexit.register(shutdown_workers)
    log_loaded_models()
    APP.run(host=ARGS.host, port=ARGS.port, debug=False, use_reloader=False, threaded=True)
