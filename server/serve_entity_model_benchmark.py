from __future__ import annotations

"""Serve one entity-action policy model for quick simulator benchmarks.

This server is intentionally narrow:
    - it serves a single entity-family policy model
    - it accepts the simulator's public battle snapshot
    - it returns the same move/switch/revive response shape as flask_api_multi.py

The purpose is to let us benchmark a freshly trained entity model before we fully
fold the family into the main multi-model runtime.
"""

import argparse
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
from flask import Flask, jsonify, request


REPO_PATH = Path(__file__).resolve().parent.parent
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from ActionLegality import filter_legal_revive_targets, filter_legal_switches
from EntityServerRuntime import (
    entity_runtime_health,
    load_entity_runtime_artifacts,
    predict_entity_candidate_logits,
    predict_entity_logits,
)
from core.ActionSelection import pick_best_action as select_best_action
from core.ActionSelection import pick_best_slot_target
from core.ActionSelection import FORCED_SWITCH_REASONS, softmax


APP = Flask(__name__)
SERVER_STATE: dict[str, Any] = {}
SERVER_METRICS_LOCK = threading.Lock()
SERVER_METRICS: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one entity_action_bc_v1 model for local benchmarks.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Training metadata JSON for the entity model release to serve.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def load_server_state(metadata_path: Path) -> dict[str, Any]:
    return load_entity_runtime_artifacts(metadata_path)


def reset_server_metrics() -> None:
    global SERVER_METRICS
    SERVER_METRICS = {
        "request_count": 0,
        "success_count": 0,
        "error_count": 0,
        "move_count": 0,
        "switch_count": 0,
        "revive_count": 0,
        "none_count": 0,
        "avg_inference_ms": 0.0,
        "avg_elapsed_ms": 0.0,
        "max_inference_ms": 0.0,
        "max_elapsed_ms": 0.0,
        "_total_inference_ms": 0.0,
        "_total_elapsed_ms": 0.0,
    }


def record_server_metric(
    *,
    inference_ms: float,
    elapsed_ms: float,
    result_type: str | None,
    success: bool,
) -> None:
    with SERVER_METRICS_LOCK:
        SERVER_METRICS["request_count"] += 1
        if success:
            SERVER_METRICS["success_count"] += 1
        else:
            SERVER_METRICS["error_count"] += 1
        if result_type == "move":
            SERVER_METRICS["move_count"] += 1
        elif result_type == "switch":
            SERVER_METRICS["switch_count"] += 1
        elif result_type == "revive":
            SERVER_METRICS["revive_count"] += 1
        elif result_type == "none":
            SERVER_METRICS["none_count"] += 1
        SERVER_METRICS["_total_inference_ms"] += float(inference_ms)
        SERVER_METRICS["_total_elapsed_ms"] += float(elapsed_ms)
        SERVER_METRICS["max_inference_ms"] = max(SERVER_METRICS["max_inference_ms"], float(inference_ms))
        SERVER_METRICS["max_elapsed_ms"] = max(SERVER_METRICS["max_elapsed_ms"], float(elapsed_ms))
        request_count = float(SERVER_METRICS["request_count"])
        SERVER_METRICS["avg_inference_ms"] = SERVER_METRICS["_total_inference_ms"] / request_count
        SERVER_METRICS["avg_elapsed_ms"] = SERVER_METRICS["_total_elapsed_ms"] / request_count


def snapshot_server_metrics() -> dict[str, Any]:
    with SERVER_METRICS_LOCK:
        return {
            key: value
            for key, value in SERVER_METRICS.items()
            if not key.startswith("_")
        }


def predict_logits(battle_state: dict[str, Any], perspective_player: str) -> np.ndarray:
    return predict_entity_logits(SERVER_STATE, battle_state, perspective_player)


def select_best_v2_candidate(
    logits: np.ndarray,
    candidate_tokens: list[str],
    *,
    legal_moves: list[dict[str, Any]],
    legal_switches: list[dict[str, Any]],
    switch_reason: str | None,
    switch_logit_bias: float,
) -> tuple[dict[str, Any] | None, float]:
    adjusted = np.asarray(logits, dtype=np.float32).copy()
    if (
        switch_logit_bias > 0
        and switch_reason not in FORCED_SWITCH_REASONS
        and legal_moves
        and legal_switches
    ):
        for idx, token in enumerate(candidate_tokens):
            if token.startswith("switch:"):
                adjusted[idx] -= float(switch_logit_bias)

    probs = softmax(adjusted)
    best_idx = -1
    best_prob = -1.0
    for idx, token in enumerate(candidate_tokens):
        if not token:
            continue
        probability = float(probs[idx])
        if probability > best_prob:
            best_prob = probability
            best_idx = idx

    if best_idx < 0:
        return None, -1.0

    token = candidate_tokens[best_idx]
    if token.startswith("move:"):
        move_id = token.split(":", 1)[1]
        for move in legal_moves:
            if str(move.get("id") or move.get("move") or "").lower().replace(" ", "") == move_id:
                return {"type": "move", "payload": move, "token": token}, best_prob
    elif token.startswith("switch:"):
        slot = int(token.split(":", 1)[1])
        for switch in legal_switches:
            if int(switch.get("slot") or 0) == slot:
                return {"type": "switch", "payload": switch, "token": token}, best_prob
    return None, best_prob


@APP.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        model_id=SERVER_STATE["model_id"],
        model_name=SERVER_STATE["model_name"],
        switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
        runtime=entity_runtime_health(SERVER_STATE),
        request_metrics=snapshot_server_metrics(),
    )


@APP.route("/predict", methods=["POST"])
def predict():
    request_started = time.perf_counter()
    inference_ms = 0.0
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(error="invalid JSON"), 400

        requested_model_id = data.get("model_id")
        if requested_model_id and str(requested_model_id) != SERVER_STATE["model_id"]:
            return jsonify(
                error=f"Unsupported model_id '{requested_model_id}'.",
                supported_model_ids=[SERVER_STATE["model_id"]],
            ), 400

        battle_state = data.get("battle_state")
        if not isinstance(battle_state, dict):
            return jsonify(error="missing battle_state"), 400

        perspective_player = data.get("perspective_player")
        if perspective_player not in {"p1", "p2"}:
            return jsonify(error="perspective_player must be 'p1' or 'p2'"), 400

        legal_moves = data.get("legal_moves") or []
        revive_targets, revive_reason = filter_legal_revive_targets(
            data,
            data.get("legal_revives", []) or data.get("legal_switches", []) or [],
        )
        legal_switches, switch_reason = filter_legal_switches(data, data.get("legal_switches", []) or [])

        # Apply value-based switch gating if we have a value head
        switch_logit_bias = float(SERVER_STATE.get("switch_logit_bias", 0.0))
        if SERVER_STATE.get("has_value_head"):
            value_estimate = SERVER_STATE.get("_last_value_estimate", 0.5)
            if value_estimate > 0.5:
                # Winning: strongly penalize switches
                switch_logit_bias += (value_estimate - 0.5) * 18
            # Losing: keep base penalty unchanged (don't boost switches)

        # Soft cooldown: bias against switches after a switch, but do NOT hard-block.
        # History of this knob:
        #   - Originally a hard block of 999.0 (consecutive switches effectively impossible).
        #     That hard block caused the agent to get stuck in losing positions where the
        #     only good move was a second switch (e.g. forced-out then chaining into a pivot).
        #   - 3.0 is the softer penalty that preserves the 42% WR baseline established in
        #     commit b7209b3 ("Optimize entity_action_v2 serving path") and 65a65cd ("ONNX
        #     inference (13.5x speedup) + value-head switch gating").
        #   - The value-head gating above (lines ~218-223) already adds up to +9 when winning,
        #     so 3.0 compounds cleanly without swamping the signal.
        # If you re-introduce a hard block, document the regression trade-off above this line.
        if SERVER_STATE.get("_last_action_was_switch"):
            switch_logit_bias += 3.0

        inference_started = time.perf_counter()
        if SERVER_STATE.get("input_mode") == "entity_action_v2":
            logits, candidate_tokens = predict_entity_candidate_logits(
                SERVER_STATE,
                battle_state,
                perspective_player,
                legal_moves=legal_moves,
                legal_switches=legal_switches,
            )
        else:
            logits = predict_logits(battle_state, perspective_player)
            candidate_tokens = []
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        if revive_targets:
            best_revive, best_prob = pick_best_slot_target(
                SERVER_STATE["action_vocab"],
                logits,
                revive_targets,
                "revive",
            )
            if best_revive is None:
                record_server_metric(
                    inference_ms=inference_ms,
                    elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
                    result_type="none",
                    success=True,
                )
                return jsonify(
                    type="none",
                    note="revive target request detected but no legal revive targets were available",
                    revive_reason=revive_reason,
                    model_id=SERVER_STATE["model_id"],
                    supported_model_ids=[SERVER_STATE["model_id"]],
                )
            record_server_metric(
                inference_ms=inference_ms,
                elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
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
                model_id=SERVER_STATE["model_id"],
                supported_model_ids=[SERVER_STATE["model_id"]],
                switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
            )

        if SERVER_STATE.get("input_mode") == "entity_action_v2":
            best_action, best_prob = select_best_v2_candidate(
                logits,
                candidate_tokens,
                legal_moves=legal_moves,
                legal_switches=legal_switches,
                switch_reason=switch_reason,
                switch_logit_bias=switch_logit_bias,
            )
        else:
            best_action, best_prob = select_best_action(
                SERVER_STATE["action_vocab"],
                logits,
                legal_moves,
                legal_switches,
                switch_logit_bias=switch_logit_bias,
            )
        if best_action is None:
            record_server_metric(
                inference_ms=inference_ms,
                elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
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
                model_id=SERVER_STATE["model_id"],
                supported_model_ids=[SERVER_STATE["model_id"]],
                switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
            )

        if best_action["type"] == "move":
            SERVER_STATE["_last_action_was_switch"] = False
            record_server_metric(
                inference_ms=inference_ms,
                elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
                result_type="move",
                success=True,
            )
            return jsonify(
                best_move=best_action["payload"],
                type="move",
                probability=best_prob,
                action_token=best_action["token"],
                model_id=SERVER_STATE["model_id"],
                supported_model_ids=[SERVER_STATE["model_id"]],
                switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
            )

        SERVER_STATE["_last_action_was_switch"] = True
        record_server_metric(
            inference_ms=inference_ms,
            elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
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
            model_id=SERVER_STATE["model_id"],
            supported_model_ids=[SERVER_STATE["model_id"]],
            switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
        )
    except Exception as error:
        record_server_metric(
            inference_ms=inference_ms,
            elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
            result_type="error",
            success=False,
        )
        import traceback

        traceback.print_exc()
        return jsonify(error=f"Server error: {error}"), 500


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path).resolve()
    global SERVER_STATE
    SERVER_STATE = load_server_state(metadata_path)
    reset_server_metrics()
    print(f"[entity-server] model_id={SERVER_STATE['model_id']}")
    print(f"[entity-server] metadata_path={metadata_path}")
    APP.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
