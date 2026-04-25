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
import os
from collections import deque
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
    predict_entity_auxiliary_outputs,
    predict_entity_auxiliary_outputs_batch,
    predict_entity_candidate_logits_with_metadata,
    predict_entity_logits_with_metadata,
)
from core.ActionSelection import FORCED_SWITCH_REASONS, softmax
from core.ActionSelection import adjust_logits_for_switch_bias
from core.ActionSelection import pick_best_action as select_best_action
from core.ActionSelection import pick_best_slot_target
from core.ActionSelection import build_move_tokens, build_switch_tokens
from core.SequencePlanning import (
    combine_policy_and_auxiliary_scores,
    score_sequence_tokens,
    summarize_auxiliary_prediction,
)


APP = Flask(__name__)
SERVER_STATE: dict[str, Any] = {}
SERVER_METRICS_LOCK = threading.Lock()
SERVER_METRICS: dict[str, Any] = {}
DEFAULT_VOLUNTARY_SWITCH_RERANK_PENALTY = float(os.environ.get("ENTITY_RERANK_VOLUNTARY_SWITCH_PENALTY", "0.2"))

_HISTORY_BUFFERS: dict[str, deque] = {}
_HISTORY_LOCK = threading.Lock()


def _get_history_buffer(battle_id: str, max_turns: int) -> deque:
    with _HISTORY_LOCK:
        if battle_id not in _HISTORY_BUFFERS:
            _HISTORY_BUFFERS[battle_id] = deque(maxlen=max_turns)
        return _HISTORY_BUFFERS[battle_id]


def _append_aux_log(record: dict) -> None:
    log_path = SERVER_STATE.get("_capture_aux_log_path")
    if not log_path:
        return
    import json
    lock = SERVER_STATE.get("_capture_aux_log_lock")
    line = json.dumps(record, default=str) + "\n"
    if lock:
        with lock:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
    else:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one entity_action_bc_v1 model for local benchmarks.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Training metadata JSON for the entity model release to serve.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--capture-aux-log",
        default=None,
        metavar="PATH",
        help="If set, write one JSONL record per /predict call to this file.",
    )
    parser.add_argument(
        "--disable-history-buffer",
        action="store_true",
        help="Force past_turn_events=[] for aux-head rerank. Ablation: isolates "
             "live-history contribution while keeping the rest of the rerank pipeline.",
    )
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
    logits, _ = predict_entity_logits_with_metadata(SERVER_STATE, battle_state, perspective_player)
    return logits


def select_best_v2_candidate(
    logits: np.ndarray,
    candidate_tokens: list[str],
    *,
    legal_moves: list[dict[str, Any]],
    legal_switches: list[dict[str, Any]],
    switch_reason: str | None,
    switch_logit_bias: float,
) -> tuple[dict[str, Any] | None, float]:
    adjusted = np.atleast_1d(np.asarray(logits, dtype=np.float32).copy())
    # Guard: model may return fewer logits than candidates (e.g. scalar squeeze).
    candidate_tokens = candidate_tokens[:len(adjusted)]
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

def _candidate_actions(
    action_vocab: dict[str, int],
    *,
    legal_moves: list[dict[str, Any]],
    legal_switches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for move in legal_moves:
        move_name = str(move.get("move") or move.get("id") or "")
        for token in build_move_tokens(action_vocab, move_name):
            index = action_vocab.get(token)
            if index is None:
                continue
            candidates.append(
                {
                    "type": "move",
                    "payload": move,
                    "token": token,
                    "index": int(index),
                }
            )

    for switch in legal_switches:
        slot = switch.get("slot")
        if slot is None:
            continue
        for token in build_switch_tokens(action_vocab, int(slot)):
            index = action_vocab.get(token)
            if index is None:
                continue
            candidates.append(
                {
                    "type": "switch",
                    "payload": switch,
                    "token": token,
                    "index": int(index),
                }
            )

    return candidates


def _select_best_action_with_auxiliary(
    *,
    battle_state: dict[str, Any],
    perspective_player: str,
    logits: np.ndarray,
    legal_moves: list[dict[str, Any]],
    legal_switches: list[dict[str, Any]],
    switch_reason: str | None,
    switch_logit_bias: float,
    opponent_action_token: str | None,
    auxiliary_scale: float,
    value_scale: float,
    sequence_scale: float,
    voluntary_switch_penalty: float,
    past_turn_events: list[list[dict]] | None = None,
) -> tuple[dict[str, Any] | None, float | None, list[dict[str, Any]]]:
    action_vocab = SERVER_STATE["action_vocab"]
    adjusted_logits = adjust_logits_for_switch_bias(
        logits,
        action_vocab,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        switch_reason=switch_reason,
        switch_logit_bias=switch_logit_bias,
    )
    candidates = _candidate_actions(
        action_vocab,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
    )
    analyses: list[dict[str, Any]] = []
    combined_scores: list[float] = []
    best_candidate = None
    best_score = None
    auxiliary_outputs = predict_entity_auxiliary_outputs_batch(
        SERVER_STATE,
        battle_state,
        perspective_player,
        my_action_tokens=[str(candidate["token"]) for candidate in candidates],
        opp_action_token=opponent_action_token,
        past_turn_events=past_turn_events,
    )

    for index, candidate in enumerate(candidates):
        base_logit = float(adjusted_logits[candidate["index"]])
        auxiliary = auxiliary_outputs[index] if index < len(auxiliary_outputs) else None
        value_prediction = None
        sequence_tokens: list[str] = []
        sequence_score = None
        if auxiliary is not None:
            value_prediction = auxiliary.get("value_prediction")
            sequence_tokens = list(auxiliary.get("sequence_tokens") or [])
            if sequence_tokens:
                sequence_score = score_sequence_tokens(
                    sequence_tokens,
                    perspective_player=perspective_player,
                )

        combined_score = combine_policy_and_auxiliary_scores(
            policy_logit=base_logit,
            value_prediction=value_prediction,
            sequence_score=sequence_score,
            auxiliary_scale=auxiliary_scale,
            value_scale=value_scale,
            sequence_scale=sequence_scale,
        )
        switch_penalty_applied = 0.0
        if (
            candidate["type"] == "switch"
            and voluntary_switch_penalty > 0
            and switch_reason not in FORCED_SWITCH_REASONS
            and legal_moves
            and legal_switches
        ):
            switch_penalty_applied = float(voluntary_switch_penalty)
            combined_score -= switch_penalty_applied
        summary = summarize_auxiliary_prediction(
            sequence_tokens=sequence_tokens,
            sequence_score=sequence_score,
            value_prediction=value_prediction,
            used_opponent_action_token=opponent_action_token,
        )
        combined_scores.append(float(combined_score))
        analyses.append(
            {
                "_candidate_index": index,
                "action_token": candidate["token"],
                "action_type": candidate["type"],
                "base_logit": base_logit,
                "combined_score": combined_score,
                "voluntary_switch_penalty_applied": switch_penalty_applied,
                **summary,
                "history_attention": (
                    auxiliary.get("history_attention").tolist()
                    if auxiliary is not None and auxiliary.get("history_attention") is not None
                    else None
                ),
            }
        )
        if best_score is None or combined_score > best_score:
            best_score = combined_score
            best_candidate = candidate

    if analyses:
        probabilities = np.asarray(softmax(np.asarray(combined_scores, dtype=np.float32)), dtype=np.float32)
        for index, analysis in enumerate(analyses):
            probability = float(probabilities[index]) if index < len(probabilities) else 0.0
            analysis["selection_probability"] = probability
            analysis["selection_percentage"] = probability * 100.0

        base_rank_by_index = {
            int(entry["_candidate_index"]): rank
            for rank, entry in enumerate(
                sorted(analyses, key=lambda entry: float(entry["base_logit"]), reverse=True),
                start=1,
            )
        }
        combined_rank_by_index = {
            int(entry["_candidate_index"]): rank
            for rank, entry in enumerate(
                sorted(analyses, key=lambda entry: float(entry["combined_score"]), reverse=True),
                start=1,
            )
        }
        for analysis in analyses:
            candidate_index = int(analysis["_candidate_index"])
            analysis["base_rank"] = base_rank_by_index[candidate_index]
            analysis["combined_rank"] = combined_rank_by_index[candidate_index]
            del analysis["_candidate_index"]

    analyses.sort(key=lambda entry: float(entry["combined_score"]), reverse=True)
    return best_candidate, best_score, analyses


@APP.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        model_id=SERVER_STATE["model_id"],
        model_name=SERVER_STATE["model_name"],
        switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
        runtime=entity_runtime_health(SERVER_STATE),
        request_metrics=snapshot_server_metrics(),
        voluntary_switch_rerank_penalty=DEFAULT_VOLUNTARY_SWITCH_RERANK_PENALTY,
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

        battle_id = str(data.get("battle_id") or "")
        last_turn_events = data.get("last_turn_events")  # list of event dicts, optional
        past_turn_events: list = []

        if SERVER_STATE.get("use_history") and battle_id and not SERVER_STATE.get("_disable_history_buffer"):
            K = int(SERVER_STATE.get("history_turns", 8))
            buf = _get_history_buffer(battle_id, K)
            if last_turn_events is not None:
                buf.append(list(last_turn_events))
            past_turn_events = list(buf)

        perspective_player = data.get("perspective_player")
        if perspective_player not in {"p1", "p2"}:
            return jsonify(error="perspective_player must be 'p1' or 'p2'"), 400

        legal_moves = data.get("legal_moves") or []
        revive_targets, revive_reason = filter_legal_revive_targets(
            data,
            data.get("legal_revives", []) or data.get("legal_switches", []) or [],
        )
        legal_switches, switch_reason = filter_legal_switches(data, data.get("legal_switches", []) or [])
        auxiliary_mode = str(data.get("auxiliary_mode") or "none").lower()
        opponent_action_token = data.get("opponent_action_token")
        if opponent_action_token is not None:
            opponent_action_token = str(opponent_action_token)
        auxiliary_scale = float(data.get("auxiliary_scale", 1.0))
        value_scale = float(data.get("value_scale", 1.0))
        sequence_scale = float(data.get("sequence_scale", 0.35))
        voluntary_switch_penalty = float(data.get("voluntary_switch_penalty", DEFAULT_VOLUNTARY_SWITCH_RERANK_PENALTY))

        inference_started = time.perf_counter()
        if SERVER_STATE.get("input_mode") == "entity_action_v2":
            logits, candidate_tokens, prediction_metadata = predict_entity_candidate_logits_with_metadata(
                SERVER_STATE,
                battle_state,
                perspective_player,
                legal_moves=legal_moves,
                legal_switches=legal_switches,
            )
        else:
            logits, prediction_metadata = predict_entity_logits_with_metadata(
                SERVER_STATE,
                battle_state,
                perspective_player,
            )
            candidate_tokens = []
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        switch_logit_bias = float(SERVER_STATE.get("switch_logit_bias", 0.0))
        value_estimate = prediction_metadata.get("value_estimate")
        if value_estimate is not None and float(value_estimate) > 0.5:
            switch_logit_bias += (float(value_estimate) - 0.5) * 18.0

        # Soft cooldown: bias against switches after a switch, but do NOT hard-block.
        # History of this knob:
        #   - Originally a hard block of 999.0 (consecutive switches effectively impossible).
        #     That hard block caused the agent to get stuck in losing positions where the
        #     only good move was a second switch (e.g. forced-out then chaining into a pivot).
        #   - 3.0 is the softer penalty that preserves the 42% WR baseline established in
        #     commit b7209b3 ("Optimize entity_action_v2 serving path") and 65a65cd ("ONNX
        #     inference (13.5x speedup) + value-head switch gating").
        #   - The value-head gating above already adds up to +9 when winning,
        #     so 3.0 compounds cleanly without swamping the signal.
        # If you re-introduce a hard block, document the regression trade-off above this line.
        if SERVER_STATE.get("_last_action_was_switch"):
            switch_logit_bias += 3.0
        auxiliary_analysis = None

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
                switch_logit_bias=switch_logit_bias,
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
            auxiliary_analysis = None
            if auxiliary_mode == "rerank":
                best_action, best_prob, auxiliary_analysis = _select_best_action_with_auxiliary(
                    battle_state=battle_state,
                    perspective_player=perspective_player,
                    logits=logits,
                    legal_moves=legal_moves,
                    legal_switches=legal_switches,
                    switch_reason=switch_reason,
                    switch_logit_bias=switch_logit_bias,
                    opponent_action_token=opponent_action_token,
                    auxiliary_scale=auxiliary_scale,
                    value_scale=value_scale,
                    sequence_scale=sequence_scale,
                    voluntary_switch_penalty=voluntary_switch_penalty,
                    past_turn_events=past_turn_events,
                )
                _append_aux_log({
                    "battle_id": battle_id,
                    "turn_number": battle_state.get("turn_index"),
                    "perspective_player": perspective_player,
                    "past_turns_in_buffer": len(past_turn_events),
                    "selected_action_token": best_action.get("token") if best_action else None,
                    "analyses": auxiliary_analysis,
                })
            else:
                best_action, best_prob = select_best_action(
                    SERVER_STATE["action_vocab"],
                    logits,
                    legal_moves,
                    legal_switches,
                    switch_reason=switch_reason,
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
                switch_logit_bias=switch_logit_bias,
                auxiliary_mode=auxiliary_mode,
                voluntary_switch_penalty=voluntary_switch_penalty,
                auxiliary_analysis=auxiliary_analysis,
            )

        selected_auxiliary = None
        selected_probability = None
        if auxiliary_mode == "rerank" and auxiliary_analysis:
            for analysis in auxiliary_analysis:
                if analysis.get("action_token") == best_action["token"]:
                    selected_probability = analysis.get("selection_probability")
                    selected_auxiliary = summarize_auxiliary_prediction(
                        sequence_tokens=list(analysis.get("sequence_tokens") or []),
                        sequence_score=analysis.get("sequence_score"),
                        value_prediction=analysis.get("value_prediction"),
                        used_opponent_action_token=analysis.get("used_opponent_action_token"),
                    )
                    break
        if selected_auxiliary is None and auxiliary_mode in {"inspect", "rerank"}:
            selected_outputs = predict_entity_auxiliary_outputs(
                SERVER_STATE,
                battle_state,
                perspective_player,
                my_action_token=str(best_action["token"]),
                opp_action_token=opponent_action_token,
            )
            if selected_outputs is not None:
                selected_tokens = list(selected_outputs.get("sequence_tokens") or [])
                selected_sequence_score = None
                if selected_tokens:
                    selected_sequence_score = score_sequence_tokens(
                        selected_tokens,
                        perspective_player=perspective_player,
                    )
                selected_auxiliary = summarize_auxiliary_prediction(
                    sequence_tokens=selected_tokens,
                    sequence_score=selected_sequence_score,
                    value_prediction=selected_outputs.get("value_prediction"),
                    used_opponent_action_token=opponent_action_token,
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
                selection_probability=selected_probability,
                action_token=best_action["token"],
                model_id=SERVER_STATE["model_id"],
                supported_model_ids=[SERVER_STATE["model_id"]],
                switch_logit_bias=switch_logit_bias,
                auxiliary_mode=auxiliary_mode,
                voluntary_switch_penalty=voluntary_switch_penalty,
                auxiliary=selected_auxiliary,
                auxiliary_analysis=auxiliary_analysis,
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
            selection_probability=selected_probability,
            action_token=best_action["token"],
            switch_reason=switch_reason,
            model_id=SERVER_STATE["model_id"],
            supported_model_ids=[SERVER_STATE["model_id"]],
            switch_logit_bias=switch_logit_bias,
            auxiliary_mode=auxiliary_mode,
            voluntary_switch_penalty=voluntary_switch_penalty,
            auxiliary=selected_auxiliary,
            auxiliary_analysis=auxiliary_analysis,
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
    SERVER_STATE["_capture_aux_log_path"] = getattr(args, "capture_aux_log", None)
    SERVER_STATE["_capture_aux_log_lock"] = threading.Lock()
    SERVER_STATE["_disable_history_buffer"] = bool(getattr(args, "disable_history_buffer", False))
    if SERVER_STATE["_disable_history_buffer"]:
        print("[entity-server] --disable-history-buffer: past_turn_events will be forced empty")
    reset_server_metrics()
    print(f"[entity-server] model_id={SERVER_STATE['model_id']}")
    print(f"[entity-server] metadata_path={metadata_path}")
    APP.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
