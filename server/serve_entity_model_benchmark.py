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
from typing import Any

import numpy as np
from flask import Flask, jsonify, request


REPO_PATH = Path(__file__).resolve().parent.parent
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from ActionLegality import filter_legal_revive_targets, filter_legal_switches
from EntityServerRuntime import load_entity_runtime_artifacts, predict_entity_logits
from core.ActionSelection import pick_best_action as select_best_action
from core.ActionSelection import pick_best_slot_target


APP = Flask(__name__)
SERVER_STATE: dict[str, Any] = {}


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


def predict_logits(battle_state: dict[str, Any], perspective_player: str) -> np.ndarray:
    return predict_entity_logits(SERVER_STATE, battle_state, perspective_player)


@APP.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        model_id=SERVER_STATE["model_id"],
        model_name=SERVER_STATE["model_name"],
        switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
    )


@APP.route("/predict", methods=["POST"])
def predict():
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

        logits = predict_logits(battle_state, perspective_player)

        # Apply value-based switch gating if we have a value head
        switch_logit_bias = float(SERVER_STATE.get("switch_logit_bias", 0.0))
        if SERVER_STATE.get("has_value_head"):
            value_estimate = SERVER_STATE.get("_last_value_estimate", 0.5)
            # If winning (value > 0.5), strongly penalize switches
            # If losing (value < 0.5), allow switches
            if value_estimate > 0.5:
                # Winning: penalize switches more (value 0.51 -> penalty 4.5, value 1.0 -> penalty 9)
                switch_logit_bias += (value_estimate - 0.5) * 18
            else:
                # Losing: reduce penalty (value 0.5 -> penalty 0.2, value 0.0 -> penalty -1.0)
                switch_logit_bias = max(-0.5, switch_logit_bias - (0.5 - value_estimate) * 3.4)

        if revive_targets:
            best_revive, best_prob = pick_best_slot_target(
                SERVER_STATE["action_vocab"],
                logits,
                revive_targets,
                "revive",
            )
            if best_revive is None:
                return jsonify(
                    type="none",
                    note="revive target request detected but no legal revive targets were available",
                    revive_reason=revive_reason,
                    model_id=SERVER_STATE["model_id"],
                    supported_model_ids=[SERVER_STATE["model_id"]],
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

        best_action, best_prob = select_best_action(
            SERVER_STATE["action_vocab"],
            logits,
            legal_moves,
            legal_switches,
            switch_logit_bias=switch_logit_bias,
        )
        if best_action is None:
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
            return jsonify(
                best_move=best_action["payload"],
                type="move",
                probability=best_prob,
                action_token=best_action["token"],
                model_id=SERVER_STATE["model_id"],
                supported_model_ids=[SERVER_STATE["model_id"]],
                switch_logit_bias=float(SERVER_STATE.get("switch_logit_bias", 0.0)),
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
        import traceback

        traceback.print_exc()
        return jsonify(error=f"Server error: {error}"), 500


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path).resolve()
    global SERVER_STATE
    SERVER_STATE = load_server_state(metadata_path)
    print(f"[entity-server] model_id={SERVER_STATE['model_id']}")
    print(f"[entity-server] metadata_path={metadata_path}")
    APP.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
