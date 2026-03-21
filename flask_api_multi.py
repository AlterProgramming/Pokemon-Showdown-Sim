from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one or more Pokemon Showdown policy models.")
    parser.add_argument("--repo-path", help="Training repo root containing artifacts and ActionLegality.py")
    parser.add_argument("--mode", choices=["model1", "model2", "model2_large", "multi"], default="multi")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


ARGS = parse_args()
REPO_PATH = Path(ARGS.repo_path).resolve() if ARGS.repo_path else Path(__file__).resolve().parent
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from ActionLegality import filter_legal_revive_targets, filter_legal_switches  # noqa: E402


def resolve_vocab_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve action vocab from: {candidates}")


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


def load_model_artifacts(model_id: str, model_path: Path, vocab_candidates: list[Path]) -> dict[str, Any]:
    model = tf.keras.models.load_model(model_path)
    expected_input_dim = None
    if getattr(model, "input_shape", None):
        shape = model.input_shape
        if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] is not None:
            expected_input_dim = int(shape[-1])

    vocab_path = resolve_vocab_path(vocab_candidates)
    with open(vocab_path, "r", encoding="utf-8") as handle:
        action_vocab = json.load(handle)

    return {
        "model_id": model_id,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "model": model,
        "expected_input_dim": expected_input_dim,
        "action_vocab": action_vocab,
    }


def load_models(mode: str) -> tuple[dict[str, dict[str, Any]], str]:
    base_configs: dict[str, tuple[Path, list[Path]]] = {
        "model1": (
            REPO_PATH / "artifacts" / "model_1.keras",
            [
                REPO_PATH / "artifacts" / "action_vocab_1.json",
                REPO_PATH / "artifacts" / "move_vocab.json",
                REPO_PATH / "move_vocab.json",
            ],
        ),
        "model2": (
            REPO_PATH / "artifacts" / "model_2.keras",
            [
                REPO_PATH / "artifacts" / "action_vocab_2.json",
                REPO_PATH / "artifacts" / "move_vocab.json",
                REPO_PATH / "move_vocab.json",
            ],
        ),
        "model2_large": (
            REPO_PATH / "artifacts" / "model_2_large.keras",
            [
                REPO_PATH / "artifacts" / "action_vocab_2_large.json",
                REPO_PATH / "artifacts" / "move_vocab.json",
                REPO_PATH / "move_vocab.json",
            ],
        ),
    }

    model_ids = ["model1", "model2", "model2_large"] if mode == "multi" else [mode]
    artifacts = {
        model_id: load_model_artifacts(model_id, *base_configs[model_id])
        for model_id in model_ids
    }
    default_model_id = "model1" if "model1" in artifacts else model_ids[0]
    return artifacts, default_model_id


MODEL_ARTIFACTS, DEFAULT_MODEL_ID = load_models(ARGS.mode)
APP = Flask(__name__)


def choose_model_artifacts(request_data: dict[str, Any]) -> dict[str, Any]:
    requested_model_id = request_data.get("model_id") or DEFAULT_MODEL_ID
    artifacts = MODEL_ARTIFACTS.get(str(requested_model_id))
    if artifacts is None:
        raise KeyError(
            f"Unsupported model_id '{requested_model_id}'. "
            f"Supported values: {', '.join(sorted(MODEL_ARTIFACTS))}"
        )
    return artifacts


def predict_logits(model_artifacts: dict[str, Any], state_vector: list[float]) -> np.ndarray:
    arr = np.asarray([state_vector], dtype=np.float32)
    return model_artifacts["model"].predict(arr, verbose=0)[0]


def pick_best_slot_target(
    model_artifacts: dict[str, Any],
    logits: np.ndarray,
    slot_targets: list[dict[str, Any]],
    target_type: str,
) -> tuple[dict[str, Any] | None, float | None]:
    probs = tf.nn.softmax(logits).numpy()
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
    probs = tf.nn.softmax(logits).numpy()
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
    return jsonify(
        status="ok",
        mode=ARGS.mode,
        default_model_id=DEFAULT_MODEL_ID,
        supported_model_ids=sorted(MODEL_ARTIFACTS.keys()),
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

        logits = predict_logits(model_artifacts, state_vector)

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
    except Exception as error:
        import traceback

        traceback.print_exc()
        return jsonify(error=f"Server error: {error}"), 500


if __name__ == "__main__":
    APP.run(host=ARGS.host, port=ARGS.port, debug=False, use_reloader=False)
