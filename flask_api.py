from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

app = Flask(__name__)

# paths to the serialized artifacts created by the notebook
MODEL_PATH = "artifacts/model.keras"
VOCAB_CANDIDATES = [
    Path("artifacts/action_vocab.json"),
    Path("artifacts/move_vocab.json"),
    Path("action_vocab.json"),
    Path("move_vocab.json"),
]

# load once at startup
model = tf.keras.models.load_model(MODEL_PATH)
EXPECTED_INPUT_DIM = None
if getattr(model, "input_shape", None):
    shape = model.input_shape
    if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] is not None:
        EXPECTED_INPUT_DIM = int(shape[-1])

VOCAB_PATH = next((path for path in VOCAB_CANDIDATES if path.exists()), VOCAB_CANDIDATES[-1])
with open(VOCAB_PATH, "r") as f:
    action_vocab = json.load(f)

# invert vocab for convenience
idx_to_action = {v: k for k, v in action_vocab.items()}

def vocab_uses_action_tokens() -> bool:
    return any(
        token.startswith(("move:", "switch:"))
        for token in action_vocab
        if token != "<UNK>"
    )


def move_vocab_tokens(move_name: str) -> list[str]:
    normalized = move_name.lower().replace(" ", "")
    if vocab_uses_action_tokens():
        return [f"move:{normalized}"]
    return [normalized]


def switch_vocab_tokens(slot: int) -> list[str]:
    if vocab_uses_action_tokens():
        return [f"switch:{slot}"]
    return []


def pick_best_action(
    logits: np.ndarray,
    legal_moves: list[dict],
    legal_switches: list[dict],
) -> tuple[dict | None, float]:
    """Return the highest-probability legal action supported by the current vocab."""
    probs = tf.nn.softmax(logits).numpy()
    best_action = None
    best_prob = -1.0

    for mv in legal_moves:
        print('Move is : ', mv)
        move_name = mv.get("move", "")
        for token in move_vocab_tokens(move_name):
            idx = action_vocab.get(token)
            print(idx)
            if idx is None:
                continue
            p = float(probs[idx])
            if p > best_prob:
                best_prob = p
                best_action = {"type": "move", "payload": mv, "token": token}

    for sw in legal_switches:
        slot = sw.get("slot")
        if slot is None:
            continue
        for token in switch_vocab_tokens(int(slot)):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            p = float(probs[idx])
            if p > best_prob:
                best_prob = p
                best_action = {"type": "switch", "payload": sw, "token": token}

    return best_action, best_prob

@app.route("/test", methods=["GET"])
def test():
    print("This is a test")

import random
from flask import request, jsonify

# Temporary Separation of Switch behavior.. Later, predict should also account for this.
# Or perhaps not? Will evaluate if switch model might be best as a standalone module
@app.route("/predict/switch", methods=["POST"])
def choose_switch():
    data = request.json

    legal_switches = data.get("legal_switches", [])

    if not legal_switches:
        return jsonify({"error": "No legal switches provided"}), 400

    # Pick random legal switch
    chosen = random.choice(legal_switches)

    slot = chosen.get("slot")

    if slot is None:
        return jsonify({"error": "Malformed legal switch object"}), 400

    print({"type": "switch",
        "slot": slot
    })
    # Showdown format
    return jsonify({"type": "switch",
        "slot": slot
    })

@app.route("/predict", methods=["POST"])
def predict():
    """expects JSON with keys:
        - state_vector: list[float]  (already vectorized, shape = D)
        - legal_moves: list[str]    (move ids)
        - legal_switches: list[str] (optional, not used by model)
    """
    print('begin request')
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(error="invalid JSON"), 400

        vec = data.get("state_vector")
        if vec is None:
            return jsonify(error="missing state_vector"), 400
        if EXPECTED_INPUT_DIM is not None and len(vec) != EXPECTED_INPUT_DIM:
            return jsonify(
                error=(
                    f"state_vector has length {len(vec)} but model expects {EXPECTED_INPUT_DIM}. "
                    "Retrain or swap the saved model if the vectorizer changed."
                )
            ), 400

        legal_moves = data.get("legal_moves") or []
        legal_switches = data.get("legal_switches", []) or []

        # shape (1, D)
        arr = np.asarray([vec], dtype=np.float32)
        logits = model.predict(arr, verbose=0)[0]

        best_action, best_prob = pick_best_action(logits, legal_moves, legal_switches)
        print(best_action)
    
        if best_action is None:
            return jsonify(
                best_action=None,
                type='none',
                note="no legal actions found in vocabulary",
                legal_moves=legal_moves,
                legal_switches=legal_switches,
            )

        if best_action["type"] == "move":
            return jsonify(
                best_move=best_action["payload"],
                type='move',
                probability=best_prob,
                action_token=best_action["token"],
            )

        return jsonify(
            best_switch=best_action["payload"],
            slot=best_action["payload"].get("slot"),
            type='switch',
            probability=best_prob,
            action_token=best_action["token"],
        )
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Server error: {str(e)}"), 500


if __name__ == "__main__":
    # simple development server
    app.run(host="0.0.0.0", port=5000, debug=True)
