from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)

# paths to the serialized artifacts created by the notebook
MODEL_PATH = "model.h5"
VOCAB_PATH = "move_vocab.json"

# load once at startup
model = tf.keras.models.load_model(MODEL_PATH)
with open(VOCAB_PATH, "r") as f:
    move_vocab = json.load(f)

# invert vocab for convenience
idx_to_move = {v: k for k, v in move_vocab.items()}


def pick_best_move(logits: np.ndarray, legal_moves: list[str]) -> tuple[str | None, float]:
    """Return (move, probability). If no legal move is in the vocab we return (None,0)."""
    probs = tf.nn.softmax(logits).numpy()
    best_move = None
    best_prob = -1.0
    # print(legal_moves)
    for mv in legal_moves:
        print('Move is : ', mv)
        move_name = mv['move'].lower().replace(' ', '')
        idx = move_vocab.get(move_name)
        print(idx)
        if idx is None:
            continue
        print("Found move in trained vocab")
        p = float(probs[idx])

        if p > best_prob:
            best_prob = p
            best_move = mv
        print("best move is ", best_move, " probability: ", best_prob)
    return best_move, best_prob

@app.route("/test", methods=["GET"])
def test():
    print("This is a test")


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
        print(data)
        if 'forceSwitch' in data and data['forceSwitch']:
            legal_switches = data['side']  
            selected_idx = np.random.choice(len(legal_switches))
            
            print(legal_switches[selected_idx]['p'].slot)
            return jsonify(type= 'switch', best_switch=legal_switches[selected_idx]['p'].slot)
        if data is None:
            return jsonify(error="invalid JSON"), 400

        vec = data.get("state_vector")
        if vec is None:
            return jsonify(error="missing state_vector"), 400

        legal_moves = data.get("legal_moves") or []
        legal_switches = data.get("legal_switches", []) or []

        # shape (1, D)
        arr = np.asarray([vec], dtype=np.float32)
        logits = model.predict(arr, verbose=0)[0]

        best_move, best_prob = pick_best_move(logits, legal_moves)
        print(best_move)
        print('close to end request')
    
        if best_move is None:
            # no legal move found in vocab, signal a switch recommendation
            return jsonify(
                best_move=None,
                type= 'move',
                note="no legal moves found in vocabulary; consider switching",
                legal_switches=legal_switches,
            )

        return jsonify(best_move=best_move, type='move', probability=best_prob)
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Server error: {str(e)}"), 500


if __name__ == "__main__":
    # simple development server
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
