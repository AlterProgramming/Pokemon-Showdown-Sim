from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as _ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

from EntityInvarianceTensorization import to_numpy_invariance_inputs
from EntityTensorization import encode_entity_state, to_single_example_entity_inputs
from EntityTensorizationV2 import encode_entity_state_with_candidates, to_single_example_entity_v2_inputs
from ModelRegistry import resolve_artifact_path
from core.ActionSelection import resolve_switch_logit_bias


build_entity_invariance_models = None
build_entity_action_models = None
build_entity_action_v2_models = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(logits)
    return exp / denom


def _entity_dummy_inputs() -> dict[str, np.ndarray]:
    return {
        "global_conditions":      np.zeros((1, 5),      dtype=np.int32),
        "global_numeric":         np.zeros((1, 17),     dtype=np.float32),
        "pokemon_ability":        np.zeros((1, 12),     dtype=np.int32),
        "pokemon_item":           np.zeros((1, 12),     dtype=np.int32),
        "pokemon_numeric":        np.zeros((1, 12, 13), dtype=np.float32),
        "pokemon_observed_moves": np.zeros((1, 12, 4),  dtype=np.int32),
        "pokemon_side":           np.zeros((1, 12),     dtype=np.int32),
        "pokemon_slot":           np.zeros((1, 12),     dtype=np.int32),
        "pokemon_species":        np.zeros((1, 12),     dtype=np.int32),
        "pokemon_status":         np.zeros((1, 12),     dtype=np.int32),
        "pokemon_tera":           np.zeros((1, 12),     dtype=np.int32),
        "weather":                np.zeros((1, 1),      dtype=np.int32),
    }


def _entity_v2_dummy_inputs(max_candidates: int) -> dict[str, np.ndarray]:
    dummy = _entity_dummy_inputs()
    dummy.update(
        {
            "candidate_type": np.zeros((1, max_candidates), dtype=np.int32),
            "candidate_move": np.zeros((1, max_candidates), dtype=np.int32),
            "candidate_switch_slot": np.zeros((1, max_candidates), dtype=np.int32),
            "candidate_mask": np.zeros((1, max_candidates), dtype=np.float32),
        }
    )
    return dummy


def _cast_for_onnx_input(spec: Any, arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if spec.type == "tensor(int32)" and arr.dtype != np.int32:
        return arr.astype(np.int32)
    if spec.type == "tensor(int64)" and arr.dtype != np.int64:
        return arr.astype(np.int64)
    if spec.type == "tensor(float)" and arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def _run_onnx_outputs(onnx_session: Any, batched_inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    onnx_inputs = {i.name: _cast_for_onnx_input(i, batched_inputs[i.name]) for i in onnx_session.get_inputs()}
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    output_names = [o.name for o in onnx_session.get_outputs()]
    return dict(zip(output_names, onnx_outputs))


def load_entity_runtime_artifacts(
    metadata_path: Path,
    *,
    repo_path: Path | None = None,
) -> dict[str, Any]:
    metadata_path = metadata_path.resolve()
    repo_path = repo_path.resolve() if repo_path is not None else metadata_path.parent.parent.resolve()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    family_id = str(metadata.get("family_id") or "")
    if family_id == "entity_action_v2":
        model_path_key = "training_model_path" if bool(metadata.get("predict_value")) and metadata.get("training_model_path") else "policy_model_path"
    else:
        model_path_key = "policy_value_model_path" if "policy_value_model_path" in metadata else "policy_model_path"

    model_path = resolve_artifact_path(repo_path, metadata_path, str(metadata[model_path_key]))
    policy_vocab_path = None
    if metadata.get("policy_vocab_path"):
        policy_vocab_path = resolve_artifact_path(repo_path, metadata_path, str(metadata["policy_vocab_path"]))
    token_vocab_path = resolve_artifact_path(repo_path, metadata_path, str(metadata["entity_token_vocab_path"]))

    action_vocab = None
    if policy_vocab_path is not None and policy_vocab_path.exists():
        with open(policy_vocab_path, "r", encoding="utf-8") as handle:
            action_vocab = json.load(handle)
    with open(token_vocab_path, "r", encoding="utf-8") as handle:
        token_vocabs = json.load(handle)

    has_value_head = False
    try:
        if family_id == "entity_action_bc":
            global build_entity_action_models
            if build_entity_action_models is None:
                from EntityModelV1 import build_entity_action_models as _build_entity_action_models

                build_entity_action_models = _build_entity_action_models
            # Rebuild the model architecture from metadata, handling both policy-only and policy-value variants.
            # The first entity family uses Lambda helpers that are unreliable to deserialize directly.
            # Instead, we rebuild the known family architecture and load weights into it.
            if model_path_key == "policy_value_model_path":
                # Policy-value model: rebuild with value head enabled
                _, policy_only_model, policy_value_model = build_entity_action_models(
                    vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
                    num_policy_classes=int(metadata["num_action_classes"]),
                    hidden_dim=int(metadata["hidden_dim"]),
                    depth=int(metadata["depth"]),
                    dropout=float(metadata["dropout"]),
                    learning_rate=float(metadata["learning_rate"]),
                    token_embed_dim=int(metadata.get("token_embed_dim", 24)),
                    predict_value=True,
                    value_hidden_dim=int(metadata.get("value_hidden_dim", 128)),
                    value_weight=float(metadata.get("value_weight", 0.25)),
                )
                policy_value_model.load_weights(model_path)
                model = policy_value_model
                has_value_head = True
                input_mode = "entity_action"
            else:
                # Policy-only model: rebuild without value head
                _, model, _ = build_entity_action_models(
                    vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
                    num_policy_classes=int(metadata["num_action_classes"]),
                    hidden_dim=int(metadata["hidden_dim"]),
                    depth=int(metadata["depth"]),
                    dropout=float(metadata["dropout"]),
                    learning_rate=float(metadata["learning_rate"]),
                    token_embed_dim=int(metadata.get("token_embed_dim", 24)),
                )
                model.load_weights(model_path)
                input_mode = "entity_action"
        elif family_id == "entity_action_v2":
            global build_entity_action_v2_models
            if build_entity_action_v2_models is None:
                from EntityModelV2 import build_entity_action_v2_models as _build_entity_action_v2_models

                build_entity_action_v2_models = _build_entity_action_v2_models
            training_model, policy_model, policy_value_model = build_entity_action_v2_models(
                vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
                hidden_dim=int(metadata["hidden_dim"]),
                depth=int(metadata["depth"]),
                dropout=float(metadata["dropout"]),
                learning_rate=float(metadata["learning_rate"]),
                token_embed_dim=int(metadata.get("token_embed_dim", 24)),
                max_candidates=int(metadata.get("max_candidates", 10)),
                predict_value=bool(metadata.get("predict_value")),
                value_weight=float(metadata.get("value_weight", 0.25)),
            )
            if bool(metadata.get("predict_value")) and model_path_key == "training_model_path":
                training_model.load_weights(model_path)
                model = training_model
                has_value_head = True
            else:
                policy_model.load_weights(model_path)
                model = policy_model
            input_mode = "entity_action_v2"
        elif family_id == "entity_invariance_aux":
            global build_entity_invariance_models
            if build_entity_invariance_models is None:
                from EntityInvarianceModelV1 import build_entity_invariance_models as _build_entity_invariance_models

                build_entity_invariance_models = _build_entity_invariance_models
            _, model, _ = build_entity_invariance_models(
                vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
                num_policy_classes=int(metadata["num_action_classes"]),
                hidden_dim=int(metadata["hidden_dim"]),
                depth=int(metadata["depth"]),
                dropout=float(metadata["dropout"]),
                learning_rate=float(metadata["learning_rate"]),
                token_embed_dim=int(metadata.get("token_embed_dim", 24)),
                latent_dim=int(metadata.get("latent_dim", 64)),
                transition_dim=None,
                action_context_vocab_size=None,
                predict_value=False,
            )
            model.load_weights(model_path)
            input_mode = "entity_invariance"
        else:
            raise SystemExit(
                "This server only supports entity_action_bc, entity_action_v2, and entity_invariance_aux family models."
            )
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is required to serve entity models.") from exc
    # Attempt ONNX export for fast inference, including the legal-candidate v2 family.
    onnx_session = None
    if _ORT_AVAILABLE:
        onnx_path = model_path.parent / (model_path.stem + ".onnx")
        if not onnx_path.exists():
            try:
                import tensorflow as tf
                import tf2onnx

                dummy = (
                    _entity_v2_dummy_inputs(int(metadata.get("max_candidates", 10)))
                    if family_id == "entity_action_v2"
                    else _entity_dummy_inputs()
                )
                input_signature = {k: tf.TensorSpec(v.shape, dtype=v.dtype, name=k) for k, v in dummy.items()}

                @tf.function(input_signature=[input_signature])
                def _infer(inputs):
                    return model(inputs, training=False)

                _ = _infer(dummy)  # trace
                tf2onnx.convert.from_function(_infer, input_signature=[input_signature], output_path=str(onnx_path))
                print(f"[runtime] ONNX model exported to {onnx_path}")
            except Exception as exc:
                print(f"[runtime] ONNX export failed, falling back to Keras: {exc}")
                onnx_path = None

        if onnx_path and onnx_path.exists():
            try:
                onnx_session = _ort.InferenceSession(str(onnx_path))
                dummy = (
                    _entity_v2_dummy_inputs(int(metadata.get("max_candidates", 10)))
                    if family_id == "entity_action_v2"
                    else _entity_dummy_inputs()
                )
                _ = _run_onnx_outputs(onnx_session, dummy)
                print(f"[runtime] ONNX session loaded — fast inference enabled (~0.4ms/request)")
            except Exception as exc:
                print(f"[runtime] ONNX session load failed: {exc}")
    if onnx_session is None:
        try:
            _ = model(_entity_dummy_inputs(), training=False)
        except Exception as exc:
            print(f"[runtime] Keras warmup failed: {exc}")

    model_id = str(metadata.get("model_release_id") or metadata.get("model_name") or model_path.stem)
    family_default_switch_bias = 0.20 if family_id == "entity_action_bc" else 0.0
    return {
        "kind": "entity",
        "model": model,
        "onnx_session": onnx_session,
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "family_id": family_id,
        "family_version": metadata.get("family_version"),
        "family_name": metadata.get("family_name"),
        "input_mode": input_mode,
        "metadata_path": str(metadata_path),
        "model_path": str(model_path),
        "policy_vocab_path": str(policy_vocab_path) if policy_vocab_path is not None else None,
        "entity_token_vocab_path": str(token_vocab_path),
        "action_vocab": action_vocab,
        "token_vocabs": token_vocabs,
        "switch_logit_bias": resolve_switch_logit_bias(metadata, default=family_default_switch_bias),
        "has_value_head": has_value_head,
        "max_candidates": int(metadata.get("max_candidates", 10)),
    }


def to_single_example_invariance_inputs(current_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Batch one invariance example, using zero previous-turn context at inference."""
    raw_inputs: dict[str, Any] = {}
    for key, value in current_inputs.items():
        raw_inputs[key] = [value]
        raw_inputs[f"prev_{key}"] = [np.zeros_like(np.asarray(value)).tolist()]
    return to_numpy_invariance_inputs(raw_inputs)


def predict_entity_logits(
    runtime: dict[str, Any],
    battle_state: dict[str, Any],
    perspective_player: str,
) -> np.ndarray:
    encoded = encode_entity_state(
        battle_state,
        perspective_player=perspective_player,
        token_vocabs=runtime["token_vocabs"],
    )
    if runtime["input_mode"] == "entity_invariance":
        batched_inputs = to_single_example_invariance_inputs(encoded)
    else:
        batched_inputs = to_single_example_entity_inputs(encoded)
    # Use ONNX if available (~0.4ms), else direct model() call (~55ms), never model.predict() (~153ms)
    onnx_session = runtime.get("onnx_session")
    if onnx_session is not None:
        raw_output = _run_onnx_outputs(onnx_session, batched_inputs)
    else:
        raw_output = runtime["model"](batched_inputs, training=False)
        # Direct call returns tensors — convert to numpy
        if isinstance(raw_output, dict):
            raw_output = {k: v.numpy() for k, v in raw_output.items()}
        elif isinstance(raw_output, (list, tuple)):
            raw_output = [v.numpy() if hasattr(v, "numpy") else v for v in raw_output]
        elif hasattr(raw_output, "numpy"):
            raw_output = raw_output.numpy()

    # Handle policy-value model output (dict with policy and value keys)
    if runtime.get("has_value_head") and isinstance(raw_output, dict) and "value" in raw_output:
        policy_logits = np.asarray(raw_output["policy"][0], dtype=np.float32)
        value_estimate = np.asarray(raw_output["value"][0][0], dtype=np.float32)
        runtime["_last_value_estimate"] = value_estimate
        return policy_logits

    # Handle policy-value model output (tuple of policy, value)
    if runtime.get("has_value_head") and isinstance(raw_output, (list, tuple)) and len(raw_output) >= 2:
        policy_logits = np.asarray(raw_output[0][0], dtype=np.float32)
        value_estimate = np.asarray(raw_output[1][0][0], dtype=np.float32)
        runtime["_last_value_estimate"] = value_estimate
        return policy_logits

    # Handle dict output (policy extracted)
    if isinstance(raw_output, dict):
        raw_output = raw_output.get("policy")

    # Handle list/tuple output (first element is policy)
    if isinstance(raw_output, (list, tuple)):
        raw_output = raw_output[0]

    return np.asarray(raw_output[0], dtype=np.float32)


def predict_entity_candidate_logits(
    runtime: dict[str, Any],
    battle_state: dict[str, Any],
    perspective_player: str,
    *,
    legal_moves: list[dict[str, Any]] | None = None,
    legal_switches: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if runtime.get("input_mode") != "entity_action_v2":
        raise ValueError("predict_entity_candidate_logits requires an entity_action_v2 runtime")

    encoded = encode_entity_state_with_candidates(
        battle_state,
        perspective_player=perspective_player,
        token_vocabs=runtime["token_vocabs"],
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        max_candidates=int(runtime.get("max_candidates", 10)),
    )
    candidate_tokens = list(encoded["candidate_tokens"])
    batched_inputs = to_single_example_entity_v2_inputs(encoded)

    onnx_session = runtime.get("onnx_session")
    if onnx_session is not None:
        raw_output = _run_onnx_outputs(onnx_session, batched_inputs)
    else:
        raw_output = runtime["model"](batched_inputs, training=False)
        if isinstance(raw_output, dict):
            raw_output = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in raw_output.items()}
        elif isinstance(raw_output, (list, tuple)):
            raw_output = [v.numpy() if hasattr(v, "numpy") else v for v in raw_output]
        elif hasattr(raw_output, "numpy"):
            raw_output = raw_output.numpy()

    if runtime.get("has_value_head") and isinstance(raw_output, dict) and "value" in raw_output:
        runtime["_last_value_estimate"] = np.asarray(raw_output["value"][0][0], dtype=np.float32)
        raw_output = raw_output["policy"]
    elif runtime.get("has_value_head") and isinstance(raw_output, (list, tuple)) and len(raw_output) >= 2:
        runtime["_last_value_estimate"] = np.asarray(raw_output[1][0][0], dtype=np.float32)
        raw_output = raw_output[0]
    elif isinstance(raw_output, dict):
        raw_output = raw_output.get("policy")
    elif isinstance(raw_output, (list, tuple)):
        raw_output = raw_output[0]

    return np.asarray(raw_output[0], dtype=np.float32), candidate_tokens


def entity_runtime_health(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": runtime["model_id"],
        "alive": True,
        "kind": runtime.get("kind", "entity"),
        "family_id": runtime.get("family_id"),
        "family_version": runtime.get("family_version"),
        "family_name": runtime.get("family_name"),
        "model_name": runtime.get("model_name"),
        "input_mode": runtime.get("input_mode"),
        "metadata_path": runtime.get("metadata_path"),
        "model_path": runtime.get("model_path"),
        "policy_vocab_path": runtime.get("policy_vocab_path"),
        "entity_token_vocab_path": runtime.get("entity_token_vocab_path"),
        "switch_logit_bias": runtime.get("switch_logit_bias", 0.0),
    }
