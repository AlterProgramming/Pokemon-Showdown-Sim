"""Network of Theseus utilities for the Model1-to-Elman conversion.

The pure NumPy helpers in this module deliberately do not import TensorFlow so
that sequence-contract and alignment tests can run in lightweight environments.
Keras builders import TensorFlow lazily at call time.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


MODEL1_INPUT_DIM = 582
MODEL1_ACTION_CLASSES = 357
MODEL1_GUIDE_ID = "model1"
TARGET_MODEL_ID = "model1_not_elman3"
TARGET_FAMILY_ID = "vector_joint_bc_not_elman"
TARGET_FAMILY_VERSION = 1
DEFAULT_ELMAN_HIDDEN_DIM = 256
DEFAULT_ELMAN_DEPTH = 3
DEFAULT_SEQUENCE_LENGTH = 16


def progressive_replacement_schedule(depth: int) -> tuple[tuple[int, ...], ...]:
    """Return the progressive prefix schedule used by Network of Theseus."""
    if depth < 0:
        raise ValueError("depth must be non-negative")
    return tuple(tuple(range(stage + 1)) for stage in range(depth))


def _coerce_history(history: Sequence[Sequence[float]] | np.ndarray, feature_dim: int) -> np.ndarray:
    array = np.asarray(history, dtype=np.float32)
    if array.size == 0:
        array = np.empty((0, feature_dim), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("history must be a rank-2 array of state vectors")
    if array.shape[1] != feature_dim:
        raise ValueError(
            f"history feature dimension {array.shape[1]} does not match expected {feature_dim}"
        )
    return array


def pad_state_history(
    history: Sequence[Sequence[float]] | np.ndarray,
    *,
    sequence_length: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate a history using the paper-reproduction repeat-first rule.

    The returned mask is one for real observations and zero for left padding.
    A completely empty history is represented by zero vectors and an all-zero
    mask; normal inference callers should provide at least the current state.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    array = _coerce_history(history, feature_dim)
    if len(array) > sequence_length:
        array = array[-sequence_length:]

    valid_length = len(array)
    if valid_length == 0:
        return (
            np.zeros((sequence_length, feature_dim), dtype=np.float32),
            np.zeros((sequence_length,), dtype=np.float32),
        )

    padding_length = sequence_length - valid_length
    if padding_length:
        padding = np.repeat(array[:1], padding_length, axis=0)
        array = np.concatenate((padding, array), axis=0)
    mask = np.concatenate(
        (
            np.zeros((padding_length,), dtype=np.float32),
            np.ones((valid_length,), dtype=np.float32),
        )
    )
    return array.astype(np.float32, copy=False), mask


def normalize_model1_state_vector(state_vector: Sequence[float]) -> list[float]:
    """Reduce current simulator layouts to Model1's historical 582 features."""
    values = list(state_vector)
    if len(values) == MODEL1_INPUT_DIM:
        return values
    if len(values) == 678:
        # Older augmented layout: the 96-dimensional team block was inserted
        # immediately before the final 26 base features.
        return values[:556] + values[652:]
    if len(values) == 690:
        # Current layout adds one slot_exists feature to each of twelve bench
        # slots and inserts the 96-dimensional team block before field features.
        normalized = values[:232]
        for block_start in (232, 400):
            block = values[block_start:block_start + 168]
            for offset in range(0, 168, 28):
                normalized.extend(block[offset + 1:offset + 28])
        normalized.extend(values[664:])
        if len(normalized) != MODEL1_INPUT_DIM:
            raise ValueError("failed to normalize the current 690-dimensional state layout")
        return normalized
    raise ValueError(
        f"state vector has length {len(values)}; expected {MODEL1_INPUT_DIM}, 678, or 690"
    )


def flatten_state_history(
    history: Sequence[Sequence[float]],
    *,
    sequence_length: int,
    feature_dim: int = MODEL1_INPUT_DIM,
) -> list[float]:
    """Normalize, pad, and flatten a request history for the serving wrapper."""
    if not history:
        raise ValueError("state history must contain at least one observation")
    normalized = [
        normalize_model1_state_vector(vector) if feature_dim == MODEL1_INPUT_DIM else list(vector)
        for vector in history
    ]
    padded, _ = pad_state_history(
        normalized,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
    )
    return padded.reshape(-1).astype(np.float32).tolist()


def prepare_sequence_request_state(
    model_entry: dict[str, Any],
    request_data: dict[str, Any],
) -> list[float] | None:
    """Prepare the flat worker input for a metadata-declared sequence model."""
    if not bool(model_entry.get("sequence_model")):
        return None
    history = request_data.get("state_history")
    if not isinstance(history, list) or not history:
        raise ValueError("sequence model requests require a non-empty state_history list")
    sequence_length = int(model_entry.get("sequence_length") or 0)
    feature_dim = int(model_entry.get("base_feature_dim") or MODEL1_INPUT_DIM)
    if sequence_length <= 0:
        raise ValueError("sequence model metadata requires a positive sequence_length")
    return flatten_state_history(
        history,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
    )


def _validate_representations(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("representations must be rank-2 arrays")
    if left_array.shape[0] != right_array.shape[0]:
        raise ValueError("representations must contain the same number of rows")
    if left_array.shape[0] < 2:
        raise ValueError("representations require at least two rows")
    return left_array, right_array


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the row-normalized linear CKA used by the paper."""
    left_array, right_array = _validate_representations(left, right)
    left_norm = np.linalg.norm(left_array, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right_array, axis=1, keepdims=True)
    left_normalized = left_array / np.maximum(left_norm, 1e-8)
    right_normalized = right_array / np.maximum(right_norm, 1e-8)

    left_gram = left_normalized @ left_normalized.T
    right_gram = right_normalized @ right_normalized.T
    centering = np.eye(len(left_array), dtype=np.float32) - (1.0 / len(left_array))
    left_centered = centering @ left_gram @ centering
    right_centered = centering @ right_gram @ centering

    numerator = float(np.sum(left_centered * right_centered))
    denominator = float(
        np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered))
    )
    if denominator <= 1e-8:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def cka_dissimilarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the paper's CKA dissimilarity, ``1 - CKA``."""
    return 1.0 - linear_cka(left, right)


def build_elman_policy_model(
    *,
    input_dim: int,
    num_classes: int,
    sequence_length: int,
    hidden_dim: int = DEFAULT_ELMAN_HIDDEN_DIM,
    depth: int = DEFAULT_ELMAN_DEPTH,
    dropout: float = 0.1,
    name: str = TARGET_MODEL_ID,
) -> Any:
    """Build a fixed-length sequence policy whose output is per-turn logits."""
    if input_dim <= 0 or num_classes <= 0 or sequence_length <= 0:
        raise ValueError("input_dim, num_classes, and sequence_length must be positive")
    if hidden_dim <= 0 or depth <= 0:
        raise ValueError("hidden_dim and depth must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1)")

    from tensorflow import keras
    from tensorflow.keras import layers

    state_sequence = layers.Input(
        shape=(sequence_length, input_dim),
        name="state_sequence",
    )
    hidden = state_sequence
    for layer_index in range(depth):
        hidden = layers.SimpleRNN(
            hidden_dim,
            activation="tanh",
            return_sequences=True,
            name=f"elman_{layer_index + 1}",
        )(hidden)
        if dropout > 0.0:
            hidden = layers.Dropout(dropout, name=f"elman_dropout_{layer_index + 1}")(hidden)
    logits = layers.TimeDistributed(
        layers.Dense(num_classes, name="policy"),
        name="policy_sequence",
    )(hidden)
    return keras.Model(state_sequence, logits, name=name)


def build_serving_model(
    sequence_model: Any,
    *,
    input_dim: int,
    sequence_length: int,
    num_classes: int,
    name: str = TARGET_MODEL_ID,
) -> Any:
    """Wrap a sequence policy in the existing worker's flat-vector interface."""
    from tensorflow import keras
    from tensorflow.keras import layers

    flat_state = layers.Input(
        shape=(sequence_length * input_dim,),
        name="flattened_state_history",
    )
    sequence = layers.Reshape((sequence_length, input_dim), name="state_history")(flat_state)
    sequence_logits = sequence_model(sequence, training=False)
    last_logits = layers.Cropping1D(
        cropping=(sequence_length - 1, 0),
        name="last_timestep",
    )(sequence_logits)
    last_logits = layers.Reshape((num_classes,), name="policy_logits")(last_logits)
    return keras.Model(flat_state, last_logits, name=f"{name}_serving")


def tf_linear_cka_dissimilarity(left: Any, right: Any, mask: Any | None = None) -> Any:
    """TensorFlow counterpart used by the staged alignment trainer."""
    import tensorflow as tf

    left_tensor = tf.cast(left, tf.float32)
    right_tensor = tf.cast(right, tf.float32)
    if mask is not None:
        valid = tf.cast(mask, tf.bool)
        left_tensor = tf.boolean_mask(left_tensor, valid)
        right_tensor = tf.boolean_mask(right_tensor, valid)
    left_tensor = tf.reshape(left_tensor, (-1, tf.shape(left_tensor)[-1]))
    right_tensor = tf.reshape(right_tensor, (-1, tf.shape(right_tensor)[-1]))
    left_tensor = tf.math.l2_normalize(left_tensor, axis=1)
    right_tensor = tf.math.l2_normalize(right_tensor, axis=1)
    left_gram = tf.matmul(left_tensor, left_tensor, transpose_b=True)
    right_gram = tf.matmul(right_tensor, right_tensor, transpose_b=True)
    row_count = tf.shape(left_gram)[0]
    centering = tf.eye(row_count) - tf.ones((row_count, row_count), dtype=tf.float32) / tf.cast(row_count, tf.float32)
    left_centered = centering @ left_gram @ centering
    right_centered = centering @ right_gram @ centering
    numerator = tf.reduce_sum(left_centered * right_centered)
    denominator = tf.sqrt(
        tf.reduce_sum(left_centered * left_centered) * tf.reduce_sum(right_centered * right_centered)
    )
    similarity = tf.where(denominator > 1e-8, numerator / denominator, tf.zeros_like(denominator))
    return 1.0 - tf.clip_by_value(similarity, 0.0, 1.0)
