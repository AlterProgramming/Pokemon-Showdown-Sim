from __future__ import annotations

"""Keras model builder for the first trainable entity-centric family.

This is intentionally a "graph-ready" encoder rather than a full message-passing
GNN. It already gives us the important project shifts:
    - learned embeddings for named entities
    - slot-wise shared encoding
    - pooled active / bench / global summaries
    - optional transition and value auxiliary heads

That makes it a safe first entity generation while keeping the door open for a
future true graph encoder under a new family version.
"""

import math
from typing import Any, Dict, List

from .EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES
from .StateVectorization import SIDE_CONDITION_ORDER, STAT_ORDER


POKEMON_NUMERIC_DIM = 6 + len(STAT_ORDER)
GLOBAL_NUMERIC_DIM = 1 + 2 * len(SIDE_CONDITION_ORDER)


# ---------------------------------------------------------------------------
# Registered serialisable layers — replacing all Lambda layers so that
# keras.models.load_model() can reconstruct the graph without safe_mode=False.
# ---------------------------------------------------------------------------

def _keras():
    import keras
    return keras

def _tf():
    import tensorflow as tf
    return tf


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class MaskedAverage(_keras().layers.Layer):
    """Average-pool embeddings over `axis`, ignoring PAD (id==0) entries."""

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        tf = _tf()
        embeddings, token_ids = inputs
        mask = tf.cast(tf.not_equal(token_ids, 0), tf.float32)
        while len(mask.shape) < len(embeddings.shape):
            mask = tf.expand_dims(mask, axis=-1)
        numer = tf.reduce_sum(embeddings * mask, axis=self.axis)
        denom = tf.maximum(tf.reduce_sum(mask, axis=self.axis), 1.0)
        return numer / denom

    def get_config(self):
        cfg = super().get_config()
        cfg["axis"] = self.axis
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class MaskedPool(_keras().layers.Layer):
    """Average-pool slot embeddings over axis=1 using a float mask."""

    def call(self, inputs):
        tf = _tf()
        values, mask = inputs
        mask = tf.cast(mask, tf.float32)
        if len(mask.shape) < len(values.shape):
            mask = tf.expand_dims(mask, axis=-1)
        numer = tf.reduce_sum(values * mask, axis=1)
        denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        return numer / denom


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class SlotSlice(_keras().layers.Layer):
    """Slice `inputs[:, start:end, :]` — replaces non-serialisable slice Lambdas."""

    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[:, self.start:self.end, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"start": self.start, "end": self.end})
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class ReduceMean1(_keras().layers.Layer):
    """tf.reduce_mean(x, axis=1) — replaces the all_slot_pool Lambda."""

    def call(self, inputs):
        return _tf().reduce_mean(inputs, axis=1)


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class ExtractColumn(_keras().layers.Layer):
    """Extract one column from a 3-D tensor: (batch, slots, features) -> (batch, slots)."""

    def __init__(self, col_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.col_idx = col_idx

    def call(self, inputs):
        return inputs[:, :, self.col_idx]

    def get_config(self):
        cfg = super().get_config()
        cfg["col_idx"] = self.col_idx
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class BenchMask(_keras().layers.Layer):
    """Bench mask: slots that are neither active (col 3) nor fainted (col 4)."""

    def call(self, inputs):
        numeric = inputs  # shape (batch, slots, POKEMON_NUMERIC_DIM)
        active  = numeric[:, :, 3]
        fainted = numeric[:, :, 4]
        return (1.0 - active) * (1.0 - fainted)


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class ExpandBoolMask(_keras().layers.Layer):
    """[B, K] float → [B, 1, K] bool for MHA attention_mask."""

    def call(self, inputs):
        tf = _tf()
        return tf.cast(inputs[:, tf.newaxis, :], tf.bool)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[1])




# ---------------------------------------------------------------------------
# Registered loss / metric functions — plain Python functions saved as Lambdas
# by Keras unless decorated with register_keras_serializable.
# ---------------------------------------------------------------------------

@_keras().saving.register_keras_serializable(package="EntityModelV1")
def masked_sequence_loss(y_true, y_pred):
    """SparseCategoricalCrossentropy that ignores PAD_ID=0 positions."""
    tf = _tf()
    keras = _keras()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")
    per_token_loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=per_token_loss.dtype)
    masked = per_token_loss * mask
    denom = tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)
    return tf.reduce_mean(tf.reduce_sum(masked, axis=-1) / denom)


@_keras().saving.register_keras_serializable(package="EntityModelV1")
def masked_token_accuracy(y_true, y_pred):
    """Token accuracy that ignores PAD_ID=0 positions."""
    tf = _tf()
    predicted = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    correct = tf.cast(tf.equal(predicted, y_true), dtype=tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(correct * mask) / denom


def build_entity_action_models(
    *,
    vocab_sizes: Dict[str, int],
    num_policy_classes: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    learning_rate: float,
    token_embed_dim: int = 24,
    transition_dim: int | None = None,
    action_context_vocab_size: int | None = None,
    action_embed_dim: int = 16,
    transition_hidden_dim: int | None = None,
    transition_weight: float = 0.25,
    predict_value: bool = False,
    value_hidden_dim: int | None = None,
    value_weight: float = 0.25,
    predict_sequence: bool = False,
    sequence_vocab_size: int | None = None,
    sequence_hidden_dim: int = 128,
    sequence_weight: float = 0.1,
    max_seq_len: int = 32,
    use_history: bool = False,
    history_vocab_size: int | None = None,
    history_embed_dim: int = 32,
    history_lstm_dim: int = 64,
    history_turns: int = 8,
    history_events_per_turn: int = 24,
):
    """Build the multitask entity-action models.

    Returns:
        training_model: full multitask model used during fitting
        policy_model: lightweight policy-only inference artifact
        policy_value_model: optional state-only policy+value artifact
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Pooling and slicing are handled by the registered layer classes above;
    # no local closures needed here.

    inputs: Dict[str, Any] = {
        # The tensor contract mirrors EntityTensorization.entity_tensor_layout().
        "pokemon_species": layers.Input(shape=(12,), dtype="int32", name="pokemon_species"),
        "pokemon_item": layers.Input(shape=(12,), dtype="int32", name="pokemon_item"),
        "pokemon_ability": layers.Input(shape=(12,), dtype="int32", name="pokemon_ability"),
        "pokemon_tera": layers.Input(shape=(12,), dtype="int32", name="pokemon_tera"),
        "pokemon_status": layers.Input(shape=(12,), dtype="int32", name="pokemon_status"),
        "pokemon_side": layers.Input(shape=(12,), dtype="int32", name="pokemon_side"),
        "pokemon_slot": layers.Input(shape=(12,), dtype="int32", name="pokemon_slot"),
        "pokemon_observed_moves": layers.Input(
            shape=(12, MAX_OBSERVED_MOVES),
            dtype="int32",
            name="pokemon_observed_moves",
        ),
        "pokemon_numeric": layers.Input(
            shape=(12, POKEMON_NUMERIC_DIM),
            dtype="float32",
            name="pokemon_numeric",
        ),
        "weather": layers.Input(shape=(1,), dtype="int32", name="weather"),
        "global_conditions": layers.Input(
            shape=(MAX_GLOBAL_CONDITIONS,),
            dtype="int32",
            name="global_conditions",
        ),
        "global_numeric": layers.Input(
            shape=(GLOBAL_NUMERIC_DIM,),
            dtype="float32",
            name="global_numeric",
        ),
    }

    species_embedding = layers.Embedding(vocab_sizes["species"], token_embed_dim, mask_zero=True, name="species_embedding")
    item_embedding = layers.Embedding(vocab_sizes["item"], max(8, token_embed_dim // 2), mask_zero=True, name="item_embedding")
    ability_embedding = layers.Embedding(vocab_sizes["ability"], max(8, token_embed_dim // 2), mask_zero=True, name="ability_embedding")
    tera_embedding = layers.Embedding(vocab_sizes["tera"], max(8, token_embed_dim // 3), mask_zero=True, name="tera_embedding")
    status_embedding = layers.Embedding(vocab_sizes["status"], max(8, token_embed_dim // 3), mask_zero=True, name="status_embedding")
    move_embedding = layers.Embedding(vocab_sizes["move"], max(8, token_embed_dim // 2), mask_zero=True, name="move_embedding")
    weather_embedding = layers.Embedding(vocab_sizes["weather"], max(8, token_embed_dim // 3), mask_zero=True, name="weather_embedding")
    global_condition_embedding = layers.Embedding(
        vocab_sizes["global_condition"],
        max(8, token_embed_dim // 3),
        mask_zero=True,
        name="global_condition_embedding",
    )
    side_embedding = layers.Embedding(3, max(4, token_embed_dim // 4), name="side_embedding")
    slot_embedding = layers.Embedding(7, max(4, token_embed_dim // 4), name="slot_embedding")

    # Observed moves arrive as a short token list per slot. We embed each move and then
    # average only the non-pad entries to get one learned "observed move summary" vector.
    move_embedded = move_embedding(inputs["pokemon_observed_moves"])
    move_pooled = MaskedAverage(axis=2, name="observed_move_pool")(
        [move_embedded, inputs["pokemon_observed_moves"]]
    )

    weather_x = layers.Flatten(name="weather_flat")(weather_embedding(inputs["weather"]))
    global_condition_embedded = global_condition_embedding(inputs["global_conditions"])
    global_condition_x = MaskedAverage(axis=1, name="global_condition_pool")(
        [global_condition_embedded, inputs["global_conditions"]]
    )

    pokemon_x = layers.Concatenate(axis=-1, name="pokemon_concat")(
        [
            species_embedding(inputs["pokemon_species"]),
            item_embedding(inputs["pokemon_item"]),
            ability_embedding(inputs["pokemon_ability"]),
            tera_embedding(inputs["pokemon_tera"]),
            status_embedding(inputs["pokemon_status"]),
            side_embedding(inputs["pokemon_side"]),
            slot_embedding(inputs["pokemon_slot"]),
            move_pooled,
            inputs["pokemon_numeric"],
        ]
    )
    # Shared per-slot encoder: each Pokemon slot gets contextualized by the same weights
    # before we summarize self active / opp active / bench state.
    pokemon_x = layers.Dense(hidden_dim, activation="relu", name="pokemon_dense_0")(pokemon_x)
    if dropout > 0:
        pokemon_x = layers.Dropout(dropout, name="pokemon_dropout_0")(pokemon_x)
    pokemon_x = layers.Dense(hidden_dim, activation="relu", name="pokemon_dense_1")(pokemon_x)

    self_slots   = SlotSlice(0, 6,  name="self_slots")(pokemon_x)
    opp_slots    = SlotSlice(6, 12, name="opp_slots")(pokemon_x)
    self_numeric = SlotSlice(0, 6,  name="self_numeric")(inputs["pokemon_numeric"])
    opp_numeric  = SlotSlice(6, 12, name="opp_numeric")(inputs["pokemon_numeric"])

    # These pooled summaries are the key simplification in v1:
    #   - one summary for my active
    #   - one summary for opponent active
    #   - one summary for my bench
    #   - one summary for opponent bench
    # This keeps the model entity-centric without requiring a full graph library yet.
    self_active = MaskedPool(name="self_active_pool")(
        [self_slots, ExtractColumn(3, name="self_active_flag")(self_numeric)]
    )
    opp_active = MaskedPool(name="opp_active_pool")(
        [opp_slots, ExtractColumn(3, name="opp_active_flag")(opp_numeric)]
    )
    self_bench = MaskedPool(name="self_bench_pool")(
        [self_slots, BenchMask(name="self_bench_mask")(self_numeric)]
    )
    opp_bench = MaskedPool(name="opp_bench_pool")(
        [opp_slots, BenchMask(name="opp_bench_mask")(opp_numeric)]
    )
    all_slots = ReduceMean1(name="all_slot_pool")(pokemon_x)

    global_x = layers.Concatenate(name="global_concat")(
        [
            weather_x,
            global_condition_x,
            inputs["global_numeric"],
        ]
    )
    global_x = layers.Dense(hidden_dim, activation="relu", name="global_dense")(global_x)

    # The shared trunk mixes matchup, bench, and global context before any head-specific
    # prediction happens.
    x = layers.Concatenate(name="state_concat")(
        [self_active, opp_active, self_bench, opp_bench, all_slots, global_x]
    )
    for layer_idx in range(depth):
        x = layers.Dense(hidden_dim, activation="relu", name=f"trunk_dense_{layer_idx}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"trunk_dropout_{layer_idx}")(x)

    shared = x

    # model_inputs starts as a copy of inputs; history and action heads extend it.
    # It must be initialised here so the history block below can add to it before
    # the policy_model line (which uses the bare `inputs` dict, not model_inputs).
    model_inputs: Dict[str, Any] = dict(inputs)

    # --- Optional event history encoder ---
    history_context = None

    if use_history and history_vocab_size is not None:
        _attn_dim = history_lstm_dim * 2
        hist_tokens_input = layers.Input(
            shape=(history_turns, history_events_per_turn),
            dtype="int32",
            name="event_history_tokens",
        )
        hist_mask_input = layers.Input(
            shape=(history_turns,),
            dtype="float32",
            name="event_history_mask",
        )
        model_inputs["event_history_tokens"] = hist_tokens_input
        model_inputs["event_history_mask"] = hist_mask_input

        history_event_embedding = layers.Embedding(
            history_vocab_size,
            history_embed_dim,
            name="history_event_embedding",
        )
        hist_embedded = history_event_embedding(hist_tokens_input)

        # PAD-masked mean pool over the event axis.
        # MaskedAverage is a registered serialisable layer (no Lambda needed).
        hist_pooled = MaskedAverage(axis=2, name="history_turn_pool")(
            [hist_embedded, hist_tokens_input]
        )

        # recurrent_dropout=0.0 is explicit — any non-zero value disables cuDNN path.
        hist_lstm_out = layers.Bidirectional(
            layers.LSTM(history_lstm_dim, return_sequences=True, recurrent_dropout=0.0),
            name="history_bilstm",
        )(hist_pooled)

        # hist_mask_input [B,K] float → [B,1,K] bool for MHA attention_mask.
        # ExpandBoolMask is registered and implements compute_output_shape.
        shared_q = layers.Reshape((1, hidden_dim), name="shared_query")(shared)
        hist_attn_mask = ExpandBoolMask(name="history_attn_mask")(hist_mask_input)
        hist_context_seq, hist_attn_scores = layers.MultiHeadAttention(
            num_heads=1,
            key_dim=_attn_dim,
            output_shape=_attn_dim,
            name="history_attention_layer",
        )(query=shared_q, key=hist_lstm_out, value=hist_lstm_out,
          attention_mask=hist_attn_mask, return_attention_scores=True)
        # [B, 1, _attn_dim] → [B, _attn_dim]; Reshape is safe here since dim-1 is fixed.
        history_context = layers.Reshape((_attn_dim,), name="history_context")(hist_context_seq)
        # hist_attn_scores: [B, 1, 1, K] → [B, K] for the attention extractor model.
        history_attention_weights = layers.Reshape(
            (history_turns,), name="history_attention_weights"
        )(hist_attn_scores)

    # Build shared_with_history for auxiliary heads
    if history_context is not None:
        _fused = layers.Concatenate(name="shared_history_fuse")([shared, history_context])
        shared_with_history = layers.Dense(
            hidden_dim, activation="relu", name="history_fuse_proj"
        )(_fused)
    else:
        shared_with_history = shared

    # v1 still predicts the global action vocabulary because offline logs do not contain
    # the per-turn legal request objects needed for true action-wise legal scoring.
    # Policy head uses `shared` (not shared_with_history) for backward compatibility.
    policy_logits = layers.Dense(num_policy_classes, name="policy")(shared)
    policy_model = keras.Model(inputs, policy_logits, name="entity_action_policy_model")

    policy_metrics: List[Any] = [keras.metrics.SparseCategoricalAccuracy(name="top1")]
    if num_policy_classes >= 3:
        policy_metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"))

    use_transition = transition_dim is not None and action_context_vocab_size is not None
    use_sequence = predict_sequence and sequence_vocab_size is not None
    if not use_transition and not predict_value and not use_sequence:
        policy_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=policy_metrics,
        )
        return policy_model, policy_model, None, None

    outputs: Dict[str, Any] = {"policy": policy_logits}
    losses: Dict[str, Any] = {
        "policy": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    loss_weights: Dict[str, float] = {"policy": 1.0}
    metrics: Dict[str, List[Any]] = {"policy": policy_metrics}

    policy_value_model = None
    if predict_value:
        # Keep value state-only and conservative: it estimates eventual win probability.
        value_x = layers.Dense(value_hidden_dim or max(64, hidden_dim // 2), activation="relu", name="value_dense")(shared_with_history)
        _value_dropout = max(dropout, 0.25)
        if _value_dropout > 0:
            value_x = layers.Dropout(_value_dropout, name="value_dropout")(value_x)
        value_out = layers.Dense(1, activation="sigmoid", name="value")(value_x)
        outputs["value"] = value_out
        losses["value"] = keras.losses.BinaryCrossentropy()
        loss_weights["value"] = value_weight
        metrics["value"] = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="brier"),
        ]
        # policy and value don't depend on action inputs (my_action/opp_action),
        # so exclude them — Keras raises if inputs are disconnected from outputs.
        _action_keys = {"my_action", "opp_action"}
        pv_inputs = (
            {k: v for k, v in model_inputs.items() if k not in _action_keys}
            if history_context is not None
            else inputs
        )
        policy_value_model = keras.Model(
            pv_inputs,
            {"policy": policy_logits, "value": value_out},
            name="entity_action_policy_value_model",
        )

    # Both the transition head and the sequence head are conditioned on action embeddings.
    # We create shared action inputs and a shared embedding layer whenever either head is on.
    need_action_inputs = use_transition or use_sequence
    my_action_input = None
    opp_action_input = None
    action_embedding = None
    if need_action_inputs:
        my_action_input = layers.Input(shape=(), dtype="int32", name="my_action")
        opp_action_input = layers.Input(shape=(), dtype="int32", name="opp_action")
        action_embedding = layers.Embedding(
            input_dim=action_context_vocab_size,
            output_dim=max(8, action_embed_dim),
            name="action_embedding",
        )
        model_inputs["my_action"] = my_action_input
        model_inputs["opp_action"] = opp_action_input

    if use_transition:
        transition_x = layers.Concatenate(name="transition_features")(
            [
                shared_with_history,
                # The transition head gets both action identities so it can explain what
                # public state should change after the simultaneous turn resolves.
                action_embedding(my_action_input),
                action_embedding(opp_action_input),
            ]
        )
        transition_x = layers.Dense(transition_hidden_dim or hidden_dim, activation="relu", name="transition_dense")(transition_x)
        if dropout > 0:
            transition_x = layers.Dropout(dropout, name="transition_dropout")(transition_x)
        transition_out = layers.Dense(transition_dim, name="transition")(transition_x)
        outputs["transition"] = transition_out
        losses["transition"] = keras.losses.MeanSquaredError()
        loss_weights["transition"] = transition_weight
        metrics["transition"] = [keras.metrics.MeanAbsoluteError(name="mae")]

    if use_sequence:
        # Non-autoregressive sequence head: concatenate trunk + action embeddings,
        # repeat across time steps, then apply LSTM to produce per-step distributions.
        sequence_context = layers.Concatenate(name="sequence_context")(
            [
                shared_with_history,
                action_embedding(my_action_input),
                action_embedding(opp_action_input),
            ]
        )
        sequence_repeated = layers.RepeatVector(max_seq_len, name="sequence_repeat")(sequence_context)
        sequence_lstm = layers.LSTM(sequence_hidden_dim, return_sequences=True, name="sequence_lstm")(sequence_repeated)
        sequence_out = layers.TimeDistributed(
            layers.Dense(sequence_vocab_size, activation="softmax"),
            name="sequence",
        )(sequence_lstm)
        outputs["sequence"] = sequence_out
        losses["sequence"] = masked_sequence_loss
        loss_weights["sequence"] = sequence_weight
        metrics["sequence"] = [masked_token_accuracy]

    # Attention extractor: separate model sharing the same graph, no training targets needed.
    # Excludes action inputs (not reachable from history_attention_weights).
    if history_context is not None:
        _attn_inputs = {k: v for k, v in model_inputs.items()
                        if k not in {"my_action", "opp_action"}}
        history_attention_model = keras.Model(
            _attn_inputs, history_attention_weights,
            name="entity_history_attention_model",
        )
    else:
        history_attention_model = None

    # The training artifact is the richest object because it includes every enabled
    # auxiliary head. The policy and policy+value artifacts stay serving-friendly.
    training_model = keras.Model(model_inputs, outputs, name="entity_action_training_model")
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return training_model, policy_model, policy_value_model, history_attention_model
