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
        self.supports_masking = True

    def call(self, inputs, mask=None):
        ops = _keras().ops
        embeddings, token_ids = inputs
        mask = ops.cast(ops.not_equal(token_ids, 0), "float32")
        # Expand mask to match embeddings rank (e.g. [B,K,E] → [B,K,E,1])
        for _ in range(len(embeddings.shape) - len(mask.shape)):
            mask = ops.expand_dims(mask, axis=-1)
        numer = ops.sum(embeddings * mask, axis=self.axis)
        denom = ops.maximum(ops.sum(mask, axis=self.axis), 1.0)
        return numer / denom

    def compute_output_shape(self, input_shape):
        emb_shape = input_shape[0]
        return emb_shape[: self.axis] + emb_shape[self.axis + 1 :]

    def compute_mask(self, inputs, mask=None):
        return None  # absorbs upstream mask; does not propagate one

    def get_config(self):
        cfg = super().get_config()
        cfg["axis"] = self.axis
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class MaskedPool(_keras().layers.Layer):
    """Average-pool slot embeddings over axis=1 using a float mask."""

    supports_masking = True

    def call(self, inputs, mask=None):
        ops = _keras().ops
        values, float_mask = inputs
        float_mask = ops.cast(float_mask, "float32")
        if len(float_mask.shape) < len(values.shape):
            float_mask = ops.expand_dims(float_mask, axis=-1)
        numer = ops.sum(values * float_mask, axis=1)
        denom = ops.maximum(ops.sum(float_mask, axis=1), 1.0)
        return numer / denom

    def compute_mask(self, inputs, mask=None):
        return None


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class SlotSlice(_keras().layers.Layer):
    """Slice `inputs[:, start:end, :]` — replaces non-serialisable slice Lambdas."""

    supports_masking = True

    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs, mask=None):
        return inputs[:, self.start:self.end, :]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"start": self.start, "end": self.end})
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class ReduceMean1(_keras().layers.Layer):
    """Mean over axis=1 — replaces the all_slot_pool Lambda."""

    supports_masking = True

    def call(self, inputs, mask=None):
        return _keras().ops.mean(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class ExtractColumn(_keras().layers.Layer):
    """Extract one column from a 3-D tensor: (batch, slots, features) -> (batch, slots)."""

    supports_masking = True

    def __init__(self, col_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.col_idx = col_idx

    def call(self, inputs, mask=None):
        return inputs[:, :, self.col_idx]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg["col_idx"] = self.col_idx
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class BenchMask(_keras().layers.Layer):
    """Bench mask: slots that are neither active (col 3) nor fainted (col 4)."""

    supports_masking = True

    def call(self, inputs, mask=None):
        numeric = inputs  # shape (batch, slots, POKEMON_NUMERIC_DIM)
        active  = numeric[:, :, 3]
        fainted = numeric[:, :, 4]
        return (1.0 - active) * (1.0 - fainted)

    def compute_mask(self, inputs, mask=None):
        return None


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class HistoryAttention(_keras().layers.Layer):
    """Single-head scaled dot-product attention over history turns.

    Takes [query [B,1,D], keys [B,K,D], values [B,K,D], mask [B,K] float]
    and returns (context [B,1,out_dim], attn_weights [B,1,K]).

    This avoids Keras 3's MultiHeadAttention shape-inference bug where
    attention_scores always get key_steps=1 when return_attention_scores=True
    is used in a functional-API graph.
    """

    def __init__(self, key_dim: int, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        q_dim  = input_shape[0][-1]
        kv_dim = input_shape[1][-1]
        self.Wq = self.add_weight(shape=(q_dim,  self.key_dim),    initializer="glorot_uniform", name="Wq")
        self.Wk = self.add_weight(shape=(kv_dim, self.key_dim),    initializer="glorot_uniform", name="Wk")
        self.Wv = self.add_weight(shape=(kv_dim, self.output_dim), initializer="glorot_uniform", name="Wv")
        self.scale = self.key_dim ** -0.5

    supports_masking = True

    def call(self, inputs, mask=None):
        ops = _keras().ops
        q, k, v, float_mask = inputs  # float_mask is the explicit turn mask; mask kwarg is Keras propagated mask (ignored)
        q_p = ops.matmul(q, self.Wq)                                              # [B,1,key_dim]
        k_p = ops.matmul(k, self.Wk)                                              # [B,K,key_dim]
        v_p = ops.matmul(v, self.Wv)                                              # [B,K,out_dim]
        scores = ops.matmul(q_p, ops.transpose(k_p, axes=(0, 2, 1))) * self.scale    # [B,1,K]
        scores = scores + (1.0 - ops.cast(ops.expand_dims(float_mask, axis=1), q_p.dtype)) * -1e9
        weights = ops.softmax(scores, axis=-1)                                     # [B,1,K]
        context = ops.matmul(weights, v_p)                                         # [B,1,out_dim]
        return context, weights

    def compute_output_shape(self, input_shape):
        q_shape, kv_shape = input_shape[0], input_shape[1]
        return (
            (q_shape[0], q_shape[1], self.output_dim),
            (q_shape[0], q_shape[1], kv_shape[1]),
        )

    def compute_mask(self, inputs, mask=None):
        return None  # returns two tensors; neither carries a mask forward

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"key_dim": self.key_dim, "output_dim": self.output_dim})
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class ActionDecoder(_keras().layers.Layer):
    """Decode LSTM hidden states back into action logits.

    Reconstructs what actions were taken in the past by projecting LSTM output
    through a dense layer to produce per-turn action logits. Padded positions
    are masked out with large negative values.
    """

    def __init__(self, action_vocab_size: int, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.action_vocab_size = action_vocab_size
        self.hidden_dim = hidden_dim
        self.supports_masking = True

    def build(self, input_shape):
        # input_shape = (lstm_output_shape, action_mask_shape)
        lstm_output_shape = input_shape[0]
        lstm_dim = lstm_output_shape[-1]

        self.dense = _keras().layers.Dense(
            self.hidden_dim, activation="relu", name=f"{self.name}_dense"
        )
        self.logits = _keras().layers.Dense(
            self.action_vocab_size, name=f"{self.name}_logits"
        )

    def call(self, inputs, mask=None):
        tf = _tf()
        lstm_output, action_mask = inputs

        # Project and decode
        hidden = self.dense(lstm_output)  # [batch, K, hidden_dim]
        logits = self.logits(hidden)      # [batch, K, action_vocab_size]

        # Mask: set logits to large negative for padded positions
        if action_mask is not None:
            mask_expanded = tf.expand_dims(action_mask, axis=-1)  # [batch, K, 1]
            logits = logits + (1.0 - mask_expanded) * -1e9

        return logits

    def compute_output_shape(self, input_shape):
        lstm_output_shape = input_shape[0]
        return (lstm_output_shape[0], lstm_output_shape[1], self.action_vocab_size)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "action_vocab_size": self.action_vocab_size,
            "hidden_dim": self.hidden_dim,
        })
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV1")
class PastActionContext(_keras().layers.Layer):
    """Extract context from decoded past actions via cross-attention.

    Attends over decoded action embeddings using the shared state as query
    to produce a context vector that captures which past actions are relevant.
    """

    def __init__(self, attn_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.attn_dim = attn_dim
        self.supports_masking = True

    def build(self, input_shape):
        # input_shape = (shared_shape, action_logits_shape, action_mask_shape)
        self.action_embed_dense = _keras().layers.Dense(
            self.attn_dim, activation="relu", name=f"{self.name}_action_embed"
        )
        self.attn_Wq = _keras().layers.Dense(
            self.attn_dim, use_bias=False, name=f"{self.name}_attn_Wq"
        )
        self.attn_Wk = _keras().layers.Dense(
            self.attn_dim, use_bias=False, name=f"{self.name}_attn_Wk"
        )
        self.attn_Wv = _keras().layers.Dense(
            self.attn_dim, use_bias=False, name=f"{self.name}_attn_Wv"
        )

    def call(self, inputs, mask=None):
        ops = _keras().ops
        shared, action_logits, action_mask = inputs

        # Embed the action logits: [batch, K, action_vocab_size] → [batch, K, attn_dim]
        action_embed = self.action_embed_dense(action_logits)

        # Attend: shared as query [batch, shared_dim] → [batch, 1, shared_dim]
        shared_q = ops.reshape(shared, (ops.shape(shared)[0], 1, ops.shape(shared)[-1]))

        # Project query and keys
        _q_p = self.attn_Wq(shared_q)  # [batch, 1, attn_dim]
        _k_p = self.attn_Wk(action_embed)  # [batch, K, attn_dim]
        _v_p = self.attn_Wv(action_embed)  # [batch, K, attn_dim]

        # Scaled dot-product attention
        _scale = self.attn_dim ** -0.5
        _scores = ops.matmul(_q_p, ops.transpose(_k_p, axes=(0, 2, 1))) * _scale  # [batch, 1, K]
        _fmask = ops.cast(ops.expand_dims(action_mask, axis=1), _q_p.dtype)  # [batch, 1, K]
        _scores = _scores + (1.0 - _fmask) * -1e9
        _attn_w = ops.softmax(_scores, axis=-1)  # [batch, 1, K]
        context_seq = ops.matmul(_attn_w, _v_p)  # [batch, 1, attn_dim]

        # Reshape to [batch, attn_dim]
        context = ops.reshape(context_seq, (ops.shape(context_seq)[0], self.attn_dim))

        return context

    def compute_output_shape(self, input_shape):
        shared_shape = input_shape[0]
        return (shared_shape[0], self.attn_dim)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attn_dim": self.attn_dim})
        return cfg




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
    policy_reads_history: bool = False,
    use_history_decoding: bool = False,
    action_vocab_size: int | None = None,
    decoded_action_weight: float = 0.15,
    predict_threat: bool = False,
    threat_hidden_dim: int | None = None,
    threat_weight: float = 0.1,
    predict_type_effectiveness: bool = False,
    type_eff_hidden_dim: int | None = None,
    type_eff_weight: float = 0.1,
):
    """Build the multitask entity-action models with optional β-2 auxiliary heads.

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
    weather_embedding = layers.Embedding(vocab_sizes["weather"], max(8, token_embed_dim // 3), name="weather_embedding")
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
    # Inline masked mean-pool for observed moves — avoids list-input Lambda on GPU.
    move_embedded = move_embedding(inputs["pokemon_observed_moves"])
    _ops = _keras().ops
    _mv_mask   = _ops.cast(_ops.not_equal(inputs["pokemon_observed_moves"], 0), "float32")
    _mv_mask3d = _ops.expand_dims(_mv_mask, axis=-1)
    move_pooled = (
        _ops.sum(move_embedded * _mv_mask3d, axis=2)
        / _ops.maximum(_ops.sum(_mv_mask, axis=2, keepdims=True), 1.0)
    )

    # weather_embedding does not use mask_zero — mask was immediately destroyed by Flatten.
    weather_x = layers.Flatten(name="weather_flat")(weather_embedding(inputs["weather"]))

    # Inline masked mean-pool for global conditions — avoids list-input Lambda on GPU.
    global_condition_embedded = global_condition_embedding(inputs["global_conditions"])
    _gc_mask   = _ops.cast(_ops.not_equal(inputs["global_conditions"], 0), "float32")
    _gc_mask2d = _ops.expand_dims(_gc_mask, axis=-1)
    global_condition_x = (
        _ops.sum(global_condition_embedded * _gc_mask2d, axis=1)
        / _ops.maximum(_ops.sum(_gc_mask, axis=1, keepdims=True), 1.0)
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
    history_attention_weights = None
    hist_tokens_input = None
    hist_mask_input = None
    past_action_context = None
    action_logits_decoded = None
    past_action_ids_input = None
    past_action_mask_input = None

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

        # PAD-masked mean pool over the event axis using inline keras.ops.
        # A list-input custom layer triggers GPU graph-tracing Lambdas in all
        # Keras 3.x versions; inline ops have unambiguous shapes and no Lambdas.
        _ops = _keras().ops
        _pad_mask   = _ops.cast(_ops.not_equal(hist_tokens_input, 0), "float32")  # [B,K,E]
        _pad_mask4d = _ops.expand_dims(_pad_mask, axis=-1)                          # [B,K,E,1]
        _numer      = _ops.sum(hist_embedded * _pad_mask4d, axis=2)                 # [B,K,D]
        _denom      = _ops.maximum(_ops.sum(_pad_mask, axis=2, keepdims=True), 1.0) # [B,K,1]
        hist_pooled = _numer / _denom                                                # [B,K,D]

        # recurrent_dropout=0.0 is explicit — any non-zero value disables cuDNN path.
        hist_lstm_out = layers.Bidirectional(
            layers.LSTM(history_lstm_dim, return_sequences=True, recurrent_dropout=0.0),
            name="history_bilstm",
        )(hist_pooled)

        # Scaled dot-product attention — inlined via keras.ops + Dense projections.
        # A list-input custom layer (HistoryAttention([q,k,v,mask])) triggers the
        # same GPU graph-tracing Lambda wrapping that broke MaskedAverage.  Using
        # Dense(use_bias=False) for projections and keras.ops for the attention
        # arithmetic avoids any Lambda and is mathematically identical.
        shared_q = layers.Reshape((1, hidden_dim), name="shared_query")(shared)
        _q_p = layers.Dense(_attn_dim, use_bias=False, name="history_attn_Wq")(shared_q)      # [B,1,attn_dim]
        _k_p = layers.Dense(_attn_dim, use_bias=False, name="history_attn_Wk")(hist_lstm_out) # [B,K,attn_dim]
        _v_p = layers.Dense(_attn_dim, use_bias=False, name="history_attn_Wv")(hist_lstm_out) # [B,K,attn_dim]
        _scale   = _attn_dim ** -0.5
        _scores  = _ops.matmul(_q_p, _ops.transpose(_k_p, axes=(0, 2, 1))) * _scale           # [B,1,K]
        _fmask   = _ops.cast(_ops.expand_dims(hist_mask_input, axis=1), _q_p.dtype)            # [B,1,K]
        _scores  = _scores + (1.0 - _fmask) * -1e9
        _attn_w  = _ops.softmax(_scores, axis=-1)                                              # [B,1,K]
        hist_context_seq = _ops.matmul(_attn_w, _v_p)                                          # [B,1,attn_dim]
        # [B, 1, _attn_dim] → [B, _attn_dim]
        history_context = layers.Reshape((_attn_dim,), name="history_context")(hist_context_seq)
        # [B, 1, K] → [B, K] for the attention extractor model
        history_attention_weights = layers.Reshape(
            (history_turns,), name="history_attention_weights"
        )(_attn_w)

        # --- NEW: Optional action decoder sub-graph ---
        # Decode past actions from LSTM hidden states, then extract context.
        # Note: use_history_decoding REQUIRES use_history to be True (hist_lstm_out dependency)
        if use_history_decoding:
            if action_vocab_size is None or action_vocab_size == 0:
                raise ValueError("action_vocab_size required for history decoding")
            if not use_history:
                raise ValueError("use_history_decoding requires use_history=True")

            # Add action sequence inputs to model_inputs
            past_action_ids_input = layers.Input(
                shape=(history_turns,),
                dtype="int32",
                name="past_action_ids"
            )
            past_action_mask_input = layers.Input(
                shape=(history_turns,),
                dtype="float32",
                name="past_action_mask"
            )
            model_inputs["past_action_ids"] = past_action_ids_input
            model_inputs["past_action_mask"] = past_action_mask_input

            # Action decoder: reconstruct what actions were taken
            action_logits_decoded = ActionDecoder(
                action_vocab_size,
                hidden_dim=64,
                name="action_decoder"
            )([hist_lstm_out, past_action_mask_input])
            # Output: [batch, K, action_vocab_size]

            # Extract context from decoded actions (what actions matter?)
            past_action_context = PastActionContext(
                _attn_dim,
                name="past_action_context"
            )([shared, action_logits_decoded, past_action_mask_input])
            # Output: [batch, attn_dim]

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
    #
    # Policy head variants:
    #   - default (policy_reads_history=False): reads `shared` (history-free, backward
    #     compatible). The policy-only artifact then takes only the state inputs.
    #   - policy_reads_history=True + use_history=True: reads a raw
    #     Concatenate([shared, history_context]) of shape [hidden_dim + attn_dim].
    #     This is the "decoder" variant — the policy head actually consumes the
    #     history encoding, not just the aux heads. The policy-only artifact is
    #     NO LONGER history-free: it needs event_history_tokens/event_history_mask.
    #   - use_history_decoding=True: policy head fuses [shared + event_context + action_context]
    #     and is trained jointly with action decoder reconstruction loss.

    # NEW: Handle action decoding variant
    if use_history_decoding and past_action_context is not None:
        # Policy head attends to: current state + event history + decoded actions
        policy_input_features = layers.Concatenate(name="policy_all_context")(
            [shared, history_context, past_action_context]
        )
        policy_hidden = layers.Dense(hidden_dim, activation="relu", name="policy_fusion_dense")(policy_input_features)
        policy_logits = layers.Dense(num_policy_classes, name="policy")(policy_hidden)
    elif policy_reads_history and history_context is not None:
        # Event history only: fuse into policy
        policy_input_features = layers.Concatenate(name="policy_history_concat")(
            [shared, history_context]
        )
        policy_logits = layers.Dense(num_policy_classes, name="policy")(policy_input_features)
    else:
        # No history
        policy_logits = layers.Dense(num_policy_classes, name="policy")(shared)

    # Policy-only inference artifact.
    if use_history_decoding:
        # Action decoding variant: include both event and action inputs
        policy_model_inputs = dict(inputs)
        policy_model_inputs["event_history_tokens"] = hist_tokens_input
        policy_model_inputs["event_history_mask"] = hist_mask_input
        policy_model_inputs["past_action_ids"] = past_action_ids_input
        policy_model_inputs["past_action_mask"] = past_action_mask_input
        policy_model = keras.Model(
            policy_model_inputs, policy_logits, name="entity_action_policy_model"
        )
    elif policy_reads_history and history_context is not None:
        # Event history only: include event inputs
        policy_model_inputs = dict(inputs)
        policy_model_inputs["event_history_tokens"] = hist_tokens_input
        policy_model_inputs["event_history_mask"] = hist_mask_input
        policy_model = keras.Model(
            policy_model_inputs, policy_logits, name="entity_action_policy_model"
        )
    else:
        # State only
        policy_model = keras.Model(inputs, policy_logits, name="entity_action_policy_model")

    policy_metrics: List[Any] = [keras.metrics.SparseCategoricalAccuracy(name="top1")]
    if num_policy_classes >= 3:
        policy_metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"))

    use_transition = transition_dim is not None and action_context_vocab_size is not None
    use_sequence = predict_sequence and sequence_vocab_size is not None
    if not use_transition and not predict_value and not use_sequence and not use_history_decoding and not predict_threat and not predict_type_effectiveness:
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

    # β-2: Threat-awareness auxiliary head
    if predict_threat:
        threat_x = layers.Dense(
            threat_hidden_dim or max(64, hidden_dim // 2),
            activation="relu",
            name="threat_dense_0"
        )(shared_with_history)
        if dropout > 0:
            threat_x = layers.Dropout(dropout, name="threat_dropout_0")(threat_x)
        threat_out = layers.Dense(1, activation="relu", name="threat")(threat_x)
        outputs["threat"] = threat_out
        losses["threat"] = keras.losses.MeanSquaredError()
        loss_weights["threat"] = threat_weight
        metrics["threat"] = [keras.metrics.MeanAbsoluteError(name="mae")]

    # β-2: Type-effectiveness auxiliary head (5-class classification: 0.25, 0.5, 1.0, 2.0, 4.0)
    if predict_type_effectiveness:
        type_eff_x = layers.Dense(
            type_eff_hidden_dim or max(64, hidden_dim // 2),
            activation="relu",
            name="type_eff_dense_0"
        )(shared_with_history)
        if dropout > 0:
            type_eff_x = layers.Dropout(dropout, name="type_eff_dropout_0")(type_eff_x)
        type_eff_out = layers.Dense(num_policy_classes, activation="softmax", name="type_eff")(type_eff_x)
        outputs["type_eff"] = type_eff_out
        losses["type_eff"] = keras.losses.MeanSquaredError()
        loss_weights["type_eff"] = type_eff_weight
        metrics["type_eff"] = [keras.metrics.MeanAbsoluteError(name="mae")]

    # NEW: Action decoder losses (if history decoding is enabled)
    if use_history_decoding and action_logits_decoded is not None:
        outputs["decoded_actions"] = action_logits_decoded  # [batch, K, action_vocab_size]
        losses["decoded_actions"] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_weights["decoded_actions"] = decoded_action_weight
        metrics["decoded_actions"] = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

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
