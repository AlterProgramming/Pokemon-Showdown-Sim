from __future__ import annotations

"""Keras model builder for legal-action-conditioned entity scoring."""

from typing import Any, Dict, List

from .EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES
from .EntityTensorizationV2 import MAX_LEGAL_ACTIONS
from .StateVectorization import SIDE_CONDITION_ORDER, STAT_ORDER


POKEMON_NUMERIC_DIM = 6 + len(STAT_ORDER)
GLOBAL_NUMERIC_DIM = 1 + 2 * len(SIDE_CONDITION_ORDER)


def _keras():
    try:
        import keras
        return keras
    except ImportError:
        from tensorflow import keras
        return keras


def _tf():
    import tensorflow as tf
    return tf


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class GatherSelfSwitchSlot(_keras().layers.Layer):
    """Gather encoded self-side slot vectors for switch candidates."""

    def call(self, inputs):
        tf = _tf()
        self_slots, switch_slot_ids = inputs
        zero_slot = tf.zeros_like(self_slots[:, :1, :])
        padded_slots = tf.concat([zero_slot, self_slots], axis=1)
        return tf.gather(padded_slots, tf.cast(switch_slot_ids, tf.int32), batch_dims=1)


@_keras().saving.register_keras_serializable(package="EntityModelV2")
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


@_keras().saving.register_keras_serializable(package="EntityModelV2")
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


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class SlotSlice(_keras().layers.Layer):
    """Slice `inputs[:, start:end, :]`."""

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


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class ReduceMean1(_keras().layers.Layer):
    """Reduce mean over axis=1."""

    def call(self, inputs):
        return _tf().reduce_mean(inputs, axis=1)


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class ExtractColumn(_keras().layers.Layer):
    """Extract one column from a 3-D tensor."""

    def __init__(self, col_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.col_idx = col_idx

    def call(self, inputs):
        return inputs[:, :, self.col_idx]

    def get_config(self):
        cfg = super().get_config()
        cfg["col_idx"] = self.col_idx
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class BenchMask(_keras().layers.Layer):
    """Bench mask: slots that are neither active nor fainted."""

    def call(self, inputs):
        numeric = inputs
        active = numeric[:, :, 3]
        fainted = numeric[:, :, 4]
        return (1.0 - active) * (1.0 - fainted)


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class CandidateTypeMask(_keras().layers.Layer):
    """Emit an expanded float mask for one candidate type."""

    def __init__(self, match_value: int, **kwargs):
        super().__init__(**kwargs)
        self.match_value = int(match_value)

    def call(self, inputs):
        tf = _tf()
        mask = tf.cast(tf.equal(inputs, self.match_value), tf.float32)
        return tf.expand_dims(mask, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["match_value"] = self.match_value
        return cfg


@_keras().saving.register_keras_serializable(package="EntityModelV2")
class MaskCandidateLogits(_keras().layers.Layer):
    """Set padded candidate logits to a large negative value."""

    def call(self, inputs):
        tf = _tf()
        logits, candidate_mask = inputs
        mask = tf.cast(candidate_mask, logits.dtype)
        large_negative = tf.cast(-1e9, logits.dtype)
        return logits * mask + (1.0 - mask) * large_negative


def build_entity_action_v2_models(
    *,
    vocab_sizes: Dict[str, int],
    hidden_dim: int,
    depth: int,
    dropout: float,
    learning_rate: float,
    token_embed_dim: int = 24,
    max_candidates: int = MAX_LEGAL_ACTIONS,
    predict_value: bool = False,
    value_hidden_dim: int | None = None,
    value_weight: float = 0.25,
):
    """Build the v2 policy model that scores only current-turn legal actions."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs: Dict[str, Any] = {
        "pokemon_species": layers.Input(shape=(12,), dtype="int32", name="pokemon_species"),
        "pokemon_item": layers.Input(shape=(12,), dtype="int32", name="pokemon_item"),
        "pokemon_ability": layers.Input(shape=(12,), dtype="int32", name="pokemon_ability"),
        "pokemon_tera": layers.Input(shape=(12,), dtype="int32", name="pokemon_tera"),
        "pokemon_status": layers.Input(shape=(12,), dtype="int32", name="pokemon_status"),
        "pokemon_side": layers.Input(shape=(12,), dtype="int32", name="pokemon_side"),
        "pokemon_slot": layers.Input(shape=(12,), dtype="int32", name="pokemon_slot"),
        "pokemon_observed_moves": layers.Input(shape=(12, MAX_OBSERVED_MOVES), dtype="int32", name="pokemon_observed_moves"),
        "pokemon_numeric": layers.Input(shape=(12, POKEMON_NUMERIC_DIM), dtype="float32", name="pokemon_numeric"),
        "weather": layers.Input(shape=(1,), dtype="int32", name="weather"),
        "global_conditions": layers.Input(shape=(MAX_GLOBAL_CONDITIONS,), dtype="int32", name="global_conditions"),
        "global_numeric": layers.Input(shape=(GLOBAL_NUMERIC_DIM,), dtype="float32", name="global_numeric"),
        "candidate_type": layers.Input(shape=(max_candidates,), dtype="int32", name="candidate_type"),
        "candidate_move": layers.Input(shape=(max_candidates,), dtype="int32", name="candidate_move"),
        "candidate_switch_slot": layers.Input(shape=(max_candidates,), dtype="int32", name="candidate_switch_slot"),
        "candidate_mask": layers.Input(shape=(max_candidates,), dtype="float32", name="candidate_mask"),
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
    candidate_type_embedding = layers.Embedding(3, max(4, token_embed_dim // 4), name="candidate_type_embedding")

    move_embedded = move_embedding(inputs["pokemon_observed_moves"])
    move_pooled = MaskedAverage(axis=2, name="observed_move_pool")([move_embedded, inputs["pokemon_observed_moves"]])

    weather_x = layers.Flatten(name="weather_flat")(weather_embedding(inputs["weather"]))
    global_condition_embedded = global_condition_embedding(inputs["global_conditions"])
    global_condition_x = MaskedAverage(axis=1, name="global_condition_pool")([global_condition_embedded, inputs["global_conditions"]])

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
    pokemon_x = layers.Dense(hidden_dim, activation="relu", name="pokemon_dense_0")(pokemon_x)
    if dropout > 0:
        pokemon_x = layers.Dropout(dropout, name="pokemon_dropout_0")(pokemon_x)
    pokemon_x = layers.Dense(hidden_dim, activation="relu", name="pokemon_dense_1")(pokemon_x)

    self_slots = SlotSlice(0, 6, name="self_slots")(pokemon_x)
    opp_slots = SlotSlice(6, 12, name="opp_slots")(pokemon_x)
    self_numeric = SlotSlice(0, 6, name="self_numeric")(inputs["pokemon_numeric"])
    opp_numeric = SlotSlice(6, 12, name="opp_numeric")(inputs["pokemon_numeric"])

    self_active = MaskedPool(name="self_active_pool")([self_slots, ExtractColumn(3, name="self_active_flag")(self_numeric)])
    opp_active = MaskedPool(name="opp_active_pool")([opp_slots, ExtractColumn(3, name="opp_active_flag")(opp_numeric)])
    self_bench = MaskedPool(name="self_bench_pool")([self_slots, BenchMask(name="self_bench_mask")(self_numeric)])
    opp_bench = MaskedPool(name="opp_bench_pool")([opp_slots, BenchMask(name="opp_bench_mask")(opp_numeric)])
    all_slots = ReduceMean1(name="all_slot_pool")(pokemon_x)

    global_x = layers.Concatenate(name="global_concat")([weather_x, global_condition_x, inputs["global_numeric"]])
    global_x = layers.Dense(hidden_dim, activation="relu", name="global_dense")(global_x)

    state_x = layers.Concatenate(name="state_concat")([self_active, opp_active, self_bench, opp_bench, all_slots, global_x])
    for layer_idx in range(depth):
        state_x = layers.Dense(hidden_dim, activation="relu", name=f"trunk_dense_{layer_idx}")(state_x)
        if dropout > 0:
            state_x = layers.Dropout(dropout, name=f"trunk_dropout_{layer_idx}")(state_x)

    shared_state = state_x

    candidate_move_x = move_embedding(inputs["candidate_move"])
    candidate_switch_slot_x = slot_embedding(inputs["candidate_switch_slot"])
    switch_slot_state_x = GatherSelfSwitchSlot(name="gather_self_switch_slot")(
        [self_slots, inputs["candidate_switch_slot"]]
    )
    move_type_mask = CandidateTypeMask(1, name="move_type_mask")(inputs["candidate_type"])
    switch_type_mask = CandidateTypeMask(2, name="switch_type_mask")(inputs["candidate_type"])

    move_candidate_x = layers.Multiply(name="move_candidate_repr")([candidate_move_x, move_type_mask])
    switch_candidate_x = layers.Multiply(name="switch_candidate_repr")(
        [layers.Concatenate(name="switch_candidate_concat")([candidate_switch_slot_x, switch_slot_state_x]), switch_type_mask]
    )
    repeated_state = layers.RepeatVector(max_candidates, name="repeat_state_for_candidates")(shared_state)
    candidate_x = layers.Concatenate(name="candidate_concat")(
        [
            repeated_state,
            candidate_type_embedding(inputs["candidate_type"]),
            move_candidate_x,
            switch_candidate_x,
        ]
    )
    candidate_x = layers.Dense(hidden_dim, activation="relu", name="candidate_dense")(candidate_x)
    if dropout > 0:
        candidate_x = layers.Dropout(dropout, name="candidate_dropout")(candidate_x)
    raw_policy_logits = layers.TimeDistributed(
        layers.Dense(1),
        name="candidate_score",
    )(candidate_x)
    policy_logits = layers.Flatten(name="candidate_logits_flat")(raw_policy_logits)
    policy_logits = MaskCandidateLogits(name="policy")([policy_logits, inputs["candidate_mask"]])

    policy_model = keras.Model(inputs, policy_logits, name="entity_action_v2_policy_model")
    policy_metrics: List[Any] = [keras.metrics.SparseCategoricalAccuracy(name="top1")]
    if max_candidates >= 3:
        policy_metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"))

    if not predict_value:
        policy_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=policy_metrics,
        )
        return policy_model, policy_model, None

    value_x = layers.Dense(value_hidden_dim or max(64, hidden_dim // 2), activation="relu", name="value_dense")(shared_state)
    if dropout > 0:
        value_x = layers.Dropout(dropout, name="value_dropout")(value_x)
    value_out = layers.Dense(1, activation="sigmoid", name="value")(value_x)

    training_model = keras.Model(
        inputs,
        {"policy": policy_logits, "value": value_out},
        name="entity_action_v2_training_model",
    )
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "policy": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "value": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"policy": 1.0, "value": value_weight},
        metrics={
            "policy": policy_metrics,
            "value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.MeanSquaredError(name="brier"),
            ],
        },
    )
    policy_value_model = keras.Model(
        inputs,
        {"policy": policy_logits, "value": value_out},
        name="entity_action_v2_policy_value_model",
    )
    return training_model, policy_model, policy_value_model
