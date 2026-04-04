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

from typing import Any, Dict, List

from .EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES
from .StateVectorization import SIDE_CONDITION_ORDER, STAT_ORDER


POKEMON_NUMERIC_DIM = 6 + len(STAT_ORDER)
GLOBAL_NUMERIC_DIM = 1 + 2 * len(SIDE_CONDITION_ORDER)


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

    def masked_average(sequence_embeddings, token_ids, axis: int):
        # Pools variable-length token lists such as observed moves while ignoring PAD ids.
        mask = tf.cast(tf.not_equal(token_ids, 0), tf.float32)
        while len(mask.shape) < len(sequence_embeddings.shape):
            mask = tf.expand_dims(mask, axis=-1)
        masked = sequence_embeddings * mask
        denom = tf.reduce_sum(mask, axis=axis, keepdims=False)
        denom = tf.maximum(denom, 1.0)
        numer = tf.reduce_sum(masked, axis=axis, keepdims=False)
        return numer / denom

    def masked_pool(values, mask):
        # Reused for active/bench pooling so we can keep the first entity family simple
        # without committing to a full message-passing graph yet.
        mask = tf.cast(mask, tf.float32)
        if len(mask.shape) < len(values.shape):
            mask = tf.expand_dims(mask, axis=-1)
        numer = tf.reduce_sum(values * mask, axis=1)
        denom = tf.reduce_sum(mask, axis=1)
        denom = tf.maximum(denom, 1.0)
        return numer / denom

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
    move_pooled = layers.Lambda(
        lambda tensors: masked_average(tensors[0], tensors[1], axis=2),
        name="observed_move_pool",
    )([move_embedded, inputs["pokemon_observed_moves"]])

    weather_x = layers.Flatten(name="weather_flat")(weather_embedding(inputs["weather"]))
    global_condition_embedded = global_condition_embedding(inputs["global_conditions"])
    global_condition_x = layers.Lambda(
        lambda tensors: masked_average(tensors[0], tensors[1], axis=1),
        name="global_condition_pool",
    )([global_condition_embedded, inputs["global_conditions"]])

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

    self_slots = layers.Lambda(lambda x: x[:, :6, :], name="self_slots")(pokemon_x)
    opp_slots = layers.Lambda(lambda x: x[:, 6:, :], name="opp_slots")(pokemon_x)
    self_numeric = layers.Lambda(lambda x: x[:, :6, :], name="self_numeric")(inputs["pokemon_numeric"])
    opp_numeric = layers.Lambda(lambda x: x[:, 6:, :], name="opp_numeric")(inputs["pokemon_numeric"])

    # These pooled summaries are the key simplification in v1:
    #   - one summary for my active
    #   - one summary for opponent active
    #   - one summary for my bench
    #   - one summary for opponent bench
    # This keeps the model entity-centric without requiring a full graph library yet.
    self_active = layers.Lambda(
        lambda tensors: masked_pool(tensors[0], tensors[1][:, :, 3]),
        name="self_active_pool",
    )([self_slots, self_numeric])
    opp_active = layers.Lambda(
        lambda tensors: masked_pool(tensors[0], tensors[1][:, :, 3]),
        name="opp_active_pool",
    )([opp_slots, opp_numeric])
    self_bench = layers.Lambda(
        lambda tensors: masked_pool(tensors[0], (1.0 - tensors[1][:, :, 3]) * (1.0 - tensors[1][:, :, 4])),
        name="self_bench_pool",
    )([self_slots, self_numeric])
    opp_bench = layers.Lambda(
        lambda tensors: masked_pool(tensors[0], (1.0 - tensors[1][:, :, 3]) * (1.0 - tensors[1][:, :, 4])),
        name="opp_bench_pool",
    )([opp_slots, opp_numeric])
    all_slots = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), name="all_slot_pool")(pokemon_x)

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
    # v1 still predicts the global action vocabulary because offline logs do not contain
    # the per-turn legal request objects needed for true action-wise legal scoring.
    policy_logits = layers.Dense(num_policy_classes, name="policy")(shared)
    policy_model = keras.Model(inputs, policy_logits, name="entity_action_policy_model")

    policy_metrics: List[Any] = [keras.metrics.SparseCategoricalAccuracy(name="top1")]
    if num_policy_classes >= 3:
        policy_metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"))

    use_transition = transition_dim is not None and action_context_vocab_size is not None
    if not use_transition and not predict_value:
        policy_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=policy_metrics,
        )
        return policy_model, policy_model, None

    outputs: Dict[str, Any] = {"policy": policy_logits}
    losses: Dict[str, Any] = {
        "policy": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    loss_weights: Dict[str, float] = {"policy": 1.0}
    metrics: Dict[str, List[Any]] = {"policy": policy_metrics}

    policy_value_model = None
    if predict_value:
        # Keep value state-only and conservative: it estimates eventual win probability.
        value_x = layers.Dense(value_hidden_dim or max(64, hidden_dim // 2), activation="relu", name="value_dense")(shared)
        if dropout > 0:
            value_x = layers.Dropout(dropout, name="value_dropout")(value_x)
        value_out = layers.Dense(1, activation="sigmoid", name="value")(value_x)
        outputs["value"] = value_out
        losses["value"] = keras.losses.BinaryCrossentropy()
        loss_weights["value"] = value_weight
        metrics["value"] = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="brier"),
        ]
        policy_value_model = keras.Model(
            inputs,
            {"policy": policy_logits, "value": value_out},
            name="entity_action_policy_value_model",
        )

    model_inputs: Dict[str, Any] = dict(inputs)
    if use_transition:
        my_action_input = layers.Input(shape=(), dtype="int32", name="my_action")
        opp_action_input = layers.Input(shape=(), dtype="int32", name="opp_action")
        action_embedding = layers.Embedding(
            input_dim=action_context_vocab_size,
            output_dim=max(8, action_embed_dim),
            name="action_embedding",
        )
        transition_x = layers.Concatenate(name="transition_features")(
            [
                shared,
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
        model_inputs["my_action"] = my_action_input
        model_inputs["opp_action"] = opp_action_input

    # The training artifact is the richest object because it includes every enabled
    # auxiliary head. The policy and policy+value artifacts stay serving-friendly.
    training_model = keras.Model(model_inputs, outputs, name="entity_action_training_model")
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return training_model, policy_model, policy_value_model
