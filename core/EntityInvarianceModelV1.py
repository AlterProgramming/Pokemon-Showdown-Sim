from __future__ import annotations

"""Keras model builder for entity_invariance_aux_v1.

This family extends the current entity encoder with:
    - current-turn entity inputs
    - one-step previous-turn entity inputs
    - a latent summary derived from the previous turn and fused into the current turn

It is a scaffold, not the final recurrence design. The purpose is to make the
identity-invariance family trainable now while keeping later recurrent upgrades
possible under the same family line.
"""

from typing import Any, Dict, List

from .EntityModelV1 import GLOBAL_NUMERIC_DIM, POKEMON_NUMERIC_DIM
from .EntityTensorization import MAX_GLOBAL_CONDITIONS, MAX_OBSERVED_MOVES


def build_entity_invariance_models(
    *,
    vocab_sizes: Dict[str, int],
    num_policy_classes: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    learning_rate: float,
    token_embed_dim: int = 24,
    latent_dim: int = 64,
    transition_dim: int | None = None,
    action_context_vocab_size: int | None = None,
    action_embed_dim: int = 16,
    transition_hidden_dim: int | None = None,
    transition_weight: float = 0.25,
    predict_value: bool = False,
    value_hidden_dim: int | None = None,
    value_weight: float = 0.25,
):
    """Build the multitask entity_invariance_aux_v1 models."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    def masked_average(sequence_embeddings, token_ids, axis: int):
        mask = tf.cast(tf.not_equal(token_ids, 0), tf.float32)
        while len(mask.shape) < len(sequence_embeddings.shape):
            mask = tf.expand_dims(mask, axis=-1)
        masked = sequence_embeddings * mask
        denom = tf.reduce_sum(mask, axis=axis, keepdims=False)
        denom = tf.maximum(denom, 1.0)
        numer = tf.reduce_sum(masked, axis=axis, keepdims=False)
        return numer / denom

    def masked_pool(values, mask):
        mask = tf.cast(mask, tf.float32)
        if len(mask.shape) < len(values.shape):
            mask = tf.expand_dims(mask, axis=-1)
        numer = tf.reduce_sum(values * mask, axis=1)
        denom = tf.reduce_sum(mask, axis=1)
        denom = tf.maximum(denom, 1.0)
        return numer / denom

    def make_branch_inputs(prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}pokemon_species": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_species"),
            f"{prefix}pokemon_item": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_item"),
            f"{prefix}pokemon_ability": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_ability"),
            f"{prefix}pokemon_tera": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_tera"),
            f"{prefix}pokemon_status": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_status"),
            f"{prefix}pokemon_side": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_side"),
            f"{prefix}pokemon_slot": layers.Input(shape=(12,), dtype="int32", name=f"{prefix}pokemon_slot"),
            f"{prefix}pokemon_observed_moves": layers.Input(
                shape=(12, MAX_OBSERVED_MOVES),
                dtype="int32",
                name=f"{prefix}pokemon_observed_moves",
            ),
            f"{prefix}pokemon_numeric": layers.Input(
                shape=(12, POKEMON_NUMERIC_DIM),
                dtype="float32",
                name=f"{prefix}pokemon_numeric",
            ),
            f"{prefix}weather": layers.Input(shape=(1,), dtype="int32", name=f"{prefix}weather"),
            f"{prefix}global_conditions": layers.Input(
                shape=(MAX_GLOBAL_CONDITIONS,),
                dtype="int32",
                name=f"{prefix}global_conditions",
            ),
            f"{prefix}global_numeric": layers.Input(
                shape=(GLOBAL_NUMERIC_DIM,),
                dtype="float32",
                name=f"{prefix}global_numeric",
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

    pokemon_dense_0 = layers.Dense(hidden_dim, activation="relu", name="pokemon_dense_0")
    pokemon_dropout_0 = layers.Dropout(dropout, name="pokemon_dropout_0") if dropout > 0 else None
    pokemon_dense_1 = layers.Dense(hidden_dim, activation="relu", name="pokemon_dense_1")
    global_dense = layers.Dense(hidden_dim, activation="relu", name="global_dense")

    def encode_state(branch_inputs: Dict[str, Any], branch_prefix: str):
        move_embedded = move_embedding(branch_inputs[f"{branch_prefix}pokemon_observed_moves"])
        move_pooled = layers.Lambda(
            lambda tensors: masked_average(tensors[0], tensors[1], axis=2),
            name=f"{branch_prefix}observed_move_pool",
        )([move_embedded, branch_inputs[f"{branch_prefix}pokemon_observed_moves"]])

        weather_x = layers.Flatten(name=f"{branch_prefix}weather_flat")(
            weather_embedding(branch_inputs[f"{branch_prefix}weather"])
        )
        global_condition_embedded = global_condition_embedding(branch_inputs[f"{branch_prefix}global_conditions"])
        global_condition_x = layers.Lambda(
            lambda tensors: masked_average(tensors[0], tensors[1], axis=1),
            name=f"{branch_prefix}global_condition_pool",
        )([global_condition_embedded, branch_inputs[f"{branch_prefix}global_conditions"]])

        pokemon_x = layers.Concatenate(axis=-1, name=f"{branch_prefix}pokemon_concat")(
            [
                species_embedding(branch_inputs[f"{branch_prefix}pokemon_species"]),
                item_embedding(branch_inputs[f"{branch_prefix}pokemon_item"]),
                ability_embedding(branch_inputs[f"{branch_prefix}pokemon_ability"]),
                tera_embedding(branch_inputs[f"{branch_prefix}pokemon_tera"]),
                status_embedding(branch_inputs[f"{branch_prefix}pokemon_status"]),
                side_embedding(branch_inputs[f"{branch_prefix}pokemon_side"]),
                slot_embedding(branch_inputs[f"{branch_prefix}pokemon_slot"]),
                move_pooled,
                branch_inputs[f"{branch_prefix}pokemon_numeric"],
            ]
        )
        pokemon_x = pokemon_dense_0(pokemon_x)
        if pokemon_dropout_0 is not None:
            pokemon_x = pokemon_dropout_0(pokemon_x)
        pokemon_x = pokemon_dense_1(pokemon_x)

        self_slots = layers.Lambda(lambda x: x[:, :6, :], name=f"{branch_prefix}self_slots")(pokemon_x)
        opp_slots = layers.Lambda(lambda x: x[:, 6:, :], name=f"{branch_prefix}opp_slots")(pokemon_x)
        self_numeric = layers.Lambda(
            lambda x: x[:, :6, :],
            name=f"{branch_prefix}self_numeric",
        )(branch_inputs[f"{branch_prefix}pokemon_numeric"])
        opp_numeric = layers.Lambda(
            lambda x: x[:, 6:, :],
            name=f"{branch_prefix}opp_numeric",
        )(branch_inputs[f"{branch_prefix}pokemon_numeric"])

        self_active = layers.Lambda(
            lambda tensors: masked_pool(tensors[0], tensors[1][:, :, 3]),
            name=f"{branch_prefix}self_active_pool",
        )([self_slots, self_numeric])
        opp_active = layers.Lambda(
            lambda tensors: masked_pool(tensors[0], tensors[1][:, :, 3]),
            name=f"{branch_prefix}opp_active_pool",
        )([opp_slots, opp_numeric])
        self_bench = layers.Lambda(
            lambda tensors: masked_pool(tensors[0], (1.0 - tensors[1][:, :, 3]) * (1.0 - tensors[1][:, :, 4])),
            name=f"{branch_prefix}self_bench_pool",
        )([self_slots, self_numeric])
        opp_bench = layers.Lambda(
            lambda tensors: masked_pool(tensors[0], (1.0 - tensors[1][:, :, 3]) * (1.0 - tensors[1][:, :, 4])),
            name=f"{branch_prefix}opp_bench_pool",
        )([opp_slots, opp_numeric])
        all_slots = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1),
            name=f"{branch_prefix}all_slot_pool",
        )(pokemon_x)

        global_x = layers.Concatenate(name=f"{branch_prefix}global_concat")(
            [
                weather_x,
                global_condition_x,
                branch_inputs[f"{branch_prefix}global_numeric"],
            ]
        )
        global_x = global_dense(global_x)

        return layers.Concatenate(name=f"{branch_prefix}state_summary")(
            [self_active, opp_active, self_bench, opp_bench, all_slots, global_x]
        )

    current_inputs = make_branch_inputs("")
    previous_inputs = make_branch_inputs("prev_")
    model_inputs: Dict[str, Any] = {**current_inputs, **previous_inputs}

    current_state = encode_state(current_inputs, "")
    previous_state = encode_state(previous_inputs, "prev_")
    latent_prev = layers.Dense(latent_dim, activation="tanh", name="latent_prev")(previous_state)
    latent_gate = layers.Dense(latent_dim, activation="sigmoid", name="latent_gate")(current_state)
    latent_context = layers.Multiply(name="latent_context")([latent_prev, latent_gate])
    state_delta = layers.Subtract(name="state_delta")([current_state, previous_state])

    x = layers.Concatenate(name="state_concat")([current_state, previous_state, state_delta, latent_context])
    for layer_idx in range(depth):
        x = layers.Dense(hidden_dim, activation="relu", name=f"trunk_dense_{layer_idx}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"trunk_dropout_{layer_idx}")(x)

    shared = x
    policy_logits = layers.Dense(num_policy_classes, name="policy")(shared)
    policy_model = keras.Model(model_inputs, policy_logits, name="entity_invariance_policy_model")

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
            model_inputs,
            {"policy": policy_logits, "value": value_out},
            name="entity_invariance_policy_value_model",
        )

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
                action_embedding(my_action_input),
                action_embedding(opp_action_input),
            ]
        )
        transition_x = layers.Dense(
            transition_hidden_dim or hidden_dim,
            activation="relu",
            name="transition_dense",
        )(transition_x)
        if dropout > 0:
            transition_x = layers.Dropout(dropout, name="transition_dropout")(transition_x)
        transition_out = layers.Dense(transition_dim, name="transition")(transition_x)
        outputs["transition"] = transition_out
        losses["transition"] = keras.losses.MeanSquaredError()
        loss_weights["transition"] = transition_weight
        metrics["transition"] = [keras.metrics.MeanAbsoluteError(name="mae")]
        model_inputs["my_action"] = my_action_input
        model_inputs["opp_action"] = opp_action_input

    training_model = keras.Model(model_inputs, outputs, name="entity_invariance_training_model")
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return training_model, policy_model, policy_value_model
