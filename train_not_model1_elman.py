from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from NoTModel1Elman import (
	DEFAULT_ELMAN_DEPTH,
	DEFAULT_ELMAN_HIDDEN_DIM,
	MODEL1_ACTION_CLASSES,
	MODEL1_GUIDE_ID,
    MODEL1_INPUT_DIM,
    TARGET_MODEL_ID,
    build_elman_policy_model,
    build_serving_model,
    linear_cka,
    normalize_model1_state_vector,
    pad_state_history,
    progressive_replacement_schedule,
    tf_linear_cka_dissimilarity,
)


def build_sequence_dataset(
    examples: Iterable[dict[str, Any]],
    action_vocab: dict[str, int],
    *,
    sequence_length: int,
    feature_dim: int = MODEL1_INPUT_DIM,
    state_encoder: Callable[[dict[str, Any], str], Sequence[float]] | None = None,
) -> dict[str, np.ndarray]:
    """Turn per-turn examples into prefix windows grouped by battle and player."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    unknown_id = int(action_vocab.get("<UNK>", 0))
    grouped: dict[tuple[str, str], list[tuple[int, int, np.ndarray, int]]] = defaultdict(list)

    if state_encoder is None:
        from core.StateVectorization import encode_state_v0

        state_encoder = encode_state_v0

    for source_index, example in enumerate(examples):
        battle_id = str(example.get("battle_id") or "")
        player = str(example.get("player") or "")
        if not battle_id or player not in {"p1", "p2"}:
            continue
        raw_vector = example.get("state_vector")
        if raw_vector is None:
            raw_state = example.get("state")
            if not isinstance(raw_state, dict):
                raise ValueError("examples require state_vector or a state mapping")
            raw_vector = state_encoder(raw_state, player)
        if feature_dim == MODEL1_INPUT_DIM and len(raw_vector) != feature_dim:
            raw_vector = normalize_model1_state_vector(raw_vector)
        vector = np.asarray(raw_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != feature_dim:
            raise ValueError(
                f"example state vector has shape {vector.shape}; expected ({feature_dim},)"
            )
        token = str(example.get("action_token") or "<UNK>")
        label = int(action_vocab.get(token, unknown_id))
        turn_number = int(example.get("turn_number") or 0)
        grouped[(battle_id, player)].append((turn_number, source_index, vector, label))

    state_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    label_windows: list[np.ndarray] = []
    battle_ids: list[str] = []
    players: list[str] = []
    final_turns: list[int] = []

    for (battle_id, player), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row[0], row[1]))
        for end_index in range(len(rows)):
            window_rows = rows[max(0, end_index + 1 - sequence_length): end_index + 1]
            window_vectors = np.asarray([row[2] for row in window_rows], dtype=np.float32)
            padded, mask = pad_state_history(
                window_vectors,
                sequence_length=sequence_length,
                feature_dim=feature_dim,
            )
            labels = np.zeros((sequence_length,), dtype=np.int64)
            padding_length = sequence_length - len(window_rows)
            labels[padding_length:] = np.asarray([row[3] for row in window_rows], dtype=np.int64)
            state_windows.append(padded)
            masks.append(mask)
            label_windows.append(labels)
            battle_ids.append(battle_id)
            players.append(player)
            final_turns.append(int(window_rows[-1][0]))

    if not state_windows:
        return {
            "states": np.empty((0, sequence_length, feature_dim), dtype=np.float32),
            "masks": np.empty((0, sequence_length), dtype=np.float32),
            "labels": np.empty((0, sequence_length), dtype=np.int64),
            "battle_ids": np.empty((0,), dtype=object),
            "players": np.empty((0,), dtype=object),
            "final_turns": np.empty((0,), dtype=np.int64),
        }

    return {
        "states": np.asarray(state_windows, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.float32),
        "labels": np.asarray(label_windows, dtype=np.int64),
        "battle_ids": np.asarray(battle_ids, dtype=object),
        "players": np.asarray(players, dtype=object),
        "final_turns": np.asarray(final_turns, dtype=np.int64),
    }


def masked_policy_accuracy(logits: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> float:
    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels)
    masks_array = np.asarray(masks, dtype=np.float32)
    if logits_array.ndim != 3 or labels_array.shape != masks_array.shape or logits_array.shape[:2] != labels_array.shape:
        raise ValueError("logits, labels, and masks have incompatible shapes")
    valid = masks_array > 0.0
    if not np.any(valid):
        return 0.0
    predicted = np.argmax(logits_array, axis=-1)
    return float(np.mean(predicted[valid] == labels_array[valid]))


def alignment_loss_from_representations(
    guide_activations: Sequence[np.ndarray],
    target_activations: Sequence[np.ndarray],
    masks: np.ndarray,
) -> float:
    """Compute mean CKA dissimilarity across a replaced prefix."""
    if len(guide_activations) != len(target_activations) or not guide_activations:
        raise ValueError("guide and target activation lists must have equal non-zero length")
    mask_array = np.asarray(masks, dtype=np.float32)
    if mask_array.ndim != 2:
        raise ValueError("masks must be a rank-2 array")
    valid = mask_array.reshape(-1) > 0.0
    losses: list[float] = []
    for guide, target in zip(guide_activations, target_activations):
        guide_array = np.asarray(guide, dtype=np.float32)
        target_array = np.asarray(target, dtype=np.float32)
        if guide_array.shape[:2] != mask_array.shape or target_array.shape[:2] != mask_array.shape:
            raise ValueError("activation and mask shapes do not match")
        losses.append(
            1.0
            - linear_cka(
                guide_array.reshape(-1, guide_array.shape[-1])[valid],
                target_array.reshape(-1, target_array.shape[-1])[valid],
            )
        )
    return float(np.mean(losses))


def build_not_metadata(
    *,
    guide_model_path: str,
    target_model_path: str,
    policy_vocab_path: str,
    sequence_length: int,
    base_feature_dim: int,
    num_action_classes: int,
    train_examples: int,
    val_examples: int,
    stage_metrics: list[dict[str, Any]],
    depth: int = DEFAULT_ELMAN_DEPTH,
    hidden_dim: int = DEFAULT_ELMAN_HIDDEN_DIM,
    dropout: float = 0.1,
    guide_vocab_path: str = "artifacts/action_vocab_1.json",
    target_training_model_path: str | None = None,
) -> dict[str, Any]:
    return {
        "model_path": target_model_path,
        "policy_model_path": target_model_path,
        "training_model_path": target_training_model_path,
        "policy_vocab_path": policy_vocab_path,
        "guide_model_path": guide_model_path,
        "guide_vocab_path": guide_vocab_path,
        "feature_dim": int(sequence_length * base_feature_dim),
        "base_feature_dim": int(base_feature_dim),
        "sequence_length": int(sequence_length),
        "sequence_model": True,
        "sequence_padding": "repeat_first",
        "state_schema_version": "flat_public_sequence_v1",
        "num_action_classes": int(num_action_classes),
        "action_space": "joint",
        "include_switches": True,
        "model_release_id": TARGET_MODEL_ID,
        "parent_release_id": MODEL1_GUIDE_ID,
        "family_id": "vector_joint_bc_not_elman",
        "family_version": 1,
        "family_name": "vector_joint_bc_not_elman_v1",
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "dropout": float(dropout),
        "training_regime": "offline_not_alignment_then_bc",
        "information_policy": "public_only",
        "action_parameterization": "joint_vocab",
        "objective_set": ["policy", "representation_alignment"],
        "alignment_metric": "linear_cka",
        "replacement_schedule": [list(stage) for stage in progressive_replacement_schedule(depth)],
        "alignment_stage_metrics": stage_metrics,
        "train_examples": int(train_examples),
        "val_examples": int(val_examples),
        "initialization_source": {
            "type": "network_of_theseus",
            "guide_release_id": MODEL1_GUIDE_ID,
        },
        "registry_visibility": "runnable_policy",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _dense_layers_from_guide(guide_model: Any, depth: int) -> list[Any]:
    dense_layers = [layer for layer in guide_model.layers if layer.__class__.__name__ == "Dense"]
    if len(dense_layers) < depth + 1:
        raise ValueError(
            f"guide model exposes {len(dense_layers)} Dense layers; expected at least {depth + 1}"
        )
    return dense_layers[: depth + 1]


def _copy_time_distributed_dense(source_layer: Any, inputs: Any, *, name: str) -> Any:
    from tensorflow.keras import layers

    copied_dense = layers.Dense(
        int(source_layer.units),
        activation=getattr(source_layer, "activation", None),
        use_bias=bool(source_layer.use_bias),
        name=f"{name}_dense",
    )
    output = layers.TimeDistributed(copied_dense, name=name)(inputs)
    copied_dense.set_weights(source_layer.get_weights())
    copied_dense.trainable = False
    return output


def build_guide_activation_model(guide_model: Any, *, sequence_length: int, depth: int = DEFAULT_ELMAN_DEPTH) -> Any:
    """Apply frozen Model1 dense blocks independently to every sequence row."""
    from tensorflow import keras
    from tensorflow.keras import layers

    dense_layers = _dense_layers_from_guide(guide_model, depth)
    input_dim = int(guide_model.input_shape[-1])
    inputs = layers.Input(shape=(sequence_length, input_dim), name="state_sequence")
    hidden = inputs
    activations: list[Any] = []
    for index in range(depth):
        hidden = _copy_time_distributed_dense(dense_layers[index], hidden, name=f"guide_dense_{index + 1}")
        activations.append(hidden)
    logits = _copy_time_distributed_dense(dense_layers[depth], hidden, name="guide_policy")
    return keras.Model(inputs, activations + [logits], name="model1_guide_sequence")


def initialize_target_policy_model(
    guide_model: Any,
    *,
    sequence_length: int,
    hidden_dim: int = DEFAULT_ELMAN_HIDDEN_DIM,
    depth: int = DEFAULT_ELMAN_DEPTH,
    dropout: float = 0.1,
    name: str = TARGET_MODEL_ID,
) -> Any:
    """Build the target and initialize its policy head from Model1."""
    target = build_elman_policy_model(
        input_dim=int(guide_model.input_shape[-1]),
        num_classes=int(guide_model.output_shape[-1]),
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        name=name,
    )
    guide_dense_layers = _dense_layers_from_guide(guide_model, depth)
    target.get_layer("policy_sequence").layer.set_weights(guide_dense_layers[depth].get_weights())
    return target


def build_hybrid_stage_model(
    guide_model: Any,
    target_model: Any,
    *,
    sequence_length: int,
    replaced_layer_count: int,
    depth: int = DEFAULT_ELMAN_DEPTH,
) -> Any:
    """Construct the guide/target hybrid for one progressive alignment stage."""
    from tensorflow import keras
    from tensorflow.keras import layers

    if not 1 <= replaced_layer_count <= depth:
        raise ValueError("replaced_layer_count must identify a non-empty target prefix")
    guide_dense_layers = _dense_layers_from_guide(guide_model, depth)
    input_dim = int(guide_model.input_shape[-1])
    inputs = layers.Input(shape=(sequence_length, input_dim), name="state_sequence")
    hidden = inputs
    target_activations: list[Any] = []
    for index in range(replaced_layer_count):
        hidden = target_model.get_layer(f"elman_{index + 1}")(hidden)
        target_activations.append(hidden)
        try:
            hidden = target_model.get_layer(f"elman_dropout_{index + 1}")(hidden)
        except ValueError:
            pass

    for index in range(replaced_layer_count, depth):
        hidden = _copy_time_distributed_dense(guide_dense_layers[index], hidden, name=f"guide_suffix_{index + 1}")
    logits = _copy_time_distributed_dense(guide_dense_layers[depth], hidden, name="guide_policy_suffix")
    return keras.Model(
        inputs,
        target_activations + [logits],
        name=f"model1_not_hybrid_stage_{replaced_layer_count}",
    )


def _batch_slices(size: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, size, batch_size)]


def train_alignment_stage(
    stage_model: Any,
    guide_activation_model: Any,
    states: np.ndarray,
    masks: np.ndarray,
    *,
    replaced_layer_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 42,
) -> dict[str, float]:
    """Optimize only the current target prefix against frozen guide activations."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if len(states) == 0:
        raise ValueError("alignment requires at least one sequence")

    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    final_losses: list[float] = []
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for batch_indices in _batch_slices(len(states), batch_size, shuffle=True, seed=seed + epoch):
            batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
            batch_masks = tf.convert_to_tensor(masks[batch_indices], dtype=tf.float32)
            with tf.GradientTape() as tape:
                guide_outputs = guide_activation_model(batch_states, training=False)
                target_outputs = stage_model(batch_states, training=True)
                stage_losses = [
                    tf_linear_cka_dissimilarity(
                        target_outputs[layer_index],
                        guide_outputs[layer_index],
                        batch_masks,
                    )
                    for layer_index in range(replaced_layer_count)
                ]
                loss = tf.add_n(stage_losses) / float(replaced_layer_count)
            gradients = tape.gradient(loss, stage_model.trainable_variables)
            optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, stage_model.trainable_variables)
                if gradient is not None
            )
            epoch_losses.append(float(loss.numpy()))
        final_losses.append(float(np.mean(epoch_losses)))

    return {
        "replaced_layer_count": float(replaced_layer_count),
        "epochs": float(epochs),
        "final_loss": final_losses[-1],
        "best_loss": float(np.min(final_losses)),
    }


def train_target_policy(
    target_model: Any,
    states: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 42,
) -> dict[str, float]:
    """Fine-tune the completed recurrent target on masked action labels."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if len(states) == 0:
        raise ValueError("policy training requires at least one sequence")

    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    final_losses: list[float] = []
    final_accuracies: list[float] = []
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        epoch_logits: list[np.ndarray] = []
        epoch_labels: list[np.ndarray] = []
        epoch_masks: list[np.ndarray] = []
        for batch_indices in _batch_slices(len(states), batch_size, shuffle=True, seed=seed + epoch):
            batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
            batch_masks = tf.convert_to_tensor(masks[batch_indices], dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(labels[batch_indices], dtype=tf.int64)
            with tf.GradientTape() as tape:
                logits = target_model(batch_states, training=True)
                per_step_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    batch_labels,
                    logits,
                    from_logits=True,
                )
                loss = tf.reduce_sum(per_step_loss * batch_masks) / tf.maximum(
                    tf.reduce_sum(batch_masks), 1.0
                )
            gradients = tape.gradient(loss, target_model.trainable_variables)
            optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, target_model.trainable_variables)
                if gradient is not None
            )
            epoch_losses.append(float(loss.numpy()))
            epoch_logits.append(np.asarray(logits.numpy()))
            epoch_labels.append(np.asarray(batch_labels.numpy()))
            epoch_masks.append(np.asarray(batch_masks.numpy()))

        final_losses.append(float(np.mean(epoch_losses)))
        final_accuracies.append(
            masked_policy_accuracy(
                np.concatenate(epoch_logits, axis=0),
                np.concatenate(epoch_labels, axis=0),
                np.concatenate(epoch_masks, axis=0),
            )
        )

    return {
        "epochs": float(epochs),
        "final_loss": final_losses[-1],
        "best_loss": float(np.min(final_losses)),
        "final_accuracy": final_accuracies[-1],
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_target_policy(target_model: Any, dataset: dict[str, np.ndarray]) -> dict[str, float]:
    if len(dataset["states"]) == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    import tensorflow as tf

    states = tf.convert_to_tensor(dataset["states"], dtype=tf.float32)
    labels = tf.convert_to_tensor(dataset["labels"], dtype=tf.int64)
    masks = tf.convert_to_tensor(dataset["masks"], dtype=tf.float32)
    logits = target_model(states, training=False)
    per_step_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_sum(per_step_loss * masks) / tf.maximum(tf.reduce_sum(masks), 1.0)
    return {
        "loss": float(loss.numpy()),
        "accuracy": masked_policy_accuracy(logits.numpy(), dataset["labels"], dataset["masks"]),
    }


def run_not_training(args: argparse.Namespace) -> dict[str, Any]:
    from tensorflow import keras

    from core.BattleStateTracker import BattleStateTracker
    from core.ModelRegistry import write_model_registry
    from core.TrainingSplit import group_split_by_battle_id, ingest_battles_to_examples
    from train_policy import discover_json_paths, resolve_data_paths

    guide_path = Path(args.guide_model_path).resolve()
    vocab_path = Path(args.guide_vocab_path).resolve()
    repo_path = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    guide_model = keras.models.load_model(guide_path, compile=False)
    guide_input_dim = int(guide_model.input_shape[-1])
    guide_num_classes = int(guide_model.output_shape[-1])
    if guide_input_dim != MODEL1_INPUT_DIM or guide_num_classes != MODEL1_ACTION_CLASSES:
        raise SystemExit(
            f"guide contract mismatch: expected {MODEL1_INPUT_DIM}->{MODEL1_ACTION_CLASSES}, "
            f"got {guide_input_dim}->{guide_num_classes}"
        )
    action_vocab = _load_json(vocab_path)
    if len(action_vocab) != guide_num_classes:
        raise SystemExit(
            f"guide vocabulary has {len(action_vocab)} classes but guide output has {guide_num_classes}"
        )

    data_paths = resolve_data_paths(list(args.data_paths))
    json_paths = discover_json_paths(data_paths)
    if not json_paths:
        raise SystemExit("No JSON battle logs were found in the provided data paths.")
    tracker = BattleStateTracker(form_change_species={"Palafin"})
    examples = ingest_battles_to_examples(
        tracker,
        json_paths,
        max_battles=args.max_battles,
        verbose_every=args.verbose_every,
        include_switches=True,
    )
    if not examples:
        raise SystemExit("No training examples were produced from the provided battle logs.")
    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    train_examples = [examples[int(index)] for index in train_idx]
    val_examples = [examples[int(index)] for index in val_idx]
    train_dataset = build_sequence_dataset(
        train_examples,
        action_vocab,
        sequence_length=args.sequence_length,
        feature_dim=guide_input_dim,
    )
    val_dataset = build_sequence_dataset(
        val_examples,
        action_vocab,
        sequence_length=args.sequence_length,
        feature_dim=guide_input_dim,
    )
    if len(train_dataset["states"]) == 0:
        raise SystemExit("The training split did not produce any sequence windows.")

    target_model = initialize_target_policy_model(
        guide_model,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        name=args.model_name,
    )
    guide_activation_model = build_guide_activation_model(
        guide_model,
        sequence_length=args.sequence_length,
        depth=args.depth,
    )

    stage_metrics: list[dict[str, Any]] = []
    for stage in progressive_replacement_schedule(args.depth):
        replaced_layer_count = len(stage)
        hybrid_model = build_hybrid_stage_model(
            guide_model,
            target_model,
            sequence_length=args.sequence_length,
            replaced_layer_count=replaced_layer_count,
            depth=args.depth,
        )
        metrics = train_alignment_stage(
            hybrid_model,
            guide_activation_model,
            train_dataset["states"],
            train_dataset["masks"],
            replaced_layer_count=replaced_layer_count,
            epochs=args.alignment_epochs,
            batch_size=args.batch_size,
            learning_rate=args.alignment_learning_rate,
            seed=args.seed,
        )
        metrics["stage"] = float(replaced_layer_count)
        stage_metrics.append(metrics)
        print(
            f"alignment_stage={replaced_layer_count} "
            f"loss={metrics['final_loss']:.6f} best={metrics['best_loss']:.6f}"
        )

    fine_tune_metrics = train_target_policy(
        target_model,
        train_dataset["states"],
        train_dataset["masks"],
        train_dataset["labels"],
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.fine_tune_learning_rate,
        seed=args.seed,
    )
    validation_metrics = _evaluate_target_policy(target_model, val_dataset)

    target_path = output_dir / f"{args.model_name}.keras"
    training_path = output_dir / f"training_{args.model_name}.keras"
    target_vocab_path = output_dir / f"action_vocab_{args.model_name}.json"
    metadata_path = output_dir / f"training_metadata_{args.model_name}.json"
    metrics_path = output_dir / f"not_alignment_metrics_{args.model_name}.json"

    target_model.save(training_path)
    serving_model = build_serving_model(
        target_model,
        input_dim=guide_input_dim,
        sequence_length=args.sequence_length,
        num_classes=guide_num_classes,
        name=args.model_name,
    )
    serving_model.save(target_path)
    target_vocab_path.write_text(json.dumps(action_vocab, indent=2), encoding="utf-8")
    metrics_payload = {
        "guide_model_id": MODEL1_GUIDE_ID,
        "target_model_id": args.model_name,
        "alignment_stages": stage_metrics,
        "fine_tune": fine_tune_metrics,
        "validation": validation_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    metadata = build_not_metadata(
        guide_model_path=str(guide_path.relative_to(repo_path) if guide_path.is_relative_to(repo_path) else guide_path),
        target_model_path=str(target_path.relative_to(repo_path) if target_path.is_relative_to(repo_path) else target_path),
        policy_vocab_path=str(target_vocab_path.relative_to(repo_path) if target_vocab_path.is_relative_to(repo_path) else target_vocab_path),
        guide_vocab_path=str(vocab_path.relative_to(repo_path) if vocab_path.is_relative_to(repo_path) else vocab_path),
        sequence_length=args.sequence_length,
        base_feature_dim=guide_input_dim,
        num_action_classes=guide_num_classes,
        train_examples=len(train_dataset["states"]),
        val_examples=len(val_dataset["states"]),
        stage_metrics=stage_metrics,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        target_training_model_path=str(training_path.relative_to(repo_path) if training_path.is_relative_to(repo_path) else training_path),
    )
    metadata["model_name"] = args.model_name
    metadata["alignment_metrics_path"] = str(metrics_path.relative_to(repo_path) if metrics_path.is_relative_to(repo_path) else metrics_path)
    metadata["fine_tune_metrics"] = fine_tune_metrics
    metadata["validation_metrics"] = validation_metrics
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if output_dir == (repo_path / "artifacts").resolve():
        registry_path = write_model_registry(repo_path)
        print(f"registry_path={registry_path}")
    print(f"target_model_path={target_path}")
    print(f"metadata_path={metadata_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Model1 to an Elman policy with Network of Theseus alignment.")
    parser.add_argument("data_paths", nargs="*", help="Battle JSON files or directories containing battle logs.")
    parser.add_argument("--guide-model-path", default="artifacts/model_1.keras")
    parser.add_argument("--guide-vocab-path", default="artifacts/action_vocab_1.json")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-name", default=TARGET_MODEL_ID)
    parser.add_argument("--max-battles", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-every", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_ELMAN_HIDDEN_DIM)
    parser.add_argument("--depth", type=int, default=DEFAULT_ELMAN_DEPTH)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alignment-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alignment-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run_not_training(parse_args())
