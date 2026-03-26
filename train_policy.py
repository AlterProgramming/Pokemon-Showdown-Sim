from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from BattleStateTracker import BattleStateTracker
from ModelRegistry import write_model_registry
from RewardSignals import (
    TRACE_SCHEMA_VERSION,
    RewardConfig,
    build_move_reward_profile,
)
from StateVectorization import (
    build_action_context_vocab,
    build_action_vocab,
    build_move_vocab,
    state_vector_layout,
    turn_outcome_layout,
    turn_outcome_dim,
    vectorize_multitask_dataset,
    vectorize_dataset,
)
from TrainingSplit import group_split_by_battle_id, ingest_battles_to_examples

MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "model_2_large": {
        "model_name": "model_2_large",
        "hidden_dim": 480,
        "depth": 4,
        "transition_hidden_dim": 480,
        "description": "Approximately 3x the parameter count of model_2.",
    },"model_4_large": {
        "model_name": "model_4_large",
        "hidden_dim": 480,
        "depth": 4,
        "transition_hidden_dim": 480,
        "description": "Approximately 3x the parameter count of model_4.",
    },
}
DEFAULT_KAGGLE_DATASET = "thephilliplin/pokemon-showdown-battles-gen9-randbats"


def discover_json_paths(inputs: List[str]) -> List[str]:
    json_paths: List[str] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            json_paths.extend(str(p) for p in sorted(path.rglob("*.json")))
        elif path.suffix.lower() == ".json" and path.exists():
            json_paths.append(str(path))
    return json_paths


def resolve_data_paths(raw_paths: List[str]) -> List[str]:
    if raw_paths:
        return list(raw_paths)

    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "No data paths were provided and kagglehub is not installed. "
            "Install kagglehub or pass one or more battle-log paths explicitly."
        ) from exc

    dataset_dir = kagglehub.dataset_download(DEFAULT_KAGGLE_DATASET)
    print(f"using_default_kaggle_dataset={DEFAULT_KAGGLE_DATASET}")
    print(f"downloaded_dataset_path={dataset_dir}")
    return [str(dataset_dir)]


def use_action_vocab(args: argparse.Namespace) -> bool:
    return bool(args.include_switches or args.predict_turn_outcome)


def requires_training_model(args: argparse.Namespace) -> bool:
    return bool(args.predict_turn_outcome or args.predict_value)


def make_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        hp_weight=args.reward_hp_weight,
        ko_weight=args.reward_ko_weight,
        wasted_move_penalty=args.reward_wasted_move_penalty,
        terminal_weight=args.reward_terminal_weight,
        offensive_move_min_uses=args.reward_offensive_move_min_uses,
        offensive_move_min_damage_rate=args.reward_offensive_move_min_damage_rate,
    )


def build_policy_models(
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    learning_rate: float,
    *,
    transition_dim: int | None = None,
    action_context_vocab_size: int | None = None,
    action_embed_dim: int = 32,
    transition_hidden_dim: int | None = None,
    transition_weight: float = 0.25,
    predict_value: bool = False,
    value_hidden_dim: int | None = None,
    value_weight: float = 0.25,
):
    from tensorflow import keras
    from tensorflow.keras import layers

    state_input = layers.Input(shape=(input_dim,), name="state")
    x = state_input
    for _ in range(depth):
        x = layers.Dense(hidden_dim, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    shared = x
    policy_logits = layers.Dense(num_classes, name="policy")(shared)
    policy_model = keras.Model(state_input, policy_logits, name="policy_model")

    policy_metrics = [keras.metrics.SparseCategoricalAccuracy(name="top1")]
    if num_classes >= 3:
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
        value_x = layers.Dense(value_hidden_dim or max(64, hidden_dim // 2), activation="relu")(shared)
        if dropout > 0:
            value_x = layers.Dropout(dropout)(value_x)
        value_out = layers.Dense(1, activation="sigmoid", name="value")(value_x)
        outputs["value"] = value_out
        losses["value"] = keras.losses.BinaryCrossentropy()
        loss_weights["value"] = value_weight
        metrics["value"] = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="brier"),
        ]
        policy_value_model = keras.Model(
            state_input,
            {"policy": policy_logits, "value": value_out},
            name="policy_value_model",
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
        transition_x = layers.Dense(transition_hidden_dim or hidden_dim, activation="relu")(transition_x)
        if dropout > 0:
            transition_x = layers.Dropout(dropout)(transition_x)
        transition_out = layers.Dense(transition_dim, name="transition")(transition_x)
        outputs["transition"] = transition_out
        losses["transition"] = keras.losses.MeanSquaredError()
        loss_weights["transition"] = transition_weight
        metrics["transition"] = [keras.metrics.MeanAbsoluteError(name="mae")]

    model_inputs: Any
    if use_transition:
        model_inputs = {
            "state": state_input,
            "my_action": my_action_input,
            "opp_action": opp_action_input,
        }
    else:
        model_inputs = state_input

    training_model = keras.Model(model_inputs, outputs, name="policy_multitask_model")
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return training_model, policy_model, policy_value_model


def subset_examples(examples: List[dict], indices: np.ndarray) -> List[dict]:
    return [examples[int(i)] for i in indices]


def to_numpy_inputs(raw_inputs: Any) -> Any:
    if isinstance(raw_inputs, dict):
        out: Dict[str, np.ndarray] = {}
        for key, values in raw_inputs.items():
            dtype = np.float32 if key == "state" else np.int64
            out[key] = np.asarray(values, dtype=dtype)
        return out
    return np.asarray(raw_inputs, dtype=np.float32)


def slice_inputs(inputs: Any, indices: np.ndarray) -> Any:
    if isinstance(inputs, dict):
        return {key: values[indices] for key, values in inputs.items()}
    return inputs[indices]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_artifact_paths(
    output_dir: Path,
    *,
    policy_vocab_stem: str,
    save_training_model: bool,
    save_action_context_vocab: bool,
    save_policy_value_model: bool,
) -> Dict[str, Path]:
    idx = 0
    while True:
        run_suffix = "" if idx == 0 else f"_{idx}"
        paths: Dict[str, Path] = {
            "policy_model": output_dir / f"model{run_suffix}.keras",
            "policy_vocab": output_dir / f"{policy_vocab_stem}{run_suffix}.json",
            "metadata": output_dir / f"training_metadata{run_suffix}.json",
            "reward_profile": output_dir / f"move_reward_profile{run_suffix}.json",
        }
        if save_training_model:
            paths["training_model"] = output_dir / f"training_model{run_suffix}.keras"
        if save_action_context_vocab:
            paths["action_context_vocab"] = output_dir / f"action_context_vocab{run_suffix}.json"
        if save_policy_value_model:
            paths["policy_value_model"] = output_dir / f"policy_value_model{run_suffix}.keras"

        if all(not path.exists() for path in paths.values()):
            return paths
        idx += 1


def build_named_artifact_paths(
    output_dir: Path,
    *,
    model_name: str,
    policy_vocab_stem: str,
    save_training_model: bool,
    save_action_context_vocab: bool,
    save_policy_value_model: bool,
) -> Dict[str, Path]:
    if not model_name:
        raise ValueError("model_name must be non-empty")

    suffix = model_name.removeprefix("model")
    if not suffix:
        raise ValueError(
            "model_name must include a suffix after 'model' so related artifact names can be derived."
        )

    paths: Dict[str, Path] = {
        "policy_model": output_dir / f"{model_name}.keras",
        "policy_vocab": output_dir / f"{policy_vocab_stem}{suffix}.json",
        "metadata": output_dir / f"training_metadata{suffix}.json",
        "reward_profile": output_dir / f"move_reward_profile{suffix}.json",
    }
    if save_training_model:
        paths["training_model"] = output_dir / f"training_model{suffix}.keras"
    if save_action_context_vocab:
        paths["action_context_vocab"] = output_dir / f"action_context_vocab{suffix}.json"
    if save_policy_value_model:
        paths["policy_value_model"] = output_dir / f"policy_value_model{suffix}.keras"

    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing artifacts for "
            f"{model_name}: {', '.join(existing)}"
        )
    return paths


def make_training_metadata(
    args: argparse.Namespace,
    *,
    feature_dim: int,
    action_vocab: dict,
    train_size: int,
    val_size: int,
    history,
    artifact_paths: Dict[str, Path],
    action_context_vocab: dict | None = None,
    policy_param_count: int | None = None,
    training_param_count: int | None = None,
    reward_config: RewardConfig,
    move_reward_profile: dict,
) -> dict:
    history_dict = history.history if history is not None else {}
    payload = {
        "policy_model_path": str(artifact_paths["policy_model"]),
        "policy_vocab_path": str(artifact_paths["policy_vocab"]),
        "metadata_path": str(artifact_paths["metadata"]),
        "reward_profile_path": str(artifact_paths["reward_profile"]),
        "feature_dim": feature_dim,
        "feature_layout": state_vector_layout(),
        "num_action_classes": len(action_vocab),
        "policy_label_format": "action_tokens" if use_action_vocab(args) else "move_ids",
        "action_space": "joint" if args.include_switches else "move_only",
        "train_examples": train_size,
        "val_examples": val_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_dict.get("loss", [])),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
        "min_move_count": args.min_move_count,
        "max_battles": args.max_battles,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "include_switches": args.include_switches,
        "predict_turn_outcome": args.predict_turn_outcome,
        "predict_value": args.predict_value,
        "transition_weight": args.transition_weight,
        "value_weight": args.value_weight,
        "value_hidden_dim": args.value_hidden_dim or max(64, args.hidden_dim // 2),
        "action_embed_dim": args.action_embed_dim,
        "transition_hidden_dim": args.transition_hidden_dim,
        "model_variant": args.model_variant,
        "model_name": args.model_name,
        "reward_config": reward_config.to_dict(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "num_reward_profiled_moves": len(move_reward_profile),
        "num_offensive_reward_moves": sum(
            1 for entry in move_reward_profile.values() if entry.get("is_offensive")
        ),
    }

    if policy_param_count is not None:
        payload["policy_param_count"] = int(policy_param_count)
    if training_param_count is not None:
        payload["training_param_count"] = int(training_param_count)

    if requires_training_model(args):
        payload["training_model_path"] = str(artifact_paths["training_model"])
    if args.predict_turn_outcome:
        payload["turn_outcome_dim"] = turn_outcome_dim()
        payload["turn_outcome_layout"] = turn_outcome_layout()
        payload["action_context_vocab_path"] = str(artifact_paths["action_context_vocab"])
        payload["num_action_context_classes"] = len(action_context_vocab or {})
    if args.predict_value:
        payload["policy_value_model_path"] = str(artifact_paths["policy_value_model"])

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Pokemon Showdown action policy model.")
    parser.add_argument(
        "data_paths",
        nargs="*",
        help=(
            "Battle JSON files or directories containing battle JSON logs. "
            f"When omitted, defaults to Kaggle dataset '{DEFAULT_KAGGLE_DATASET}'."
        ),
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model and metadata artifacts.")
    parser.add_argument("--max-battles", type=int, default=5000, help="Maximum number of battles to load.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of battles reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for battle-level split.")
    parser.add_argument("--min-move-count", type=int, default=1, help="Minimum move frequency to keep in vocab.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden width for each dense layer.")
    parser.add_argument("--depth", type=int, default=3, help="Number of hidden dense layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout between dense layers.")
    parser.add_argument(
        "--model-variant",
        choices=sorted(MODEL_VARIANTS.keys()),
        default="default",
        help="Named architecture/artifact preset. Use model_2_large for the larger joint-policy model.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Explicit policy artifact base name (for example model_2_large). "
            "When set, related vocab/metadata names are derived from it."
        ),
    )
    parser.add_argument(
        "--include-switches",
        action="store_true",
        help="Train a joint action model with both move and switch classes.",
    )
    parser.add_argument(
        "--predict-turn-outcome",
        action="store_true",
        help="Add an auxiliary action-conditioned head that predicts structured end-of-turn public state.",
    )
    parser.add_argument(
        "--predict-value",
        action="store_true",
        help="Add a state-value head that predicts the acting player's terminal win probability.",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=0.25,
        help="Loss weight for the auxiliary turn-outcome head.",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=0.25,
        help="Loss weight for the auxiliary value head.",
    )
    parser.add_argument(
        "--value-hidden-dim",
        type=int,
        default=0,
        help="Hidden width for the value head MLP block. Defaults to max(64, hidden_dim // 2).",
    )
    parser.add_argument(
        "--action-embed-dim",
        type=int,
        default=32,
        help="Embedding width for the action-conditioned transition head.",
    )
    parser.add_argument(
        "--transition-hidden-dim",
        type=int,
        default=256,
        help="Hidden width for the transition head MLP block.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience on validation top-3 when validation exists.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=200,
        help="Print ingest progress every N battles. Set 0 to disable.",
    )
    parser.add_argument(
        "--reward-hp-weight",
        type=float,
        default=0.25,
        help="Weight for the HP swing component in replay reward traces.",
    )
    parser.add_argument(
        "--reward-ko-weight",
        type=float,
        default=0.50,
        help="Weight for the KO swing component in replay reward traces.",
    )
    parser.add_argument(
        "--reward-wasted-move-penalty",
        type=float,
        default=0.10,
        help="Penalty weight for offensive moves that fail to cause any tracked opponent HP loss.",
    )
    parser.add_argument(
        "--reward-terminal-weight",
        type=float,
        default=1.0,
        help="Weight for the terminal reward component in replay traces.",
    )
    parser.add_argument(
        "--reward-offensive-move-min-uses",
        type=int,
        default=20,
        help="Minimum train-split uses before a move can be tagged as offensive for replay rewards.",
    )
    parser.add_argument(
        "--reward-offensive-move-min-damage-rate",
        type=float,
        default=0.5,
        help="Minimum train-split damage-turn rate before a move is tagged as offensive for replay rewards.",
    )
    return parser.parse_args()


def apply_model_variant(args: argparse.Namespace) -> None:
    variant = MODEL_VARIANTS.get(args.model_variant, {})
    if not variant:
        return

    if args.hidden_dim == 256 and "hidden_dim" in variant:
        args.hidden_dim = int(variant["hidden_dim"])
    if args.depth == 3 and "depth" in variant:
        args.depth = int(variant["depth"])
    if args.transition_hidden_dim == 256 and "transition_hidden_dim" in variant:
        args.transition_hidden_dim = int(variant["transition_hidden_dim"])
    if args.model_name is None and variant.get("model_name"):
        args.model_name = str(variant["model_name"])


def main() -> None:
    args = parse_args()
    apply_model_variant(args)

    data_paths = resolve_data_paths(args.data_paths)
    json_paths = discover_json_paths(data_paths)
    
    if not json_paths:
        raise SystemExit("No JSON files found in the provided path(s).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = BattleStateTracker(form_change_species={"Palafin"})
    examples = ingest_battles_to_examples(
        tracker,
        json_paths,
        max_battles=args.max_battles,
        verbose_every=args.verbose_every,
        include_switches=args.include_switches,
    )
    if not examples:
        raise SystemExit("No training examples were produced from the provided battle logs.")
    if args.predict_value:
        with_terminal = [ex for ex in examples if ex.get("terminal_result") is not None]
        dropped_missing_terminal = len(examples) - len(with_terminal)
        if not with_terminal:
            raise SystemExit("Value training requires battle outcomes, but no examples carried terminal_result.")
        if dropped_missing_terminal:
            print(f"dropped_examples_missing_terminal_result={dropped_missing_terminal}")
        examples = with_terminal

    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    train_examples = subset_examples(examples, train_idx)
    vocab_source = train_examples if train_examples else examples
    reward_config = make_reward_config(args)
    move_reward_profile = build_move_reward_profile(vocab_source, reward_config)

    action_context_vocab = None
    if use_action_vocab(args):
        action_vocab = build_action_vocab(
            vocab_source,
            min_count=args.min_move_count,
            include_switches=args.include_switches,
        )
        if args.predict_turn_outcome:
            action_context_vocab = build_action_context_vocab(vocab_source)
        X_raw, targets_raw = vectorize_multitask_dataset(
            examples,
            action_vocab,
            include_switches=args.include_switches,
            use_action_tokens=True,
            action_context_vocab=action_context_vocab,
            include_transition=args.predict_turn_outcome,
            include_value=args.predict_value,
        )
    else:
        action_vocab = build_move_vocab(vocab_source, min_count=args.min_move_count)
        X_raw, targets_raw = vectorize_multitask_dataset(
            examples,
            action_vocab,
            include_switches=False,
            use_action_tokens=False,
            include_transition=False,
            include_value=args.predict_value,
        )

    X = to_numpy_inputs(X_raw)
    y_policy = np.asarray(targets_raw["policy"], dtype=np.int64)
    y_transition_np = (
        None
        if "transition" not in targets_raw
        else np.asarray(targets_raw["transition"], dtype=np.float32)
    )
    y_value_np = (
        None
        if "value" not in targets_raw
        else np.asarray(targets_raw["value"], dtype=np.float32).reshape(-1, 1)
    )

    X_train = slice_inputs(X, train_idx)
    y_train_policy = y_policy[train_idx]
    X_val = slice_inputs(X, val_idx) if len(val_idx) else None
    y_val_policy = y_policy[val_idx] if len(val_idx) else None

    if y_transition_np is not None:
        y_train_transition = y_transition_np[train_idx]
        y_val_transition = y_transition_np[val_idx] if len(val_idx) else None
    else:
        y_train_transition = None
        y_val_transition = None

    if y_value_np is not None:
        y_train_value = y_value_np[train_idx]
        y_val_value = y_value_np[val_idx] if len(val_idx) else None
    else:
        y_train_value = None
        y_val_value = None

    switch_examples = sum(1 for ex in examples if ex["action"][0] == "switch")
    state_input_dim = X["state"].shape[1] if isinstance(X, dict) else X.shape[1]
    print(
        f"examples={len(examples)} train={len(train_idx)} val={len(val_idx)}"
        f" switches={switch_examples}"
    )
    print(
        f"feature_dim={state_input_dim} num_classes={len(action_vocab)}"
        f" action_space={'joint' if args.include_switches else 'move_only'}"
    )
    print("feature_layout:")
    for block in state_vector_layout():
        print(f"  - {block['name']}: {block['size']} :: {block['description']}")
    if args.predict_turn_outcome:
        print(f"turn_outcome_dim={turn_outcome_dim()} action_context_classes={len(action_context_vocab or {})}")
        print("turn_outcome_layout:")
        for block in turn_outcome_layout():
            print(f"  - {block['name']}: {block['size']} :: {block['description']}")
    print(
        "reward_profile:"
        f" moves={len(move_reward_profile)}"
        f" offensive={sum(1 for entry in move_reward_profile.values() if entry.get('is_offensive'))}"
    )

    model, policy_model, policy_value_model = build_policy_models(
        input_dim=state_input_dim,
        num_classes=len(action_vocab),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        transition_dim=turn_outcome_dim() if args.predict_turn_outcome else None,
        action_context_vocab_size=len(action_context_vocab or {}) if args.predict_turn_outcome else None,
        action_embed_dim=args.action_embed_dim,
        transition_hidden_dim=args.transition_hidden_dim,
        transition_weight=args.transition_weight,
        predict_value=args.predict_value,
        value_hidden_dim=args.value_hidden_dim or None,
        value_weight=args.value_weight,
    )
    policy_param_count = int(policy_model.count_params())
    training_param_count = int(model.count_params())
    model.summary()
    print(
        f"model_variant={args.model_variant} model_name={args.model_name or '(auto)'} "
        f"policy_params={policy_param_count} training_params={training_param_count}"
    )

    from tensorflow import keras

    callbacks = []
    fit_kwargs: Dict[str, Any] = {
        "x": X_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "verbose": 1,
    }
    multitask_targets = requires_training_model(args)
    if multitask_targets:
        train_targets: Dict[str, Any] = {"policy": y_train_policy}
        if args.predict_turn_outcome:
            train_targets["transition"] = y_train_transition
        if args.predict_value:
            train_targets["value"] = y_train_value
        fit_kwargs["y"] = train_targets
    else:
        fit_kwargs["y"] = y_train_policy

    if X_val is not None and y_val_policy is not None and len(val_idx):
        if multitask_targets:
            monitor_metric = "val_policy_top3" if len(action_vocab) >= 3 else "val_policy_top1"
            val_targets: Dict[str, Any] = {"policy": y_val_policy}
            if args.predict_turn_outcome:
                val_targets["transition"] = y_val_transition
            if args.predict_value:
                val_targets["value"] = y_val_value
            fit_kwargs["validation_data"] = (
                X_val,
                val_targets,
            )
        else:
            monitor_metric = "val_top3" if len(action_vocab) >= 3 else "val_top1"
            fit_kwargs["validation_data"] = (X_val, y_val_policy)

        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=args.patience,
                mode="max",
                restore_best_weights=True,
            )
        )
    else:
        print("Validation split is empty; training without validation or early stopping.")

    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(**fit_kwargs)

    artifact_paths = (
        build_named_artifact_paths(
            output_dir,
            model_name=args.model_name,
            policy_vocab_stem="action_vocab" if use_action_vocab(args) else "move_vocab",
            save_training_model=requires_training_model(args),
            save_action_context_vocab=args.predict_turn_outcome,
            save_policy_value_model=args.predict_value,
        )
        if args.model_name else
        build_artifact_paths(
            output_dir,
            policy_vocab_stem="action_vocab" if use_action_vocab(args) else "move_vocab",
            save_training_model=requires_training_model(args),
            save_action_context_vocab=args.predict_turn_outcome,
            save_policy_value_model=args.predict_value,
        )
    )

    policy_model.save(artifact_paths["policy_model"])
    save_json(artifact_paths["policy_vocab"], action_vocab)
    save_json(artifact_paths["reward_profile"], move_reward_profile)

    if args.predict_value and policy_value_model is not None:
        policy_value_model.save(artifact_paths["policy_value_model"])
    if requires_training_model(args):
        model.save(artifact_paths["training_model"])
    if args.predict_turn_outcome:
        save_json(artifact_paths["action_context_vocab"], action_context_vocab or {})

    save_json(
        artifact_paths["metadata"],
        make_training_metadata(
            args,
            feature_dim=int(state_input_dim),
            action_vocab=action_vocab,
            train_size=int(len(train_idx)),
            val_size=int(len(val_idx)),
            history=history,
            artifact_paths=artifact_paths,
            action_context_vocab=action_context_vocab,
            policy_param_count=policy_param_count,
            training_param_count=training_param_count,
            reward_config=reward_config,
            move_reward_profile=move_reward_profile,
        ),
    )
    registry_path = write_model_registry(Path(__file__).resolve().parent)

    print(f"saved_policy_model={artifact_paths['policy_model']}")
    print(f"saved_policy_vocab={artifact_paths['policy_vocab']}")
    print(f"saved_reward_profile={artifact_paths['reward_profile']}")
    print(f"saved_metadata={artifact_paths['metadata']}")
    print(f"saved_model_registry={registry_path}")
    if args.predict_value:
        print(f"saved_policy_value_model={artifact_paths['policy_value_model']}")
    if requires_training_model(args):
        print(f"saved_training_model={artifact_paths['training_model']}")
    if args.predict_turn_outcome:
        print(f"saved_action_context_vocab={artifact_paths['action_context_vocab']}")


if __name__ == "__main__":
    main()
