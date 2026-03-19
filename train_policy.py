from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from BattleStateTracker import BattleStateTracker
from StateVectorization import (
    build_action_vocab,
    build_move_vocab,
    state_vector_layout,
    vectorize_action_dataset,
    vectorize_dataset,
)
from TrainingSplit import group_split_by_battle_id, ingest_battles_to_examples


def discover_json_paths(inputs: List[str]) -> List[str]:
    json_paths: List[str] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            json_paths.extend(str(p) for p in sorted(path.rglob("*.json")))
        elif path.suffix.lower() == ".json" and path.exists():
            json_paths.append(str(path))
    return json_paths


def build_policy_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    learning_rate: float,
):
    from tensorflow import keras
    from tensorflow.keras import layers

    model_layers = [layers.Input(shape=(input_dim,))]
    for _ in range(depth):
        model_layers.append(layers.Dense(hidden_dim, activation="relu"))
        if dropout > 0:
            model_layers.append(layers.Dropout(dropout))
    model_layers.append(layers.Dense(num_classes))

    model = keras.Sequential(model_layers)
    metrics = [keras.metrics.SparseCategoricalAccuracy(name="top1")]
    if num_classes >= 3:
        metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )
    return model


def subset_examples(examples: List[dict], indices: np.ndarray) -> List[dict]:
    return [examples[int(i)] for i in indices]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_artifact_paths(output_dir: Path, include_switches: bool) -> tuple[Path, Path, Path]:
    vocab_stem = "action_vocab" if include_switches else "move_vocab"
    metadata_stem = "training_metadata"
    idx = 0

    while True:
        run_suffix = "" if idx == 0 else f"_{idx}"
        model_path = output_dir / f"model{run_suffix}.keras"
        vocab_path = output_dir / f"{vocab_stem}{run_suffix}.json"
        metadata_path = output_dir / f"{metadata_stem}{run_suffix}.json"

        if not model_path.exists() and not vocab_path.exists() and not metadata_path.exists():
            return model_path, vocab_path, metadata_path
        idx += 1


def make_training_metadata(
    args: argparse.Namespace,
    feature_dim: int,
    action_vocab: dict,
    train_size: int,
    val_size: int,
    history,
    model_path: Path,
    vocab_path: Path,
) -> dict:
    history_dict = history.history if history is not None else {}
    return {
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "feature_dim": feature_dim,
        "feature_layout": state_vector_layout(),
        "num_action_classes": len(action_vocab),
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Pokemon Showdown action policy model.")
    parser.add_argument(
        "data_paths",
        nargs="+",
        help="Battle JSON files or directories containing battle JSON logs.",
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
        "--include-switches",
        action="store_true",
        help="Train a joint action model with both move and switch classes.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = discover_json_paths(args.data_paths)
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

    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    train_examples = subset_examples(examples, train_idx)
    vocab_source = train_examples if train_examples else examples
    if args.include_switches:
        action_vocab = build_action_vocab(vocab_source, min_count=args.min_move_count, include_switches=True)
        X, y = vectorize_action_dataset(examples, action_vocab, include_switches=True)
    else:
        action_vocab = build_move_vocab(vocab_source, min_count=args.min_move_count)
        X, y = vectorize_dataset(examples, action_vocab)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx] if len(val_idx) else None
    y_val = y[val_idx] if len(val_idx) else None

    switch_examples = sum(1 for ex in examples if ex["action"][0] == "switch")
    print(
        f"examples={len(examples)} train={len(train_idx)} val={len(val_idx)}"
        f" switches={switch_examples}"
    )
    print(
        f"feature_dim={X.shape[1]} num_classes={len(action_vocab)}"
        f" action_space={'joint' if args.include_switches else 'move_only'}"
    )
    print("feature_layout:")
    for block in state_vector_layout():
        print(f"  - {block['name']}: {block['size']} :: {block['description']}")

    model = build_policy_model(
        input_dim=X.shape[1],
        num_classes=len(action_vocab),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    model.summary()

    from tensorflow import keras

    callbacks = []
    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "verbose": 1,
    }
    if X_val is not None and y_val is not None and len(val_idx):
        monitor_metric = "val_top3" if len(action_vocab) >= 3 else "val_top1"
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=args.patience,
                mode="max",
                restore_best_weights=True,
            )
        )
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        print("Validation split is empty; training without validation or early stopping.")

    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(**fit_kwargs)

    model_path, vocab_path, metadata_path = build_artifact_paths(output_dir, args.include_switches)

    model.save(model_path)
    save_json(vocab_path, action_vocab)
    save_json(
        metadata_path,
        make_training_metadata(
            args,
            feature_dim=int(X.shape[1]),
            action_vocab=action_vocab,
            train_size=int(len(train_idx)),
            val_size=int(len(val_idx)),
            history=history,
            model_path=model_path,
            vocab_path=vocab_path,
        ),
    )

    print(f"saved_model={model_path}")
    print(f"saved_vocab={vocab_path}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()
