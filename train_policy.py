from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from BattleStateTracker import BattleStateTracker
from StateVectorization import (
    build_move_vocab,
    state_vector_layout,
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


def make_training_metadata(
    args: argparse.Namespace,
    feature_dim: int,
    move_vocab: dict,
    train_size: int,
    val_size: int,
    history,
) -> dict:
    history_dict = history.history if history is not None else {}
    return {
        "feature_dim": feature_dim,
        "feature_layout": state_vector_layout(),
        "num_move_classes": len(move_vocab),
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Pokemon Showdown move policy model.")
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
    )
    if not examples:
        raise SystemExit("No training examples were produced from the provided battle logs.")

    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    train_examples = subset_examples(examples, train_idx)
    vocab_source = train_examples if train_examples else examples
    move_vocab = build_move_vocab(vocab_source, min_count=args.min_move_count)

    X, y = vectorize_dataset(examples, move_vocab)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx] if len(val_idx) else None
    y_val = y[val_idx] if len(val_idx) else None

    print(f"examples={len(examples)} train={len(train_idx)} val={len(val_idx)}")
    print(f"feature_dim={X.shape[1]} num_classes={len(move_vocab)}")
    print("feature_layout:")
    for block in state_vector_layout():
        print(f"  - {block['name']}: {block['size']} :: {block['description']}")

    model = build_policy_model(
        input_dim=X.shape[1],
        num_classes=len(move_vocab),
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
        monitor_metric = "val_top3" if len(move_vocab) >= 3 else "val_top1"
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

    model_path = output_dir / "model.keras"
    vocab_path = output_dir / "move_vocab.json"
    metadata_path = output_dir / "training_metadata.json"

    model.save(model_path)
    save_json(vocab_path, move_vocab)
    save_json(
        metadata_path,
        make_training_metadata(
            args,
            feature_dim=int(X.shape[1]),
            move_vocab=move_vocab,
            train_size=int(len(train_idx)),
            val_size=int(len(val_idx)),
            history=history,
        ),
    )

    print(f"saved_model={model_path}")
    print(f"saved_vocab={vocab_path}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()
