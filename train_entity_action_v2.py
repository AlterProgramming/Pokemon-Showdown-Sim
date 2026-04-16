from __future__ import annotations

"""Trainer for the first legal-action-conditioned entity policy family.

This v2 trainer intentionally keeps scope narrow:
    - offline battle logs only
    - candidate-conditioned policy over legal/fallback action lists
    - optional value head
    - simple artifact and metadata output for Colab bring-up
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from BattleStateTracker import BattleStateTracker
from EntityModelV2 import build_entity_action_v2_models
from EntityTensorization import ENTITY_INT_INPUT_KEYS
from EntityTensorizationV2 import (
    ENTITY_V2_INT_INPUT_KEYS,
    build_entity_v2_training_bundle,
    vectorize_entity_v2_policy_dataset,
)
from TrainingSplit import group_split_by_battle_id, ingest_battles_to_examples
from train_policy import (
    discover_json_paths,
    resolve_data_paths,
    save_json,
    subset_examples,
    summarize_training_history,
)


def _to_numpy_v2_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, values in raw_inputs.items():
        if key in ENTITY_INT_INPUT_KEYS or key in ENTITY_V2_INT_INPUT_KEYS:
            dtype = np.int64
        else:
            dtype = np.float32
        out[key] = np.asarray(values, dtype=dtype)
    return out


def _slice_entity_inputs(inputs: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: value[indices] for key, value in inputs.items()}


def build_entity_v2_artifact_paths(
    output_dir: Path,
    *,
    model_name: str,
    save_training_model: bool,
) -> Dict[str, Path]:
    suffix = model_name
    paths: Dict[str, Path] = {
        "policy_model": output_dir / f"{suffix}.keras",
        "entity_token_vocabs": output_dir / f"{suffix}.entity_token_vocabs.json",
        "metadata": output_dir / f"training_metadata_{suffix}.json",
        "training_history": output_dir / f"training_history_{suffix}.json",
        "evaluation_summary": output_dir / f"evaluation_summary_{suffix}.json",
    }
    if save_training_model:
        paths["training_model"] = output_dir / f"training_model_{suffix}.keras"

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
    model_name: str,
    train_size: int,
    val_size: int,
    history,
    artifact_paths: Dict[str, Path],
    token_vocabs: Dict[str, Dict[str, int]],
    max_candidates: int,
    raw_data_paths: List[str],
    resolved_data_paths: List[str],
    json_paths: List[str],
) -> dict:
    history_summary = summarize_training_history(history)
    return {
        "model_name": model_name,
        "model_release_id": model_name,
        "family_id": "entity_action_v2",
        "family_version": 1,
        "family_name": "entity_action_v2",
        "training_regime": "offline_entity_candidate_bc",
        "information_policy": "public_only_fallback_candidates",
        "state_schema_version": "entity_action_v2",
        "entity_schema_version": "entity_action_v2",
        "policy_model_path": str(artifact_paths["policy_model"]),
        "training_model_path": str(artifact_paths.get("training_model", artifact_paths["policy_model"])),
        "entity_token_vocab_path": str(artifact_paths["entity_token_vocabs"]),
        "metadata_path": str(artifact_paths["metadata"]),
        "training_history_path": str(artifact_paths["training_history"]),
        "evaluation_summary_path": str(artifact_paths["evaluation_summary"]),
        "objective_set": ["policy", "value"] if args.predict_value else ["policy"],
        "action_space": "legal_candidates",
        "action_parameterization": "candidate_index_scored_by_entity_encoder",
        "train_examples": int(train_size),
        "val_examples": int(val_size),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(history_summary["epochs_completed"]),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "token_embed_dim": int(args.token_embed_dim),
        "max_candidates": int(max_candidates),
        "predict_value": bool(args.predict_value),
        "value_weight": float(args.value_weight),
        "seed": int(args.seed),
        "raw_data_paths": list(raw_data_paths),
        "resolved_data_paths": list(resolved_data_paths),
        "json_file_count": int(len(json_paths)),
        "entity_token_vocab_sizes": {key: len(value) for key, value in token_vocabs.items()},
        "history_summary": history_summary,
        "notes": [
            "Offline replay training uses the candidate fallback path when full legal request objects are unavailable.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the entity_action_v2 candidate-conditioned policy model from battle logs."
    )
    parser.add_argument(
        "data_paths",
        nargs="+",
        help="One or more JSON file paths or directories containing replay battle JSON.",
    )
    parser.add_argument("--model-name", default="entity_action_v2", help="Artifact/model release name.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where model artifacts will be written.")
    parser.add_argument("--max-battles", type=int, default=5000, help="Maximum number of battles to ingest.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Battle-grouped validation split ratio.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Entity/state hidden width.")
    parser.add_argument("--depth", type=int, default=3, help="Number of trunk dense layers after entity pooling.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied after dense layers.")
    parser.add_argument("--token-embed-dim", type=int, default=24, help="Base embedding width for learned entity tokens.")
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum legal candidates encoded per turn.")
    parser.add_argument("--predict-value", action="store_true", help="Add the terminal win-probability head.")
    parser.add_argument("--value-weight", type=float, default=0.25, help="Loss weight for the value head.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation top-3.")
    parser.add_argument("--verbose-every", type=int, default=200, help="Print ingest progress every N battles. Set 0 to disable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "entity_v2_training_start:"
        f" model_name={args.model_name}"
        f" output_dir={args.output_dir}"
        f" predict_value={args.predict_value}",
        flush=True,
    )

    data_paths = resolve_data_paths(args.data_paths)
    json_paths = discover_json_paths(data_paths)
    if not json_paths:
        raise SystemExit("No JSON files found in the provided path(s).")

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

    if args.predict_value:
        examples = [ex for ex in examples if ex.get("terminal_result") is not None]
        if not examples:
            raise SystemExit("Value training requires battle outcomes, but no examples carried terminal_result.")

    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    train_examples = subset_examples(examples, train_idx)
    val_examples = subset_examples(examples, val_idx)
    vocab_source = train_examples if train_examples else examples

    bundle = build_entity_v2_training_bundle(vocab_source, max_candidates=args.max_candidates)
    token_vocabs = bundle["token_vocabs"]

    X_raw, targets_raw = vectorize_entity_v2_policy_dataset(
        examples,
        token_vocabs=token_vocabs,
        include_value=args.predict_value,
        max_candidates=args.max_candidates,
    )
    X_np = _to_numpy_v2_inputs(X_raw)

    y_policy_np = np.asarray(targets_raw["policy"], dtype=np.int64)
    y_value_np = (
        None
        if "value" not in targets_raw
        else np.asarray(targets_raw["value"], dtype=np.float32).reshape(-1, 1)
    )

    X_train = _slice_entity_inputs(X_np, train_idx)
    X_val = _slice_entity_inputs(X_np, val_idx) if len(val_idx) else None
    y_train_policy = y_policy_np[train_idx]
    y_val_policy = y_policy_np[val_idx] if len(val_idx) else None
    y_train_value = y_value_np[train_idx] if y_value_np is not None else None
    y_val_value = y_value_np[val_idx] if y_value_np is not None and len(val_idx) else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_entity_v2_artifact_paths(
        output_dir,
        model_name=args.model_name,
        save_training_model=args.predict_value,
    )

    try:
        model, policy_model, _ = build_entity_action_v2_models(
            vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            token_embed_dim=args.token_embed_dim,
            max_candidates=args.max_candidates,
            predict_value=args.predict_value,
            value_weight=args.value_weight,
        )
        from tensorflow import keras
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required to train entity_action_v2, but it is not installed in this environment."
        ) from exc

    callbacks: List[Any] = []
    if len(val_idx):
        monitor = "val_policy_top3" if args.predict_value else "val_top3"
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=args.patience,
                mode="max",
                restore_best_weights=True,
            )
        )

    train_targets: Dict[str, Any] | Any = {"policy": y_train_policy}
    if args.predict_value and y_train_value is not None:
        train_targets["value"] = y_train_value
    if list(train_targets.keys()) == ["policy"]:
        train_targets = y_train_policy

    val_data = None
    if len(val_idx):
        val_targets: Dict[str, Any] | Any = {"policy": y_val_policy}
        if args.predict_value and y_val_value is not None:
            val_targets["value"] = y_val_value
        if list(val_targets.keys()) == ["policy"]:
            val_targets = y_val_policy
        val_data = (X_val, val_targets)

    history = model.fit(
        X_train,
        train_targets,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    eval_summary = model.evaluate(X_val, val_targets, verbose=0, return_dict=True) if len(val_idx) else {}

    policy_model.save(artifact_paths["policy_model"])
    if args.predict_value:
        model.save(artifact_paths["training_model"])

    save_json(artifact_paths["entity_token_vocabs"], token_vocabs)
    save_json(artifact_paths["training_history"], history.history)
    save_json(artifact_paths["evaluation_summary"], eval_summary)

    metadata = make_training_metadata(
        args,
        model_name=args.model_name,
        train_size=len(train_idx),
        val_size=len(val_idx),
        history=history,
        artifact_paths=artifact_paths,
        token_vocabs=token_vocabs,
        max_candidates=args.max_candidates,
        raw_data_paths=[str(path) for path in args.data_paths],
        resolved_data_paths=[str(path) for path in data_paths],
        json_paths=[str(path) for path in json_paths],
    )
    save_json(artifact_paths["metadata"], metadata)

    print(
        "entity_v2_training_complete:"
        f" train_examples={len(train_idx)}"
        f" val_examples={len(val_idx)}"
        f" species_vocab={len(token_vocabs['species'])}"
        f" move_vocab={len(token_vocabs['move'])}"
        f" policy_model={artifact_paths['policy_model']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
