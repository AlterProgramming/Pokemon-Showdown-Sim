from __future__ import annotations

"""Trainer for the first entity-centric offline behavior-cloning family.

This script is the entity analogue of train_policy.py, but it intentionally keeps
the first generation narrow:
    - offline logs only
    - learned entity embeddings
    - optional transition/value auxiliary heads
    - strong metadata / manifest / artifact bookkeeping

The main job here is not just fitting a model. It is defining a reproducible
family release that can later be compared against both older vector baselines and
future belief-aware entity generations.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

if __name__ == "__main__":
    # Emit a tiny bootstrap marker before the TensorFlow-heavy imports below.
    # That makes live log tailing more responsive while the process is still warming up.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    print("entity_training_bootstrap: starting import phase", flush=True)

import numpy as np

from BattleStateTracker import BattleStateTracker
from EntityModelV1 import build_entity_action_models
from EntityTensorization import (
    to_numpy_entity_inputs,
    build_entity_training_bundle,
    entity_tensor_layout,
    vectorize_entity_multitask_dataset,
)
from ModelRegistry import metadata_file_candidates, resolve_artifact_path, write_model_registry
from RewardSignals import TRACE_SCHEMA_VERSION, attach_reward_targets, build_move_reward_profile
from StateVectorization import turn_outcome_dim, turn_outcome_layout
from TransferLearning import apply_transfer_metadata, describe_transfer
from TrainingSplit import group_split_by_battle_id, ingest_battles_to_examples
from train_policy import (
    DEFAULT_KAGGLE_DATASET,
    build_data_source_metadata,
    build_policy_training_sample_weights,
    build_run_manifest,
    discover_json_paths,
    make_reward_config,
    resolve_data_paths,
    save_json,
    subset_examples,
    summarize_training_history,
)


def slice_entity_inputs(inputs: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice every named input tensor by a shared set of example indices."""
    return {key: value[indices] for key, value in inputs.items()}


def build_entity_artifact_paths(
    output_dir: Path,
    *,
    model_name: str,
    save_training_model: bool,
    save_action_context_vocab: bool,
    save_policy_value_model: bool,
    save_sequence_vocab: bool = False,
) -> Dict[str, Path]:
    """Reserve the artifact paths for one release and fail fast on collisions.

    We never silently overwrite a named release. That keeps model generations
    inspectable and reproducible.
    """
    suffix = model_name
    paths: Dict[str, Path] = {
        "policy_model": output_dir / f"{suffix}.keras",
        "policy_vocab": output_dir / f"{suffix}.policy_vocab.json",
        "entity_token_vocabs": output_dir / f"{suffix}.entity_token_vocabs.json",
        "metadata": output_dir / f"training_metadata_{suffix}.json",
        "reward_profile": output_dir / f"{suffix}.move_reward_profile.json",
        "training_history": output_dir / f"training_history_{suffix}.json",
        "epoch_metrics": output_dir / f"training_metrics_{suffix}.csv",
        "evaluation_summary": output_dir / f"evaluation_summary_{suffix}.json",
        "run_manifest": output_dir / f"run_manifest_{suffix}.json",
    }
    if save_training_model:
        paths["training_model"] = output_dir / f"training_model_{suffix}.keras"
    if save_action_context_vocab:
        paths["action_context_vocab"] = output_dir / f"action_context_vocab_{suffix}.json"
    if save_policy_value_model:
        paths["policy_value_model"] = output_dir / f"policy_value_model_{suffix}.keras"
    if save_sequence_vocab:
        paths["sequence_vocab"] = output_dir / f"sequence_vocab_{suffix}.json"

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
    policy_vocab: Dict[str, int],
    action_context_vocab: Dict[str, int] | None,
    token_vocabs: Dict[str, Dict[str, int]],
    sequence_vocab: Dict[str, int] | None = None,
    reward_config,
    move_reward_profile: Dict[str, Any],
    policy_weight_stats: Dict[str, Any] | None,
    raw_data_paths: List[str],
    resolved_data_paths: List[str],
    json_paths: List[str],
) -> dict:
    """Assemble the persisted recipe metadata for this entity-family run."""
    history_dict = history.history if history is not None else {}
    data_source_id, data_source_details = build_data_source_metadata(
        raw_data_paths=raw_data_paths,
        resolved_data_paths=resolved_data_paths,
        json_paths=json_paths,
    )
    objective_set = ["policy"]
    if args.predict_turn_outcome:
        objective_set.append("transition")
    if args.predict_value:
        objective_set.append("value")
    if getattr(args, "predict_turn_sequence", False):
        objective_set.append("sequence_auxiliary")
    if getattr(args, "predict_from_history", False):
        objective_set.append("history_encoder")
    training_regime = "offline_entity_bc"
    if len(objective_set) > 1:
        training_regime = "offline_entity_bc_aux"
    if args.policy_return_weighting != "none":
        training_regime = f"{training_regime}_weighted"

    metadata = {
        "policy_model_path": str(artifact_paths["policy_model"]),
        "policy_vocab_path": str(artifact_paths["policy_vocab"]),
        "entity_token_vocab_path": str(artifact_paths["entity_token_vocabs"]),
        "metadata_path": str(artifact_paths["metadata"]),
        "reward_profile_path": str(artifact_paths["reward_profile"]),
        "training_history_path": str(artifact_paths["training_history"]),
        "epoch_metrics_path": str(artifact_paths["epoch_metrics"]),
        "evaluation_summary_path": str(artifact_paths["evaluation_summary"]),
        "run_manifest_path": str(artifact_paths["run_manifest"]),
        "model_name": model_name,
        "model_release_id": model_name,
        "parent_release_id": None,
        "family_id": "entity_action_bc",
        "family_version": 1,
        "family_name": "entity_action_bc_v1",
        # This first entity line is intentionally research-only so the current Flask
        # runtime keeps serving the stable vector families until we opt in later.
        "training_regime": training_regime,
        "information_policy": "public_only",
        "state_schema_version": "entity_action_v1",
        "entity_schema_version": "entity_action_v1",
        "history_schema_version": None,
        "entity_tensor_layout": entity_tensor_layout(),
        "feature_layout": None,
        "num_action_classes": len(policy_vocab),
        "policy_label_format": "action_tokens",
        "action_space": "joint" if not args.move_only else "move_only",
        "action_parameterization": (
            "joint_vocab_scored_by_entity_encoder" if not args.move_only else "move_vocab_scored_by_entity_encoder"
        ),
        # objective_set is the quickest human-readable summary of what this release
        # is actually being optimized to do.
        "objective_set": objective_set,
        "train_examples": int(train_size),
        "val_examples": int(val_size),
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history_dict.get("loss", [])),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "token_embed_dim": int(args.token_embed_dim),
        "min_move_count": int(args.min_move_count),
        "max_battles": int(args.max_battles),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "include_switches": not args.move_only,
        "predict_turn_outcome": bool(args.predict_turn_outcome),
        "predict_value": bool(args.predict_value),
        "transition_weight": float(args.transition_weight),
        "value_weight": float(args.value_weight),
        "value_hidden_dim": int(args.value_hidden_dim or max(64, args.hidden_dim // 2)),
        "value_target": "terminal_win_probability",
        "value_target_definition": "terminal_win_probability" if args.predict_value else None,
        "value_head_activation": "sigmoid",
        "action_embed_dim": int(args.action_embed_dim),
        "transition_hidden_dim": int(args.transition_hidden_dim),
        "reward_config": reward_config.to_dict(),
        "reward_definition_id": "dense_reward_v2",
        "policy_return_weighting": str(args.policy_return_weighting),
        "policy_return_weight_scale": float(args.policy_return_weight_scale),
        "policy_return_weight_min": float(args.policy_return_weight_min),
        "policy_return_weight_max": float(args.policy_return_weight_max),
        "policy_weighting_definition": {
            "mode": str(args.policy_return_weighting),
            "scale": float(args.policy_return_weight_scale),
            "min_weight": float(args.policy_return_weight_min),
            "max_weight": float(args.policy_return_weight_max),
        },
        "switch_logit_bias": float(args.switch_logit_bias),
        "action_selection": {
            "switch_logit_bias": float(args.switch_logit_bias),
            "voluntary_switch_only": True,
        },
        "data_source_id": data_source_id,
        "data_source_details": data_source_details,
        "split_definition": {
            "method": "group_split_by_battle_id",
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
        },
        "environment_definition": {
            "battle_log_source": "pokemon_showdown_json",
            "observation_scope": "public_only",
            "format_hint": "gen9randombattle",
        },
        "initialization_source": {
            "type": "scratch",
            "checkpoint": None,
        },
        "evaluation_bundle_id": "offline_validation_v1",
        "registry_visibility": "research_only",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "num_reward_profiled_moves": len(move_reward_profile),
        "num_offensive_reward_moves": sum(
            1 for entry in move_reward_profile.values() if entry.get("is_offensive")
        ),
        "entity_token_vocab_sizes": {key: len(value) for key, value in token_vocabs.items()},
        "use_history": False,
    }

    predict_sequence = getattr(args, "predict_turn_sequence", False)
    if args.predict_turn_outcome:
        metadata["training_model_path"] = str(artifact_paths["training_model"])
        metadata["turn_outcome_dim"] = turn_outcome_dim()
        metadata["turn_outcome_layout"] = turn_outcome_layout()
        metadata["action_context_vocab_path"] = str(artifact_paths["action_context_vocab"])
        metadata["num_action_context_classes"] = len(action_context_vocab or {})
        metadata["transition_target_definition"] = "public_turn_outcome_summary_v1"
    elif args.predict_value or predict_sequence:
        metadata["training_model_path"] = str(artifact_paths["training_model"])

    if args.predict_value:
        metadata["policy_value_model_path"] = str(artifact_paths["policy_value_model"])

    if predict_sequence:
        metadata["sequence_target_definition"] = "turn_events_v1_composite_key"
        metadata["sequence_vocab_path"] = str(artifact_paths["sequence_vocab"])
        metadata["sequence_vocab_size"] = len(sequence_vocab or {})
        metadata["sequence_hidden_dim"] = int(getattr(args, "sequence_hidden_dim", 128))
        metadata["sequence_weight"] = float(getattr(args, "sequence_weight", 0.1))
        metadata["max_seq_len"] = int(getattr(args, "max_seq_len", 32))
        # Record action_context_vocab when sequence is enabled but transition is not.
        if not args.predict_turn_outcome:
            metadata["action_context_vocab_path"] = str(artifact_paths["action_context_vocab"])
            metadata["num_action_context_classes"] = len(action_context_vocab or {})

    if getattr(args, "predict_from_history", False):
        metadata["use_history"] = True
        metadata["history_turns"] = int(args.history_turns)
        metadata["history_events_per_turn"] = int(args.history_events_per_turn)
        metadata["history_embed_dim"] = int(args.history_embed_dim)
        metadata["history_lstm_dim"] = int(args.history_lstm_dim)

    if policy_weight_stats is not None:
        metadata["policy_weight_stats"] = policy_weight_stats

    return metadata


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk and validate the top-level type."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {path}, but found {type(payload).__name__}.")
    return payload


def resolve_transfer_source_metadata(
    repo_path: Path,
    *,
    init_from_metadata: str | None,
    init_from_release: str | None,
) -> dict[str, Any] | None:
    """Resolve optional source-release metadata for warm starts and transfer."""
    if init_from_metadata and init_from_release:
        raise SystemExit("Use only one of --init-from-metadata or --init-from-release.")
    if not init_from_metadata and not init_from_release:
        return None

    if init_from_metadata:
        metadata_path = Path(init_from_metadata)
        if not metadata_path.is_absolute():
            metadata_path = (repo_path / metadata_path).resolve()
        if not metadata_path.exists():
            raise SystemExit(f"Initialization metadata not found: {metadata_path}")
        payload = load_json_dict(metadata_path)
        payload.setdefault("metadata_path", str(metadata_path))
        return payload

    release_id = str(init_from_release)
    matches: list[dict[str, Any]] = []
    for metadata_path in metadata_file_candidates(repo_path / "artifacts"):
        payload = load_json_dict(metadata_path)
        model_release_id = str(payload.get("model_release_id") or "")
        model_name = str(payload.get("model_name") or "")
        if release_id not in {model_release_id, model_name}:
            continue
        payload.setdefault("metadata_path", str(metadata_path.relative_to(repo_path)))
        matches.append(payload)

    if not matches:
        raise SystemExit(f"Could not find a source release named '{release_id}' in artifacts/training_metadata*.json.")
    if len(matches) > 1:
        matching_labels = ", ".join(str(match.get("model_release_id") or match.get("model_name")) for match in matches)
        raise SystemExit(
            f"Release name '{release_id}' matched multiple metadata files: {matching_labels}. "
            "Use --init-from-metadata to disambiguate."
        )
    return matches[0]


def load_checkpoint_model(keras, checkpoint_path: Path):
    """Load a Keras checkpoint while tolerating Lambda-based saved models."""
    try:
        return keras.models.load_model(checkpoint_path, compile=False, safe_mode=False)
    except TypeError:
        return keras.models.load_model(checkpoint_path, compile=False)


def _source_metadata_path(repo_path: Path, source_metadata: Mapping[str, Any]) -> Path:
    raw_path = source_metadata.get("metadata_path")
    if not raw_path:
        return repo_path / "artifacts"
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (repo_path / path).resolve()


def build_transfer_source_model(
    *,
    checkpoint_path: Path,
    source_metadata: Mapping[str, Any],
    repo_path: Path,
):
    """Rebuild a known source-family architecture and load its weights."""
    family_id = str(source_metadata.get("family_id") or "")
    family_version = int(source_metadata.get("family_version") or 0)
    metadata_path = _source_metadata_path(repo_path, source_metadata)
    token_vocab_path = source_metadata.get("entity_token_vocab_path")
    if not token_vocab_path:
        raise ValueError("source metadata is missing entity_token_vocab_path")
    resolved_vocab_path = resolve_artifact_path(repo_path, metadata_path, str(token_vocab_path))
    token_vocabs = load_json_dict(resolved_vocab_path)
    vocab_sizes = {key: len(value) for key, value in token_vocabs.items()}

    if family_id == "entity_action_bc" and family_version == 1:
        source_model, _, _ = build_entity_action_models(
            vocab_sizes=vocab_sizes,
            num_policy_classes=int(source_metadata["num_action_classes"]),
            hidden_dim=int(source_metadata["hidden_dim"]),
            depth=int(source_metadata["depth"]),
            dropout=float(source_metadata["dropout"]),
            learning_rate=float(source_metadata.get("learning_rate") or 1e-3),
            token_embed_dim=int(source_metadata.get("token_embed_dim") or 24),
            transition_dim=int(source_metadata.get("turn_outcome_dim") or 0) or None,
            action_context_vocab_size=int(source_metadata.get("num_action_context_classes") or 0) or None,
            action_embed_dim=int(source_metadata.get("action_embed_dim") or 16),
            transition_hidden_dim=int(source_metadata.get("transition_hidden_dim") or source_metadata["hidden_dim"]),
            transition_weight=float(source_metadata.get("transition_weight") or 0.25),
            predict_value=bool(source_metadata.get("predict_value")),
            value_hidden_dim=int(source_metadata.get("value_hidden_dim") or max(64, int(source_metadata["hidden_dim"]) // 2)),
            value_weight=float(source_metadata.get("value_weight") or 0.25),
        )
        source_model.load_weights(checkpoint_path)
        return source_model

    if family_id == "entity_invariance_aux" and family_version == 1:
        from EntityInvarianceModelV1 import build_entity_invariance_models

        source_model, _, _ = build_entity_invariance_models(
            vocab_sizes=vocab_sizes,
            num_policy_classes=int(source_metadata["num_action_classes"]),
            hidden_dim=int(source_metadata["hidden_dim"]),
            depth=int(source_metadata["depth"]),
            dropout=float(source_metadata["dropout"]),
            learning_rate=float(source_metadata.get("learning_rate") or 1e-3),
            token_embed_dim=int(source_metadata.get("token_embed_dim") or 24),
            latent_dim=int(source_metadata.get("latent_dim") or 64),
            transition_dim=int(source_metadata.get("turn_outcome_dim") or 0) or None,
            action_context_vocab_size=int(source_metadata.get("num_action_context_classes") or 0) or None,
            action_embed_dim=int(source_metadata.get("action_embed_dim") or 16),
            transition_hidden_dim=int(source_metadata.get("transition_hidden_dim") or source_metadata["hidden_dim"]),
            transition_weight=float(source_metadata.get("transition_weight") or 0.25),
            predict_value=bool(source_metadata.get("predict_value")),
            value_hidden_dim=int(source_metadata.get("value_hidden_dim") or max(64, int(source_metadata["hidden_dim"]) // 2)),
            value_weight=float(source_metadata.get("value_weight") or 0.25),
        )
        source_model.load_weights(checkpoint_path)
        return source_model

    raise ValueError(f"unsupported transfer source family: {family_id}_v{family_version}")


def apply_keras_warm_start(
    *,
    keras,
    checkpoint_path: Path,
    target_models: List[Any],
    source_metadata: Mapping[str, Any] | None = None,
    repo_path: Path | None = None,
) -> dict[str, Any]:
    """Copy compatible layer weights from a saved checkpoint into the current models."""
    source_model = None
    if source_metadata is not None and repo_path is not None:
        try:
            source_model = build_transfer_source_model(
                checkpoint_path=checkpoint_path,
                source_metadata=source_metadata,
                repo_path=repo_path,
            )
        except Exception:
            source_model = None
    if source_model is None:
        source_model = load_checkpoint_model(keras, checkpoint_path)
    source_layers = {layer.name: layer for layer in source_model.layers}

    unique_target_layers: dict[str, Any] = {}
    seen_layer_ids: set[int] = set()
    for target_model in target_models:
        if target_model is None:
            continue
        for layer in target_model.layers:
            layer_id = id(layer)
            if layer_id in seen_layer_ids:
                continue
            seen_layer_ids.add(layer_id)
            unique_target_layers[layer.name] = layer

    applied_layers: list[str] = []
    skipped_reasons: Counter[str] = Counter()
    skipped_examples: list[dict[str, Any]] = []

    for layer_name, target_layer in unique_target_layers.items():
        source_layer = source_layers.get(layer_name)
        target_weights = target_layer.get_weights()
        if source_layer is None:
            if target_weights:
                skipped_reasons["missing_source_layer"] += 1
                if len(skipped_examples) < 24:
                    skipped_examples.append({"layer": layer_name, "reason": "missing_source_layer"})
            continue

        source_weights = source_layer.get_weights()
        if not target_weights and not source_weights:
            continue
        if len(source_weights) != len(target_weights):
            skipped_reasons["weight_count_mismatch"] += 1
            if len(skipped_examples) < 24:
                skipped_examples.append({"layer": layer_name, "reason": "weight_count_mismatch"})
            continue
        if any(src.shape != dst.shape for src, dst in zip(source_weights, target_weights)):
            skipped_reasons["shape_mismatch"] += 1
            if len(skipped_examples) < 24:
                skipped_examples.append({"layer": layer_name, "reason": "shape_mismatch"})
            continue

        target_layer.set_weights(source_weights)
        applied_layers.append(layer_name)

    return {
        "requested_checkpoint_path": str(checkpoint_path),
        "source_model_name": source_model.name,
        "source_layer_count": len(source_model.layers),
        "target_layer_count": len(unique_target_layers),
        "applied_layer_count": len(applied_layers),
        "applied_layers": applied_layers,
        "skipped_layer_count": int(sum(skipped_reasons.values())),
        "skipped_reason_counts": dict(skipped_reasons),
        "skipped_examples": skipped_examples,
    }


def parse_args() -> argparse.Namespace:
    """Define the CLI for entity_action_bc_v1 training runs."""
    parser = argparse.ArgumentParser(description="Train the first entity-centric Pokemon Showdown policy family.")
    parser.add_argument(
        "data_paths",
        nargs="*",
        help=(
            "Battle JSON files or directories containing battle JSON logs. "
            f"When omitted, defaults to Kaggle dataset '{DEFAULT_KAGGLE_DATASET}'."
        ),
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model and metadata artifacts.")
    parser.add_argument("--model-name", default="entity_action_bc_v1", help="Artifact/model release name.")
    parser.add_argument("--max-battles", type=int, default=5000, help="Maximum number of battles to load.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of battles reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the battle-level split.")
    parser.add_argument("--min-move-count", type=int, default=1, help="Minimum action frequency to keep in the vocab.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden width for the shared entity encoder and trunk.")
    parser.add_argument("--depth", type=int, default=3, help="Number of trunk dense layers after entity pooling.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied after dense layers.")
    parser.add_argument("--token-embed-dim", type=int, default=24, help="Base embedding width for learned entity tokens.")
    parser.add_argument("--move-only", action="store_true", help="Train only on move labels instead of joint move+switch action tokens.")
    parser.add_argument("--predict-turn-outcome", action="store_true", help="Add the auxiliary transition head.")
    parser.add_argument("--predict-value", action="store_true", help="Add the auxiliary terminal win-probability head.")
    parser.add_argument("--predict-turn-sequence", action="store_true", help="Add the auxiliary turn-event sequence head.")
    parser.add_argument("--transition-weight", type=float, default=0.25, help="Loss weight for the transition head.")
    parser.add_argument("--value-weight", type=float, default=0.25, help="Loss weight for the value head.")
    parser.add_argument("--sequence-weight", type=float, default=0.1, help="Loss weight for the sequence head.")
    parser.add_argument("--sequence-hidden-dim", type=int, default=128, help="LSTM hidden width for the sequence head.")
    parser.add_argument("--max-seq-len", type=int, default=32, help="Maximum token sequence length for the sequence head.")
    parser.add_argument(
        "--predict-from-history",
        action="store_true",
        help="Enable the event history encoder auxiliary module.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=8,
        help="K: number of past turns to retain in the rolling history buffer.",
    )
    parser.add_argument(
        "--history-events-per-turn",
        type=int,
        default=24,
        help="E: maximum event tokens per turn row in the history matrix.",
    )
    parser.add_argument(
        "--history-embed-dim",
        type=int,
        default=32,
        help="Embedding width for history event tokens.",
    )
    parser.add_argument(
        "--history-lstm-dim",
        type=int,
        default=64,
        help="LSTM hidden width for the history Bi-LSTM encoder.",
    )
    parser.add_argument("--value-hidden-dim", type=int, default=0, help="Hidden width for the value head block.")
    parser.add_argument("--action-embed-dim", type=int, default=16, help="Embedding width for action-conditioned transition inputs.")
    parser.add_argument("--transition-hidden-dim", type=int, default=256, help="Hidden width for the transition head block.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation top-3 when validation exists.")
    parser.add_argument("--verbose-every", type=int, default=200, help="Print ingest progress every N battles. Set 0 to disable.")
    parser.add_argument(
        "--init-from-metadata",
        default=None,
        help="Path to a source training_metadata JSON file for warm start or transfer.",
    )
    parser.add_argument(
        "--init-from-release",
        default=None,
        help="Source model_release_id or model_name to resolve from artifacts/training_metadata*.json.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Explicit checkpoint path to load instead of the source release's preferred artifact.",
    )
    parser.add_argument("--reward-hp-weight", type=float, default=0.25)
    parser.add_argument("--reward-ko-weight", type=float, default=0.50)
    parser.add_argument("--reward-wasted-move-penalty", type=float, default=0.10)
    parser.add_argument("--reward-redundant-setup-penalty", type=float, default=0.25)
    parser.add_argument("--reward-wasted-setup-penalty", type=float, default=1.0)
    parser.add_argument("--reward-terminal-weight", type=float, default=1.0)
    parser.add_argument("--reward-return-discount", type=float, default=1.0)
    parser.add_argument(
        "--policy-return-weighting",
        choices=["none", "exp"],
        default="exp",
        help="How to weight policy examples using return-to-go. 'exp' upweights higher-return examples and downweights lower-return ones.",
    )
    parser.add_argument(
        "--policy-return-weight-scale",
        type=float,
        default=0.75,
        help="Strength of return-to-go weighting when policy weighting is enabled.",
    )
    parser.add_argument(
        "--policy-return-weight-min",
        type=float,
        default=0.25,
        help="Lower clip for policy example sample weights.",
    )
    parser.add_argument(
        "--policy-return-weight-max",
        type=float,
        default=4.0,
        help="Upper clip for policy example sample weights.",
    )
    parser.add_argument(
        "--switch-logit-bias",
        type=float,
        default=0.20,
        help="Serving-time logit penalty applied to voluntary switch actions to reduce over-switching.",
    )
    parser.add_argument("--reward-offensive-move-min-uses", type=int, default=20)
    parser.add_argument("--reward-offensive-move-min-damage-rate", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run one full entity-family training job from logs to saved artifacts."""
    args = parse_args()
    repo_path = Path(__file__).resolve().parent
    print(
        "entity_training_start:"
        f" model_name={args.model_name}"
        f" output_dir={args.output_dir}"
        f" move_only={args.move_only}"
        f" predict_value={args.predict_value}"
        f" predict_turn_outcome={args.predict_turn_outcome}",
        flush=True,
    )

    data_paths = resolve_data_paths(args.data_paths)
    json_paths = discover_json_paths(data_paths)
    if not json_paths:
        raise SystemExit("No JSON files found in the provided path(s).")
    print(
        "entity_data_source:"
        f" resolved_paths={len(data_paths)}"
        f" json_files={len(json_paths)}",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_metadata = resolve_transfer_source_metadata(
        repo_path,
        init_from_metadata=args.init_from_metadata,
        init_from_release=args.init_from_release,
    )
    transfer_seed_metadata = apply_transfer_metadata(
        {
            "family_id": "entity_action_bc",
            "family_version": 1,
            "model_release_id": args.model_name,
            "model_name": args.model_name,
        },
        source_metadata=source_metadata,
        base_dir=repo_path,
        explicit_checkpoint_path=args.init_checkpoint,
    )
    init_source = transfer_seed_metadata["initialization_source"]
    checkpoint_path = init_source.get("checkpoint_path")
    if args.init_checkpoint and (not checkpoint_path or not Path(checkpoint_path).exists()):
        raise SystemExit(f"Initialization checkpoint not found: {checkpoint_path or args.init_checkpoint}")
    if source_metadata is not None and init_source.get("mode") == "scratch":
        raise SystemExit(
            "A source release was requested, but no compatible checkpoint path could be resolved from its metadata."
        )
    print(
        "entity_initialization:"
        f" {describe_transfer({'initialization_source': init_source})}"
        f" checkpoint={checkpoint_path}",
        flush=True,
    )

    tracker = BattleStateTracker(
        form_change_species={"Palafin"},
        history_turns=args.history_turns if args.predict_from_history else 0,
    )
    # Reuse the battle ingester so the entity family stays comparable to the older
    # vector families at the data-loading layer.
    examples = ingest_battles_to_examples(
        tracker,
        json_paths,
        max_battles=args.max_battles,
        verbose_every=args.verbose_every,
        include_switches=not args.move_only,
    )
    if not examples:
        raise SystemExit("No training examples were produced from the provided battle logs.")

    if args.predict_value:
        # The current value head is trained only on terminal outcome, so examples
        # without a resolved battle result are removed up front.
        with_terminal = [ex for ex in examples if ex.get("terminal_result") is not None]
        dropped_missing_terminal = len(examples) - len(with_terminal)
        if not with_terminal:
            raise SystemExit("Value training requires battle outcomes, but no examples carried terminal_result.")
        if dropped_missing_terminal:
            print(f"dropped_examples_missing_terminal_result={dropped_missing_terminal}")
        examples = with_terminal

    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    # Vocabularies are built from the train split only so future traces and replay
    # tools line up with what the model actually saw during fitting.
    train_examples = subset_examples(examples, train_idx)
    vocab_source = train_examples if train_examples else examples

    reward_config = make_reward_config(args)
    move_reward_profile = build_move_reward_profile(vocab_source, reward_config)
    # Reward traces are attached now for observability and future policy refinement, even
    # though the current value head still trains on terminal win probability only.
    examples = attach_reward_targets(examples, reward_config, move_reward_profile)

    bundle = build_entity_training_bundle(
        vocab_source,
        include_switches=not args.move_only,
        min_move_count=args.min_move_count,
        include_transition=args.predict_turn_outcome,
        include_value=args.predict_value,
        include_sequence=args.predict_turn_sequence,
        max_seq_len=args.max_seq_len,
        include_history=args.predict_from_history,
        history_turns=args.history_turns,
        history_events_per_turn=args.history_events_per_turn,
    )
    policy_vocab = bundle["policy_vocab"]
    action_context_vocab = bundle["action_context_vocab"]
    token_vocabs = bundle["token_vocabs"]
    sequence_vocab = bundle.get("sequence_vocab")

    X_raw, targets_raw = vectorize_entity_multitask_dataset(
        examples,
        policy_vocab=policy_vocab,
        token_vocabs=token_vocabs,
        action_context_vocab=action_context_vocab,
        include_switches=not args.move_only,
        include_transition=args.predict_turn_outcome,
        include_value=args.predict_value,
        include_sequence=args.predict_turn_sequence,
        sequence_vocab=sequence_vocab,
        max_seq_len=args.max_seq_len,
        include_history=args.predict_from_history,
        history_turns=args.history_turns,
        history_events_per_turn=args.history_events_per_turn,
    )
    X_np = to_numpy_entity_inputs(X_raw)
    del X_raw  # pre-numpy lists no longer needed

    # Policy is sparse categorical, transition is dense regression, value is a
    # scalar binary target in [0, 1], and sequence is an integer token matrix.
    y_policy_np = np.asarray(targets_raw["policy"], dtype=np.int64)
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
    y_sequence_np = (
        None
        if "sequence" not in targets_raw
        else np.asarray(targets_raw["sequence"], dtype=np.int64)
    )
    # Free the raw Python examples and target lists — numpy arrays now own the data.
    del examples, targets_raw
    gc.collect()

    X_train = slice_entity_inputs(X_np, train_idx)
    X_val = slice_entity_inputs(X_np, val_idx) if len(val_idx) else None
    y_train_policy = y_policy_np[train_idx]
    y_val_policy = y_policy_np[val_idx] if len(val_idx) else None
    y_train_transition = y_transition_np[train_idx] if y_transition_np is not None else None
    y_val_transition = y_transition_np[val_idx] if y_transition_np is not None and len(val_idx) else None
    y_train_value = y_value_np[train_idx] if y_value_np is not None else None
    y_val_value = y_value_np[val_idx] if y_value_np is not None and len(val_idx) else None
    y_train_sequence = y_sequence_np[train_idx] if y_sequence_np is not None else None
    y_val_sequence = y_sequence_np[val_idx] if y_sequence_np is not None and len(val_idx) else None
    # Full arrays are now split into train/val; free the originals.
    del X_np, y_policy_np, y_transition_np, y_value_np, y_sequence_np
    gc.collect()

    policy_train_weights = build_policy_training_sample_weights(
        train_examples,
        include_switches=not args.move_only,
        use_action_tokens=True,
        weighting_mode=args.policy_return_weighting,
        scale=args.policy_return_weight_scale,
        min_weight=args.policy_return_weight_min,
        max_weight=args.policy_return_weight_max,
    )
    if policy_train_weights is not None and len(policy_train_weights) != len(y_train_policy):
        raise ValueError(
            "policy sample weights length does not match the policy training targets; "
            "example filtering and vectorization fell out of sync"
        )
    del train_examples  # vocab + weights built; raw examples no longer needed
    policy_weight_stats = None
    if policy_train_weights is not None:
        policy_weight_stats = {
            "count": int(len(policy_train_weights)),
            "mean": float(np.mean(policy_train_weights)),
            "min": float(np.min(policy_train_weights)),
            "max": float(np.max(policy_train_weights)),
        }

    print(
        "entity_dataset:"
        f" train_examples={len(train_idx)}"
        f" val_examples={len(val_idx)}"
        f" policy_classes={len(policy_vocab)}"
        f" species_vocab={len(token_vocabs['species'])}"
        f" move_vocab={len(token_vocabs['move'])}"
    ,
        flush=True,
    )
    if policy_weight_stats is not None:
        print(
            "entity_policy_weighting:"
            f" mode={args.policy_return_weighting}"
            f" mean={policy_weight_stats['mean']:.3f}"
            f" min={policy_weight_stats['min']:.3f}"
            f" max={policy_weight_stats['max']:.3f}",
            flush=True,
        )

    artifact_paths = build_entity_artifact_paths(
        output_dir,
        model_name=args.model_name,
        save_training_model=(
            args.predict_turn_outcome
            or args.predict_value
            or args.predict_turn_sequence
            or args.predict_from_history
        ),
        save_action_context_vocab=(args.predict_turn_outcome or args.predict_turn_sequence),
        save_policy_value_model=args.predict_value,
        save_sequence_vocab=(args.predict_turn_sequence or args.predict_from_history),
    )

    try:
        # Import TensorFlow lazily so this script can still be inspected, linted, and
        # partially exercised in environments that do not have the training stack.
        model, policy_model, policy_value_model, _ = build_entity_action_models(
            vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
            num_policy_classes=len(policy_vocab),
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            token_embed_dim=args.token_embed_dim,
            transition_dim=turn_outcome_dim() if args.predict_turn_outcome else None,
            action_context_vocab_size=len(action_context_vocab or {}) if (args.predict_turn_outcome or args.predict_turn_sequence) else None,
            action_embed_dim=args.action_embed_dim,
            transition_hidden_dim=args.transition_hidden_dim,
            transition_weight=args.transition_weight,
            predict_value=args.predict_value,
            value_hidden_dim=args.value_hidden_dim or max(64, args.hidden_dim // 2),
            value_weight=args.value_weight,
            predict_sequence=args.predict_turn_sequence,
            sequence_vocab_size=len(sequence_vocab or {}) if args.predict_turn_sequence else None,
            sequence_hidden_dim=args.sequence_hidden_dim,
            sequence_weight=args.sequence_weight,
            max_seq_len=args.max_seq_len,
            use_history=args.predict_from_history,
            history_vocab_size=len(sequence_vocab or {}) if args.predict_from_history else None,
            history_embed_dim=args.history_embed_dim,
            history_lstm_dim=args.history_lstm_dim,
            history_turns=args.history_turns,
            history_events_per_turn=args.history_events_per_turn,
        )
        from tensorflow import keras
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required to train entity_action_bc_v1, but it is not installed in this environment."
        ) from exc

    transfer_report = None
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.is_absolute():
            checkpoint = (repo_path / checkpoint).resolve()
        if not checkpoint.exists():
            raise SystemExit(f"Initialization checkpoint not found: {checkpoint}")
        transfer_report = apply_keras_warm_start(
            keras=keras,
            checkpoint_path=checkpoint,
            target_models=[model, policy_model, policy_value_model],
            source_metadata=source_metadata,
            repo_path=repo_path,
        )
        relationship = str(init_source.get("relationship") or "unknown")
        if transfer_report["applied_layer_count"] == 0 and relationship != "cross_family_transfer":
            raise SystemExit(
                "Warm start resolved a checkpoint but applied zero compatible layers. "
                f"checkpoint={checkpoint} relationship={relationship}"
            )
        print(
            "entity_transfer_applied:"
            f" relationship={relationship}"
            f" applied_layers={transfer_report['applied_layer_count']}"
            f" skipped_layers={transfer_report['skipped_layer_count']}"
            f" source_model={transfer_report['source_model_name']}",
            flush=True,
        )

    callbacks: List[Any] = []
    # CSVLogger gives us a stable, low-friction artifact we can watch while training is live.
    callbacks.append(keras.callbacks.CSVLogger(str(artifact_paths["epoch_metrics"])))
    if len(val_idx):
        monitor = "val_policy_top3" if len(policy_vocab) >= 3 else "val_policy_top1"
        # Validation is grouped by battle id, so early stopping is watching genuine
        # held-out games rather than mixed turns from the same battle.
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=args.patience,
                mode="max",
                restore_best_weights=True,
            )
        )

    train_targets: Dict[str, Any] | Any = {"policy": y_train_policy}
    if args.predict_turn_outcome and y_train_transition is not None:
        train_targets["transition"] = y_train_transition
    if args.predict_value and y_train_value is not None:
        train_targets["value"] = y_train_value
    if args.predict_turn_sequence and y_train_sequence is not None:
        train_targets["sequence"] = y_train_sequence
    if list(train_targets.keys()) == ["policy"]:
        train_targets = y_train_policy

    val_data = None
    if len(val_idx):
        val_targets: Dict[str, Any] | Any = {"policy": y_val_policy}
        if args.predict_turn_outcome and y_val_transition is not None:
            val_targets["transition"] = y_val_transition
        if args.predict_value and y_val_value is not None:
            val_targets["value"] = y_val_value
        if args.predict_turn_sequence and y_val_sequence is not None:
            val_targets["sequence"] = y_val_sequence
        if list(val_targets.keys()) == ["policy"]:
            val_targets = y_val_policy
        val_data = (X_val, val_targets)

    multitask_targets = isinstance(train_targets, dict)
    sample_weight = None
    if policy_train_weights is not None:
        if multitask_targets:
            sample_weight = {"policy": policy_train_weights}
            if args.predict_turn_outcome and y_train_transition is not None:
                sample_weight["transition"] = np.ones(len(y_train_policy), dtype=np.float32)
            if args.predict_value and y_train_value is not None:
                sample_weight["value"] = np.ones(len(y_train_policy), dtype=np.float32)
            if args.predict_turn_sequence and y_train_sequence is not None:
                sample_weight["sequence"] = np.ones(len(y_train_policy), dtype=np.float32)
        else:
            sample_weight = policy_train_weights

    history = model.fit(
        X_train,
        train_targets,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_weight=sample_weight,
        callbacks=callbacks,
        verbose=1,
    )

    policy_model.save(artifact_paths["policy_model"])
    if (
        args.predict_turn_outcome
        or args.predict_value
        or args.predict_turn_sequence
        or args.predict_from_history
    ):
        model.save(artifact_paths["training_model"])
    if args.predict_value and policy_value_model is not None:
        policy_value_model.save(artifact_paths["policy_value_model"])

    # Persist both vocabularies: policy labels for action decoding and entity token ids
    # so later tools can inspect the learned embedding tables.
    save_json(artifact_paths["policy_vocab"], policy_vocab)
    save_json(artifact_paths["entity_token_vocabs"], token_vocabs)
    if (args.predict_turn_outcome or args.predict_turn_sequence) and action_context_vocab is not None:
        save_json(artifact_paths["action_context_vocab"], action_context_vocab)
    if (args.predict_turn_sequence or args.predict_from_history) and sequence_vocab is not None:
        save_json(artifact_paths["sequence_vocab"], sequence_vocab)
    save_json(artifact_paths["reward_profile"], move_reward_profile)

    metadata = make_training_metadata(
        args,
        model_name=args.model_name,
        train_size=len(train_idx),
        val_size=len(val_idx),
        history=history,
        artifact_paths=artifact_paths,
        policy_vocab=policy_vocab,
        action_context_vocab=action_context_vocab,
        token_vocabs=token_vocabs,
        sequence_vocab=sequence_vocab,
        reward_config=reward_config,
        move_reward_profile=move_reward_profile,
        policy_weight_stats=policy_weight_stats,
        raw_data_paths=list(args.data_paths),
        resolved_data_paths=list(data_paths),
        json_paths=list(json_paths),
    )
    if policy_weight_stats is not None:
        metadata["policy_weight_stats"] = policy_weight_stats
    metadata = apply_transfer_metadata(
        metadata,
        source_metadata=source_metadata,
        base_dir=repo_path,
        explicit_checkpoint_path=args.init_checkpoint,
    )
    if transfer_report is not None:
        metadata["initialization_source"]["transfer_report"] = transfer_report
    if source_metadata is not None and source_metadata.get("metadata_path") is not None:
        metadata["initialization_source"]["source_metadata_path"] = str(source_metadata["metadata_path"])
    save_json(artifact_paths["metadata"], metadata)
    save_json(artifact_paths["training_history"], history.history)

    evaluation_summary = summarize_training_history(history)
    save_json(artifact_paths["evaluation_summary"], evaluation_summary)

    registry_path = write_model_registry(Path(__file__).resolve().parent)
    # The run manifest is the reproducibility anchor for this exact training recipe.
    run_manifest = build_run_manifest(
        metadata=metadata,
        artifact_paths=artifact_paths,
        evaluation_summary=evaluation_summary,
        registry_path=registry_path,
    )
    save_json(artifact_paths["run_manifest"], run_manifest)

    print(f"saved_entity_policy_model={artifact_paths['policy_model']}")
    if args.predict_value:
        print(f"saved_entity_policy_value_model={artifact_paths['policy_value_model']}")
    if args.predict_turn_outcome or args.predict_value:
        print(f"saved_entity_training_model={artifact_paths['training_model']}")
    print(f"saved_entity_vocab={artifact_paths['entity_token_vocabs']}")
    print(f"saved_policy_vocab={artifact_paths['policy_vocab']}")
    print(f"saved_training_metadata={artifact_paths['metadata']}")
    print(f"saved_evaluation_summary={artifact_paths['evaluation_summary']}")
    print(f"saved_run_manifest={artifact_paths['run_manifest']}")
    print(f"updated_model_registry={registry_path}")
    print(f"completed_at_utc={datetime.now(timezone.utc).isoformat()}", flush=True)


if __name__ == "__main__":
    main()
