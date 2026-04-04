from __future__ import annotations

"""Trainer for the first identity-invariance entity family.

This scaffold keeps the existing public entity state, then adds:
    - one-step previous-turn inputs
    - identity-regime experiments (real, placeholder, mixed)
    - a latent summary derived from previous-turn state

It is the first runnable entrypoint for entity_invariance_aux_v1.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from BattleStateTracker import BattleStateTracker
from EntityInvarianceModelV1 import build_entity_invariance_models
from EntityInvarianceTensorization import (
    IDENTITY_REGIMES,
    concat_invariance_batches,
    concat_target_batches,
    invariance_tensor_layout,
    to_numpy_invariance_inputs,
    vectorize_entity_invariance_dataset,
)
from EntityTensorization import build_entity_training_bundle
from ModelRegistry import write_model_registry
from RewardSignals import TRACE_SCHEMA_VERSION, attach_reward_targets, build_move_reward_profile
from StateVectorization import turn_outcome_dim, turn_outcome_layout
from TransferLearning import apply_transfer_metadata, describe_transfer
from TrainingSplit import group_split_by_battle_id, ingest_battles_to_examples
from train_entity_action import (
    apply_keras_warm_start,
    build_entity_artifact_paths,
    resolve_transfer_source_metadata,
)
from train_policy import (
    DEFAULT_KAGGLE_DATASET,
    build_data_source_metadata,
    build_run_manifest,
    discover_json_paths,
    make_reward_config,
    resolve_data_paths,
    save_json,
    subset_examples,
    summarize_training_history,
)


def make_training_metadata(
    args: argparse.Namespace,
    *,
    model_name: str,
    train_size: int,
    base_train_size: int,
    val_size: int,
    history,
    artifact_paths: Dict[str, Path],
    policy_vocab: Dict[str, int],
    action_context_vocab: Dict[str, int] | None,
    token_vocabs: Dict[str, Dict[str, int]],
    reward_config,
    move_reward_profile: Dict[str, Any],
    raw_data_paths: List[str],
    resolved_data_paths: List[str],
    json_paths: List[str],
) -> dict:
    """Assemble the persisted recipe metadata for one invariance-family run."""
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
        "family_id": "entity_invariance_aux",
        "family_version": 1,
        "family_name": "entity_invariance_aux_v1",
        "training_regime": "offline_entity_invariance_aux",
        "information_policy": "public_only",
        "state_schema_version": "entity_invariance_v1",
        "entity_schema_version": "entity_invariance_v1",
        "history_schema_version": "entity_turn_minus_1_v1",
        "entity_tensor_layout": invariance_tensor_layout(),
        "feature_layout": None,
        "identity_regime": args.identity_regime,
        "placeholder_seed": int(args.placeholder_seed),
        "latent_dim": int(args.latent_dim),
        "num_action_classes": len(policy_vocab),
        "policy_label_format": "action_tokens",
        "action_space": "joint" if not args.move_only else "move_only",
        "action_parameterization": (
            "joint_vocab_scored_by_entity_invariance_encoder"
            if not args.move_only
            else "move_vocab_scored_by_entity_invariance_encoder"
        ),
        "objective_set": objective_set,
        "base_train_examples": int(base_train_size),
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
        "policy_weighting_definition": {"mode": "none"},
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
    }

    if args.predict_turn_outcome:
        metadata["training_model_path"] = str(artifact_paths["training_model"])
        metadata["turn_outcome_dim"] = turn_outcome_dim()
        metadata["turn_outcome_layout"] = turn_outcome_layout()
        metadata["action_context_vocab_path"] = str(artifact_paths["action_context_vocab"])
        metadata["num_action_context_classes"] = len(action_context_vocab or {})
        metadata["transition_target_definition"] = "public_turn_outcome_summary_v1"
    elif args.predict_value:
        metadata["training_model_path"] = str(artifact_paths["training_model"])

    if args.predict_value:
        metadata["policy_value_model_path"] = str(artifact_paths["policy_value_model"])

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the entity_invariance_aux_v1 Pokemon Showdown family.")
    parser.add_argument(
        "data_paths",
        nargs="*",
        help=(
            "Battle JSON files or directories containing battle JSON logs. "
            f"When omitted, defaults to Kaggle dataset '{DEFAULT_KAGGLE_DATASET}'."
        ),
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model and metadata artifacts.")
    parser.add_argument("--model-name", default="entity_invariance_aux_v1", help="Artifact/model release name.")
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
    parser.add_argument("--latent-dim", type=int, default=64, help="Width of the previous-turn latent summary.")
    parser.add_argument(
        "--identity-regime",
        default="real_id",
        choices=sorted(IDENTITY_REGIMES),
        help="Identity presentation regime: real ids, placeholder ids, or mixed training augmentation.",
    )
    parser.add_argument("--placeholder-seed", type=int, default=13, help="Seed for placeholder-id remapping.")
    parser.add_argument("--move-only", action="store_true", help="Train only on move labels instead of joint move+switch action tokens.")
    parser.add_argument("--predict-turn-outcome", action="store_true", help="Add the auxiliary transition head.")
    parser.add_argument("--predict-value", action="store_true", help="Add the auxiliary terminal win-probability head.")
    parser.add_argument("--transition-weight", type=float, default=0.25, help="Loss weight for the transition head.")
    parser.add_argument("--value-weight", type=float, default=0.25, help="Loss weight for the value head.")
    parser.add_argument("--value-hidden-dim", type=int, default=0, help="Hidden width for the value head block.")
    parser.add_argument("--action-embed-dim", type=int, default=16, help="Embedding width for action-conditioned transition inputs.")
    parser.add_argument("--transition-hidden-dim", type=int, default=256, help="Hidden width for the transition head block.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation top-3 when validation exists.")
    parser.add_argument("--verbose-every", type=int, default=200, help="Print ingest progress every N battles. Set 0 to disable.")
    parser.add_argument("--init-from-metadata", default=None, help="Path to a source training_metadata JSON file for warm start or transfer.")
    parser.add_argument("--init-from-release", default=None, help="Source model_release_id or model_name to resolve from artifacts/training_metadata*.json.")
    parser.add_argument("--init-checkpoint", default=None, help="Explicit checkpoint path to load instead of the source release's preferred artifact.")
    parser.add_argument("--reward-hp-weight", type=float, default=0.25)
    parser.add_argument("--reward-ko-weight", type=float, default=0.50)
    parser.add_argument("--reward-wasted-move-penalty", type=float, default=0.10)
    parser.add_argument("--reward-redundant-setup-penalty", type=float, default=0.25)
    parser.add_argument("--reward-wasted-setup-penalty", type=float, default=1.0)
    parser.add_argument("--reward-terminal-weight", type=float, default=1.0)
    parser.add_argument("--reward-return-discount", type=float, default=1.0)
    parser.add_argument("--reward-offensive-move-min-uses", type=int, default=20)
    parser.add_argument("--reward-offensive-move-min-damage-rate", type=float, default=0.5)
    return parser.parse_args()


def numpy_targets_from_raw(targets_raw: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {
        "policy": np.asarray(targets_raw["policy"], dtype=np.int64),
    }
    if "transition" in targets_raw:
        targets["transition"] = np.asarray(targets_raw["transition"], dtype=np.float32)
    if "value" in targets_raw:
        targets["value"] = np.asarray(targets_raw["value"], dtype=np.float32).reshape(-1, 1)
    return targets


def main() -> None:
    args = parse_args()
    repo_path = Path(__file__).resolve().parent
    print(
        "entity_invariance_training_start:"
        f" model_name={args.model_name}"
        f" identity_regime={args.identity_regime}"
        f" predict_value={args.predict_value}"
        f" predict_turn_outcome={args.predict_turn_outcome}",
        flush=True,
    )

    data_paths = resolve_data_paths(args.data_paths)
    json_paths = discover_json_paths(data_paths)
    if not json_paths:
        raise SystemExit("No JSON files found in the provided path(s).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_metadata = resolve_transfer_source_metadata(
        repo_path,
        init_from_metadata=args.init_from_metadata,
        init_from_release=args.init_from_release,
    )
    transfer_seed_metadata = apply_transfer_metadata(
        {
            "family_id": "entity_invariance_aux",
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
        raise SystemExit("A source release was requested, but no compatible checkpoint path could be resolved.")
    print(
        "entity_invariance_initialization:"
        f" {describe_transfer({'initialization_source': init_source})}"
        f" checkpoint={checkpoint_path}",
        flush=True,
    )

    tracker = BattleStateTracker(form_change_species={"Palafin"})
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
        with_terminal = [ex for ex in examples if ex.get("terminal_result") is not None]
        if not with_terminal:
            raise SystemExit("Value training requires battle outcomes, but no examples carried terminal_result.")
        examples = with_terminal

    train_idx, val_idx = group_split_by_battle_id(examples, val_ratio=args.val_ratio, seed=args.seed)
    train_examples = subset_examples(examples, train_idx)
    val_examples = subset_examples(examples, val_idx) if len(val_idx) else []
    vocab_source = train_examples if train_examples else examples

    reward_config = make_reward_config(args)
    move_reward_profile = build_move_reward_profile(vocab_source, reward_config)
    examples = attach_reward_targets(examples, reward_config, move_reward_profile)
    train_examples = subset_examples(examples, train_idx)
    val_examples = subset_examples(examples, val_idx) if len(val_idx) else []

    bundle = build_entity_training_bundle(
        vocab_source,
        include_switches=not args.move_only,
        min_move_count=args.min_move_count,
        include_transition=args.predict_turn_outcome,
        include_value=args.predict_value,
    )
    policy_vocab = bundle["policy_vocab"]
    action_context_vocab = bundle["action_context_vocab"]
    token_vocabs = bundle["token_vocabs"]

    train_inputs_raw, train_targets_raw = vectorize_entity_invariance_dataset(
        train_examples,
        policy_vocab=policy_vocab,
        token_vocabs=token_vocabs,
        action_context_vocab=action_context_vocab,
        include_switches=not args.move_only,
        include_transition=args.predict_turn_outcome,
        include_value=args.predict_value,
        include_history=True,
        identity_regime="placeholder_id" if args.identity_regime == "placeholder_id" else "real_id",
        placeholder_seed=args.placeholder_seed,
    )
    X_train = to_numpy_invariance_inputs(train_inputs_raw)
    y_train = numpy_targets_from_raw(train_targets_raw)

    if args.identity_regime == "mixed_id":
        placeholder_inputs_raw, placeholder_targets_raw = vectorize_entity_invariance_dataset(
            train_examples,
            policy_vocab=policy_vocab,
            token_vocabs=token_vocabs,
            action_context_vocab=action_context_vocab,
            include_switches=not args.move_only,
            include_transition=args.predict_turn_outcome,
            include_value=args.predict_value,
            include_history=True,
            identity_regime="placeholder_id",
            placeholder_seed=args.placeholder_seed,
        )
        X_train = concat_invariance_batches(
            X_train,
            to_numpy_invariance_inputs(placeholder_inputs_raw),
        )
        y_train = concat_target_batches(
            y_train,
            numpy_targets_from_raw(placeholder_targets_raw),
        )

    X_val = None
    y_val = None
    if val_examples:
        val_identity_regime = args.identity_regime if args.identity_regime != "mixed_id" else "real_id"
        val_inputs_raw, val_targets_raw = vectorize_entity_invariance_dataset(
            val_examples,
            policy_vocab=policy_vocab,
            token_vocabs=token_vocabs,
            action_context_vocab=action_context_vocab,
            include_switches=not args.move_only,
            include_transition=args.predict_turn_outcome,
            include_value=args.predict_value,
            include_history=True,
            identity_regime=val_identity_regime,
            placeholder_seed=args.placeholder_seed,
        )
        X_val = to_numpy_invariance_inputs(val_inputs_raw)
        y_val = numpy_targets_from_raw(val_targets_raw)

    print(
        "entity_invariance_dataset:"
        f" base_train_examples={len(train_examples)}"
        f" train_examples={len(y_train['policy'])}"
        f" val_examples={len(y_val['policy']) if y_val is not None else 0}"
        f" policy_classes={len(policy_vocab)}"
        f" identity_regime={args.identity_regime}",
        flush=True,
    )

    artifact_paths = build_entity_artifact_paths(
        output_dir,
        model_name=args.model_name,
        save_training_model=(args.predict_turn_outcome or args.predict_value),
        save_action_context_vocab=args.predict_turn_outcome,
        save_policy_value_model=args.predict_value,
    )

    try:
        model, policy_model, policy_value_model = build_entity_invariance_models(
            vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
            num_policy_classes=len(policy_vocab),
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            token_embed_dim=args.token_embed_dim,
            latent_dim=args.latent_dim,
            transition_dim=turn_outcome_dim() if args.predict_turn_outcome else None,
            action_context_vocab_size=len(action_context_vocab or {}) if args.predict_turn_outcome else None,
            action_embed_dim=args.action_embed_dim,
            transition_hidden_dim=args.transition_hidden_dim,
            transition_weight=args.transition_weight,
            predict_value=args.predict_value,
            value_hidden_dim=args.value_hidden_dim or max(64, args.hidden_dim // 2),
            value_weight=args.value_weight,
        )
        from tensorflow import keras
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required to train entity_invariance_aux_v1, but it is not installed in this environment."
        ) from exc

    transfer_report = None
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.is_absolute():
            checkpoint = (repo_path / checkpoint).resolve()
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
            "entity_invariance_transfer_applied:"
            f" relationship={relationship}"
            f" applied_layers={transfer_report['applied_layer_count']}"
            f" skipped_layers={transfer_report['skipped_layer_count']}",
            flush=True,
        )

    callbacks: List[Any] = [keras.callbacks.CSVLogger(str(artifact_paths["epoch_metrics"]))]
    if y_val is not None:
        monitor = "val_policy_top3" if len(policy_vocab) >= 3 else "val_policy_top1"
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=args.patience,
                mode="max",
                restore_best_weights=True,
            )
        )

    train_targets: Dict[str, Any] | Any = dict(y_train)
    if list(train_targets.keys()) == ["policy"]:
        train_targets = y_train["policy"]

    val_data = None
    if y_val is not None and X_val is not None:
        val_targets: Dict[str, Any] | Any = dict(y_val)
        if list(val_targets.keys()) == ["policy"]:
            val_targets = y_val["policy"]
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

    policy_model.save(artifact_paths["policy_model"])
    if args.predict_turn_outcome or args.predict_value:
        model.save(artifact_paths["training_model"])
    if args.predict_value and policy_value_model is not None:
        policy_value_model.save(artifact_paths["policy_value_model"])

    save_json(artifact_paths["policy_vocab"], policy_vocab)
    save_json(artifact_paths["entity_token_vocabs"], token_vocabs)
    if args.predict_turn_outcome and action_context_vocab is not None:
        save_json(artifact_paths["action_context_vocab"], action_context_vocab)
    save_json(artifact_paths["reward_profile"], move_reward_profile)

    metadata = make_training_metadata(
        args,
        model_name=args.model_name,
        train_size=len(y_train["policy"]),
        base_train_size=len(train_examples),
        val_size=len(y_val["policy"]) if y_val is not None else 0,
        history=history,
        artifact_paths=artifact_paths,
        policy_vocab=policy_vocab,
        action_context_vocab=action_context_vocab,
        token_vocabs=token_vocabs,
        reward_config=reward_config,
        move_reward_profile=move_reward_profile,
        raw_data_paths=list(args.data_paths),
        resolved_data_paths=list(data_paths),
        json_paths=list(json_paths),
    )
    metadata = apply_transfer_metadata(
        metadata,
        source_metadata=source_metadata,
        base_dir=repo_path,
        explicit_checkpoint_path=args.init_checkpoint,
    )
    if transfer_report is not None:
        metadata["initialization_source"]["transfer_report"] = transfer_report
    save_json(artifact_paths["metadata"], metadata)
    save_json(artifact_paths["training_history"], history.history)

    evaluation_summary = summarize_training_history(history)
    save_json(artifact_paths["evaluation_summary"], evaluation_summary)

    registry_path = write_model_registry(repo_path)
    run_manifest = build_run_manifest(
        metadata=metadata,
        artifact_paths=artifact_paths,
        evaluation_summary=evaluation_summary,
        registry_path=registry_path,
    )
    save_json(artifact_paths["run_manifest"], run_manifest)

    print(f"saved_entity_invariance_policy_model={artifact_paths['policy_model']}")
    if args.predict_value:
        print(f"saved_entity_invariance_policy_value_model={artifact_paths['policy_value_model']}")
    if args.predict_turn_outcome or args.predict_value:
        print(f"saved_entity_invariance_training_model={artifact_paths['training_model']}")
    print(f"saved_entity_invariance_metadata={artifact_paths['metadata']}")
    print(f"saved_entity_invariance_run_manifest={artifact_paths['run_manifest']}")
    print(f"updated_model_registry={registry_path}")
    print(f"completed_at_utc={datetime.now(timezone.utc).isoformat()}", flush=True)


if __name__ == "__main__":
    main()
