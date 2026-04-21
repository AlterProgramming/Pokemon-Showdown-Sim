from __future__ import annotations

import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, Sequence


REGISTRY_FILENAME = "model_registry.json"


def metadata_file_candidates(artifacts_dir: Path) -> list[Path]:
    return sorted(artifacts_dir.rglob("training_metadata*.json"))


def parse_model_id_list(raw_value: str | None) -> list[str]:
    if raw_value is None or not raw_value.strip():
        return []

    model_ids: list[str] = []
    seen: set[str] = set()
    for raw_entry in raw_value.split(","):
        model_id = raw_entry.strip()
        if not model_id or model_id in seen:
            continue
        model_ids.append(model_id)
        seen.add(model_id)
    return model_ids


def select_registered_models(
    registered_models: dict[str, Any],
    *,
    mode: str,
    requested_model_ids: Sequence[str] | None = None,
) -> list[str]:
    requested = list(requested_model_ids or [])
    if mode != "multi" and requested:
        raise ValueError("--model-ids can only be used together with --mode multi")

    selected_ids = requested if requested else ([mode] if mode != "multi" else list(registered_models.keys()))
    missing = [model_id for model_id in selected_ids if model_id not in registered_models]
    if missing:
        raise KeyError(
            f"Unknown model_id(s): {', '.join(missing)}. "
            f"Supported values: {', '.join(sorted(registered_models))}"
        )
    return selected_ids


def model_id_from_name(raw_name: str) -> str:
    stem = Path(raw_name).stem
    if stem.startswith("model_"):
        return f"model{stem[len('model_'):]}"
    return stem


def infer_action_space(metadata: dict[str, Any]) -> str:
    existing = metadata.get("action_space")
    if existing:
        return str(existing)

    include_switches = metadata.get("include_switches")
    if include_switches is True:
        return "joint"
    if include_switches is False:
        return "move_only"
    return "unknown"


def infer_action_parameterization(metadata: dict[str, Any]) -> str:
    existing = metadata.get("action_parameterization")
    if existing:
        return str(existing)

    action_space = infer_action_space(metadata)
    if action_space == "joint":
        return "joint_vocab"
    if action_space == "move_only":
        return "move_vocab"
    return "unknown"


def infer_objective_set(metadata: dict[str, Any]) -> list[str]:
    existing = metadata.get("objective_set")
    if isinstance(existing, list) and existing:
        return [str(item) for item in existing]

    objective_set = ["policy"]
    if bool(metadata.get("predict_turn_outcome")):
        objective_set.append("transition")
    if bool(metadata.get("predict_value")):
        objective_set.append("value")
    return objective_set


def infer_family_identity(metadata: dict[str, Any]) -> tuple[str, int]:
    existing_family_id = metadata.get("family_id")
    existing_family_version = metadata.get("family_version")
    if existing_family_id and existing_family_version is not None:
        return str(existing_family_id), int(existing_family_version)

    action_space = infer_action_space(metadata)
    predict_turn_outcome = bool(metadata.get("predict_turn_outcome"))
    predict_value = bool(metadata.get("predict_value"))

    if action_space == "joint":
        if predict_turn_outcome and predict_value:
            return "vector_joint_bc_transition_value", 1
        if predict_turn_outcome:
            return "vector_joint_bc_transition", 1
        if predict_value:
            return "vector_joint_bc_value", 1
        return "vector_joint_bc", 1

    if action_space == "move_only":
        if predict_turn_outcome and predict_value:
            return "vector_move_bc_transition_value", 1
        if predict_turn_outcome:
            return "vector_move_bc_transition", 1
        if predict_value:
            return "vector_move_bc_value", 1
        return "vector_move_bc", 1

    return "unknown_family", 1


def infer_training_regime(metadata: dict[str, Any]) -> str:
    existing = metadata.get("training_regime")
    if existing:
        return str(existing)

    objective_set = infer_objective_set(metadata)
    has_aux = len(objective_set) > 1
    weighting_mode = str(metadata.get("policy_return_weighting") or "none")

    if has_aux and weighting_mode != "none":
        return "offline_bc_aux_weighted"
    if has_aux:
        return "offline_bc_aux"
    if weighting_mode != "none":
        return "offline_bc_weighted"
    return "offline_bc"


def infer_reward_definition_id(metadata: dict[str, Any]) -> str:
    existing = metadata.get("reward_definition_id")
    if existing:
        return str(existing)

    reward_config = metadata.get("reward_config") or {}
    if any(key in reward_config for key in ("redundant_setup_penalty", "wasted_setup_penalty", "return_discount")):
        return "dense_reward_v2"
    return "dense_reward_v1"


def enrich_training_metadata_recipe_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    # Older vector-family runs used the simpler model_path / vocab_path names before we
    # standardized on policy_* artifact fields. Keep them discoverable so generation-1
    # baselines remain benchmarkable alongside newer families.
    payload.setdefault("policy_model_path", payload.get("model_path"))
    payload.setdefault("policy_vocab_path", payload.get("vocab_path"))
    if not payload.get("policy_vocab_path") and payload.get("entity_token_vocab_path"):
        payload["policy_vocab_path"] = payload.get("entity_token_vocab_path")

    family_id, family_version = infer_family_identity(payload)
    payload["family_id"] = family_id
    payload["family_version"] = int(family_version)
    payload.setdefault("family_name", f"{family_id}_v{family_version}")

    model_name = payload.get("model_name") or payload.get("policy_model_path") or "model"
    payload.setdefault("model_release_id", model_id_from_name(str(model_name)))
    payload.setdefault("parent_release_id", None)
    payload["training_regime"] = infer_training_regime(payload)
    payload.setdefault("information_policy", "public_only")
    payload.setdefault("state_schema_version", "flat_public_v1")
    payload.setdefault("entity_schema_version", None)
    payload.setdefault("history_schema_version", None)
    payload["action_parameterization"] = infer_action_parameterization(payload)
    payload["objective_set"] = infer_objective_set(payload)
    payload["reward_definition_id"] = infer_reward_definition_id(payload)

    if payload.get("value_target_definition") is None:
        value_target = payload.get("value_target")
        if value_target is not None:
            payload["value_target_definition"] = str(value_target)
        elif bool(payload.get("predict_value")):
            payload["value_target_definition"] = "terminal_win_probability"

    if payload.get("transition_target_definition") is None and bool(payload.get("predict_turn_outcome")):
        payload["transition_target_definition"] = "public_turn_outcome_summary_v1"

    if payload.get("policy_weighting_definition") is None:
        weighting_mode = payload.get("policy_return_weighting")
        if weighting_mode is not None:
            payload["policy_weighting_definition"] = {
                "mode": str(weighting_mode),
                "scale": payload.get("policy_return_weight_scale"),
                "min_weight": payload.get("policy_return_weight_min"),
                "max_weight": payload.get("policy_return_weight_max"),
            }

    payload.setdefault("initialization_source", {"type": "scratch"})
    payload.setdefault("evaluation_bundle_id", "offline_validation_v1")
    payload.setdefault("registry_visibility", "runnable_policy")
    return payload


def resolve_artifact_path(repo_path: Path, metadata_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    looks_posix_absolute = str(raw_path).startswith("/")
    absolute_parts = candidate.parts
    if looks_posix_absolute and not candidate.is_absolute():
        absolute_parts = PurePosixPath(str(raw_path)).parts

    if candidate.is_absolute() or looks_posix_absolute:
        if candidate.is_absolute() and candidate.exists():
            return candidate

        parts_lower = [part.lower() for part in absolute_parts]
        if "artifacts" in parts_lower:
            artifacts_index = parts_lower.index("artifacts")
            relative_from_artifacts = Path(*absolute_parts[artifacts_index:])
            repo_candidate = (repo_path / relative_from_artifacts).resolve()
            if repo_candidate.exists():
                return repo_candidate

        metadata_sibling_candidate = (metadata_path.parent / candidate.name).resolve()
        if metadata_sibling_candidate.exists():
            return metadata_sibling_candidate

        return metadata_sibling_candidate

    repo_candidate = (repo_path / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    metadata_candidate = (metadata_path.parent / candidate).resolve()
    if metadata_candidate.exists():
        return metadata_candidate

    return repo_candidate


def build_model_registry(repo_path: Path) -> dict[str, Any]:
    repo_path = repo_path.resolve()
    artifacts_dir = repo_path / "artifacts"
    models: Dict[str, Dict[str, Any]] = {}

    for metadata_path in metadata_file_candidates(artifacts_dir):
        metadata = enrich_training_metadata_recipe_fields(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
        registry_visibility = str(metadata.get("registry_visibility") or "runnable_policy")
        if registry_visibility not in {"runnable_policy", "runnable_policy_value"}:
            continue
        policy_model_path = metadata.get("policy_model_path")
        policy_vocab_path = metadata.get("policy_vocab_path")
        if not policy_model_path or not policy_vocab_path:
            continue

        resolved_model_path = resolve_artifact_path(repo_path, metadata_path, str(policy_model_path))
        resolved_vocab_path = resolve_artifact_path(repo_path, metadata_path, str(policy_vocab_path))
        if not resolved_model_path.exists() or not resolved_vocab_path.exists():
            continue

        model_name = metadata.get("model_name") or Path(str(policy_model_path)).stem
        model_id = model_id_from_name(str(model_name))
        models[model_id] = {
            "model_id": model_id,
            "model_name": metadata.get("model_name"),
            "model_variant": metadata.get("model_variant"),
            "metadata_path": metadata_path.relative_to(repo_path).as_posix(),
            "policy_model_path": str(policy_model_path),
            "policy_vocab_path": str(policy_vocab_path),
            "policy_value_model_path": metadata.get("policy_value_model_path"),
            "training_model_path": metadata.get("training_model_path"),
            "evaluation_summary_path": metadata.get("evaluation_summary_path"),
            "run_manifest_path": metadata.get("run_manifest_path"),
            "action_space": metadata.get("action_space"),
            "predict_value": bool(metadata.get("predict_value")),
            "predict_turn_outcome": bool(metadata.get("predict_turn_outcome")),
            "feature_dim": metadata.get("feature_dim"),
            "num_action_classes": metadata.get("num_action_classes"),
            "model_release_id": metadata.get("model_release_id"),
            "parent_release_id": metadata.get("parent_release_id"),
            "family_id": metadata.get("family_id"),
            "family_version": metadata.get("family_version"),
            "family_name": metadata.get("family_name"),
            "training_regime": metadata.get("training_regime"),
            "information_policy": metadata.get("information_policy"),
            "action_parameterization": metadata.get("action_parameterization"),
            "objective_set": metadata.get("objective_set"),
            "policy_return_weighting": metadata.get("policy_return_weighting"),
            "policy_return_weight_scale": metadata.get("policy_return_weight_scale"),
            "policy_return_weight_min": metadata.get("policy_return_weight_min"),
            "policy_return_weight_max": metadata.get("policy_return_weight_max"),
            "policy_weighting_definition": metadata.get("policy_weighting_definition"),
            "policy_weight_stats": metadata.get("policy_weight_stats"),
            "switch_logit_bias": metadata.get("switch_logit_bias"),
            "action_selection": metadata.get("action_selection"),
            "value_target_definition": metadata.get("value_target_definition"),
            "transition_target_definition": metadata.get("transition_target_definition"),
            "registry_visibility": registry_visibility,
        }

    default_model_id = "model1" if "model1" in models else (sorted(models.keys())[0] if models else None)
    return {
        "repo_path": str(repo_path),
        "registry_path": (artifacts_dir / REGISTRY_FILENAME).relative_to(repo_path).as_posix(),
        "default_model_id": default_model_id,
        "models": dict(sorted(models.items())),
    }


def write_model_registry(repo_path: Path) -> Path:
    repo_path = repo_path.resolve()
    registry = build_model_registry(repo_path)
    registry_path = repo_path / "artifacts" / REGISTRY_FILENAME
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry_path
