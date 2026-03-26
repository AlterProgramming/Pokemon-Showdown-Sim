from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence


REGISTRY_FILENAME = "model_registry.json"


def metadata_file_candidates(artifacts_dir: Path) -> list[Path]:
    return sorted(artifacts_dir.glob("training_metadata*.json"))


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


def resolve_artifact_path(repo_path: Path, metadata_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

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
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
            "metadata_path": str(metadata_path.relative_to(repo_path)),
            "policy_model_path": str(policy_model_path),
            "policy_vocab_path": str(policy_vocab_path),
            "policy_value_model_path": metadata.get("policy_value_model_path"),
            "training_model_path": metadata.get("training_model_path"),
            "action_space": metadata.get("action_space"),
            "predict_value": bool(metadata.get("predict_value")),
            "predict_turn_outcome": bool(metadata.get("predict_turn_outcome")),
            "feature_dim": metadata.get("feature_dim"),
            "num_action_classes": metadata.get("num_action_classes"),
        }

    default_model_id = "model1" if "model1" in models else (sorted(models.keys())[0] if models else None)
    return {
        "repo_path": str(repo_path),
        "registry_path": str((artifacts_dir / REGISTRY_FILENAME).relative_to(repo_path)),
        "default_model_id": default_model_id,
        "models": dict(sorted(models.items())),
    }


def write_model_registry(repo_path: Path) -> Path:
    repo_path = repo_path.resolve()
    registry = build_model_registry(repo_path)
    registry_path = repo_path / "artifacts" / REGISTRY_FILENAME
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry_path
