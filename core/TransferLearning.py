from __future__ import annotations

"""Reusable helpers for transfer-learning and warm-start metadata.

This module stays intentionally framework-light. It does not know how to build
or load a neural network. Instead, it standardizes the metadata and path
decisions needed to initialize a run from:

- scratch
- another release in the same family/version line
- a different family entirely

The main training scripts can use these helpers to keep initialization
conventions consistent across generations.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ARTIFACT_PATH_KEYS: tuple[str, ...] = (
    "policy_model_path",
    "policy_value_model_path",
    "training_model_path",
    "policy_vocab_path",
    "action_context_vocab_path",
    "entity_token_vocab_path",
    "reward_profile_path",
    "evaluation_summary_path",
    "run_manifest_path",
    "metadata_path",
)


@dataclass(frozen=True)
class ReleaseIdentity:
    """Compact identity record for one trained release."""

    family_id: str | None
    family_version: int | None
    model_release_id: str | None
    parent_release_id: str | None = None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ReleaseIdentity":
        family_id = metadata.get("family_id")
        family_version = metadata.get("family_version")
        model_release_id = metadata.get("model_release_id")
        parent_release_id = metadata.get("parent_release_id")
        return cls(
            family_id=str(family_id) if family_id is not None else None,
            family_version=int(family_version) if family_version is not None else None,
            model_release_id=str(model_release_id) if model_release_id is not None else None,
            parent_release_id=str(parent_release_id) if parent_release_id is not None else None,
        )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitializationPlan:
    """Normalized description of how a run should be initialized."""

    mode: str
    relationship: str
    source: ReleaseIdentity | None = None
    target: ReleaseIdentity | None = None
    checkpoint_path: str | None = None
    artifact_key: str | None = None
    notes: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.source is not None:
            payload["source"] = self.source.to_metadata()
        if self.target is not None:
            payload["target"] = self.target.to_metadata()
        return payload


def collect_artifact_paths(metadata: Mapping[str, Any], keys: Iterable[str] = ARTIFACT_PATH_KEYS) -> dict[str, str]:
    """Return the non-empty artifact paths from a metadata dictionary."""

    paths: dict[str, str] = {}
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            paths[key] = text
    return paths


def resolve_artifact_path(base_dir: Path | str, raw_path: str | None) -> Path | None:
    """Resolve a possibly-relative artifact path against a base directory."""

    if raw_path is None:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def select_primary_checkpoint_path(metadata: Mapping[str, Any], *, base_dir: Path | str | None = None) -> Path | None:
    """Select the most useful artifact path for transfer initialization.

    Preference order favors trainable model checkpoints first, then serving
    artifacts, then any recorded artifact path.
    """

    paths = collect_artifact_paths(metadata)
    preferred_keys = (
        "training_model_path",
        "policy_value_model_path",
        "policy_model_path",
    )
    for key in preferred_keys:
        raw_path = paths.get(key)
        if not raw_path:
            continue
        return resolve_artifact_path(base_dir or ".", raw_path)

    for raw_path in paths.values():
        resolved = resolve_artifact_path(base_dir or ".", raw_path)
        if resolved is not None:
            return resolved
    return None


def same_family(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Check whether two metadata dictionaries belong to the same family."""

    left_family = left.get("family_id")
    right_family = right.get("family_id")
    if left_family is None or right_family is None:
        return False
    return str(left_family) == str(right_family)


def same_generation(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Check whether two releases share family and version."""

    return same_family(left, right) and left.get("family_version") == right.get("family_version")


def build_initialization_source(
    target_metadata: Mapping[str, Any],
    *,
    source_metadata: Mapping[str, Any] | None = None,
    base_dir: Path | str | None = None,
    explicit_checkpoint_path: str | Path | None = None,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    """Build the standardized initialization-source payload.

    If no source metadata or checkpoint is provided, the result is a scratch
    initialization. If a source release is supplied, the function classifies it
    as same-generation warm-start or cross-generation transfer.
    """

    target = ReleaseIdentity.from_metadata(target_metadata)
    source = ReleaseIdentity.from_metadata(source_metadata) if source_metadata is not None else None

    if explicit_checkpoint_path is not None:
        checkpoint_path = Path(explicit_checkpoint_path)
        if not checkpoint_path.is_absolute() and base_dir is not None:
            checkpoint_path = Path(base_dir) / checkpoint_path
        return InitializationPlan(
            mode="checkpoint",
            relationship="explicit_checkpoint",
            source=source,
            target=target,
            checkpoint_path=str(checkpoint_path),
            artifact_key=artifact_key,
            notes=("explicit checkpoint path provided",),
        ).to_metadata()

    if source is None:
        return InitializationPlan(
            mode="scratch",
            relationship="none",
            source=None,
            target=target,
            checkpoint_path=None,
            artifact_key=None,
            notes=("no source release supplied",),
        ).to_metadata()

    checkpoint = select_primary_checkpoint_path(source_metadata, base_dir=base_dir)
    if checkpoint is None:
        return InitializationPlan(
            mode="scratch",
            relationship="source_missing_artifact",
            source=source,
            target=target,
            checkpoint_path=None,
            artifact_key=None,
            notes=("source metadata did not expose a usable checkpoint path",),
        ).to_metadata()

    if same_generation(target_metadata, source_metadata):
        relationship = "same_generation_warm_start"
    elif same_family(target_metadata, source_metadata):
        relationship = "same_family_cross_generation"
    else:
        relationship = "cross_family_transfer"

    return InitializationPlan(
        mode="transfer",
        relationship=relationship,
        source=source,
        target=target,
        checkpoint_path=str(checkpoint),
        artifact_key=artifact_key or _infer_artifact_key(source_metadata),
        notes=(
            "source release supplied",
            "checkpoint resolved from source metadata",
        ),
    ).to_metadata()


def describe_transfer(metadata: Mapping[str, Any]) -> str:
    """Produce a compact human-readable description of initialization metadata."""

    init = metadata.get("initialization_source") or {}
    mode = init.get("mode") or "unknown"
    relationship = init.get("relationship") or "unknown"
    source = init.get("source") or {}
    source_label = source.get("model_release_id") or source.get("family_id") or "none"
    return f"{mode}:{relationship}:{source_label}"


def apply_transfer_metadata(
    metadata: Mapping[str, Any],
    *,
    source_metadata: Mapping[str, Any] | None = None,
    base_dir: Path | str | None = None,
    explicit_checkpoint_path: str | Path | None = None,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    """Return a copy of metadata with transfer-learning fields filled in."""

    payload = dict(metadata)
    target_identity = ReleaseIdentity.from_metadata(payload)
    if target_identity.model_release_id is None and payload.get("model_name") is not None:
        payload["model_release_id"] = str(payload["model_name"])
        target_identity = ReleaseIdentity.from_metadata(payload)

    if payload.get("parent_release_id") is None and source_metadata is not None:
        source_identity = ReleaseIdentity.from_metadata(source_metadata)
        payload["parent_release_id"] = source_identity.model_release_id

    payload["initialization_source"] = build_initialization_source(
        payload,
        source_metadata=source_metadata,
        base_dir=base_dir,
        explicit_checkpoint_path=explicit_checkpoint_path,
        artifact_key=artifact_key,
    )
    return payload


def _infer_artifact_key(metadata: Mapping[str, Any]) -> str | None:
    """Choose the most likely artifact field for a source release."""

    for key in ("training_model_path", "policy_value_model_path", "policy_model_path"):
        value = metadata.get(key)
        if value:
            return key
    for key in ARTIFACT_PATH_KEYS:
        if metadata.get(key):
            return key
    return None
