from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ModelRegistry import enrich_training_metadata_recipe_fields, metadata_file_candidates, resolve_artifact_path, write_model_registry
from train_policy import build_run_manifest, save_json


def default_companion_path(metadata_path: Path, prefix: str) -> Path:
    metadata_name = metadata_path.name
    if metadata_name.startswith("training_metadata"):
        suffix = metadata_name[len("training_metadata"):]
        return metadata_path.with_name(f"{prefix}{suffix}")
    return metadata_path.with_name(f"{prefix}_{metadata_name}")


def backfill_evaluation_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "backfilled_without_training_history",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": int(metadata.get("epochs_completed") or 0),
        "train_examples": metadata.get("train_examples"),
        "val_examples": metadata.get("val_examples"),
        "metrics": {},
        "note": (
            "Historical run did not save per-epoch history. "
            "This summary was backfilled from available metadata only."
        ),
    }


def backfill_repo(repo_path: Path) -> dict[str, int]:
    repo_path = repo_path.resolve()
    artifacts_dir = repo_path / "artifacts"
    updated_metadata_files = 0
    created_evaluation_summaries = 0
    created_run_manifests = 0

    for metadata_path in metadata_file_candidates(artifacts_dir):
        metadata = enrich_training_metadata_recipe_fields(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )

        evaluation_summary_path = metadata.get("evaluation_summary_path")
        if evaluation_summary_path:
            evaluation_summary_file = resolve_artifact_path(repo_path, metadata_path, str(evaluation_summary_path))
        else:
            evaluation_summary_file = default_companion_path(metadata_path, "evaluation_summary")
            metadata["evaluation_summary_path"] = str(evaluation_summary_file.relative_to(repo_path))

        run_manifest_path = metadata.get("run_manifest_path")
        if run_manifest_path:
            run_manifest_file = resolve_artifact_path(repo_path, metadata_path, str(run_manifest_path))
        else:
            run_manifest_file = default_companion_path(metadata_path, "run_manifest")
            metadata["run_manifest_path"] = str(run_manifest_file.relative_to(repo_path))

        if not evaluation_summary_file.exists():
            save_json(evaluation_summary_file, backfill_evaluation_summary(metadata))
            created_evaluation_summaries += 1

        evaluation_summary = json.loads(evaluation_summary_file.read_text(encoding="utf-8"))

        artifact_paths: dict[str, Path] = {
            "metadata": metadata_path,
            "evaluation_summary": evaluation_summary_file,
            "run_manifest": run_manifest_file,
        }
        for field_name in (
            "policy_model_path",
            "policy_vocab_path",
            "policy_value_model_path",
            "training_model_path",
            "action_context_vocab_path",
            "reward_profile_path",
        ):
            raw_path = metadata.get(field_name)
            if raw_path:
                artifact_name = field_name.removesuffix("_path")
                artifact_paths[artifact_name] = resolve_artifact_path(repo_path, metadata_path, str(raw_path))

        metadata["evaluation_summary_path"] = str(evaluation_summary_file.relative_to(repo_path))
        metadata["run_manifest_path"] = str(run_manifest_file.relative_to(repo_path))
        save_json(metadata_path, metadata)
        updated_metadata_files += 1

        if not run_manifest_file.exists():
            save_json(
                run_manifest_file,
                build_run_manifest(
                    metadata=metadata,
                    artifact_paths=artifact_paths,
                    evaluation_summary=evaluation_summary,
                    registry_path=repo_path / "artifacts" / "model_registry.json",
                ),
            )
            created_run_manifests += 1

    write_model_registry(repo_path)
    return {
        "updated_metadata_files": updated_metadata_files,
        "created_evaluation_summaries": created_evaluation_summaries,
        "created_run_manifests": created_run_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill governance artifacts for historical training runs.")
    parser.add_argument(
        "--repo-path",
        default=Path(__file__).resolve().parent,
        help="Repo root containing the artifacts directory.",
    )
    args = parser.parse_args()

    stats = backfill_repo(Path(args.repo_path))
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
