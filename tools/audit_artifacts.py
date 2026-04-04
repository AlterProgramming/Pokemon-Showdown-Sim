from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ModelRegistry import (
    enrich_training_metadata_recipe_fields,
    metadata_file_candidates,
    resolve_artifact_path,
)


def expected_artifact_paths(repo_path: Path, metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    artifact_fields = [
        "policy_model_path",
        "policy_vocab_path",
        "policy_value_model_path",
        "training_model_path",
        "action_context_vocab_path",
        "reward_profile_path",
        "evaluation_summary_path",
        "run_manifest_path",
    ]
    paths: dict[str, Path] = {}
    for field in artifact_fields:
        raw_path = metadata.get(field)
        if raw_path:
            paths[field] = resolve_artifact_path(repo_path, metadata_path, str(raw_path))
    return paths


def audit_artifacts(repo_path: Path) -> dict[str, Any]:
    repo_path = repo_path.resolve()
    artifacts_dir = repo_path / "artifacts"
    governance_fields = ("evaluation_summary_path", "run_manifest_path")

    releases: list[dict[str, Any]] = []
    for metadata_path in metadata_file_candidates(artifacts_dir):
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = enrich_training_metadata_recipe_fields(raw_metadata)
        artifact_paths = expected_artifact_paths(repo_path, metadata_path, metadata)
        missing_artifacts = sorted(
            field_name
            for field_name, path in artifact_paths.items()
            if not path.exists()
        )
        missing_governance_artifacts = sorted(
            field_name
            for field_name in governance_fields
            if field_name not in artifact_paths or not artifact_paths[field_name].exists()
        )
        releases.append(
            {
                "metadata_path": str(metadata_path.relative_to(repo_path)),
                "model_release_id": metadata.get("model_release_id"),
                "family_id": metadata.get("family_id"),
                "family_version": metadata.get("family_version"),
                "training_regime": metadata.get("training_regime"),
                "registry_visibility": metadata.get("registry_visibility"),
                "missing_artifacts": missing_artifacts,
                "missing_governance_artifacts": missing_governance_artifacts,
                "has_run_manifest": (
                    "run_manifest_path" in artifact_paths
                    and artifact_paths["run_manifest_path"].exists()
                ),
                "has_evaluation_summary": (
                    "evaluation_summary_path" in artifact_paths
                    and artifact_paths["evaluation_summary_path"].exists()
                ),
            }
        )

    releases_sorted = sorted(releases, key=lambda entry: str(entry.get("model_release_id") or ""))
    summary = {
        "repo_path": str(repo_path),
        "artifacts_dir": str(artifacts_dir),
        "release_count": int(len(releases_sorted)),
        "releases_with_missing_artifacts": int(sum(1 for entry in releases_sorted if entry["missing_artifacts"])),
        "releases_missing_governance_artifacts": int(
            sum(1 for entry in releases_sorted if entry["missing_governance_artifacts"])
        ),
        "releases_with_run_manifest": int(sum(1 for entry in releases_sorted if entry["has_run_manifest"])),
        "releases_with_evaluation_summary": int(sum(1 for entry in releases_sorted if entry["has_evaluation_summary"])),
        "releases": releases_sorted,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit training artifacts for governance completeness.")
    parser.add_argument(
        "--repo-path",
        default=Path(__file__).resolve().parent,
        help="Repo root containing the artifacts directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full audit report as JSON.",
    )
    args = parser.parse_args()

    report = audit_artifacts(Path(args.repo_path))
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"release_count={report['release_count']}")
    print(f"releases_with_missing_artifacts={report['releases_with_missing_artifacts']}")
    print(f"releases_missing_governance_artifacts={report['releases_missing_governance_artifacts']}")
    print(f"releases_with_run_manifest={report['releases_with_run_manifest']}")
    print(f"releases_with_evaluation_summary={report['releases_with_evaluation_summary']}")
    for release in report["releases"]:
        missing = ",".join(release["missing_artifacts"]) if release["missing_artifacts"] else "(none)"
        missing_governance = (
            ",".join(release["missing_governance_artifacts"])
            if release["missing_governance_artifacts"]
            else "(none)"
        )
        print(
            "release "
            f"model_release_id={release['model_release_id']} "
            f"family_id={release['family_id']} "
            f"training_regime={release['training_regime']} "
            f"missing_artifacts={missing} "
            f"missing_governance_artifacts={missing_governance}"
        )


if __name__ == "__main__":
    main()
