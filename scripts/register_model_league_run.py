#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a model league run into the docs ledger.")
    parser.add_argument("--result-json", required=True, help="Path to the raw league result JSON.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier, for example 2026-04-17_medium_full.")
    parser.add_argument("--label", required=True, help="Short descriptive label.")
    parser.add_argument("--status", default="validated", help="Run status, for example validated or invalid.")
    parser.add_argument("--run-type", default="mixed", help="Run type, for example mixed or vector_only.")
    parser.add_argument("--generated-at", required=True, help="Date string for the run, for example 2026-04-17.")
    parser.add_argument("--scheduled-games", type=int, help="Expected total scheduled games when different from completed.")
    parser.add_argument("--note", action="append", default=[], help="Repeatable note to attach to the run.")
    parser.add_argument("--failure-mode", help="Optional failure mode summary.")
    parser.add_argument("--history-path", default="docs/model_league_history.json")
    parser.add_argument("--latest-path", default="docs/model_league_latest.json")
    parser.add_argument("--set-latest", action="store_true", help="Promote this run to model_league_latest.json.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_run_entry(args: argparse.Namespace, result_payload: dict[str, Any]) -> dict[str, Any]:
    ranking = list(result_payload.get("ranking") or [])
    models = [entry["model"] for entry in ranking if "model" in entry]
    completed = int(result_payload.get("completed", result_payload.get("totalGames", 0)))
    failures = int(result_payload.get("failures", 0))
    scheduled_games = int(args.scheduled_games or completed)
    failure_rate = (failures / scheduled_games) if scheduled_games else 0.0
    return {
        "run_id": args.run_id,
        "label": args.label,
        "status": args.status,
        "run_type": args.run_type,
        "notes": list(args.note),
        "models": models,
        "perPairGames": int(result_payload.get("perPairGames", 0)),
        "scheduledGames": scheduled_games,
        "completed": completed,
        "failures": failures,
        "concurrency": int(result_payload.get("concurrency", 0)),
        "wallSeconds": float(result_payload.get("wallSeconds", 0.0)),
        "ranking": ranking,
        "pairings": list(result_payload.get("pairings") or []),
        "reliability": {
            "failureRate": round(failure_rate, 6),
            "servedCleanly": failures == 0 and completed == scheduled_games,
            "failureMode": args.failure_mode,
        },
    }


def upsert_run(history_payload: dict[str, Any], run_entry: dict[str, Any]) -> dict[str, Any]:
    runs = list(history_payload.get("runs") or [])
    filtered = [entry for entry in runs if entry.get("run_id") != run_entry["run_id"]]
    filtered.append(run_entry)
    filtered.sort(key=lambda entry: str(entry.get("run_id")))
    history_payload["runs"] = filtered
    history_payload["generated_at"] = run_entry["run_id"].split("_", 1)[0]
    return history_payload


def build_latest_payload(run_entry: dict[str, Any], history_path: Path) -> dict[str, Any]:
    return {
        "generated_at": run_entry["run_id"].split("_", 1)[0],
        "label": run_entry["label"],
        "status": run_entry["status"],
        "snapshot_path": str(history_path),
        "run_id": run_entry["run_id"],
        "totalGames": run_entry["scheduledGames"],
        "completed": run_entry["completed"],
        "failures": run_entry["failures"],
        "wallSeconds": run_entry["wallSeconds"],
        "perPairGames": run_entry["perPairGames"],
        "concurrency": run_entry["concurrency"],
        "models": run_entry["models"],
        "ranking": run_entry["ranking"],
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_json)
    history_path = Path(args.history_path)
    latest_path = Path(args.latest_path)

    result_payload = load_json(result_path)
    history_payload = load_json(history_path)
    run_entry = build_run_entry(args, result_payload)
    updated_history = upsert_run(history_payload, run_entry)
    write_json(history_path, updated_history)

    if args.set_latest:
        write_json(latest_path, build_latest_payload(run_entry, history_path))


if __name__ == "__main__":
    main()
