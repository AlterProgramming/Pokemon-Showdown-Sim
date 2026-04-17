#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_SERVICES = [
    ("vector_multi", "http://127.0.0.1:5000/health"),
    ("entity_benchmark", "http://127.0.0.1:5001/health"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect service health snapshots for league-serving endpoints.")
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help="Repeatable service definition in the form name=url. Defaults to the standard local endpoints.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--latest-path", default="docs/model_service_health_latest.json")
    parser.add_argument("--history-path", default="docs/model_service_health_history.json")
    parser.add_argument("--generated-at", help="Optional timestamp/date override.")
    parser.add_argument("--max-history", type=int, default=200)
    return parser.parse_args()


def parse_services(raw_values: list[str]) -> list[tuple[str, str]]:
    if not raw_values:
        return list(DEFAULT_SERVICES)
    services: list[tuple[str, str]] = []
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"invalid --service '{raw}', expected name=url")
        name, url = raw.split("=", 1)
        services.append((name.strip(), url.strip()))
    return services


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_service(name: str, url: str, timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            payload = json.loads(body)
            healthy = 200 <= response.status < 300 and payload.get("status") in {"ok", "degraded"}
            return {
                "name": name,
                "url": url,
                "httpStatus": int(response.status),
                "latencyMs": latency_ms,
                "status": str(payload.get("status", "unknown")),
                "healthy": healthy and payload.get("status") == "ok",
                "payload": payload,
                "error": None,
            }
    except error.HTTPError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return {
            "name": name,
            "url": url,
            "httpStatus": int(exc.code),
            "latencyMs": latency_ms,
            "status": "http_error",
            "healthy": False,
            "payload": None,
            "error": str(exc),
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return {
            "name": name,
            "url": url,
            "httpStatus": None,
            "latencyMs": latency_ms,
            "status": "unreachable",
            "healthy": False,
            "payload": None,
            "error": str(exc),
        }


def build_snapshot(args: argparse.Namespace, services: list[tuple[str, str]]) -> dict[str, Any]:
    entries = [collect_service(name, url, args.timeout_seconds) for name, url in services]
    healthy_services = sum(1 for entry in entries if entry["healthy"])
    degraded_services = sum(1 for entry in entries if not entry["healthy"])
    if not entries:
        overall_status = "no_snapshot"
    elif degraded_services == 0:
        overall_status = "ok"
    elif healthy_services == 0:
        overall_status = "degraded"
    else:
        overall_status = "partial"
    return {
        "generated_at": args.generated_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": overall_status,
        "summary": {
            "totalServices": len(entries),
            "healthyServices": healthy_services,
            "degradedServices": degraded_services,
        },
        "services": entries,
    }


def append_history(history_payload: dict[str, Any], snapshot: dict[str, Any], max_history: int) -> dict[str, Any]:
    snapshots = list(history_payload.get("snapshots") or [])
    snapshots.append(snapshot)
    snapshots = snapshots[-max_history:]
    return {
        "generated_at": snapshot["generated_at"],
        "snapshots": snapshots,
    }


def main() -> None:
    args = parse_args()
    services = parse_services(args.service)
    latest_path = Path(args.latest_path)
    history_path = Path(args.history_path)

    snapshot = build_snapshot(args, services)
    write_json(latest_path, snapshot)

    history_payload = load_json(history_path)
    updated_history = append_history(history_payload, snapshot, args.max_history)
    write_json(history_path, updated_history)


if __name__ == "__main__":
    main()
