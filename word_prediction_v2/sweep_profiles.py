from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .policy_config import PolicyConfig, policy_config_from_profile_name, policy_profile_names


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "pokemon-showdown-model-feature" / "dist" / "sim" / "examples" / "statistical-runner.js"
DEFAULT_SWEEP_DIR = REPO_ROOT / "artifacts" / "word_prediction_model" / "sweeps"
SUMMARY_PATTERNS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "configured_games": (r"^Configured Games:\s+(\d+)$", int),
    "configured_concurrency": (r"^Configured Concurrency:\s+(\d+)$", int),
    "completed_games": (r"^Completed Games:\s+(\d+)$", int),
    "rl_wins": (r"^RL Wins:\s+(\d+)$", int),
    "random_wins": (r"^Random Wins:\s+(\d+)$", int),
    "ties": (r"^Ties:\s+(\d+)$", int),
    "failed_games": (r"^Failed Games:\s+(\d+)$", int),
    "timed_out_games": (r"^Timed Out Games:\s+(\d+)$", int),
    "rl_switches": (r"^RL Switches:\s+(\d+)$", int),
    "random_switches": (r"^Random Switches:\s+(\d+)$", int),
    "elapsed_wall_time": (r"^Elapsed Wall Time:\s+(.+)$", str),
    "rl_win_rate_percent": (r"^RL Win Rate:\s+([0-9.]+)%$", float),
    "avg_rl_switches_per_game": (r"^Avg RL Switches/Game:\s+([0-9.]+)$", float),
    "throughput_games_per_min": (r"^Throughput:\s+([0-9.]+) games/min$", float),
    "rl_decisions": (r"^RL Decisions:\s+(\d+)$", int),
    "avg_rl_decision_ms": (r"^Avg RL Decision Time:\s+([0-9.]+) ms", float),
    "avg_model_request_latency_ms": (r"^Avg Model Request Latency:\s+([0-9.]+) ms", float),
    "model_requests": (r"^Model Requests:\s+(\d+)$", int),
    "model_move_choices": (r"^Model Move Choices:\s+(\d+)$", int),
    "model_voluntary_switch_choices": (r"^Model Voluntary Switch Choices:\s+(\d+)$", int),
    "model_force_switch_choices": (r"^Model Force Switch Choices:\s+(\d+)$", int),
    "suppressed_voluntary_switch_opportunities": (r"^Suppressed Voluntary Switch Opportunities:\s+(\d+)$", int),
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "profile"


def _parse_last_summary(output: str) -> dict[str, Any]:
    marker = "===== PARTIAL RESULTS ====="
    end_marker = "==========================="
    start = output.rfind(marker)
    if start < 0:
        raise ValueError("benchmark output did not contain a summary block")
    end = output.find(end_marker, start)
    block = output[start:end] if end >= 0 else output[start:]
    summary: dict[str, Any] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        for key, (pattern, caster) in SUMMARY_PATTERNS.items():
            match = re.match(pattern, line)
            if match:
                summary[key] = caster(match.group(1))
                break
    if "rl_win_rate_percent" not in summary:
        raise ValueError("benchmark summary was missing RL Win Rate")
    return summary


def _write_profile_file(directory: Path, profile_name: str, config: PolicyConfig) -> Path:
    profile_path = directory / f"{_slugify(profile_name)}.json"
    profile_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile_path


def _run_profile(
    *,
    profile_name: str,
    config: PolicyConfig,
    games: int,
    concurrency: int,
    output_dir: Path,
    benchmark_quiet: bool,
    allow_voluntary_switches: bool,
) -> dict[str, Any]:
    profile_path = _write_profile_file(output_dir, profile_name, config)
    log_path = output_dir / f"{_slugify(profile_name)}.log"
    env = os.environ.copy()
    env.update(
        {
            "TOTAL_GAMES": str(games),
            "CONCURRENCY": str(concurrency),
            "BENCHMARK_QUIET": "true" if benchmark_quiet else "false",
            "RL_MODEL_TRANSPORT": "ipc",
            "RL_MODEL_PROFILE": "joint-policy",
            "RL_MODEL_IPC_PYTHON": sys.executable,
            "WORD_POLICY_PROFILE_NAME": "baseline",
            "WORD_POLICY_PROFILE_PATH": str(profile_path),
            "RL_ALLOW_VOLUNTARY_SWITCHES": "true" if allow_voluntary_switches else "false",
        }
    )
    cmd = ["node", str(RUNNER_PATH)]
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    in_summary = False
    saw_summary_end = False
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        if "===== PARTIAL RESULTS =====" in line:
            in_summary = True
        elif in_summary and "===========================" in line:
            saw_summary_end = True
            break
    if saw_summary_end and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    else:
        process.wait(timeout=5)
    combined_output = "".join(lines)
    log_path.write_text(combined_output, encoding="utf-8")
    result: dict[str, Any] = {
        "profile_name": profile_name,
        "profile_path": str(profile_path),
        "log_path": str(log_path),
        "return_code": process.returncode,
    }
    try:
        result["summary"] = _parse_last_summary(combined_output)
        result["error"] = None
    except ValueError as exc:
        result["summary"] = None
        result["error"] = str(exc)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep named word-policy profiles against the statistical benchmark.")
    parser.add_argument("--profiles", nargs="*", default=["baseline", "aggressive_closeout", "anti_loop_light", "safe_conversion"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-voluntary-switches", action="store_true")
    parser.add_argument("--no-benchmark-quiet", action="store_true")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    requested_profiles = args.profiles or ["baseline"]
    unknown = sorted(set(requested_profiles) - set(policy_profile_names()))
    if unknown:
        parser.error(f"unknown profiles: {', '.join(unknown)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_SWEEP_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for profile_name in requested_profiles:
        config = policy_config_from_profile_name(profile_name)
        result = _run_profile(
            profile_name=profile_name,
            config=config,
            games=args.games,
            concurrency=args.concurrency,
            output_dir=output_dir,
            benchmark_quiet=not args.no_benchmark_quiet,
            allow_voluntary_switches=args.allow_voluntary_switches,
        )
        results.append(result)
        summary = result["summary"]
        if summary is None:
            print(f"{profile_name}: ERROR {result['error']}")
        else:
            print(
                f"{profile_name}: {summary['rl_win_rate_percent']:.2f}% "
                f"({summary['rl_wins']}/{summary['completed_games']}) "
                f"failed={summary.get('failed_games', 0)} "
                f"timeouts={summary.get('timed_out_games', 0)}"
            )

    successful = [item for item in results if item["summary"] is not None]
    ranked = sorted(
        successful,
        key=lambda item: (
            item["summary"]["rl_win_rate_percent"],
            -item["summary"].get("failed_games", 0),
            -item["summary"].get("timed_out_games", 0),
        ),
        reverse=True,
    )
    payload = {
        "games": args.games,
        "concurrency": args.concurrency,
        "transport": "ipc",
        "python": sys.executable,
        "results": results,
        "ranked_profile_names": [item["profile_name"] for item in ranked],
        "best_profile_name": ranked[0]["profile_name"] if ranked else None,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
