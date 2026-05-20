from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .loss_analysis import analyze_replay_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = PROJECT_ROOT / "pokemon-showdown-model-feature"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "word_prediction_model" / "loss_replays"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark and analyze captured loss replays.")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--capture-count", type=int, default=50)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5010/predict")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.html"):
        path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "TOTAL_GAMES": str(args.games),
            "CONCURRENCY": str(args.concurrency),
            "RL_ALLOW_VOLUNTARY_SWITCHES": "false",
            "RL_MODEL_ENDPOINT": args.endpoint,
            "REPLAY_CAPTURE_MODE": "loss",
            "REPLAY_CAPTURE_COUNT": str(args.capture_count),
            "REPLAY_OUTPUT_DIR": str(output_dir),
        }
    )
    cmd = [
        "node",
        "dist/sim/examples/statistical-runner.js",
        "--concurrency",
        str(args.concurrency),
        "--battle-timeout-ms",
        "180000",
        "--rl-model-id",
        "word_policy_v1",
        "--rl-model-profile",
        "joint-policy",
    ]
    subprocess.run(cmd, cwd=RUNNER_DIR, env=env, check=True)

    summary = analyze_replay_files(output_dir.glob("*.html"))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
