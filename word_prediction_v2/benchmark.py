from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a saved word embedding run.")
    parser.add_argument("run_dir", type=Path, help="Path to a saved run directory.")
    parser.add_argument("--show-failures", action="store_true", help="Print incorrect cases when present.")
    return parser


def summarize(run_dir: Path) -> Dict[str, Any]:
    evaluation = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))
    failures = [case for case in evaluation["cases"] if not case["correct"]]
    return {
        "run_dir": str(run_dir),
        "model_family": metadata["model_family"],
        "embedding_dim": metadata["embedding_dim"],
        "epochs": metadata["epochs"],
        "num_examples": metadata["num_examples"],
        "training_ticks": int(round(float(metadata["training_seconds"]) / 5.0)),
        "overall_accuracy": evaluation["overall_accuracy"],
        "prompt_accuracy": evaluation["prompt_accuracy"],
        "surface_form_accuracy": evaluation["surface_form_accuracy"],
        "num_cases": evaluation["num_cases"],
        "num_failures": len(failures),
        "failures": failures,
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(args.run_dir)
    print(f"Run Dir: {summary['run_dir']}")
    print(f"Model Family: {summary['model_family']}")
    print(f"Embedding Dim: {summary['embedding_dim']}")
    print(f"Epochs: {summary['epochs']}")
    print(f"Examples: {summary['num_examples']}")
    print(f"Training Ticks: {summary['training_ticks']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")
    print(f"Prompt Accuracy: {summary['prompt_accuracy']:.3f}")
    print(f"Surface Accuracy: {summary['surface_form_accuracy']:.3f}")
    print(f"Cases: {summary['num_cases']}")
    print(f"Failures: {summary['num_failures']}")

    if args.show_failures and summary["failures"]:
        print("Failure Cases:")
        for case in summary["failures"]:
            joined = " ".join(case["input"])
            print(
                f"  kind={case['kind']} input={joined} expected={case['expected_word']} "
                f"predicted={case['predicted_word']}"
            )


if __name__ == "__main__":
    main()
