from __future__ import annotations

import argparse
from pathlib import Path

from .model import TrainingConfig
from .pipeline import DEFAULT_ARTIFACT_ROOT, train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the misspelling-aware word embedding model.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--repeats-per-entry", type=int, default=72)
    parser.add_argument("--corpus-seed", type=int, default=23)
    parser.add_argument("--embedding-dim", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--alignment-weight", type=float, default=0.45)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_and_save(
        output_root=args.output_root,
        run_name=args.run_name,
        repeats_per_entry=args.repeats_per_entry,
        corpus_seed=args.corpus_seed,
        config=TrainingConfig(
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            alignment_weight=args.alignment_weight,
            l2_weight=args.l2_weight,
            seed=args.seed,
        ),
    )

    evaluation = result["evaluation"]
    print(f"run_id={result['run_id']}")
    print(f"run_dir={result['run_dir']}")
    print(f"training_seconds={result['training_seconds']:.3f}")
    print(f"overall_accuracy={evaluation['overall_accuracy']:.3f}")
    print(f"prompt_accuracy={evaluation['prompt_accuracy']:.3f}")
    print(f"surface_form_accuracy={evaluation['surface_form_accuracy']:.3f}")
    for name, path in result["artifacts"].items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
