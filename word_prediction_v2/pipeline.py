from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import time

from .dataset import TrainingCorpus, build_training_corpus
from .lexicon import default_lexicon
from .model import TrainingConfig, WordEmbeddingModel, train_model


DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "word_prediction_model"


@dataclass(frozen=True)
class EvalCase:
    kind: str
    input_tokens: tuple[str, ...]
    expected_word: str


def default_eval_cases() -> List[EvalCase]:
    lexicon = default_lexicon()
    prompt_cases = [
        EvalCase(kind="prompt", input_tokens=tuple(entry.descriptors[:3]), expected_word=entry.canonical)
        for entry in lexicon
    ]
    typo_cases = [
        EvalCase(kind="surface_form", input_tokens=(entry.surface_forms[1],), expected_word=entry.canonical)
        for entry in lexicon
        if len(entry.surface_forms) > 1
    ]
    return prompt_cases + typo_cases


def evaluate_model(model: WordEmbeddingModel, cases: Sequence[EvalCase] | None = None) -> Dict[str, Any]:
    eval_cases = list(cases or default_eval_cases())
    rows: List[Dict[str, Any]] = []
    prompt_correct = 0
    surface_correct = 0
    prompt_total = 0
    surface_total = 0

    for case in eval_cases:
        if case.kind == "prompt":
            ranked = model.predict(case.input_tokens, top_k=3)
            predicted = ranked[0].word if ranked else None
            prompt_total += 1
            prompt_correct += int(predicted == case.expected_word)
        else:
            ranked = model.nearest_words_for_surface_form(case.input_tokens[0], top_k=3)
            predicted = ranked[0].word if ranked else None
            surface_total += 1
            surface_correct += int(predicted == case.expected_word)

        rows.append(
            {
                "kind": case.kind,
                "input": list(case.input_tokens),
                "expected_word": case.expected_word,
                "predicted_word": predicted,
                "top_candidates": [{"word": item.word, "score": item.score} for item in ranked],
                "correct": bool(predicted == case.expected_word),
            }
        )

    return {
        "prompt_accuracy": (prompt_correct / prompt_total) if prompt_total else 0.0,
        "surface_form_accuracy": (surface_correct / surface_total) if surface_total else 0.0,
        "overall_accuracy": ((prompt_correct + surface_correct) / len(rows)) if rows else 0.0,
        "num_cases": len(rows),
        "cases": rows,
    }


def make_run_name(prefix: str = "word_embedding") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def train_and_save(
    *,
    output_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_name: str | None = None,
    repeats_per_entry: int = 72,
    corpus_seed: int = 23,
    config: TrainingConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or TrainingConfig()
    corpus = build_training_corpus(repeats_per_entry=repeats_per_entry, seed=corpus_seed)
    run_id = run_name or make_run_name()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    start = time.perf_counter()
    model = train_model(corpus, cfg)
    duration = time.perf_counter() - start

    evaluation = evaluate_model(model)
    artifacts = _write_artifacts(run_dir, model, corpus, cfg, evaluation, duration, corpus_seed)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "evaluation": evaluation,
        "training_seconds": duration,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_artifacts(
    run_dir: Path,
    model: WordEmbeddingModel,
    corpus: TrainingCorpus,
    config: TrainingConfig,
    evaluation: Dict[str, Any],
    duration: float,
    corpus_seed: int,
) -> Dict[str, str]:
    model_path = run_dir / "model.json"
    metadata_path = run_dir / "training_metadata.json"
    evaluation_path = run_dir / "evaluation_summary.json"
    run_manifest_path = run_dir / "run_manifest.json"

    _write_json(model_path, model.to_serializable())
    _write_json(
        metadata_path,
        {
            "model_family": "word_prediction_embedding_v1",
            "embedding_dim": config.embedding_dim,
            "max_parameters": config.max_parameters,
            "parameter_count": model.parameter_count,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "alignment_weight": config.alignment_weight,
            "l2_weight": config.l2_weight,
            "seed": config.seed,
            "corpus_seed": corpus_seed,
            "num_examples": len(corpus.examples),
            "token_vocab_size": len(corpus.token_vocab),
            "label_vocab_size": len(corpus.label_vocab),
            "ngram_vocab_size": len(corpus.ngram_vocab),
            "training_seconds": duration,
        },
    )
    _write_json(evaluation_path, evaluation)
    _write_json(
        run_manifest_path,
        {
            "artifacts": {
                "model": str(model_path),
                "training_metadata": str(metadata_path),
                "evaluation_summary": str(evaluation_path),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "family": "word_prediction_model",
            "run_dir": str(run_dir),
        },
    )
    return {
        "model": str(model_path),
        "training_metadata": str(metadata_path),
        "evaluation_summary": str(evaluation_path),
        "run_manifest": str(run_manifest_path),
    }


def load_model(path: Path) -> WordEmbeddingModel:
    return WordEmbeddingModel.from_serializable(json.loads(path.read_text(encoding="utf-8")))
