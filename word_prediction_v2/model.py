from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .dataset import TrainingCorpus, TrainingExample, encode_prompt, encode_surface_form


@dataclass
class TrainingConfig:
    embedding_dim: int = 8
    epochs: int = 80
    learning_rate: float = 0.08
    alignment_weight: float = 0.35
    l2_weight: float = 1e-4
    seed: int = 7
    max_parameters: int = 10_000


@dataclass
class Prediction:
    word: str
    score: float


@dataclass(frozen=True)
class AttentionRow:
    word: str
    prediction_score: float
    token_weights: tuple[float, ...]
    attended_score: float


class WordEmbeddingModel:
    def __init__(
        self,
        token_vocab: Dict[str, int],
        label_vocab: Dict[str, int],
        ngram_vocab: Dict[str, int],
        token_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
        ngram_embeddings: np.ndarray,
    ) -> None:
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.ngram_vocab = ngram_vocab
        self.token_embeddings = token_embeddings
        self.label_embeddings = label_embeddings
        self.ngram_embeddings = ngram_embeddings
        self.id_to_label = {idx: label for label, idx in label_vocab.items()}

    @property
    def parameter_count(self) -> int:
        return int(
            self.token_embeddings.size
            + self.label_embeddings.size
            + self.ngram_embeddings.size
        )

    def encode_prompt(self, prompt_tokens: Sequence[str]) -> np.ndarray:
        token_ids = encode_prompt(prompt_tokens, self.token_vocab)
        if not token_ids:
            return np.zeros(self.label_embeddings.shape[1], dtype=np.float64)
        return self.token_embeddings[token_ids].mean(axis=0)

    def encode_prompt_batch(self, prompt_token_batches: Sequence[Sequence[str]]) -> np.ndarray:
        batch_size = len(prompt_token_batches)
        dim = self.label_embeddings.shape[1]
        prompt_matrix = np.zeros((batch_size, dim), dtype=np.float64)
        for row_index, prompt_tokens in enumerate(prompt_token_batches):
            token_ids = encode_prompt(prompt_tokens, self.token_vocab)
            if token_ids:
                prompt_matrix[row_index] = self.token_embeddings[token_ids].mean(axis=0)
        return prompt_matrix

    def encode_surface_form(self, word: str) -> np.ndarray:
        gram_ids = encode_surface_form(word, self.ngram_vocab)
        if not gram_ids:
            return np.zeros(self.label_embeddings.shape[1], dtype=np.float64)
        return self.ngram_embeddings[gram_ids].mean(axis=0)

    def prompt_attention_logits(
        self,
        prompt_tokens: Sequence[str],
    ) -> tuple[list[str], np.ndarray]:
        normalized_tokens = [str(token) for token in prompt_tokens if str(token)]
        token_ids = encode_prompt(normalized_tokens, self.token_vocab)
        if not token_ids:
            return normalized_tokens, np.zeros((len(self.label_vocab), 0), dtype=np.float64)
        token_matrix = self.token_embeddings[token_ids]
        scale = float(np.sqrt(max(1, token_matrix.shape[1])))
        logits = (self.label_embeddings @ token_matrix.T) / scale
        return normalized_tokens, logits

    def prompt_attention_rows(
        self,
        prompt_tokens: Sequence[str],
        top_k: int = 3,
    ) -> List[AttentionRow]:
        ranked = self.predict(prompt_tokens, top_k=top_k)
        tokens, logits = self.prompt_attention_logits(prompt_tokens)
        if logits.shape[1] == 0:
            return [
                AttentionRow(
                    word=prediction.word,
                    prediction_score=prediction.score,
                    token_weights=tuple(),
                    attended_score=0.0,
                )
                for prediction in ranked
            ]

        rows: List[AttentionRow] = []
        for prediction in ranked:
            label_id = self.label_vocab[prediction.word]
            weights = _softmax(logits[label_id])
            attended_score = float(np.dot(weights, logits[label_id]))
            rows.append(
                AttentionRow(
                    word=prediction.word,
                    prediction_score=prediction.score,
                    token_weights=tuple(float(weight) for weight in weights),
                    attended_score=attended_score,
                )
            )
        return rows

    def prompt_attention_report(
        self,
        prompt_tokens: Sequence[str],
        top_k: int = 3,
    ) -> Dict[str, object]:
        tokens, logits = self.prompt_attention_logits(prompt_tokens)
        rows = self.prompt_attention_rows(prompt_tokens, top_k=top_k)
        return {
            "tokens": tokens,
            "matrix_shape": [len(rows), len(tokens)],
            "rows": [
                {
                    "word": row.word,
                    "prediction_score": row.prediction_score,
                    "attended_score": row.attended_score,
                    "token_weights": list(row.token_weights),
                }
                for row in rows
            ],
            "raw_logits": [
                list(logits[self.label_vocab[row.word]]) if logits.shape[1] else []
                for row in rows
            ],
        }

    def predict(self, prompt_tokens: Sequence[str], top_k: int = 3) -> List[Prediction]:
        prompt_vec = self.encode_prompt(prompt_tokens)
        logits = self.label_embeddings @ prompt_vec
        probs = _softmax(logits)
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [
            Prediction(word=self.id_to_label[idx], score=float(probs[idx]))
            for idx in top_indices
            if idx != 0
        ]

    def predict_batch(
        self,
        prompt_token_batches: Sequence[Sequence[str]],
        top_k: int = 3,
    ) -> List[List[Prediction]]:
        if not prompt_token_batches:
            return []
        prompt_matrix = self.encode_prompt_batch(prompt_token_batches)
        logits = prompt_matrix @ self.label_embeddings.T
        probs = _softmax_rows(logits)
        top_width = min(probs.shape[1], max(top_k + 1, top_k))
        top_indices = np.argsort(probs, axis=1)[:, ::-1][:, :top_width]
        predictions: List[List[Prediction]] = []
        for row_index, row_indices in enumerate(top_indices):
            row_predictions: List[Prediction] = []
            for idx in row_indices:
                if idx == 0:
                    continue
                row_predictions.append(
                    Prediction(word=self.id_to_label[int(idx)], score=float(probs[row_index, idx]))
                )
                if len(row_predictions) >= top_k:
                    break
            predictions.append(row_predictions)
        return predictions

    def nearest_words_for_surface_form(self, word: str, top_k: int = 3) -> List[Prediction]:
        surface_vec = self.encode_surface_form(word)
        denom = np.linalg.norm(surface_vec) + 1e-8
        label_norms = np.linalg.norm(self.label_embeddings, axis=1) + 1e-8
        sims = (self.label_embeddings @ surface_vec) / (label_norms * denom)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            Prediction(word=self.id_to_label[idx], score=float(sims[idx]))
            for idx in top_indices
            if idx != 0
        ]

    def to_serializable(self) -> Dict[str, object]:
        return {
            "token_vocab": self.token_vocab,
            "label_vocab": self.label_vocab,
            "ngram_vocab": self.ngram_vocab,
            "token_embeddings": self.token_embeddings.tolist(),
            "label_embeddings": self.label_embeddings.tolist(),
            "ngram_embeddings": self.ngram_embeddings.tolist(),
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, object]) -> "WordEmbeddingModel":
        return cls(
            token_vocab={str(k): int(v) for k, v in dict(payload["token_vocab"]).items()},
            label_vocab={str(k): int(v) for k, v in dict(payload["label_vocab"]).items()},
            ngram_vocab={str(k): int(v) for k, v in dict(payload["ngram_vocab"]).items()},
            token_embeddings=np.asarray(payload["token_embeddings"], dtype=np.float64),
            label_embeddings=np.asarray(payload["label_embeddings"], dtype=np.float64),
            ngram_embeddings=np.asarray(payload["ngram_embeddings"], dtype=np.float64),
        )


def train_model(corpus: TrainingCorpus, config: TrainingConfig | None = None) -> WordEmbeddingModel:
    cfg = config or TrainingConfig()
    rng = np.random.default_rng(cfg.seed)
    dim = cfg.embedding_dim

    parameter_count = (len(corpus.token_vocab) + len(corpus.label_vocab) + len(corpus.ngram_vocab)) * dim
    if parameter_count > cfg.max_parameters:
        raise ValueError(
            f"parameter budget exceeded: {parameter_count} > {cfg.max_parameters}"
        )

    token_embeddings = rng.normal(0.0, 0.12, size=(len(corpus.token_vocab), dim))
    label_embeddings = rng.normal(0.0, 0.12, size=(len(corpus.label_vocab), dim))
    ngram_embeddings = rng.normal(0.0, 0.12, size=(len(corpus.ngram_vocab), dim))

    token_embeddings[0] = 0.0
    label_embeddings[0] = 0.0
    ngram_embeddings[0] = 0.0

    for _ in range(cfg.epochs):
        for example in corpus.examples:
            _train_step(
                example=example,
                corpus=corpus,
                token_embeddings=token_embeddings,
                label_embeddings=label_embeddings,
                ngram_embeddings=ngram_embeddings,
                config=cfg,
            )

    return WordEmbeddingModel(
        token_vocab=corpus.token_vocab,
        label_vocab=corpus.label_vocab,
        ngram_vocab=corpus.ngram_vocab,
        token_embeddings=token_embeddings,
        label_embeddings=label_embeddings,
        ngram_embeddings=ngram_embeddings,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _train_step(
    *,
    example: TrainingExample,
    corpus: TrainingCorpus,
    token_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    ngram_embeddings: np.ndarray,
    config: TrainingConfig,
) -> None:
    token_ids = encode_prompt(example.prompt_tokens, corpus.token_vocab)
    gram_ids = encode_surface_form(example.observed_surface_form, corpus.ngram_vocab)
    label_id = corpus.label_vocab[example.canonical_word]

    if not token_ids:
        return

    prompt_vec = token_embeddings[token_ids].mean(axis=0)
    logits = label_embeddings @ prompt_vec
    probs = _softmax(logits)
    probs[label_id] -= 1.0

    grad_prompt = label_embeddings.T @ probs
    grad_labels = np.outer(probs, prompt_vec)

    grad_token_share = grad_prompt / len(token_ids)
    for token_id in token_ids:
        token_embeddings[token_id] -= config.learning_rate * (
            grad_token_share + config.l2_weight * token_embeddings[token_id]
        )

    label_embeddings -= config.learning_rate * (grad_labels + config.l2_weight * label_embeddings)

    if gram_ids:
        surface_vec = ngram_embeddings[gram_ids].mean(axis=0)
        diff = surface_vec - label_embeddings[label_id]
        grad_surface = config.alignment_weight * diff
        grad_label_align = config.alignment_weight * (-diff)

        label_embeddings[label_id] -= config.learning_rate * (
            grad_label_align + config.l2_weight * label_embeddings[label_id]
        )

        grad_ngram_share = grad_surface / len(gram_ids)
        for gram_id in gram_ids:
            ngram_embeddings[gram_id] -= config.learning_rate * (
                grad_ngram_share + config.l2_weight * ngram_embeddings[gram_id]
            )
