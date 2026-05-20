from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import hashlib
import random

from .lexicon import LexiconEntry, default_lexicon
from .text import char_ngrams, normalize_prompt


@dataclass(frozen=True)
class TrainingExample:
    prompt_tokens: tuple[str, ...]
    canonical_word: str
    observed_surface_form: str


@dataclass(frozen=True)
class TrainingCorpus:
    examples: tuple[TrainingExample, ...]
    token_vocab: dict[str, int]
    label_vocab: dict[str, int]
    ngram_vocab: dict[str, int]


def _stable_bucket_for_ngram(gram: str, bucket_count: int) -> str:
    digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % bucket_count
    return f"<bucket_{bucket}>"


def build_vocabulary(sequences: Iterable[Iterable[str]], pad_token: str = "<pad>") -> Dict[str, int]:
    vocab = {pad_token: 0}
    for seq in sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def generate_examples(
    lexicon: Sequence[LexiconEntry],
    *,
    repeats_per_entry: int = 48,
    seed: int = 13,
) -> List[TrainingExample]:
    rng = random.Random(seed)
    examples: List[TrainingExample] = []
    for entry in lexicon:
        for _ in range(repeats_per_entry):
            prompt_size = rng.randint(2, min(4, len(entry.descriptors)))
            prompt_tokens = tuple(sorted(rng.sample(list(entry.descriptors), prompt_size)))
            observed_form = rng.choice(entry.surface_forms)
            examples.append(
                TrainingExample(
                    prompt_tokens=prompt_tokens,
                    canonical_word=entry.canonical,
                    observed_surface_form=observed_form,
                )
            )
    rng.shuffle(examples)
    return examples


def build_training_corpus(
    lexicon: Sequence[LexiconEntry] | None = None,
    *,
    repeats_per_entry: int = 48,
    seed: int = 13,
    ngram_bucket_count: int | None = 128,
) -> TrainingCorpus:
    entries = list(lexicon or default_lexicon())
    examples = generate_examples(entries, repeats_per_entry=repeats_per_entry, seed=seed)

    token_vocab = build_vocabulary(example.prompt_tokens for example in examples)
    label_vocab = build_vocabulary(([entry.canonical] for entry in entries), pad_token="<unk>")
    if ngram_bucket_count is not None:
        ngram_vocab = {"<pad>": 0}
        for index in range(ngram_bucket_count):
            ngram_vocab[f"<bucket_{index}>"] = index + 1
    else:
        ngram_vocab = build_vocabulary(
            (char_ngrams(example.observed_surface_form) for example in examples),
            pad_token="<pad>",
        )

    return TrainingCorpus(
        examples=tuple(examples),
        token_vocab=token_vocab,
        label_vocab=label_vocab,
        ngram_vocab=ngram_vocab,
    )


def encode_prompt(prompt_tokens: Sequence[str], token_vocab: Dict[str, int]) -> List[int]:
    normalized = normalize_prompt(prompt_tokens)
    return [token_vocab[token] for token in normalized if token in token_vocab]


def encode_surface_form(surface_form: str, ngram_vocab: Dict[str, int]) -> List[int]:
    if any(token.startswith("<bucket_") for token in ngram_vocab):
        bucket_count = sum(1 for token in ngram_vocab if token.startswith("<bucket_"))
        return [
            ngram_vocab[_stable_bucket_for_ngram(gram, bucket_count)]
            for gram in char_ngrams(surface_form)
        ]
    return [ngram_vocab[gram] for gram in char_ngrams(surface_form) if gram in ngram_vocab]
