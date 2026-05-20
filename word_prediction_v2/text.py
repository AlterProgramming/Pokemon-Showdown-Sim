from __future__ import annotations

from typing import Iterable, List


def normalize_token(text: str) -> str:
    lowered = (text or "").lower().strip()
    return "".join(ch for ch in lowered if ch.isalpha())


def normalize_prompt(tokens: Iterable[str]) -> List[str]:
    return [token for token in (normalize_token(token) for token in tokens) if token]


def char_ngrams(word: str, min_n: int = 2, max_n: int = 4) -> List[str]:
    normalized = normalize_token(word)
    if not normalized:
        return ["<empty>"]
    padded = f"<{normalized}>"
    grams: List[str] = []
    for n in range(min_n, max_n + 1):
        if len(padded) < n:
            continue
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams or [padded]


def generate_typo_variants(word: str) -> List[str]:
    normalized = normalize_token(word)
    if len(normalized) < 3:
        return [normalized] if normalized else []

    variants = {normalized}

    for idx in range(len(normalized)):
        variants.add(normalized[:idx] + normalized[idx + 1 :])

    for idx in range(len(normalized) - 1):
        swapped = list(normalized)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        variants.add("".join(swapped))

    vowels = "aeiou"
    for idx, ch in enumerate(normalized):
        if ch in vowels:
            for repl in vowels:
                if repl != ch:
                    variants.add(normalized[:idx] + repl + normalized[idx + 1 :])

    for idx in range(len(normalized)):
        variants.add(normalized[:idx] + normalized[idx] + normalized[idx:])

    variants.discard("")
    return sorted(variants)
