from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .text import generate_typo_variants, normalize_token


@dataclass(frozen=True)
class LexiconEntry:
    canonical: str
    descriptors: tuple[str, ...]
    surface_forms: tuple[str, ...]


def _build_entry(canonical: str, descriptors: Iterable[str], explicit_surface_forms: Iterable[str]) -> LexiconEntry:
    base = normalize_token(canonical)
    forms = {base}
    forms.update(normalize_token(form) for form in explicit_surface_forms)
    forms.update(generate_typo_variants(base))
    forms.discard("")
    return LexiconEntry(
        canonical=base,
        descriptors=tuple(normalize_token(token) for token in descriptors if normalize_token(token)),
        surface_forms=tuple(sorted(forms)),
    )


def default_lexicon() -> List[LexiconEntry]:
    return [
        _build_entry("happy", ["joy", "smile", "warm", "bright", "good"], ["hapy", "happee"]),
        _build_entry("angry", ["mad", "rage", "heat", "storm", "sharp"], ["angery", "anrgy"]),
        _build_entry("swift", ["fast", "quick", "dash", "speed", "light"], ["swfit", "swyft"]),
        _build_entry("ocean", ["water", "deep", "blue", "wave", "salt"], ["ocen", "oceon"]),
        _build_entry("forest", ["tree", "green", "moss", "wild", "leaf"], ["forrest", "foresst"]),
        _build_entry("ember", ["fire", "glow", "ash", "spark", "warm"], ["embur", "embr"]),
        _build_entry("crystal", ["glass", "clear", "shard", "bright", "gem"], ["cristal", "crystl"]),
        _build_entry("shadow", ["dark", "night", "shade", "quiet", "hidden"], ["shaddow", "shado"]),
        _build_entry("dream", ["sleep", "vision", "soft", "wish", "night"], ["dreem", "drem"]),
        _build_entry("glitch", ["error", "broken", "spark", "noise", "digital"], ["glich", "gltich"]),
        _build_entry("thunder", ["storm", "sound", "electric", "flash", "sky"], ["thuner", "thundr"]),
        _build_entry("silken", ["smooth", "soft", "thread", "fabric", "fine"], ["silkin", "silkenn"]),
    ]
