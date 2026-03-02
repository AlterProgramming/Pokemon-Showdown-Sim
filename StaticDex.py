# StaticDex.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json
import os

DEFAULT_POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"


def to_id(s: str) -> str:
    """Showdown-style id normalization: lower + alnum only."""
    s = (s or "").lower()
    return "".join(ch for ch in s if ch.isalnum())


@dataclass
class DexVocab:
    species_to_idx: Dict[str, int]
    type_to_idx: Dict[str, int]


class StaticDex:
    """
    Static knowledge source:
      - loads Pokemon Showdown pokedex.json
      - provides vocabularies for species/types
      - provides lookup(species_name) -> (species_idx, t1_idx, t2_idx, stats6_scaled)
    """

    def __init__(self, pokedex: Dict[str, Any], vocab: DexVocab):
        self.pokedex = pokedex
        self.vocab = vocab

    @staticmethod
    def load_json_from_path(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_json_from_url(url: str) -> Dict[str, Any]:
        # kept inside function so importing module doesn't require network
        import urllib.request
        with urllib.request.urlopen(url) as r:
            return json.loads(r.read().decode("utf-8"))

    @classmethod
    def from_source(
        cls,
        local_path: Optional[str] = None,
    ) -> "StaticDex":
        """
        Prefer local_path if provided and exists; otherwise fetch from URL.
        """
        if local_path and os.path.exists(local_path):
            pokedex = cls.load_json_from_path(local_path)
        else:
            print('File not found')

        vocab = cls._build_vocab(pokedex)
        return cls(pokedex=pokedex, vocab=vocab)

    @staticmethod
    def _build_vocab(pokedex: Dict[str, Any]) -> DexVocab:
        # Types present in the pokedex
        all_types = sorted(
            {t for entry in pokedex.values() for t in (entry.get("types", []) or [])}
        )
        type_to_idx = {"<NONE>": 0, **{t: i + 1 for i, t in enumerate(all_types)}}

        # Species keys are already showdown ids
        species_keys = sorted(pokedex.keys())
        species_to_idx = {"<UNK>": 0, **{k: i + 1 for i, k in enumerate(species_keys)}}

        return DexVocab(species_to_idx=species_to_idx, type_to_idx=type_to_idx)

    def lookup(self, species_name: Optional[str]) -> Tuple[int, int, int, List[float]]:
        """
        Returns:
          (species_idx, type1_idx, type2_idx, [hp,atk,def,spa,spd,spe] scaled to ~[0,1])
        """
        if not species_name:
            return 0, 0, 0, [0.0] * 6

        sid = to_id(species_name)
        entry = self.pokedex.get(sid)
        if not entry:
            return 0, 0, 0, [0.0] * 6

        sp_idx = self.vocab.species_to_idx.get(sid, 0)

        types = entry.get("types", []) or []
        t1 = self.vocab.type_to_idx.get(types[0], 0) if len(types) >= 1 else 0
        t2 = self.vocab.type_to_idx.get(types[1], 0) if len(types) >= 2 else 0

        bs = entry.get("baseStats", {}) or {}
        # Use 255 as safe upper bound for scaling
        stats = [
            float(bs.get("hp", 0)) / 255.0,
            float(bs.get("atk", 0)) / 255.0,
            float(bs.get("def", 0)) / 255.0,
            float(bs.get("spa", 0)) / 255.0,
            float(bs.get("spd", 0)) / 255.0,
            float(bs.get("spe", 0)) / 255.0,
        ]

        return sp_idx, t1, t2, stats
