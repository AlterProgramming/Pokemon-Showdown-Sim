from __future__ import annotations

import unittest
from pathlib import Path

from StaticDex import StaticDex


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class StaticDexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dex = StaticDex.from_source(local_path=str(DATA_DIR / "pokedex.json"))

    def test_lookup_matches_exact_species_name(self) -> None:
        sp_idx, t1, t2, stats = self.dex.lookup("Pikachu")

        self.assertNotEqual(sp_idx, 0)
        self.assertNotEqual(t1, 0)
        self.assertEqual(len(stats), 6)

    def test_lookup_resolves_close_spelling_match(self) -> None:
        pikachu_idx = self.dex.lookup("Pikachu")[0]
        typo_idx = self.dex.lookup("Pikchu")[0]

        self.assertEqual(typo_idx, pikachu_idx)

    def test_lookup_resolves_display_name_formatting(self) -> None:
        mime_idx = self.dex.lookup("Mr Mime")[0]
        canonical_idx = self.dex.lookup("Mr. Mime")[0]
        hooh_idx = self.dex.lookup("Ho Oh")[0]
        canonical_hooh_idx = self.dex.lookup("Ho-Oh")[0]

        self.assertEqual(mime_idx, canonical_idx)
        self.assertEqual(hooh_idx, canonical_hooh_idx)

    def test_lookup_returns_unknown_for_distant_match(self) -> None:
        sp_idx, t1, t2, stats = self.dex.lookup("DefinitelyNotAPokemon")

        self.assertEqual((sp_idx, t1, t2), (0, 0, 0))
        self.assertEqual(stats, [0.0] * 6)


if __name__ == "__main__":
    unittest.main()
