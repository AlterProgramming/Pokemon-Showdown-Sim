#!/usr/bin/env python3
"""
diagnose_state_failures.py
==========================
Batch diagnostic tool for Pokemon Showdown battle JSON files.

Analyzes turn_events_v1 sequences (as produced by the sequence auxiliary
head training pipeline) and reports:

  • State-transition failures — moves that had no effect because the target
    state was already active (immunity, duplicate status, active terrain,
    active screen, active encore).
  • Move effectiveness distribution — super-effective / resisted / neutral /
    immune damage events using the Gen 9 type chart.
  • "Big failure" detection — 3+ consecutive failed transitions per battle.

Outputs
-------
  --output-json <dir>   One JSON file per battle: granular failure records.
  --output-csv  <path>  Aggregated CSV summary across all battles.

Usage
-----
  python tools/diagnose_state_failures.py /path/to/battles/ \\
      --output-json report/ \\
      --output-csv  summary.csv \\
      --sequence-vocab artifacts/sequence_vocab_5.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Static type chart (Gen 9) — offence × defence multipliers
# ---------------------------------------------------------------------------
# Keys: attacking_type -> {defending_type: multiplier}
# Only non-neutral entries are stored; missing = 1.0
_TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Rock": 0.5, "Dragon": 0.5,
                 "Grass": 2.0, "Ice": 2.0, "Bug": 2.0, "Steel": 2.0},
    "Water":    {"Water": 0.5, "Grass": 0.5, "Dragon": 0.5,
                 "Fire": 2.0, "Ground": 2.0, "Rock": 2.0},
    "Electric": {"Electric": 0.5, "Grass": 0.5, "Dragon": 0.5,
                 "Ground": 0.0, "Flying": 2.0, "Water": 2.0},
    "Grass":    {"Fire": 0.5, "Grass": 0.5, "Poison": 0.5, "Flying": 0.5,
                 "Bug": 0.5, "Dragon": 0.5, "Steel": 0.5,
                 "Water": 2.0, "Ground": 2.0, "Rock": 2.0},
    "Ice":      {"Water": 0.5, "Ice": 0.5,
                 "Fire": 0.5, "Steel": 0.5,
                 "Grass": 2.0, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Rock": 2.0, "Dark": 2.0,
                 "Steel": 2.0,
                 "Poison": 0.5, "Bug": 0.5, "Psychic": 0.5,
                 "Flying": 0.5, "Fairy": 0.5,
                 "Ghost": 0.0},
    "Poison":   {"Grass": 2.0, "Fairy": 2.0,
                 "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5,
                 "Steel": 0.0},
    "Ground":   {"Fire": 2.0, "Electric": 2.0, "Poison": 2.0,
                 "Rock": 2.0, "Steel": 2.0,
                 "Grass": 0.5, "Bug": 0.5,
                 "Flying": 0.0},
    "Flying":   {"Grass": 2.0, "Fighting": 2.0, "Bug": 2.0,
                 "Electric": 0.5, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2.0, "Poison": 2.0,
                 "Psychic": 0.5, "Steel": 0.5,
                 "Dark": 0.0},
    "Bug":      {"Grass": 2.0, "Psychic": 2.0, "Dark": 2.0,
                 "Fire": 0.5, "Fighting": 0.5, "Flying": 0.5,
                 "Ghost": 0.5, "Steel": 0.5, "Fairy": 0.5},
    "Rock":     {"Fire": 2.0, "Ice": 2.0, "Flying": 2.0, "Bug": 2.0,
                 "Fighting": 0.5, "Ground": 0.5, "Steel": 0.5},
    "Ghost":    {"Psychic": 2.0, "Ghost": 2.0,
                 "Normal": 0.0, "Dark": 0.5},
    "Dragon":   {"Dragon": 2.0,
                 "Steel": 0.5, "Fairy": 0.0},
    "Dark":     {"Psychic": 2.0, "Ghost": 2.0,
                 "Fighting": 0.5, "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Ice": 2.0, "Rock": 2.0, "Fairy": 2.0,
                 "Fire": 0.5, "Water": 0.5, "Electric": 0.5,
                 "Steel": 0.5},
    "Fairy":    {"Fighting": 2.0, "Dragon": 2.0, "Dark": 2.0,
                 "Fire": 0.5, "Poison": 0.5, "Steel": 0.5},
}


def type_effectiveness(attacking_type: str, defending_types: list[str]) -> float:
    """Return the combined multiplier for attacking_type vs a list of defending types."""
    row = _TYPE_CHART.get(attacking_type, {})
    multiplier = 1.0
    for dt in defending_types:
        multiplier *= row.get(dt, 1.0)
    return multiplier


def effectiveness_label(multiplier: float) -> str:
    if multiplier == 0.0:
        return "immune"
    if multiplier > 1.0:
        return "super_effective"
    if multiplier < 1.0:
        return "resisted"
    return "neutral"


# ---------------------------------------------------------------------------
# Minimal Pokedex (species -> types) assembled from well-known Gen 9 mons
# ---------------------------------------------------------------------------
# This is a best-effort inline table covering common randbats species.
# Unknown species default to ["Normal"] with a warning.
# Keyed by lowercase, no spaces (Showdown-normalised).
_SPECIES_TYPES: dict[str, list[str]] = {
    # A
    "abomasnow": ["Grass", "Ice"], "absol": ["Dark"], "aegislash": ["Steel", "Ghost"],
    "aerodactyl": ["Rock", "Flying"], "aggron": ["Steel", "Rock"],
    "alakazam": ["Psychic"], "alcremie": ["Fairy"],
    "alomomola": ["Water"], "altaria": ["Dragon", "Flying"],
    "ambipom": ["Normal"], "amoonguss": ["Grass", "Poison"],
    "ampharos": ["Electric"], "annihilape": ["Fighting", "Ghost"],
    "appletun": ["Grass", "Dragon"], "arcanine": ["Fire"],
    "araquanid": ["Water", "Bug"], "arctozolt": ["Electric", "Ice"],
    "arctovish": ["Water", "Ice"],
    "armarouge": ["Fire", "Psychic"], "armaldo": ["Rock", "Bug"],
    "aromatisse": ["Fairy"], "articuno": ["Ice", "Flying"],
    "aurorus": ["Rock", "Ice"],
    # B
    "baxcalibur": ["Dragon", "Ice"], "beartic": ["Ice"],
    "beedrill": ["Bug", "Poison"], "bellossom": ["Grass"],
    "bewear": ["Normal", "Fighting"], "bisharp": ["Dark", "Steel"],
    "blacephalon": ["Fire", "Ghost"], "blaziken": ["Fire", "Fighting"],
    "blissey": ["Normal"], "breloom": ["Grass", "Fighting"],
    "bronzong": ["Steel", "Psychic"],
    # C
    "cacturne": ["Grass", "Dark"], "calyrex": ["Psychic", "Grass"],
    "calyrex-ice": ["Psychic", "Ice"], "calyrex-shadow": ["Psychic", "Ghost"],
    "camerupt": ["Fire", "Ground"], "carbink": ["Rock", "Fairy"],
    "carracosta": ["Water", "Rock"], "ceruledge": ["Fire", "Ghost"],
    "chandelure": ["Ghost", "Fire"], "chansey": ["Normal"],
    "charizard": ["Fire", "Flying"], "cinccino": ["Normal"],
    "cinderace": ["Fire"], "clefable": ["Fairy"],
    "clefairy": ["Fairy"], "coalossal": ["Rock", "Fire"],
    "cofagrigus": ["Ghost"], "comfey": ["Fairy"],
    "conkeldurr": ["Fighting"], "copperajah": ["Steel"],
    "corsola-galar": ["Ghost"], "corviknight": ["Flying", "Steel"],
    "crabominable": ["Fighting", "Ice"], "cramorant": ["Flying", "Water"],
    "crawdaunt": ["Water", "Dark"], "cresselia": ["Psychic"],
    "crobat": ["Poison", "Flying"], "crustle": ["Bug", "Rock"],
    "cryogonal": ["Ice"], "cursola": ["Ghost"],
    # D
    "darkrai": ["Dark"], "darmanitan": ["Fire"],
    "darmanitan-galar": ["Ice"], "decidueye": ["Grass", "Ghost"],
    "dedenne": ["Electric", "Fairy"], "delibird": ["Ice", "Flying"],
    "delphox": ["Fire", "Psychic"], "dewgong": ["Water", "Ice"],
    "dialga": ["Steel", "Dragon"], "ditto": ["Normal"],
    "dodrio": ["Normal", "Flying"], "donphan": ["Ground"],
    "doublade": ["Steel", "Ghost"], "dragalge": ["Poison", "Dragon"],
    "dragapult": ["Dragon", "Ghost"], "dragonite": ["Dragon", "Flying"],
    "drampa": ["Normal", "Dragon"], "drapion": ["Poison", "Dark"],
    "drifblim": ["Ghost", "Flying"], "drizzile": ["Water"],
    "druddigon": ["Dragon"], "dusknoir": ["Ghost"],
    # E
    "eelektross": ["Electric"], "eiscue": ["Ice"],
    "electrode": ["Electric"], "electrode-hisui": ["Electric", "Grass"],
    "emboar": ["Fire", "Fighting"], "empoleon": ["Water", "Steel"],
    "entei": ["Fire"], "escavalier": ["Bug", "Steel"],
    "espeon": ["Psychic"], "excadrill": ["Ground", "Steel"],
    # F
    "falinks": ["Fighting"], "feraligatr": ["Water"],
    "ferrothorn": ["Grass", "Steel"], "fezandipiti": ["Poison", "Fairy"],
    "flamigo": ["Flying", "Fighting"], "flareon": ["Fire"],
    "flygon": ["Ground", "Dragon"], "forretress": ["Bug", "Steel"],
    "froslass": ["Ice", "Ghost"], "frosmoth": ["Ice", "Bug"],
    "furfrou": ["Normal"],
    # G
    "gallade": ["Psychic", "Fighting"], "garchomp": ["Dragon", "Ground"],
    "gardevoir": ["Psychic", "Fairy"], "gastrodon": ["Water", "Ground"],
    "gengar": ["Ghost", "Poison"], "gholdengo": ["Steel", "Ghost"],
    "glaceon": ["Ice"], "glalie": ["Ice"],
    "glimmora": ["Rock", "Poison"], "gliscor": ["Ground", "Flying"],
    "golisopod": ["Bug", "Water"], "goodra": ["Dragon"],
    "goodra-hisui": ["Dragon", "Steel"], "gothitelle": ["Psychic"],
    "gouging-fire": ["Fire", "Dragon"], "grafaiai": ["Poison", "Normal"],
    "granbull": ["Fairy"], "greninja": ["Water", "Dark"],
    "groudon": ["Ground"], "gyarados": ["Water", "Flying"],
    # H
    "hatterene": ["Psychic", "Fairy"], "hawlucha": ["Fighting", "Flying"],
    "heliolisk": ["Electric", "Normal"], "heracross": ["Bug", "Fighting"],
    "hippowdon": ["Ground"], "hitmonlee": ["Fighting"],
    "hitmonchan": ["Fighting"], "hitmontop": ["Fighting"],
    "ho-oh": ["Fire", "Flying"], "honchkrow": ["Dark", "Flying"],
    "hooligans": ["Normal", "Ghost"], "hoopa": ["Psychic", "Ghost"],
    "hoopa-unbound": ["Psychic", "Dark"], "houndoom": ["Dark", "Fire"],
    "hydrapple": ["Grass", "Dragon"], "hydreigon": ["Dark", "Dragon"],
    # I
    "incineroar": ["Fire", "Dark"], "indeedee": ["Psychic", "Normal"],
    "infernape": ["Fire", "Fighting"], "iron-bundle": ["Ice", "Water"],
    "iron-hands": ["Fighting", "Electric"], "iron-jugulis": ["Dark", "Flying"],
    "iron-leaves": ["Grass", "Psychic"], "iron-moth": ["Fire", "Poison"],
    "iron-thorns": ["Rock", "Electric"], "iron-treads": ["Ground", "Steel"],
    "iron-valiant": ["Fairy", "Fighting"],
    "ironbundle": ["Ice", "Water"], "ironhands": ["Fighting", "Electric"],
    "ironjugulis": ["Dark", "Flying"], "ironleaves": ["Grass", "Psychic"],
    "ironmoth": ["Fire", "Poison"], "ironthorns": ["Rock", "Electric"],
    "irontreads": ["Ground", "Steel"], "ironvaliant": ["Fairy", "Fighting"],
    # J
    "jolteon": ["Electric"], "joltik": ["Bug", "Electric"],
    "jumpluff": ["Grass", "Flying"],
    # K
    "keldeo": ["Water", "Fighting"], "kingambit": ["Dark", "Steel"],
    "klefki": ["Steel", "Fairy"], "kommo-o": ["Dragon", "Fighting"],
    "krookodile": ["Ground", "Dark"],
    # L
    "landorus": ["Ground", "Flying"], "landorus-therian": ["Ground", "Flying"],
    "lanturn": ["Water", "Electric"], "lapras": ["Water", "Ice"],
    "latias": ["Dragon", "Psychic"], "latios": ["Dragon", "Psychic"],
    "leafeon": ["Grass"], "leavanny": ["Bug", "Grass"],
    "ledian": ["Bug", "Flying"], "liepard": ["Dark"],
    "lilligant": ["Grass"], "lilligant-hisui": ["Grass", "Fighting"],
    "lokix": ["Bug", "Dark"], "lopunny": ["Normal"],
    "lucario": ["Fighting", "Steel"], "ludicolo": ["Water", "Grass"],
    "lugia": ["Psychic", "Flying"],
    # M
    "magmortar": ["Fire"], "mamoswine": ["Ice", "Ground"],
    "mandibuzz": ["Dark", "Flying"], "mantine": ["Water", "Flying"],
    "marowak-alola": ["Fire", "Ghost"], "marshadow": ["Fighting", "Ghost"],
    "masquerain": ["Bug", "Flying"], "medicham": ["Fighting", "Psychic"],
    "meloetta": ["Normal", "Psychic"], "meloetta-pirouette": ["Normal", "Fighting"],
    "meowscarada": ["Grass", "Dark"], "metagross": ["Steel", "Psychic"],
    "mew": ["Psychic"], "mewtwo": ["Psychic"],
    "mimikyu": ["Ghost", "Fairy"], "minior": ["Rock", "Flying"],
    "mienshao": ["Fighting"], "mismagius": ["Ghost"],
    "moltres": ["Fire", "Flying"], "moltres-galar": ["Dark", "Flying"],
    "morpeko": ["Electric", "Dark"], "mudsdale": ["Ground"],
    "muk-alola": ["Poison", "Dark"],
    # N
    "naganadel": ["Poison", "Dragon"], "necrozma": ["Psychic"],
    "ninetales": ["Fire"], "ninetales-alola": ["Ice", "Fairy"],
    "noivern": ["Flying", "Dragon"], "nidoking": ["Poison", "Ground"],
    "nidoqueen": ["Poison", "Ground"],
    # O
    "obstagoon": ["Dark", "Normal"], "ogerpon": ["Grass"],
    "ogerpon-hearthflame": ["Grass", "Fire"],
    "ogerpon-cornerstone": ["Grass", "Rock"],
    "ogerpon-wellspring": ["Grass", "Water"],
    "okidogi": ["Poison", "Fighting"], "oricorio": ["Fire", "Flying"],
    "oinkologne": ["Normal"],
    # P
    "pachirisu": ["Electric"], "palafin": ["Water"],
    "palossand": ["Ghost", "Ground"], "pangoro": ["Fighting", "Dark"],
    "pelipper": ["Water", "Flying"], "persian-alola": ["Dark"],
    "perrserker": ["Steel"], "pex": ["Poison", "Water"],
    "pheromosa": ["Bug", "Fighting"], "pikachu": ["Electric"],
    "politoed": ["Water"], "poliwrath": ["Water", "Fighting"],
    "porygon-z": ["Normal"], "porygon2": ["Normal"],
    "primarina": ["Water", "Fairy"], "probopass": ["Rock", "Steel"],
    "pyukumuku": ["Water"],
    # Q
    "quagsire": ["Water", "Ground"], "queenly-majesty": ["Normal", "Fairy"],
    "quaquaval": ["Water", "Fighting"], "qwilfish-hisui": ["Poison", "Dark"],
    # R
    "raichu": ["Electric"], "raichu-alola": ["Electric", "Psychic"],
    "raikou": ["Electric"], "rampardos": ["Rock"],
    "rapidash": ["Fire"], "rapidash-galar": ["Psychic", "Fairy"],
    "raticate": ["Normal"], "rillaboom": ["Grass"],
    "roaring-moon": ["Dragon", "Dark"], "rotom": ["Electric", "Ghost"],
    "rotom-wash": ["Electric", "Water"], "rotom-heat": ["Electric", "Fire"],
    "rotom-cut": ["Electric", "Grass"], "rotom-frost": ["Electric", "Ice"],
    "rotom-fan": ["Electric", "Flying"],
    # S
    "salamence": ["Dragon", "Flying"], "salazzle": ["Poison", "Fire"],
    "samurott": ["Water"], "samurott-hisui": ["Water", "Dark"],
    "sceptile": ["Grass"], "scizor": ["Bug", "Steel"],
    "scrafty": ["Dark", "Fighting"], "seismitoad": ["Water", "Ground"],
    "serperior": ["Grass"], "shaymin": ["Grass"],
    "shaymin-sky": ["Grass", "Flying"], "shiftry": ["Grass", "Dark"],
    "silvally": ["Normal"], "sinistcha": ["Grass", "Ghost"],
    "skeledirge": ["Fire", "Ghost"], "skuntank": ["Poison", "Dark"],
    "slurpuff": ["Fairy"], "smeargle": ["Normal"],
    "sneasler": ["Fighting", "Poison"], "snorlax": ["Normal"],
    "spectrier": ["Ghost"], "starmie": ["Water", "Psychic"],
    "staraptor": ["Normal", "Flying"], "steelix": ["Steel", "Ground"],
    "sudowoodo": ["Rock"], "suicune": ["Water"],
    "sunflora": ["Grass"], "swampert": ["Water", "Ground"],
    "sylveon": ["Fairy"],
    # T
    "talonflame": ["Fire", "Flying"], "tangrowth": ["Grass"],
    "tapu-bulu": ["Grass", "Fairy"], "tapu-fini": ["Water", "Fairy"],
    "tapu-koko": ["Electric", "Fairy"], "tapu-lele": ["Psychic", "Fairy"],
    "tapubulu": ["Grass", "Fairy"], "tapufini": ["Water", "Fairy"],
    "tapukoko": ["Electric", "Fairy"], "tapulele": ["Psychic", "Fairy"],
    "tentacruel": ["Water", "Poison"], "ting-lu": ["Dark", "Ground"],
    "tinkaton": ["Fairy", "Steel"], "togekiss": ["Fairy", "Flying"],
    "torkoal": ["Fire"], "tornadus": ["Flying"],
    "tornadus-therian": ["Flying"], "toxapex": ["Poison", "Water"],
    "toxtricity": ["Electric", "Poison"],
    "trevenant": ["Ghost", "Grass"], "tsareena": ["Grass"],
    "turtonator": ["Fire", "Dragon"], "toedscruel": ["Ground", "Grass"],
    "ting-lu": ["Dark", "Ground"], "tinglu": ["Dark", "Ground"],
    "typhlosion": ["Fire"], "typhlosion-hisui": ["Fire", "Ghost"],
    # U
    "umbreon": ["Dark"], "unfezant": ["Normal", "Flying"],
    "urshifu": ["Fighting", "Water"], "urshifu-rapid-strike": ["Fighting", "Water"],
    "ursaluna": ["Normal", "Ground"], "ursaluna-bloodmoon": ["Normal", "Ground"],
    # V
    "vaporeon": ["Water"], "venusaur": ["Grass", "Poison"],
    "victini": ["Psychic", "Fire"], "virizion": ["Grass", "Fighting"],
    "volcanion": ["Fire", "Water"], "volcarona": ["Bug", "Fire"],
    # W
    "walking-wake": ["Water", "Dragon"], "weavile": ["Dark", "Ice"],
    "whimsicott": ["Grass", "Fairy"], "wo-chien": ["Grass", "Dark"],
    "wochien": ["Grass", "Dark"], "wigglytuff": ["Normal", "Fairy"],
    # X
    "xerneas": ["Fairy"], "xurkitree": ["Electric"],
    # Y
    "yanmega": ["Bug", "Flying"],
    # Z
    "zamazenta": ["Fighting"], "zapdos": ["Electric", "Flying"],
    "zapdos-galar": ["Fighting", "Flying"], "zarude": ["Grass", "Dark"],
    "zekrom": ["Dragon", "Electric"], "zeraora": ["Electric"],
    "zoroark-hisui": ["Normal", "Ghost"],
}

# Move-type lookup (lowercase move id -> type string)
_MOVE_TYPES: dict[str, str] = {
    # Normal
    "bodyslam": "Normal", "boomburst": "Normal", "double-edge": "Normal",
    "extremespeed": "Normal", "facade": "Normal", "gigaimpact": "Normal",
    "headbutt": "Normal", "hyper-voice": "Normal", "hypervoice": "Normal",
    "lastrespects": "Normal", "payback": "Normal",
    "quickattack": "Normal", "return": "Normal", "seismictoss": "Normal",
    "slash": "Normal", "snore": "Normal", "struggle": "Normal",
    "superfang": "Normal", "swift": "Normal", "tailslap": "Normal",
    "triattack": "Normal", "uproar": "Normal",
    # Fire
    "armorcannon": "Fire", "blastburn": "Fire", "blazekick": "Fire",
    "eruption": "Fire", "fierydance": "Fire", "fierywrath": "Fire",
    "fireblast": "Fire", "firefang": "Fire", "firepledge": "Fire",
    "firepunch": "Fire", "flamecharge": "Fire", "flamethrower": "Fire",
    "flareblitz": "Fire", "heatwave": "Fire", "magmastorm": "Fire",
    "mindblown": "Fire", "mysticalfire": "Fire", "overheat": "Fire",
    "pyroball": "Fire", "sacredfire": "Fire", "sacredflame": "Fire",
    "scorchingsands": "Fire", "sunnyday": "Fire", "temperflare": "Fire",
    "torchsong": "Fire", "v-create": "Fire",
    # Water
    "aquajet": "Water", "aquastep": "Water", "aquacutter": "Water",
    "brine": "Water", "clamp": "Water", "crabhammer": "Water",
    "dive": "Water", "flip-turn": "Water", "flipturn": "Water",
    "hydropump": "Water", "muddywater": "Water", "mudsport": "Water",
    "originpulse": "Water", "raindance": "Water", "scald": "Water",
    "sparklingaria": "Water", "steameruption": "Water", "surf": "Water",
    "surgingstrikes": "Water", "waterfall": "Water", "waterpulse": "Water",
    "waterspout": "Water", "wavecrash": "Water", "whirlpool": "Water",
    # Electric
    "bolt-beak": "Electric", "boltbeak": "Electric", "boltstrike": "Electric",
    "charge": "Electric", "discharge": "Electric", "doubleshock": "Electric",
    "electrodrift": "Electric", "electroball": "Electric", "fusionbolt": "Electric",
    "nuzzle": "Electric", "risingvoltage": "Electric", "spark": "Electric",
    "supercellslam": "Electric", "thunder": "Electric", "thunderbolt": "Electric",
    "thunderclap": "Electric", "thunderfang": "Electric", "thunderpunch": "Electric",
    "thunderwave": "Electric", "voltswitch": "Electric", "volttackle": "Electric",
    "wildcharge": "Electric", "zap-cannon": "Electric", "zapcannon": "Electric",
    "aurawheel": "Electric",  # default (Normal if non-Morpeko, but treat Electric)
    # Grass
    "appleacid": "Grass", "bulletseed": "Grass", "energyball": "Grass",
    "frenzyplant": "Grass", "gigadrain": "Grass", "grassknot": "Grass",
    "grasspledge": "Grass", "grassyglide": "Grass", "gravapple": "Grass",
    "hornleech": "Grass", "ivycudgel": "Grass", "leafblade": "Grass",
    "leafstorm": "Grass", "leechseed": "Grass", "magicalleaf": "Grass",
    "petaldance": "Grass", "powerwhip": "Grass", "razorleaf": "Grass",
    "seedbomb": "Grass", "seedflare": "Grass", "solarbeam": "Grass",
    "solarblade": "Grass", "spore": "Grass", "stunspore": "Grass",
    "synthesis": "Grass", "trailblaze": "Grass", "woodhammer": "Grass",
    # Ice
    "auroraveil": "Ice", "avalanche": "Ice", "blizzard": "Ice",
    "freeze-dry": "Ice", "freezedry": "Ice", "freezingglare": "Ice",
    "glaciallance": "Ice", "glaciate": "Ice", "icebeam": "Ice",
    "icefang": "Ice", "icepunch": "Ice", "icespinner": "Ice",
    "iciclespear": "Ice", "iciclecrash": "Ice", "snowscape": "Ice",
    "subzeroslammer": "Ice", "triplearrows": "Ice",
    # Fighting
    "armthrust": "Fighting", "aurasphere": "Fighting", "bodypress": "Fighting",
    "brickbreak": "Fighting", "bulletpunch": "Fighting",
    "closecombat": "Fighting", "collisioncourse": "Fighting",
    "crosschop": "Fighting", "detect": "Fighting", "drainpunch": "Fighting",
    "dynamicpunch": "Fighting", "endeavor": "Fighting", "figthing": "Fighting",
    "finalgambit": "Fighting", "focusblast": "Fighting", "focuspunch": "Fighting",
    "hammerarm": "Fighting", "headlongrush": "Fighting",
    "highjumpkick": "Fighting", "jumpkick": "Fighting",
    "karatechop": "Fighting", "lowkick": "Fighting", "lowsweep": "Fighting",
    "machpunch": "Fighting", "martialarts": "Fighting",
    "qcblade": "Fighting", "revenge": "Fighting", "reversal": "Fighting",
    "rocksmash": "Fighting", "secretsword": "Fighting",
    "skyuppercut": "Fighting", "stormthrow": "Fighting",
    "submission": "Fighting", "superpower": "Fighting",
    "vacuumwave": "Fighting", "victorydance": "Fighting",
    "vitalthrow": "Fighting", "wakeupslap": "Fighting",
    "circlethrow": "Fighting",
    # Poison
    "acidspray": "Poison", "banefulbunker": "Poison", "gunkshot": "Poison",
    "mortalspinto": "Poison", "poisonfang": "Poison", "poisonjab": "Poison",
    "poisonsting": "Poison", "sludgebomb": "Poison", "sludgewave": "Poison",
    "toxic": "Poison", "toxicspikes": "Poison", "venoshock": "Poison",
    "clearsmog": "Poison",
    # Ground
    "bonemerang": "Ground", "bonerush": "Ground", "bulldoze": "Ground",
    "drillrun": "Ground", "earthquake": "Ground", "earthpower": "Ground",
    "highhorsepower": "Ground", "highhorsepower": "Ground",
    "mudbomb": "Ground", "mudshot": "Ground", "precipiceblades": "Ground",
    "sandattack": "Ground", "shore-up": "Ground", "shoreup": "Ground",
    "spikes": "Ground", "stompingtantrum": "Ground",
    # Flying
    "acrobatics": "Flying", "aeroblast": "Flying", "airslash": "Flying",
    "beakblast": "Flying", "bleakwindstorm": "Flying", "bravebird": "Flying",
    "dragonascent": "Flying", "drillpeck": "Flying", "dualwingbeat": "Flying",
    "fly": "Flying", "hurricane": "Flying", "oblivionwing": "Flying",
    "skydrop": "Flying", "skyattack": "Flying", "tailwind": "Flying",
    "wingattack": "Flying",
    # Psychic
    "calmmind": "Psychic", "cosmicpower": "Psychic", "dazzlinggleam": "Psychic",
    "expandingforce": "Psychic", "extrasensory": "Psychic",
    "futuresight": "Psychic", "gravity": "Psychic", "guardianofalo": "Psychic",
    "heartswap": "Psychic", "hypnosis": "Psychic", "kinetisis": "Psychic",
    "luster-purge": "Psychic", "lusterpurge": "Psychic",
    "mist-ball": "Psychic", "mistball": "Psychic", "photongeyser": "Psychic",
    "psyblade": "Psychic", "psychic": "Psychic", "psychicfangs": "Psychic",
    "psychicnoise": "Psychic", "psychoboost": "Psychic", "psychocut": "Psychic",
    "psychoshift": "Psychic", "psyshock": "Psychic", "psystrike": "Psychic",
    "storedpower": "Psychic", "takeheart": "Psychic",
    "telekinesis": "Psychic", "trick": "Psychic", "trickroom": "Psychic",
    "wonderroom": "Psychic", "zenheadbutt": "Psychic",
    "lunarblessing": "Psychic", "lunardance": "Psychic",
    # Bug
    "bugbite": "Bug", "bugbuzz": "Bug", "firstimpression": "Bug",
    "fury-cutter": "Bug", "furycutter": "Bug", "leechlife": "Bug",
    "megahorn": "Bug", "pinmissile": "Bug", "pollenpuff": "Bug",
    "populationbomb": "Bug", "pounce": "Bug", "silverwind": "Bug",
    "signalbeam": "Bug", "skittersmack": "Bug", "u-turn": "Bug",
    "uturn": "Bug", "x-scissor": "Bug", "xscissor": "Bug",
    # Rock
    "accelerock": "Rock", "anchorshot": "Rock", "ancientpower": "Rock",
    "diamondstorm": "Rock", "headsmash": "Rock", "meteorbeam": "Rock",
    "powergem": "Rock", "rockblast": "Rock", "rockslide": "Rock",
    "rocktomb": "Rock", "rockwrecker": "Rock", "rollout": "Rock",
    "smackdown": "Rock", "stealthrock": "Rock", "stoneaxe": "Rock",
    "stoneedge": "Rock",
    # Ghost
    "astralbarrage": "Ghost", "astonish": "Ghost", "curse": "Ghost",
    "destinybond": "Ghost", "forestscurse": "Ghost", "hex": "Ghost",
    "lick": "Ghost", "menacingmoonrazemaelstrom": "Ghost",
    "moongeistbeam": "Ghost", "nevermeltice": "Ghost",
    "ominouswind": "Ghost", "painsplit": "Ghost", "phantomforce": "Ghost",
    "poltergeist": "Ghost", "shadowball": "Ghost", "shadowclaw": "Ghost",
    "shadowforce": "Ghost", "shadowsneak": "Ghost", "sinisterarrowraid": "Ghost",
    "spectralthief": "Ghost", "spiritbreak": "Ghost",
    "spiritshackle": "Ghost", "trickorthreat": "Ghost",
    "nocteoveil": "Ghost",
    # Dragon
    "clangingscales": "Dragon", "clangoroussoul": "Dragon",
    "dragonbreath": "Dragon", "dragonclaw": "Dragon", "dragondance": "Dragon",
    "dragondarts": "Dragon", "dragonenergy": "Dragon", "dragonpulse": "Dragon",
    "dragontail": "Dragon", "dragonrush": "Dragon", "dynamaxcannon": "Dragon",
    "eternabeam": "Dragon", "glaiverush": "Dragon", "outrage": "Dragon",
    "roaroftime": "Dragon", "scaleshot": "Dragon", "spacialrend": "Dragon",
    "twister": "Dragon",
    # Dark
    "assurance": "Dark", "beatup": "Dark", "bittermalice": "Dark",
    "darkpulse": "Dark", "dirtydirtybob": "Dark", "embargo": "Dark",
    "faintattack": "Dark", "feintattack": "Dark", "foulplay": "Dark",
    "hyperspace-fury": "Dark", "hyperspacefury": "Dark",
    "jawlock": "Dark", "knockoff": "Dark", "lashout": "Dark",
    "nightdaze": "Dark", "nightslash": "Dark", "obstruct": "Dark",
    "partingshot": "Dark", "powertrip": "Dark", "punishment": "Dark",
    "pursuit": "Dark", "ragingbull": "Dark", "ruination": "Dark",
    "snarl": "Dark", "suckerpunch": "Dark", "thief": "Dark",
    "torment": "Dark", "wickedblow": "Dark",
    # Steel
    "autotomize": "Steel", "behemothblade": "Steel", "behemothbash": "Steel",
    "bodypress": "Steel", "bulletpunch": "Steel", "copperajah": "Steel",
    "corrosivegas": "Steel", "corkscrew": "Steel", "flashcannon": "Steel",
    "geargrind": "Steel", "gigatonhammer": "Steel", "gyroball": "Steel",
    "heavyslam": "Steel", "irondefense": "Steel", "ironhead": "Steel",
    "irontail": "Steel", "magnetrise": "Steel", "meteormash": "Steel",
    "mirrorshot": "Steel", "shiftgear": "Steel", "smartstrike": "Steel",
    "steelbeam": "Steel", "steelroller": "Steel", "steelwing": "Steel",
    "sunsteelstrike": "Steel", "tachyoncutter": "Steel",
    # Fairy
    "alluringvoice": "Fairy", "babydolleys": "Fairy",
    "charm": "Fairy", "craftyshield": "Fairy",
    "disarmingvoice": "Fairy", "drainingkiss": "Fairy",
    "dazzlinggleam": "Fairy", "fleurcannon": "Fairy",
    "flowertrick": "Fairy", "geomancy": "Fairy",
    "lightthatburnsthesky": "Fairy", "moonblast": "Fairy",
    "moonlight": "Fairy", "mistyterrain": "Fairy",
    "naturesmadness": "Fairy", "nuzzle": "Electric",
    "playrough": "Fairy", "sparklyswirl": "Fairy", "sweetkiss": "Fairy",
    "tinkaton": "Fairy", "wish": "Normal",  # wish is Normal type
    "xerneas": "Fairy", "zingzap": "Electric",
    # Specific overrides / multi-type moves
    "ficklebeam": "Dragon", "bitterblade": "Fire",
    "bloodmoon": "Normal", "esperwing": "Psychic",
    "psyblade": "Psychic", "saltcure": "Water",
    "ceaselessedge": "Dark", "direclaw": "Poison",
    "electrodrift": "Electric", "terablast": "Normal",  # depends on tera
    "terastarstorm": "Normal",
    "vacuumwave": "Fighting", "revivalblessing": "Normal",
    "tidyup": "Normal", "filletaway": "Normal",
    "shoreup": "Ground",
}


def get_move_type(move_id: str) -> str | None:
    """Return the type of a move by its lowercase Showdown id, or None if unknown."""
    return _MOVE_TYPES.get(move_id.lower().replace("-", "").replace(" ", ""))


def get_species_types(species: str) -> list[str]:
    """Return the type list for a species, defaulting to ['Normal'] if unknown."""
    key = species.lower().replace(" ", "-").replace("'", "")
    # Try direct lookup, then strip forme suffixes
    if key in _SPECIES_TYPES:
        return _SPECIES_TYPES[key]
    # Try without trailing parenthetical / forme markers
    base = key.split("-")[0]
    return _SPECIES_TYPES.get(base, ["Normal"])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    turn: int
    actor: str           # "p1" or "p2"
    target: str          # "p1", "p2", or ""
    move: str
    reason: str          # immunity / duplicate_status / active_terrain /
                         # active_screen / active_encore
    prev_state: str      # human-readable description of the prior state


@dataclass
class BattleReport:
    battle_id: str
    failures: list[FailureRecord] = field(default_factory=list)
    move_effectiveness: dict[str, int] = field(
        default_factory=lambda: {
            "super_effective": 0,
            "resisted": 0,
            "neutral": 0,
            "immune": 0,
        }
    )
    big_failures_count: int = 0
    summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failures"] = [asdict(f) for f in self.failures]
        return d


# ---------------------------------------------------------------------------
# Battle state tracker
# ---------------------------------------------------------------------------

class BattleStateTracker:
    """
    Minimal, forward-only battle state tracker sufficient for failure
    detection.  Consumes turn_events_v1 token sequences in order.
    """

    def __init__(self) -> None:
        # Statuses: {side: status_token} e.g. {"p1": "brn", "p2": None}
        self.status: dict[str, str | None] = {"p1": None, "p2": None}
        # Screens / veil: {side: {condition: bool}}
        self.side_conditions: dict[str, dict[str, bool]] = {
            "p1": defaultdict(bool),
            "p2": defaultdict(bool),
        }
        # Active terrain (None or string like "grassyterrain")
        self.terrain: str | None = None
        # Weather (None or string like "sunnyday")
        self.weather: str | None = None
        # Encore: {side: bool}
        self.encore: dict[str, bool] = {"p1": False, "p2": False}
        # Active pokemon species per side (needed for type chart lookups)
        self.active_species: dict[str, str | None] = {"p1": None, "p2": None}
        # Tera type if terastallised
        self.tera_type: dict[str, str | None] = {"p1": None, "p2": None}

    def apply_token(self, token: str) -> None:
        """Update tracked state based on a single sequence vocab token."""
        parts = token.split(":")
        prefix = parts[0]

        if prefix == "status_start" and len(parts) >= 3:
            side, status = parts[1], parts[2]
            self.status[side] = status

        elif prefix == "status_end" and len(parts) >= 3:
            side = parts[1]
            self.status[side] = None

        elif prefix == "side_condition" and len(parts) >= 4:
            side, cond, action = parts[1], parts[2], parts[3]
            self.side_conditions[side][cond] = (action == "start")

        elif prefix == "field" and len(parts) >= 3:
            field_name, action = parts[1], parts[2]
            if "terrain" in field_name:
                if action == "start":
                    self.terrain = field_name
                else:
                    self.terrain = None
            elif field_name == "trickroom":
                pass  # not tracked for failure detection

        elif prefix == "weather" and len(parts) >= 2:
            w = parts[1]
            self.weather = None if w == "none" else w

        elif prefix == "switch" and len(parts) >= 3:
            side, species = parts[1], parts[2]
            self.active_species[side] = species
            # Clear volatile state on switch
            self.encore[side] = False

        elif prefix == "forme_change" and len(parts) >= 4:
            side = parts[1]
            # forme_change:pN:tera:<Type>
            if parts[2] == "tera" and len(parts) >= 4:
                self.tera_type[side] = parts[3]
            elif parts[2] == "species":
                self.active_species[side] = parts[3]

    def get_active_types(self, side: str) -> list[str]:
        """Return the active type list for *side*, accounting for Tera."""
        tera = self.tera_type.get(side)
        if tera:
            return [tera]
        species = self.active_species.get(side)
        if species:
            return get_species_types(species)
        return ["Normal"]


# ---------------------------------------------------------------------------
# Pokemon Showdown battle JSON parsing
# ---------------------------------------------------------------------------

def _side_from_slot(slot: str) -> str:
    """'p1a' -> 'p1', 'p2b' -> 'p2'."""
    if slot.startswith("p1"):
        return "p1"
    if slot.startswith("p2"):
        return "p2"
    return slot[:2] if len(slot) >= 2 else slot


def _other_side(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def _parse_showdown_log(log_lines: list[str]) -> list[dict]:
    """
    Parse a raw Showdown protocol log (list of pipe-separated lines) and
    return a list of turn dicts:

        {
          "turn_num": int,
          "events": [   # ordered list of event dicts
              {"type": str, "side": str, "data": dict}
          ]
        }

    Only the event types we care about are extracted.
    """
    turns: list[dict] = []
    current_turn: dict | None = None

    for raw in log_lines:
        raw = raw.strip()
        if not raw or not raw.startswith("|"):
            continue
        parts = raw.split("|")
        if len(parts) < 2:
            continue
        kind = parts[1]

        if kind == "turn":
            if current_turn is not None:
                turns.append(current_turn)
            turn_num = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            current_turn = {"turn_num": turn_num, "events": []}
            continue

        if current_turn is None:
            continue

        ev = None

        if kind in ("move",):
            # |move|p1a: Name|Move Name|p2a: Name|[effect]
            actor_slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            move_name = parts[3].strip() if len(parts) > 3 else ""
            target_slot = parts[4].split(":")[0].strip() if len(parts) > 4 else ""
            ev = {
                "type": "move",
                "actor": _side_from_slot(actor_slot),
                "move": move_name.lower().replace(" ", "").replace("-", ""),
                "move_raw": move_name,
                "target": _side_from_slot(target_slot) if target_slot else "",
            }

        elif kind in ("-damage", "-heal"):
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            hp_str = parts[3].strip() if len(parts) > 3 else "0/0"
            ev = {
                "type": "damage" if kind == "-damage" else "heal",
                "side": _side_from_slot(slot),
                "hp_str": hp_str,
                "from": parts[4] if len(parts) > 4 else "",
            }

        elif kind == "-status":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            status = parts[3].strip() if len(parts) > 3 else ""
            ev = {"type": "status_start", "side": _side_from_slot(slot), "status": status}

        elif kind == "-curestatus":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            ev = {"type": "status_end", "side": _side_from_slot(slot)}

        elif kind == "-immune":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            ev = {"type": "immune", "side": _side_from_slot(slot)}

        elif kind in ("-sidestart", "-sideend"):
            side_str = parts[2].strip() if len(parts) > 2 else ""
            side = "p1" if side_str.startswith("p1") else "p2"
            cond = parts[3].strip().lower().replace("move: ", "").replace(" ", "") \
                if len(parts) > 3 else ""
            ev = {
                "type": "side_condition",
                "side": side,
                "condition": cond,
                "action": "start" if kind == "-sidestart" else "end",
            }

        elif kind == "-fieldstart":
            cond = parts[2].strip().lower().replace("move: ", "").replace(" ", "") \
                if len(parts) > 2 else ""
            ev = {"type": "field_start", "condition": cond}

        elif kind == "-fieldend":
            cond = parts[2].strip().lower().replace("move: ", "").replace(" ", "") \
                if len(parts) > 2 else ""
            ev = {"type": "field_end", "condition": cond}

        elif kind in ("-weather",):
            weather = parts[2].strip().lower() if len(parts) > 2 else "none"
            ev = {"type": "weather", "weather": weather}

        elif kind == "switch":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            species_full = parts[3].strip() if len(parts) > 3 else ""
            species = species_full.split(",")[0].strip()
            ev = {
                "type": "switch",
                "side": _side_from_slot(slot),
                "species": species.lower().replace(" ", "-").replace("'", ""),
            }

        elif kind == "-start":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            effect = parts[3].strip().lower().replace("move: ", "") \
                if len(parts) > 3 else ""
            ev = {"type": "volatile_start", "side": _side_from_slot(slot), "effect": effect}

        elif kind == "-end":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            effect = parts[3].strip().lower().replace("move: ", "") \
                if len(parts) > 3 else ""
            ev = {"type": "volatile_end", "side": _side_from_slot(slot), "effect": effect}

        elif kind == "faint":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            ev = {"type": "faint", "side": _side_from_slot(slot)}

        elif kind == "detailschange":
            slot = parts[2].split(":")[0].strip() if len(parts) > 2 else ""
            species_full = parts[3].strip() if len(parts) > 3 else ""
            species = species_full.split(",")[0].strip()
            ev = {
                "type": "forme_change",
                "side": _side_from_slot(slot),
                "species": species.lower().replace(" ", "-"),
            }

        if ev is not None:
            current_turn["events"].append(ev)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


def _parse_battle_json(battle: dict) -> tuple[str, list[dict]]:
    """
    Extract battle_id and a list of turn dicts from a battle JSON object.

    Handles two common layouts:
      1. {"id": ..., "log": [...]}        — Showdown server export format
      2. {"battle_id": ..., "turns": [...]}  — preprocessed training format
    """
    battle_id: str = (
        battle.get("id")
        or battle.get("battle_id")
        or battle.get("battleid")
        or "unknown"
    )

    # --- Layout 1: raw log (list of pipe strings or single joined string) ---
    raw_log = battle.get("log")
    if raw_log is not None:
        if isinstance(raw_log, str):
            lines = raw_log.splitlines()
        elif isinstance(raw_log, list):
            # Either a list of strings or a list of dicts
            if raw_log and isinstance(raw_log[0], str):
                lines = raw_log
            else:
                # Might be turn-level dicts in the log
                lines = []
                for item in raw_log:
                    if isinstance(item, str):
                        lines.append(item)
        else:
            lines = []
        turns = _parse_showdown_log(lines)
        return str(battle_id), turns

    # --- Layout 2: pre-parsed turns with turn_events_v1 token lists ----------
    turns_raw = battle.get("turns") or battle.get("turn_data") or []
    turns: list[dict] = []
    for turn_raw in turns_raw:
        turn_num = turn_raw.get("turn", turn_raw.get("turn_num", 0))
        events: list[dict] = []

        # If the turn has a token sequence, convert those back to event dicts
        token_seq: list[str] = turn_raw.get("turn_events_v1") or turn_raw.get("events") or []
        for tok in token_seq:
            parts = tok.split(":")
            ev_type = parts[0]
            if ev_type == "move" and len(parts) >= 3:
                actor = parts[1]
                move_id = parts[2]
                target = parts[3] if len(parts) > 3 else ""
                events.append({
                    "type": "move",
                    "actor": actor,
                    "move": move_id,
                    "move_raw": move_id,
                    "target": target,
                })
            elif ev_type in ("damage", "heal") and len(parts) >= 3:
                events.append({"type": ev_type, "side": parts[1], "hp_str": parts[2]})
            elif ev_type == "status_start" and len(parts) >= 3:
                events.append({"type": "status_start", "side": parts[1], "status": parts[2]})
            elif ev_type == "status_end" and len(parts) >= 3:
                events.append({"type": "status_end", "side": parts[1]})
            elif ev_type == "side_condition" and len(parts) >= 4:
                events.append({
                    "type": "side_condition",
                    "side": parts[1],
                    "condition": parts[2],
                    "action": parts[3],
                })
            elif ev_type == "field" and len(parts) >= 3:
                events.append({
                    "type": "field_start" if parts[2] == "start" else "field_end",
                    "condition": parts[1],
                })
            elif ev_type == "weather" and len(parts) >= 2:
                events.append({"type": "weather", "weather": parts[1]})
            elif ev_type == "switch" and len(parts) >= 3:
                events.append({"type": "switch", "side": parts[1], "species": parts[2]})

        turns.append({"turn_num": turn_num, "events": events})

    return str(battle_id), turns


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

# Status move ID sets — moves whose primary goal is to apply a status
_STATUS_MOVES: dict[str, str] = {
    # move_id -> status it attempts to apply
    "thunderwave": "par", "glare": "par", "stunspore": "par",
    "toxic": "tox", "poisonpowder": "psn", "poisongas": "psn",
    "poisonjab": "psn", "sludgebomb": "psn",
    "willowisp": "brn", "will-o-wisp": "brn",
    "sleeppowder": "slp", "spore": "slp", "hypnosis": "slp",
    "yawn": "slp",
    "icebeam": "frz", "blizzard": "frz",
}

# Screen / veil conditions (as they appear in side_condition events)
_SCREEN_CONDITIONS = {"reflect", "lightscreen", "auroraveil"}

# Terrain names (as they appear in field events)
_TERRAIN_NAMES = {"grassyterrain", "electricterrain", "psychicterrain", "mistyterrain"}

# Moves that set screens
_SCREEN_MOVES: dict[str, str] = {
    "reflect": "reflect",
    "lightscreen": "lightscreen",
    "auroraveil": "auroraveil",
}

# Moves that set terrain
_TERRAIN_MOVES: dict[str, str] = {
    "grassyterrain": "grassyterrain",
    "electricterrain": "electricterrain",
    "psychicterrain": "psychicterrain",
    "mistyterrain": "mistyterrain",
}


def _normalise_move_id(raw: str) -> str:
    """Lowercase, strip spaces and hyphens for lookup."""
    return raw.lower().replace(" ", "").replace("-", "")


def analyse_battle(battle_id: str, turns: list[dict]) -> BattleReport:
    """
    Walk the turn list, apply state tracking, detect failures and compute
    move effectiveness statistics.
    """
    report = BattleReport(battle_id=battle_id)
    state = BattleStateTracker()

    # Track the last N moves per (actor) to detect immunity failures
    # last_move_used[side] = (move_id, turn_num)
    last_move_used: dict[str, tuple[str, int]] = {}

    # For consecutive failure detection: track failure indices
    consecutive_window: list[int] = []  # turn indices of recent failures
    BIG_FAILURE_THRESHOLD = 3

    def _record_failure(turn_num: int, actor: str, target: str,
                        move: str, reason: str, prev_state: str) -> None:
        report.failures.append(FailureRecord(
            turn=turn_num, actor=actor, target=target,
            move=move, reason=reason, prev_state=prev_state,
        ))

    # Maps immunity events (which appear after the move) to the move that
    # caused them.  We buffer the last move and resolve on seeing -immune.
    pending_move: dict[str, dict | None] = {"p1": None, "p2": None}

    failure_turn_indices: list[int] = []

    for turn_dict in turns:
        turn_num: int = turn_dict["turn_num"]
        events: list[dict] = turn_dict["events"]
        turn_had_failure = False

        # --- Apply initial switch events to update active species
        for ev in events:
            if ev["type"] == "switch":
                state.apply_token(f"switch:{ev['side']}:{ev['species']}:1")

        for ev in events:
            ev_type = ev["type"]

            # ----------------------------------------------------------------
            # Move events — cache pending move per actor
            # ----------------------------------------------------------------
            if ev_type == "move":
                actor = ev.get("actor", "")
                move_id = _normalise_move_id(ev.get("move", ""))
                target = ev.get("target", "")
                pending_move[actor] = {
                    "actor": actor,
                    "move": move_id,
                    "move_raw": ev.get("move_raw", move_id),
                    "target": target,
                    "turn_num": turn_num,
                }
                last_move_used[actor] = (move_id, turn_num)

                # --- Status-move failure: target already has that status ---
                expected_status = _STATUS_MOVES.get(move_id)
                if expected_status and target:
                    current = state.status.get(target)
                    if current == expected_status:
                        _record_failure(
                            turn_num, actor, target, move_id,
                            reason="duplicate_status",
                            prev_state=f"{target} already has {expected_status}",
                        )
                        turn_had_failure = True

                # --- Screen / veil failure: already active ---
                screen_cond = _SCREEN_MOVES.get(move_id)
                if screen_cond:
                    if state.side_conditions[actor].get(screen_cond):
                        _record_failure(
                            turn_num, actor, actor, move_id,
                            reason="active_screen",
                            prev_state=f"{actor} {screen_cond} already active",
                        )
                        turn_had_failure = True

                # --- Terrain failure: same terrain already active ---
                terrain_target = _TERRAIN_MOVES.get(move_id)
                if terrain_target:
                    if state.terrain == terrain_target:
                        _record_failure(
                            turn_num, actor, "", move_id,
                            reason="active_terrain",
                            prev_state=f"terrain {terrain_target} already active",
                        )
                        turn_had_failure = True

                # --- Encore failure: target already in encore ---
                if move_id == "encore":
                    if state.encore.get(target, False):
                        _record_failure(
                            turn_num, actor, target, move_id,
                            reason="active_encore",
                            prev_state=f"{target} already in encore",
                        )
                        turn_had_failure = True

            # ----------------------------------------------------------------
            # Immune event — the move that preceded it had no effect
            # ----------------------------------------------------------------
            elif ev_type == "immune":
                target_side = ev.get("side", "")
                # The actor is the other side
                actor_side = _other_side(target_side) if target_side else ""
                pm = pending_move.get(actor_side)
                move_id = pm["move"] if pm else "unknown"
                _record_failure(
                    turn_num, actor_side, target_side, move_id,
                    reason="move_outcome_immunity",
                    prev_state=(
                        f"{target_side} immune to {move_id}; "
                        f"active: {state.active_species.get(target_side, '?')}"
                    ),
                )
                turn_had_failure = True

            # ----------------------------------------------------------------
            # Damage events — compute effectiveness
            # ----------------------------------------------------------------
            elif ev_type == "damage":
                target_side = ev.get("side", "")
                hp_str = ev.get("hp_str", "")
                # Find the actor (the other side's last move)
                actor_side = _other_side(target_side) if target_side else ""
                pm = pending_move.get(actor_side)
                if pm:
                    move_id = pm["move"]
                    move_type = get_move_type(move_id)
                    if move_type:
                        defending_types = state.get_active_types(target_side)
                        multiplier = type_effectiveness(move_type, defending_types)
                        label = effectiveness_label(multiplier)
                        report.move_effectiveness[label] += 1

            # ----------------------------------------------------------------
            # State-update tokens
            # ----------------------------------------------------------------
            elif ev_type == "status_start":
                side = ev.get("side", "")
                status = ev.get("status", "")
                state.apply_token(f"status_start:{side}:{status}")

            elif ev_type == "status_end":
                side = ev.get("side", "")
                state.apply_token(f"status_end:{side}:x")

            elif ev_type == "side_condition":
                side = ev.get("side", "")
                cond = ev.get("condition", "")
                action = ev.get("action", "start")
                state.apply_token(f"side_condition:{side}:{cond}:{action}")

            elif ev_type in ("field_start", "field_end"):
                cond = ev.get("condition", "")
                action = "start" if ev_type == "field_start" else "end"
                state.apply_token(f"field:{cond}:{action}")

            elif ev_type == "weather":
                state.apply_token(f"weather:{ev.get('weather', 'none')}")

            elif ev_type == "forme_change":
                side = ev.get("side", "")
                species = ev.get("species", "")
                state.apply_token(f"forme_change:{side}:species:{species}")

            elif ev_type == "volatile_start":
                effect = ev.get("effect", "")
                side = ev.get("side", "")
                if effect == "encore":
                    state.encore[side] = True

            elif ev_type == "volatile_end":
                effect = ev.get("effect", "")
                side = ev.get("side", "")
                if effect == "encore":
                    state.encore[side] = False

        # End-of-turn: note which turns had failures
        if turn_had_failure:
            failure_turn_indices.append(turn_num)

        # Clear pending moves each turn
        pending_move = {"p1": None, "p2": None}

    # -----------------------------------------------------------------------
    # Detect "big failures": 3+ consecutive failure turns
    # -----------------------------------------------------------------------
    big_failures = 0
    if len(failure_turn_indices) >= BIG_FAILURE_THRESHOLD:
        run = 1
        for i in range(1, len(failure_turn_indices)):
            if failure_turn_indices[i] == failure_turn_indices[i - 1] + 1:
                run += 1
                if run >= BIG_FAILURE_THRESHOLD:
                    big_failures += 1
                    run = 1  # Reset after counting to avoid double-counting
            else:
                run = 1
    report.big_failures_count = big_failures

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    eff = report.move_effectiveness
    total_eff = sum(eff.values())
    fail_counts: dict[str, int] = defaultdict(int)
    for f in report.failures:
        fail_counts[f.reason] += 1

    report.summary = (
        f"Battles analysed: {battle_id}. "
        f"Total failures: {len(report.failures)} "
        f"(immunity={fail_counts['move_outcome_immunity']}, "
        f"status={fail_counts['duplicate_status']}, "
        f"terrain={fail_counts['active_terrain']}, "
        f"screen={fail_counts['active_screen']}, "
        f"encore={fail_counts['active_encore']}). "
        f"Big failures (3+ consecutive turns): {big_failures}. "
        f"Damage events: {total_eff} "
        f"(SE={eff['super_effective']}, resist={eff['resisted']}, "
        f"neutral={eff['neutral']}, immune={eff['immune']})."
    )

    return report


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_battle_files(paths: list[Path]) -> list[tuple[Path, dict]]:
    """Load each JSON file and return (path, battle_dict) pairs."""
    results = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # Handle both a single battle dict and a list of battle dicts
            if isinstance(data, list):
                for item in data:
                    results.append((p, item))
            else:
                results.append((p, data))
        except Exception as exc:
            print(f"[WARN] Failed to load {p}: {exc}", file=sys.stderr)
    return results


def write_json_report(report: BattleReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = report.battle_id.replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"{safe_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


_CSV_FIELDNAMES = [
    "battle_id",
    "total_failures",
    "big_failures",
    "immunity_count",
    "status_count",
    "terrain_count",
    "reflect_count",
    "encore_count",
    "super_effective_count",
    "resisted_count",
    "neutral_count",
    "immune_damage_count",
]


def append_csv_row(writer: "csv.DictWriter", report: BattleReport) -> None:
    fail_counts: dict[str, int] = defaultdict(int)
    for f in report.failures:
        fail_counts[f.reason] += 1

    writer.writerow({
        "battle_id": report.battle_id,
        "total_failures": len(report.failures),
        "big_failures": report.big_failures_count,
        "immunity_count": fail_counts["move_outcome_immunity"],
        "status_count": fail_counts["duplicate_status"],
        "terrain_count": fail_counts["active_terrain"],
        "reflect_count": fail_counts["active_screen"],
        "encore_count": fail_counts["active_encore"],
        "super_effective_count": report.move_effectiveness["super_effective"],
        "resisted_count": report.move_effectiveness["resisted"],
        "neutral_count": report.move_effectiveness["neutral"],
        "immune_damage_count": report.move_effectiveness["immune"],
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def collect_json_files(input_path: Path) -> list[Path]:
    """Collect all .json files from a file or directory path."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        found = list(input_path.glob("*.json")) + list(input_path.glob("**/*.json"))
        return sorted(set(found))
    print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose state-transition failures and move effectiveness in battle logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single battle JSON file or a directory of battle JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write per-battle JSON reports into.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path for the aggregated CSV summary file.",
    )
    parser.add_argument(
        "--sequence-vocab",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to sequence_vocab_5.json (or equivalent). "
            "Loaded for reference; not required for analysis."
        ),
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N battles (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-battle summaries to stdout.",
    )
    args = parser.parse_args()

    # Optional: load vocab for reference (not required for analysis logic)
    vocab: dict | None = None
    if args.sequence_vocab:
        try:
            with open(args.sequence_vocab, encoding="utf-8") as f:
                vocab = json.load(f)
            print(f"[INFO] Loaded sequence vocab with {len(vocab)} tokens.")
        except Exception as exc:
            print(f"[WARN] Could not load sequence vocab: {exc}", file=sys.stderr)

    # Collect input files
    json_files = collect_json_files(args.input)
    if not json_files:
        print("[ERROR] No JSON files found at the given path.", file=sys.stderr)
        sys.exit(1)

    if args.max_battles is not None:
        json_files = json_files[: args.max_battles]

    print(f"[INFO] Processing {len(json_files)} battle file(s)...")

    # Set up CSV writer
    csv_file = None
    csv_writer = None
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES)
        csv_writer.writeheader()

    # Aggregated totals for final print summary
    totals: dict[str, int] = defaultdict(int)
    processed = 0

    try:
        for path, battle in load_battle_files(json_files):
            try:
                battle_id, turns = _parse_battle_json(battle)
                if not turns:
                    # Try treating the file-level id as battle_id
                    battle_id = str(path.stem)

                report = analyse_battle(battle_id, turns)

                if args.output_json:
                    write_json_report(report, args.output_json)

                if csv_writer:
                    append_csv_row(csv_writer, report)

                if args.verbose:
                    print(f"  {report.summary}")

                # Accumulate totals
                totals["battles"] += 1
                totals["failures"] += len(report.failures)
                totals["big_failures"] += report.big_failures_count
                for k, v in report.move_effectiveness.items():
                    totals[k] += v
                for f in report.failures:
                    totals[f"reason_{f.reason}"] += 1

                processed += 1

            except Exception as exc:
                print(f"[WARN] Error analysing {path}: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

    finally:
        if csv_file:
            csv_file.close()

    # -----------------------------------------------------------------------
    # Print aggregate summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"DIAGNOSTIC SUMMARY — {processed} battle(s) analysed")
    print("=" * 60)
    print(f"  Total state-transition failures : {totals['failures']}")
    print(f"    move_outcome_immunity          : {totals.get('reason_move_outcome_immunity', 0)}")
    print(f"    duplicate_status               : {totals.get('reason_duplicate_status', 0)}")
    print(f"    active_terrain                 : {totals.get('reason_active_terrain', 0)}")
    print(f"    active_screen (reflect/screen) : {totals.get('reason_active_screen', 0)}")
    print(f"    active_encore                  : {totals.get('reason_active_encore', 0)}")
    print(f"  Big failures (3+ consec. turns)  : {totals['big_failures']}")
    print()
    print("  Move effectiveness distribution:")
    print(f"    super_effective : {totals.get('super_effective', 0)}")
    print(f"    neutral         : {totals.get('neutral', 0)}")
    print(f"    resisted        : {totals.get('resisted', 0)}")
    print(f"    immune          : {totals.get('immune', 0)}")
    print("=" * 60)

    if args.output_json:
        print(f"  Per-battle JSON reports written to: {args.output_json}/")
    if args.output_csv:
        print(f"  Aggregated CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
