from __future__ import annotations

from typing import List

from .lexicon import LexiconEntry, _build_entry


def battle_lexicon() -> List[LexiconEntry]:
    return [
        _build_entry("attack", ["damage", "offense", "hit", "pressure", "strike"], ["atack", "attak"]),
        _build_entry("switch", ["pivot", "retreat", "swap", "rotate", "escape"], ["swtich", "swich"]),
        _build_entry("finish", ["ko", "close", "lethal", "end", "secure"], ["finsh", "finnish"]),
        _build_entry("stabilize", ["safe", "steady", "recover", "reset", "survive"], ["stablize", "stabilise"]),
        _build_entry("preserve", ["save", "protect", "keep", "conserve", "healthy"], ["preserv", "prezerve"]),
        _build_entry("setup", ["boost", "buff", "charge", "snowball", "stack"], ["setip", "setupp"]),
        _build_entry("scout", ["reveal", "learn", "probe", "info", "test"], ["scuot", "scoutt"]),
        _build_entry("risk", ["guess", "volatile", "gamble", "uncertain", "danger"], ["rsk", "riks"]),
        _build_entry("status", ["burn", "poison", "sleep", "cripple", "hinder"], ["stauts", "statuz"]),
        _build_entry("wall", ["tank", "absorb", "defend", "sponge", "endure"], ["waal", "wll"]),
        _build_entry("priority", ["quick", "first", "cleanup", "urgent", "pickoff"], ["priorty", "prority"]),
        _build_entry("tempo", ["momentum", "pace", "flow", "initiative", "cycle"], ["temop", "teempo"]),
        _build_entry("revenge", ["retaliate", "answer", "punish", "return", "respond"], ["revnge", "revange"]),
        _build_entry("safeko", ["reliable", "accurate", "clean", "convert", "certain"], ["saefko", "safeko"]),
        _build_entry("sacrifice", ["trade", "fodder", "spend", "drop", "give"], ["sacrafice", "sacrfice"]),
    ]
