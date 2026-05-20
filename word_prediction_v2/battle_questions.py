from __future__ import annotations

from typing import Any, Dict

STATUS_KEYWORDS = {
    "toxic", "poisonpowder", "stunspore", "thunderwave", "spore", "hypnosis",
    "sleeppowder", "willowisp", "leechseed", "yawn", "encore",
}
SETUP_KEYWORDS = {
    "swordsdance", "nastyplot", "bulkup", "calmmind", "dragondance",
    "quiverdance", "shellsmash", "agility", "curse", "growth", "trailblaze",
}
RECOVER_KEYWORDS = {
    "recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun",
    "synthesis", "rest",
}
HAZARD_SET_KEYWORDS = {
    "stealthrock", "spikes", "stickyweb", "toxicspikes", "stoneaxe", "ceaselessedge",
}
HAZARD_CLEAR_KEYWORDS = {
    "rapidspin", "defog", "tidyup", "courtchange", "mortalspin",
}
PHAZE_KEYWORDS = {
    "roar", "whirlwind", "dragontail", "circlethrow", "haze", "clearsmog",
}


def _hazard_pressure(conditions: dict) -> float:
    return (
        float(conditions.get("stealthrock", 0) or 0) * 1.3
        + float(conditions.get("spikes", 0) or 0) * 0.9
        + float(conditions.get("toxicspikes", 0) or 0) * 0.8
        + float(conditions.get("stickyweb", 0) or 0) * 0.7
    )


def _normalize_move_name(move_payload: dict) -> str:
    move_name = str(move_payload.get("move") or move_payload.get("id") or "")
    return move_name.lower().replace(" ", "").replace("-", "")


def _active_request_flags(data: dict) -> dict:
    active = data.get("active") or []
    if isinstance(active, list) and active:
        first = active[0]
        if isinstance(first, dict):
            return first
    return {}


def _allow_voluntary_switches(data: dict) -> bool:
    active = _active_request_flags(data)
    return not bool(active.get("trapped") or active.get("maybeTrapped"))


def _boost_total(mon: dict) -> int:
    boosts = mon.get("boosts") or {}
    return sum(int(boosts.get(stat, 0) or 0) for stat in ("atk", "spa", "spe"))


def default_question_for_state(data: Dict[str, Any]) -> str:
    legal_moves = data.get("legal_moves") or []
    legal_switches = data.get("legal_switches") or []
    can_voluntary_switch = _allow_voluntary_switches(data)
    battle_state = data.get("battle_state") or {}
    perspective = data.get("perspective_player") or "p1"
    side = battle_state.get(perspective) or {}
    active_uid = side.get("active_uid")
    opp_side = battle_state.get("p2" if perspective == "p1" else "p1") or {}
    opp_active_uid = opp_side.get("active_uid")
    mons = battle_state.get("mons") or {}
    my_active = mons.get(active_uid) or {}
    opp_active = mons.get(opp_active_uid) or {}
    my_hp = float(my_active.get("hp_frac") or 0.0)
    opp_hp = float(opp_active.get("hp_frac") or 0.0)
    opp_status = str(opp_active.get("status") or "")
    opp_observed_moves = list(opp_active.get("observed_moves") or [])
    total_boost = _boost_total(my_active)
    move_names = {_normalize_move_name(move) for move in legal_moves}
    has_setup = any(name in SETUP_KEYWORDS for name in move_names)
    has_status = any(name in STATUS_KEYWORDS for name in move_names)
    has_recover = any(name in RECOVER_KEYWORDS for name in move_names)
    my_alive = 0
    opp_alive = 0
    my_hp_total = 0.0
    opp_hp_total = 0.0
    for uid in side.get("slots") or []:
        if not uid:
            continue
        mon = mons.get(uid) or {}
        if mon.get("fainted"):
            continue
        hp = float(mon.get("hp_frac") or 0.0)
        if hp > 0.0:
            my_alive += 1
            my_hp_total += hp
    for uid in opp_side.get("slots") or []:
        if not uid:
            continue
        mon = mons.get(uid) or {}
        if mon.get("fainted"):
            continue
        hp = float(mon.get("hp_frac") or 0.0)
        if hp > 0.0:
            opp_alive += 1
            opp_hp_total += hp
    material_edge = (my_alive - opp_alive) * 1.6 + (my_hp_total - opp_hp_total)
    high_threat = bool(opp_observed_moves) and my_hp <= 0.45 and opp_hp > 0.35

    if opp_hp <= 0.35 and legal_moves:
        return "can I knock it out now?"
    if material_edge >= 2.0 and opp_hp <= 0.55 and legal_moves:
        return "how do I close this out safely?"
    if high_threat and can_voluntary_switch and legal_switches:
        return "should I preserve this and switch?"
    if my_hp <= 0.12 and can_voluntary_switch and legal_switches and not has_recover:
        return "should I switch out here?"
    if my_hp <= 0.3 and has_recover:
        return "what is the safe play?"
    if material_edge <= -1.5 and not opp_status and has_status:
        return "should I slow this down with status?"
    if my_hp >= 0.85 and opp_hp >= 0.75 and total_boost == 0 and has_setup:
        return "should I setup here?"
    if legal_moves:
        return "should I attack now?"
    return "what is the safe play?"
