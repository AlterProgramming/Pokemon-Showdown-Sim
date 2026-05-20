from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Sequence

from .battle_inquiry import InquiryAnswer, answer_battle_inquiry, build_battle_prompt_tokens
from .model import WordEmbeddingModel
from .policy_config import get_active_policy_config
from .pokemon_move_metadata import move_metadata
from .pokemon_type_utils import move_type, species_types, type_effectiveness
from .semantic import semantic_word_bias
from .semantic_actions import action_retrieval_bias as _action_retrieval_bias
from .text import normalize_token


STATUS_KEYWORDS = {
    "toxic", "poisonpowder", "stunspore", "thunderwave", "spore", "hypnosis",
    "sleeppowder", "willowisp", "leechseed", "yawn", "encore",
}
SETUP_KEYWORDS = {
    "swordsdance", "nastyplot", "bulkup", "calmmind", "dragondance",
    "quiverdance", "shellsmash", "agility", "curse", "growth", "trailblaze",
    "irondefense", "acidarmor", "cottonguard",
}
SCOUT_KEYWORDS = {
    "protect", "detect", "substitute", "uturn", "voltswitch", "flipturn",
    "partingshot", "batonpass",
}
RECOVER_KEYWORDS = {
    "recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun",
    "synthesis", "rest",
}
SPECIAL_SELF_DROP_MOVES = {"dracometeor", "overheat", "leafstorm", "makeitrain", "fleurcannon"}
DEFENSE_DROP_MOVES = {"closecombat", "superpower", "vcreate"}
HAZARD_SET_KEYWORDS = {
    "stealthrock", "spikes", "stickyweb", "toxicspikes", "stoneaxe", "ceaselessedge",
}
HAZARD_CLEAR_KEYWORDS = {
    "rapidspin", "defog", "tidyup", "courtchange", "mortalspin",
}
PHAZE_KEYWORDS = {
    "roar", "whirlwind", "dragontail", "circlethrow", "haze", "clearsmog",
}
ITEM_PROGRESS_KEYWORDS = {
    "knockoff", "trick", "switcheroo",
}
STICKY_GENERIC_ATTACKS = {
    "earthquake",
    "knockoff",
    "icebeam",
    "thunderbolt",
    "closecombat",
    "shadowball",
    "earthpower",
    "playrough",
    "bugbuzz",
    "scald",
}
RISKY_NON_PRIORITY_CLOSEOUT_MOVES = {
    "bravebird",
    "wavecrash",
    "flareblitz",
    "woodhammer",
    "doubleedge",
    "highjumpkick",
    "jumpkick",
    "closecombat",
    "superpower",
    "headlongrush",
    "glaiverush",
    "vcreate",
}


def _normalize_move_name(move_payload: Dict[str, Any]) -> str:
    move_name = str(move_payload.get("move") or move_payload.get("id") or "")
    return _normalize_move_name_text(move_name)


@lru_cache(maxsize=1024)
def _normalize_move_name_text(move_name: str) -> str:
    return move_name.lower().replace(" ", "").replace("-", "")


def _normalized_text(value: str) -> str:
    return _normalized_text_cached(value)


@lru_cache(maxsize=256)
def _normalized_text_cached(value: str) -> str:
    return value.lower().replace("?", " ").replace(",", " ").replace("-", " ")


def _move_priority_bucket(primary_word: str, move_payload: Dict[str, Any]) -> tuple[int, int]:
    name = _normalize_move_name(move_payload)
    base_slot = int(move_payload.get("slot") or 99)

    if primary_word == "status":
        if name in STATUS_KEYWORDS:
            return (0, base_slot)
        if name in RECOVER_KEYWORDS:
            return (1, base_slot)
        return (5, base_slot)

    if primary_word == "setup":
        if name in SETUP_KEYWORDS:
            return (0, base_slot)
        if name in SCOUT_KEYWORDS:
            return (2, base_slot)
        return (5, base_slot)

    if primary_word == "scout":
        if name in SCOUT_KEYWORDS:
            return (0, base_slot)
        if name in STATUS_KEYWORDS:
            return (2, base_slot)
        return (5, base_slot)

    if primary_word in {"stabilize", "preserve", "wall"}:
        if name in RECOVER_KEYWORDS:
            return (0, base_slot)
        if name in STATUS_KEYWORDS:
            return (2, base_slot)
        return (4, base_slot)

    if primary_word in {"attack", "finish", "risk"}:
        if name in STATUS_KEYWORDS or name in SETUP_KEYWORDS or name in RECOVER_KEYWORDS:
            return (4, base_slot)
        return (0, base_slot)

    return (2, base_slot)


def _move_word_score(word: str, move_payload: Dict[str, Any]) -> float:
    name = _normalize_move_name(move_payload)
    metadata = move_metadata(str(move_payload.get("id") or move_payload.get("move") or ""))
    category = str(metadata.get("category") or "")
    accuracy = float(metadata.get("accuracy") or 0.0)
    priority = float(metadata.get("priority") or 0.0)
    self_switch = bool(metadata.get("selfSwitch"))
    damaging = category in {"Physical", "Special"}

    if word == "finish":
        if name in STATUS_KEYWORDS or name in SETUP_KEYWORDS or name in RECOVER_KEYWORDS:
            return -1.5
        return 2.0
    if word == "attack":
        if name in STATUS_KEYWORDS or name in SETUP_KEYWORDS or name in RECOVER_KEYWORDS:
            return -1.0
        return 1.4
    if word == "setup":
        if name in SETUP_KEYWORDS:
            return 2.2
        if name in SCOUT_KEYWORDS:
            return 0.5
        return -0.8
    if word == "status":
        if name in STATUS_KEYWORDS:
            return 2.0
        if name in RECOVER_KEYWORDS:
            return 0.4
        return -0.7
    if word == "scout":
        if name in SCOUT_KEYWORDS:
            return 1.8
        return -0.4
    if word == "priority":
        if not damaging:
            return -0.9
        if priority > 0:
            return 2.4
        return 0.2
    if word == "tempo":
        if self_switch or name in SCOUT_KEYWORDS:
            return 1.5
        if damaging:
            return 1.0
        if name in STATUS_KEYWORDS or name in SETUP_KEYWORDS or name in RECOVER_KEYWORDS:
            return -0.7
        return 0.2
    if word == "revenge":
        if not damaging:
            return -1.0
        if priority > 0:
            return 2.0
        return 1.3
    if word == "safeko":
        if not damaging:
            return -1.3
        score = 1.1
        if accuracy >= 95:
            score += 0.8
        elif accuracy and accuracy < 90:
            score -= 0.7
        if priority > 0:
            score += 0.3
        return score
    if word == "sacrifice":
        if damaging:
            return 1.0
        if self_switch or name in SCOUT_KEYWORDS:
            return 0.4
        if name in RECOVER_KEYWORDS or name in SETUP_KEYWORDS:
            return -1.1
        return -0.2
    if word in {"stabilize", "preserve", "wall"}:
        if name in RECOVER_KEYWORDS:
            return 1.8
        if name in STATUS_KEYWORDS:
            return 1.0
        if name in SCOUT_KEYWORDS:
            return 0.5
        return -0.5
    if word == "risk":
        if name in STATUS_KEYWORDS or name in RECOVER_KEYWORDS:
            return -0.6
        return 0.8
    return 0.0


def _attack_pressure_score(
    move_payload: Dict[str, Any],
    *,
    my_species: str,
    opp_species: str,
) -> float:
    metadata = move_metadata(str(move_payload.get("id") or move_payload.get("move") or ""))
    category = str(metadata.get("category") or "")
    if category not in {"Physical", "Special"}:
        return -10.0
    base_power = float(metadata.get("basePower") or 0.0)
    accuracy = float(metadata.get("accuracy") or 0.0)
    priority = float(metadata.get("priority") or 0.0)
    inferred_move_type = str(metadata.get("type") or move_type(str(move_payload.get("id") or move_payload.get("move") or "")) or "")
    score = min(base_power / 55.0, 2.8)
    if accuracy:
        score -= max(0.0, (100.0 - accuracy) / 60.0)
    if priority > 0:
        score += 0.25
    my_types = species_types(my_species)
    opp_types = species_types(opp_species)
    if inferred_move_type and opp_types:
        multiplier = type_effectiveness(inferred_move_type, opp_types)
        if multiplier == 0.0:
            score -= 4.0
        elif multiplier >= 4.0:
            score += 3.5
        elif multiplier > 1.0:
            score += 2.0
        elif multiplier < 1.0:
            score -= 1.1
    if inferred_move_type and my_types and inferred_move_type in my_types:
        score += 0.5
    return score


def _best_move_for_predictions(predictions: Sequence[Any], moves: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored_moves: list[tuple[float, int, Dict[str, Any]]] = []
    for move in moves:
        total = 0.0
        for pred in predictions:
            total += float(pred.score) * _move_word_score(str(pred.word), move)
        slot = int(move.get("slot") or 99)
        scored_moves.append((total, -slot, move))
    scored_moves.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return scored_moves[0][2]


def _active_hp(battle_state: Dict[str, Any], perspective_player: str) -> float:
    side = battle_state.get(perspective_player) or {}
    active_uid = side.get("active_uid")
    mons = battle_state.get("mons") or {}
    active_mon = mons.get(active_uid) or {}
    return float(active_mon.get("hp_frac") or 0.0)


def _active_mon(battle_state: Dict[str, Any], perspective_player: str) -> Dict[str, Any]:
    side = battle_state.get(perspective_player) or {}
    active_uid = side.get("active_uid")
    mons = battle_state.get("mons") or {}
    return mons.get(active_uid) or {}


def _opponent_mon(battle_state: Dict[str, Any], perspective_player: str) -> Dict[str, Any]:
    other = "p2" if perspective_player == "p1" else "p1"
    side = battle_state.get(other) or {}
    active_uid = side.get("active_uid")
    mons = battle_state.get("mons") or {}
    return mons.get(active_uid) or {}


def _healthy_team_state(battle_state: Dict[str, Any], player: str) -> tuple[int, float]:
    side = battle_state.get(player) or {}
    mons = battle_state.get("mons") or {}
    healthy = 0
    hp_total = 0.0
    for uid in side.get("slots") or []:
        if not uid:
            continue
        mon = mons.get(uid) or {}
        if mon.get("fainted"):
            continue
        hp = float(mon.get("hp_frac") or 0.0)
        if hp > 0.0:
            healthy += 1
            hp_total += hp
    return healthy, hp_total


def _material_edge(battle_state: Dict[str, Any], perspective_player: str) -> float:
    other = "p2" if perspective_player == "p1" else "p1"
    my_healthy, my_hp_total = _healthy_team_state(battle_state, perspective_player)
    opp_healthy, opp_hp_total = _healthy_team_state(battle_state, other)
    return (my_healthy - opp_healthy) * 1.6 + (my_hp_total - opp_hp_total)


def _slot_mon_for_switch(
    battle_state: Dict[str, Any],
    perspective_player: str,
    switch_payload: Dict[str, Any],
) -> Dict[str, Any]:
    side = battle_state.get(perspective_player) or {}
    slots = list(side.get("slots") or [])
    raw_slot = switch_payload.get("slot")
    try:
        slot_index = int(raw_slot) - 1
    except (TypeError, ValueError):
        slot_index = -1
    if 0 <= slot_index < len(slots):
        uid = slots[slot_index]
        if uid:
            mon = dict((battle_state.get("mons") or {}).get(uid) or {})
            if switch_payload.get("species") and not mon.get("species"):
                mon["species"] = switch_payload.get("species")
            if switch_payload.get("ability") and not mon.get("ability"):
                mon["ability"] = switch_payload.get("ability")
            if switch_payload.get("item") and not mon.get("item"):
                mon["item"] = switch_payload.get("item")
            if switch_payload.get("observed_moves"):
                mon["observed_moves"] = list(switch_payload.get("observed_moves") or [])
            elif switch_payload.get("moves"):
                mon["observed_moves"] = [
                    str(move.get("move") or move.get("id") or "")
                    for move in (switch_payload.get("moves") or [])
                    if isinstance(move, dict)
                ]
            return mon
    mon = {}
    if switch_payload.get("species"):
        mon["species"] = switch_payload.get("species")
    if switch_payload.get("ability"):
        mon["ability"] = switch_payload.get("ability")
    if switch_payload.get("item"):
        mon["item"] = switch_payload.get("item")
    if switch_payload.get("observed_moves"):
        mon["observed_moves"] = list(switch_payload.get("observed_moves") or [])
    elif switch_payload.get("moves"):
        mon["observed_moves"] = [
            str(move.get("move") or move.get("id") or "")
            for move in (switch_payload.get("moves") or [])
            if isinstance(move, dict)
        ]
    return mon


def _bench_switch_score(
    switch_payload: Dict[str, Any],
    *,
    battle_state: Dict[str, Any],
    perspective_player: str,
    opp_species: str,
    opp_ability: str,
    opp_observed_moves: Sequence[str],
    material_edge: float,
) -> float:
    bench_mon = _slot_mon_for_switch(battle_state, perspective_player, switch_payload)
    if not bench_mon:
        return -10.0
    if bench_mon.get("fainted"):
        return -10.0
    bench_hp = float(bench_mon.get("hp_frac") or switch_payload.get("hp_frac") or 0.0)
    if bench_hp <= 0.0:
        return -10.0
    bench_species = str(bench_mon.get("species") or "")
    bench_types = species_types(bench_species)
    score = bench_hp * 2.8
    if bench_hp >= 0.7:
        score += 0.4
    if str(bench_mon.get("status") or ""):
        score -= 0.5
    for opp_move_name in opp_observed_moves:
        opp_meta = move_metadata(opp_move_name)
        opp_move_type = str(opp_meta.get("type") or move_type(opp_move_name) or "")
        opp_category = str(opp_meta.get("category") or "")
        opp_power = float(opp_meta.get("basePower") or 0.0)
        if opp_category not in {"Physical", "Special"} or not opp_move_type:
            continue
        multiplier = type_effectiveness(opp_move_type, bench_types)
        local = min(opp_power / 80.0, 1.8)
        if multiplier == 0.0:
            local -= 1.0
        elif multiplier >= 4.0:
            local += 2.4
        elif multiplier > 1.0:
            local += 1.0
        elif multiplier < 1.0:
            local -= 0.6
        score -= local
    raw_moves = bench_mon.get("observed_moves") or []
    if raw_moves:
        best_attack = -10.0
        best_progress = 0.0
        for move_name in raw_moves:
            move_payload = {"move": move_name, "id": move_name}
            normalized_name = _normalize_move_name(move_payload)
            metadata = move_metadata(str(move_name))
            category = str(metadata.get("category") or "")
            if category in {"Physical", "Special"}:
                pressure = _attack_pressure_score(
                    move_payload,
                    my_species=bench_species,
                    opp_species=opp_species,
                )
                move_type_name = str(metadata.get("type") or move_type(str(move_name)) or "")
                if _ability_immunity_multiplier(opp_ability, move_type_name) == 0.0:
                    pressure -= 4.0
                best_attack = max(best_attack, pressure)
            else:
                if normalized_name in ITEM_PROGRESS_KEYWORDS:
                    best_progress = max(best_progress, 0.8)
                elif normalized_name in STATUS_KEYWORDS:
                    best_progress = max(best_progress, 0.55)
                elif normalized_name in PHAZE_KEYWORDS:
                    best_progress = max(best_progress, 0.6)
                elif normalized_name in HAZARD_SET_KEYWORDS:
                    best_progress = max(best_progress, 0.45)
        if best_attack > -10.0:
            score += min(best_attack, 2.8) * 0.4
        score += best_progress
    if material_edge > 1.5:
        score += 0.3
    if not opp_observed_moves and opp_species:
        score += 0.1
    return score


def _side_conditions(battle_state: Dict[str, Any], player: str) -> Dict[str, float]:
    side = battle_state.get(player) or {}
    conditions = side.get("side_conditions") or side.get("sideConditions") or {}
    if not isinstance(conditions, dict):
        return {}
    return {str(key): float(value or 0.0) for key, value in conditions.items()}


def _hazard_pressure(conditions: Dict[str, float]) -> float:
    return (
        conditions.get("stealthrock", 0.0) * 1.3
        + conditions.get("spikes", 0.0) * 0.9
        + conditions.get("toxicspikes", 0.0) * 0.8
        + conditions.get("stickyweb", 0.0) * 0.7
    )


def _opponent_role_flags(opp_observed_moves: Sequence[str]) -> Dict[str, bool]:
    normalized = {_normalize_move_name_text(str(move_name)) for move_name in opp_observed_moves}
    return {
        "setup": any(name in SETUP_KEYWORDS for name in normalized),
        "recover": any(name in RECOVER_KEYWORDS for name in normalized),
        "hazard": any(name in HAZARD_SET_KEYWORDS for name in normalized),
        "phaze": any(name in PHAZE_KEYWORDS for name in normalized),
        "status": any(name in STATUS_KEYWORDS for name in normalized),
    }


def _real_attack_option_count(moves: Sequence[Dict[str, Any]], *, my_species: str, opp_species: str, opp_ability: str) -> int:
    total = 0
    for move in moves:
        metadata = move_metadata(str(move.get("id") or move.get("move") or ""))
        category = str(metadata.get("category") or "")
        if category not in {"Physical", "Special"}:
            continue
        move_name = str(move.get("id") or move.get("move") or "")
        move_type_name = str(metadata.get("type") or move_type(str(move_name)) or "")
        if _move_blocked_by_ability(ability_name=opp_ability, move_name=move_name, move_type_name=move_type_name):
            continue
        pressure = _attack_pressure_score(move, my_species=my_species, opp_species=opp_species)
        if pressure >= 0.85:
            total += 1
    return total


ABILITY_IMMUNITIES: dict[str, set[str]] = {
    "levitate": {"Ground"},
    "flashfire": {"Fire"},
    "waterabsorb": {"Water"},
    "stormdrain": {"Water"},
    "dryskin": {"Water"},
    "voltabsorb": {"Electric"},
    "lightningrod": {"Electric"},
    "motordrive": {"Electric"},
    "sapsipper": {"Grass"},
    "soundproof": set(),
    "eartheater": {"Ground"},
    "wellbakedbody": {"Fire"},
}
WIND_RIDER_MOVES = {"hurricane", "bleakwindstorm", "gust", "twister", "aircutter", "fairywind", "heatwave"}


def _ability_immunity_multiplier(ability_name: str, move_type_name: str) -> float:
    normalized_ability = _normalize_move_name_text(str(ability_name or ""))
    immunities = ABILITY_IMMUNITIES.get(normalized_ability)
    if not immunities:
        return 1.0
    return 0.0 if str(move_type_name or "") in immunities else 1.0


def _move_blocked_by_ability(*, ability_name: str, move_name: str, move_type_name: str) -> bool:
    if _ability_immunity_multiplier(ability_name, move_type_name) == 0.0:
        return True
    normalized_ability = _normalize_move_name_text(str(ability_name or ""))
    normalized_move = _normalize_move_name_text(str(move_name or ""))
    if normalized_ability == "windrider" and normalized_move in WIND_RIDER_MOVES:
        return True
    return False


def _boost_total(mon: Dict[str, Any]) -> int:
    boosts = mon.get("boosts") or {}
    return sum(int(boosts.get(stat, 0) or 0) for stat in ("atk", "spa", "spe"))


def _active_runtime_move_info(active_payload: Sequence[Dict[str, Any]] | None) -> Dict[str, Dict[str, float]]:
    if not active_payload:
        return {}
    first_active = active_payload[0] or {}
    moves = first_active.get("moves") or []
    info: Dict[str, Dict[str, float]] = {}
    for move in moves:
        move_id = normalize_token(str(move.get("id") or move.get("move") or ""))
        if not move_id:
            continue
        pp = float(move.get("pp") or 0.0)
        maxpp = float(move.get("maxpp") or 0.0)
        info[move_id] = {"pp": pp, "maxpp": maxpp}
    return info


def _active_priority_move_ids(active_move_info: Dict[str, Dict[str, float]]) -> set[str]:
    priority_ids: set[str] = set()
    for move_id in active_move_info:
        meta = move_metadata(move_id)
        if float(meta.get("priority") or 0.0) > 0:
            priority_ids.add(move_id)
    return priority_ids


def _state_move_score(
    move_payload: Dict[str, Any],
    *,
    my_hp: float,
    my_status: str,
    opp_hp: float,
    opp_status: str,
    total_boost: int,
    my_species: str,
    opp_species: str,
    opp_observed_moves: Sequence[str],
    my_observed_moves: Sequence[str],
    opp_ability: str,
    opp_item: str,
    boosts: Dict[str, Any],
    active_move_info: Dict[str, Dict[str, float]],
    material_edge: float,
    best_attack_pressure: float,
    my_hazard_pressure: float,
    opp_hazard_pressure: float,
    my_remaining: int,
    opp_remaining: int,
) -> float:
    config = get_active_policy_config()
    name = _normalize_move_name(move_payload)
    my_observed_names = {_normalize_move_name_text(str(move_name)) for move_name in my_observed_moves}
    runtime_info = active_move_info.get(name) or {}
    current_pp = float(runtime_info.get("pp") or 0.0)
    max_pp = float(runtime_info.get("maxpp") or 0.0)
    repeated_move = (max_pp > 0 and current_pp < max_pp) or (name in my_observed_names)
    available_priority_ids = _active_priority_move_ids(active_move_info)
    has_other_priority_option = bool(available_priority_ids - {name})
    metadata = move_metadata(str(move_payload.get("id") or move_payload.get("move") or ""))
    inferred_move_type = str(metadata.get("type") or move_type(str(move_payload.get("id") or move_payload.get("move") or "")) or "")
    category = str(metadata.get("category") or "")
    base_power = float(metadata.get("basePower") or 0.0)
    accuracy = float(metadata.get("accuracy") or 0.0)
    priority = float(metadata.get("priority") or 0.0)
    target = str(metadata.get("target") or "")
    self_switch = bool(metadata.get("selfSwitch"))
    side_condition = str(metadata.get("sideCondition") or "")
    pseudo_weather = str(metadata.get("pseudoWeather") or "")
    my_types = species_types(my_species)
    opp_types = species_types(opp_species)
    opp_role_flags = _opponent_role_flags(opp_observed_moves)
    my_move_names = {_normalize_move_name_text(str(move_name)) for move_name in my_observed_moves}
    normalized_opp_moves = {_normalize_move_name_text(str(move_name)) for move_name in opp_observed_moves}
    threat_score = 0.0
    for opp_move_name in opp_observed_moves:
        opp_meta = move_metadata(opp_move_name)
        opp_move_type = str(opp_meta.get("type") or move_type(opp_move_name) or "")
        opp_category = str(opp_meta.get("category") or "")
        opp_power = float(opp_meta.get("basePower") or 0.0)
        opp_priority = float(opp_meta.get("priority") or 0.0)
        if opp_category not in {"Physical", "Special"} or not opp_move_type:
            continue
        multiplier = type_effectiveness(opp_move_type, my_types)
        local = min(opp_power / 70.0, 2.0)
        if multiplier == 0.0:
            local -= 0.5
        elif multiplier >= 4.0:
            local += 3.0
        elif multiplier > 1.0:
            local += 1.5
        elif multiplier < 1.0:
            local -= 0.5
        if opp_priority > 0 and my_hp <= 0.35:
            local += 0.8
        threat_score = max(threat_score, local)
    if name in RECOVER_KEYWORDS:
        if best_attack_pressure >= config.recover_block_pressure and opp_hp <= config.recover_block_opp_hp:
            return -1.8
        if opp_role_flags["setup"] and my_hp <= 0.45:
            return 0.4
        if opp_role_flags["status"] and best_attack_pressure >= 2.8 and opp_hp >= 0.35:
            return -0.6
        if my_status in {"tox", "psn", "brn"} and best_attack_pressure >= 2.4:
            return -1.0
        if material_edge >= 2.0 and my_hp > 0.45 and opp_hp <= 0.4:
            return -1.2
        if my_hp <= config.recover_critical_hp_threshold:
            return config.recover_low_hp_bonus
        if my_hp <= config.recover_danger_hp_threshold:
            return config.recover_danger_base + min(threat_score, 1.6)
        if my_hp <= config.recover_mid_hp_threshold:
            return config.recover_mid_base + min(threat_score, 1.0)
        return config.recover_default_penalty
    if name in STATUS_KEYWORDS:
        if best_attack_pressure >= config.status_block_pressure and opp_hp <= config.status_block_opp_hp:
            return -2.0
        if opp_role_flags["recover"] and my_hp >= 0.55 and opp_hp >= 0.55:
            return 0.3
        if opp_role_flags["status"] and not opp_status and best_attack_pressure >= 2.6:
            return -0.7
        if material_edge >= 2.0 and opp_hp <= 0.55:
            return -1.4
        if not opp_status and opp_hp >= 0.8 and my_hp >= 0.7 and total_boost == 0:
            return 0.3
        return config.status_default_penalty - min(threat_score * 0.4, 1.0)
    if name in SETUP_KEYWORDS:
        has_body_press = "bodypress" in my_move_names
        def_stage = int(boosts.get("def", 0) or 0)
        if total_boost >= 2:
            return config.setup_repeat_penalty
        if name == "irondefense" and has_body_press:
            opp_is_body_press_mirror = "bodypress" in normalized_opp_moves and "irondefense" in normalized_opp_moves
            if def_stage >= 2:
                return config.setup_repeat_penalty - 0.3
            if opp_is_body_press_mirror and my_hp >= 0.4:
                return 11.0
            if my_hp >= 0.45 and best_attack_pressure < 2.8 and threat_score <= 1.9:
                return 1.25
            if my_hp >= 0.3 and best_attack_pressure < 2.2 and threat_score <= 2.2:
                return 0.5
        if total_boost >= 1 and best_attack_pressure >= config.setup_attack_ready_pressure:
            return config.setup_attack_ready_penalty
        if best_attack_pressure >= config.setup_block_pressure:
            return -2.4
        if opp_role_flags["phaze"]:
            return -2.0
        if opp_role_flags["status"] or opp_role_flags["recover"]:
            return -2.1 - min(threat_score * 0.3, 0.6)
        if my_status in {"tox", "psn", "brn"}:
            return -2.3
        if material_edge >= 1.5:
            return -2.2 - min(threat_score * 0.4, 0.8)
        if my_hp >= 0.92 and opp_hp >= 0.85 and total_boost == 0:
            return config.setup_opening_window_bonus
        return config.setup_default_penalty - min(threat_score * 0.6, 1.4)
    if name in SCOUT_KEYWORDS:
        if best_attack_pressure >= 3.4 and material_edge >= 0.5:
            return -1.3
        if 0.35 <= my_hp <= 0.7 and opp_hp >= 0.55:
            return config.scout_safe_window_bonus - min(threat_score * 0.2, 0.4)
        return config.scout_default_penalty - min(threat_score * 0.2, 0.4)
    if name in HAZARD_SET_KEYWORDS:
        if opp_hazard_pressure >= 1.2:
            return -1.2
        if opp_remaining >= 4 and my_hp >= 0.6 and opp_hp >= 0.45 and best_attack_pressure < 3.8:
            return config.hazard_set_large_bonus
        if opp_remaining >= 3 and material_edge <= 0.5 and opp_hp >= 0.55 and best_attack_pressure < 3.4:
            return config.hazard_set_medium_bonus
        return config.hazard_set_default_penalty
    if name in HAZARD_CLEAR_KEYWORDS:
        if my_hazard_pressure >= 2.4 and my_remaining >= 3 and my_hp >= 0.35:
            return config.hazard_clear_high_bonus
        if my_hazard_pressure >= 1.2 and my_remaining >= 3 and my_hp >= 0.4:
            return config.hazard_clear_mid_bonus
        if my_hazard_pressure >= 0.8 and my_remaining >= 2:
            return config.hazard_clear_low_bonus
        return config.hazard_clear_default_penalty
    score = 0.0
    effectiveness_multiplier = 1.0
    if category in {"Physical", "Special"}:
        direct_pressure = _attack_pressure_score(
            move_payload,
            my_species=my_species,
            opp_species=opp_species,
        )
        pressure_gap = max(0.0, best_attack_pressure - direct_pressure)
        score += direct_pressure * config.attack_pressure_weight
        score += min(base_power / config.attack_power_divisor, config.attack_power_cap)
        if accuracy and accuracy < 100:
            score -= (100.0 - accuracy) / config.attack_accuracy_divisor
        if accuracy >= 100 and direct_pressure >= best_attack_pressure - config.attack_pressure_close_margin:
            score += config.attack_perfect_accuracy_bonus
        if pressure_gap >= config.attack_pressure_gap_threshold and base_power <= config.attack_low_power_threshold and priority <= 0:
            score += config.attack_pressure_gap_penalty
        if pressure_gap >= 1.4 and base_power <= 45:
            score -= 0.9
        if priority > 0 and opp_hp <= 0.35:
            score += config.attack_priority_cleanup_bonus
        elif priority > 0:
            score += config.attack_priority_bonus
        if opp_hp <= 0.25 and priority > 0:
            score += config.attack_priority_finish_bonus
        if opp_hp <= config.attack_skip_priority_critical_hp_threshold and priority > 0:
            score += config.attack_priority_critical_finish_bonus
        if priority > 0 and opp_hp > 0.28 and pressure_gap >= config.attack_priority_overuse_gap_threshold:
            score += config.attack_priority_overuse_penalty
        if (
            priority <= 0
            and has_other_priority_option
            and opp_hp <= config.attack_skip_priority_opp_hp_threshold
        ):
            risky_closeout = (
                name in RISKY_NON_PRIORITY_CLOSEOUT_MOVES
                or accuracy < 100
                or name in DEFENSE_DROP_MOVES
                or name in SPECIAL_SELF_DROP_MOVES
            )
            if risky_closeout:
                score += config.attack_skip_priority_closeout_penalty
            if opp_hp <= config.attack_skip_priority_critical_hp_threshold and risky_closeout:
                score += config.attack_skip_priority_critical_penalty
            if risky_closeout:
                score += config.attack_skip_priority_risky_penalty
        if opp_hp <= 0.3 and accuracy >= 95 and base_power >= 40:
            score += config.attack_low_hp_accuracy_bonus
        if opp_hp <= 0.3 and accuracy and accuracy < 90:
            score += config.attack_low_hp_accuracy_penalty
        if opp_hp <= 0.2 and accuracy and accuracy < 100:
            score += config.attack_low_hp_imperfect_penalty
        if self_switch:
            score += config.attack_self_switch_penalty
            if opp_hp <= 0.45 or material_edge >= 0.5:
                score += config.attack_self_switch_closeout_penalty
    elif category == "Status":
        if target in {"self", "allySide"}:
            score -= 0.5
        if side_condition or pseudo_weather:
            score -= 0.7
    if inferred_move_type:
        multiplier = type_effectiveness(inferred_move_type, opp_types) if opp_types else 1.0
        blocked_by_ability = _move_blocked_by_ability(
            ability_name=opp_ability,
            move_name=str(move_payload.get("id") or move_payload.get("move") or ""),
            move_type_name=inferred_move_type,
        )
        if blocked_by_ability:
            multiplier = 0.0
        effectiveness_multiplier = multiplier
        if multiplier == 0.0:
            score += config.type_zero_penalty
        elif multiplier >= 4.0:
            score += config.type_double_super_bonus
        elif multiplier > 1.0:
            score += config.type_super_bonus
        elif multiplier < 1.0:
            score += config.type_resist_penalty
        if multiplier == 0.0:
            score += config.move_known_immunity_penalty
            if _normalize_move_name_text(str(opp_ability or "")) == "windrider":
                score -= 1.0
            if repeated_move:
                score += config.move_repeat_known_immunity_penalty
        elif multiplier < 1.0 and repeated_move:
            score += config.move_repeat_resist_penalty
    if inferred_move_type and my_types and inferred_move_type in my_types:
        score += config.stab_bonus
    if category in {"Physical", "Special"}:
        if repeated_move:
            score += config.move_repeat_penalty
        else:
            score += config.move_novelty_bonus
        if max_pp > 0:
            used_fraction = max(0.0, min(1.0, (max_pp - current_pp) / max_pp))
            if used_fraction >= config.move_overused_pp_fraction and len(active_move_info) >= 2:
                score += config.move_overused_penalty
            if (
                name in STICKY_GENERIC_ATTACKS
                and repeated_move
                and len(active_move_info) >= 2
                and used_fraction >= config.move_sticky_repeat_pp_fraction
                and opp_hp > 0.3
                and effectiveness_multiplier <= 1.0
                and pressure_gap >= config.move_sticky_repeat_gap_threshold
            ):
                score += config.move_sticky_repeat_penalty
        spa_stage = int(boosts.get("spa", 0) or 0)
        def_stage = int(boosts.get("def", 0) or 0)
        spd_stage = int(boosts.get("spd", 0) or 0)
        atk_stage = int(boosts.get("atk", 0) or 0)
        if name == "bodypress":
            score += min(def_stage, 4) * 0.7
            if "irondefense" in my_move_names:
                score += 0.45
        if name in SPECIAL_SELF_DROP_MOVES and spa_stage < 0:
            score += config.self_drop_repeat_penalty
        if name in DEFENSE_DROP_MOVES and (def_stage < 0 or spd_stage < 0 or atk_stage < 0):
            score += config.defense_drop_repeat_penalty
        if name == "knockoff" and opp_item:
            score += config.knock_off_item_bonus
            if opp_role_flags["recover"] or opp_role_flags["setup"]:
                score += config.knock_off_sustain_item_bonus
        elif (
            name == "knockoff"
            and repeated_move
            and not opp_item
            and pressure_gap >= config.move_sticky_repeat_gap_threshold
        ):
            score += config.move_knock_off_no_item_repeat_penalty
    if opp_role_flags["setup"] and priority > 0:
        score += config.setup_priority_response_bonus
    if opp_role_flags["recover"] and base_power >= 70:
        score += config.recover_attack_bonus
    if opp_role_flags["recover"] and category in {"Physical", "Special"} and accuracy >= 90 and base_power >= 60:
        score += config.recover_stable_attack_bonus
    if opp_role_flags["status"] and not opp_status and category in {"Physical", "Special"} and accuracy >= 90 and base_power >= 60:
        score += config.status_clean_attack_bonus
    if my_status in {"tox", "psn", "brn"} and category in {"Physical", "Special"} and base_power >= 60:
        score += config.self_status_attack_bonus
    if my_status in {"tox", "psn", "brn"} and priority > 0 and opp_hp <= 0.4:
        score += config.self_status_priority_bonus
    if opp_hp <= 0.25 and inferred_move_type and opp_types:
        multiplier = type_effectiveness(inferred_move_type, opp_types)
        if multiplier > 1.0:
            score += config.low_hp_super_effective_bonus
    if material_edge >= 1.5 and accuracy >= 90 and base_power > 0:
        score += config.material_edge_accuracy_bonus
    if material_edge >= 2.0 and accuracy and accuracy < 90 and opp_hp <= 0.45:
        score += config.material_edge_risky_accuracy_penalty
    if opp_hp <= config.attack_low_opp_hp_threshold:
        score += config.attack_low_opp_hp_bonus
        if accuracy >= 95 and base_power > 0:
            score += config.attack_low_opp_hp_perfect_accuracy_bonus
    if opp_hp <= config.attack_desperation_opp_hp_threshold and my_hp <= 0.35 and base_power > 0 and accuracy >= 95:
        score += config.attack_desperation_finish_bonus
    if my_hp <= 0.2:
        score += config.attack_critical_self_hp_bonus
    else:
        score += config.attack_default_hp_bonus
    if my_hp <= 0.35 and threat_score >= 2.0 and name not in RECOVER_KEYWORDS:
        score += config.threat_low_hp_attack_penalty
    if material_edge <= -1.5 and name in SCOUT_KEYWORDS:
        score += config.losing_scout_bonus
    if threat_score >= 2.0 and opp_hp > 0.45:
        score += config.threat_general_penalty
    return score


def _best_move_for_predictions_with_state(
    predictions: Sequence[Any],
    moves: Sequence[Dict[str, Any]],
    *,
    my_hp: float,
    my_status: str,
    opp_hp: float,
    opp_status: str,
    total_boost: int,
    my_species: str,
    opp_species: str,
    opp_observed_moves: Sequence[str],
    my_observed_moves: Sequence[str],
    opp_ability: str,
    opp_item: str,
    boosts: Dict[str, Any],
    active_move_info: Dict[str, Dict[str, float]],
    material_edge: float,
    my_hazard_pressure: float,
    opp_hazard_pressure: float,
    my_remaining: int,
    opp_remaining: int,
    best_attack_pressure: float | None = None,
) -> Dict[str, Any]:
    if best_attack_pressure is None:
        best_attack_pressure = max(
            (
                _attack_pressure_score(move, my_species=my_species, opp_species=opp_species)
                for move in moves
            ),
            default=-10.0,
        )
    scored_moves: list[tuple[float, int, Dict[str, Any]]] = []
    for move in moves:
        total = 0.0
        for pred in predictions:
            total += float(pred.score) * _move_word_score(str(pred.word), move)
        total += _state_move_score(
            move,
            my_hp=my_hp,
            my_status=my_status,
            opp_hp=opp_hp,
            opp_status=opp_status,
            total_boost=total_boost,
            my_species=my_species,
            opp_species=opp_species,
            opp_observed_moves=opp_observed_moves,
            my_observed_moves=my_observed_moves,
            opp_ability=opp_ability,
            opp_item=opp_item,
            boosts=boosts,
            active_move_info=active_move_info,
            material_edge=material_edge,
            best_attack_pressure=best_attack_pressure,
            my_hazard_pressure=my_hazard_pressure,
            opp_hazard_pressure=opp_hazard_pressure,
            my_remaining=my_remaining,
            opp_remaining=opp_remaining,
        )
        slot = int(move.get("slot") or 99)
        scored_moves.append((total, -slot, move))
    scored_moves.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return scored_moves[0][2]


def _scored_moves_for_predictions_with_state(
    predictions: Sequence[Any],
    moves: Sequence[Dict[str, Any]],
    *,
    my_hp: float,
    my_status: str,
    opp_hp: float,
    opp_status: str,
    total_boost: int,
    my_species: str,
    opp_species: str,
    opp_observed_moves: Sequence[str],
    my_observed_moves: Sequence[str],
    opp_ability: str,
    opp_item: str,
    boosts: Dict[str, Any],
    active_move_info: Dict[str, Dict[str, float]],
    material_edge: float,
    my_hazard_pressure: float,
    opp_hazard_pressure: float,
    my_remaining: int,
    opp_remaining: int,
    best_attack_pressure: float | None = None,
    retrieval_bias: Dict[str, float] | None = None,
    retrieval_weight: float = 0.0,
) -> list[dict[str, Any]]:
    if best_attack_pressure is None:
        best_attack_pressure = max(
            (
                _attack_pressure_score(move, my_species=my_species, opp_species=opp_species)
                for move in moves
            ),
            default=-10.0,
        )

    bias_active = bool(retrieval_bias) and retrieval_weight > 0
    scored_moves: list[dict[str, Any]] = []
    for move in moves:
        word_score = 0.0
        word_contributions: list[dict[str, Any]] = []
        for pred in predictions:
            contribution = float(pred.score) * _move_word_score(str(pred.word), move)
            word_score += contribution
            word_contributions.append(
                {
                    "word": str(pred.word),
                    "prediction_score": float(pred.score),
                    "contribution": contribution,
                }
            )
        state_score = _state_move_score(
            move,
            my_hp=my_hp,
            my_status=my_status,
            opp_hp=opp_hp,
            opp_status=opp_status,
            total_boost=total_boost,
            my_species=my_species,
            opp_species=opp_species,
            opp_observed_moves=opp_observed_moves,
            my_observed_moves=my_observed_moves,
            opp_ability=opp_ability,
            opp_item=opp_item,
            boosts=boosts,
            active_move_info=active_move_info,
            material_edge=material_edge,
            best_attack_pressure=best_attack_pressure,
            my_hazard_pressure=my_hazard_pressure,
            opp_hazard_pressure=opp_hazard_pressure,
            my_remaining=my_remaining,
            opp_remaining=opp_remaining,
        )
        retrieval_score = 0.0
        if bias_active:
            move_key = "move:" + _normalize_move_name(move)
            retrieval_score = retrieval_weight * retrieval_bias.get(move_key, 0.0)
        total = word_score + state_score + retrieval_score
        scored_moves.append(
            {
                "move": move,
                "total_score": total,
                "word_score": word_score,
                "state_score": state_score,
                "retrieval_score": retrieval_score,
                "word_contributions": word_contributions,
            }
        )
    scored_moves.sort(
        reverse=True,
        key=lambda item: (item["total_score"], -(int(item["move"].get("slot") or 99))),
    )
    return scored_moves


def _apply_semantic_rerank(
    answer: InquiryAnswer,
    battle_state: Dict[str, Any],
    perspective_player: str,
    config,
) -> InquiryAnswer:
    """Blend a semantic-retrieval bias into the word predictions.

    No-op (returns answer unchanged) when the weight is 0 OR when the
    embedding artifacts are missing. The semantic module enforces both
    conditions defensively; this function adds an early-exit for speed.
    """
    weight = float(getattr(config, "semantic_retrieval_weight", 0.0) or 0.0)
    if weight <= 0.0 or not answer.predictions:
        return answer
    from dataclasses import replace
    from .model import Prediction

    candidate_words = [p.word for p in answer.predictions]
    bias = semantic_word_bias(
        battle_state,
        perspective_player,
        candidate_words,
        top_k=int(getattr(config, "semantic_retrieval_top_k", 8)),
        similarity_floor=float(getattr(config, "semantic_retrieval_similarity_floor", 0.30)),
        boost=float(getattr(config, "semantic_retrieval_word_boost", 0.6)),
        penalty=float(getattr(config, "semantic_retrieval_word_penalty", -0.2)),
    )
    if not bias:
        return answer

    import re as _re
    def _key(w: str) -> str:
        return _re.sub(r"[^a-z0-9]", "", str(w or "").lower())

    new_preds = []
    for pred in answer.predictions:
        delta = bias.get(_key(pred.word), 0.0) * weight
        new_preds.append(Prediction(word=pred.word, score=float(pred.score) + delta))
    new_preds.sort(key=lambda p: -p.score)
    return replace(answer, predictions=tuple(new_preds))


def _compute_action_retrieval_bias(
    moves: Sequence[Dict[str, Any]],
    battle_state: Dict[str, Any],
    perspective_player: str,
    config,
) -> Dict[str, float]:
    """Return {move:<name>: bias} or empty dict if profile disables retrieval.

    Mirrors _apply_semantic_rerank's defensive shape: zero-weight short-circuit
    keeps the threat-aware baseline byte-identical when v3 is not active.
    """
    weight = float(getattr(config, "action_retrieval_weight", 0.0) or 0.0)
    if weight <= 0.0 or not moves:
        return {}
    legal_move_names = [_normalize_move_name(m) for m in moves]
    return _action_retrieval_bias(
        battle_state,
        perspective_player,
        legal_move_names,
        top_k=int(getattr(config, "action_retrieval_top_k", 8)),
        similarity_floor=float(getattr(config, "action_retrieval_similarity_floor", 0.30)),
        boost=float(getattr(config, "action_retrieval_boost", 0.25)),
        penalty=float(getattr(config, "action_retrieval_penalty", -0.1)),
    )


def _choose_action_from_answer(
    answer: InquiryAnswer,
    question: str,
    battle_state: Dict[str, Any],
    *,
    perspective_player: str = "p1",
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
    allow_voluntary_switches: bool = True,
    active_payload: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    config = get_active_policy_config()
    moves = list(legal_moves or [])
    switches = list(legal_switches or []) if allow_voluntary_switches else []
    answer = _apply_semantic_rerank(answer, battle_state, perspective_player, config)
    retrieval_bias = _compute_action_retrieval_bias(
        list(legal_moves or []), battle_state, perspective_player, config
    )
    retrieval_weight = float(getattr(config, "action_retrieval_weight", 0.0) or 0.0)
    primary_word = answer.predictions[0].word if answer.predictions else None
    my_active = _active_mon(battle_state, perspective_player)
    opp_active = _opponent_mon(battle_state, perspective_player)
    my_species = str(my_active.get("species") or "")
    my_observed_moves = list(my_active.get("observed_moves") or [])
    opp_species = str(opp_active.get("species") or "")
    opp_ability = str(opp_active.get("ability") or "")
    opp_item = str(opp_active.get("item") or "")
    opp_item = str(opp_active.get("item") or "")
    opp_item = str(opp_active.get("item") or "")
    hp_frac = float(my_active.get("hp_frac") or 0.0)
    my_status = str(my_active.get("status") or "")
    opp_hp = float(opp_active.get("hp_frac") or 0.0)
    opp_status = str(opp_active.get("status") or "")
    boosts = my_active.get("boosts") or {}
    active_move_info = _active_runtime_move_info(active_payload)
    total_boost = sum(int(boosts.get(stat, 0) or 0) for stat in ("atk", "spa", "spe"))
    material_edge = _material_edge(battle_state, perspective_player)
    opp_observed_moves = list(opp_active.get("observed_moves") or [])
    other_player = "p2" if perspective_player == "p1" else "p1"
    my_hazard_pressure = _hazard_pressure(_side_conditions(battle_state, perspective_player))
    opp_hazard_pressure = _hazard_pressure(_side_conditions(battle_state, other_player))
    my_remaining, _ = _healthy_team_state(battle_state, perspective_player)
    opp_remaining, _ = _healthy_team_state(battle_state, other_player)
    normalized_question = _normalized_text(question)
    opp_boost_total = _boost_total(opp_active)
    opp_role_flags = _opponent_role_flags(opp_observed_moves)
    move_names = {_normalize_move_name(move) for move in moves}
    best_attack_pressure = max(
        (
            _attack_pressure_score(move, my_species=my_species, opp_species=opp_species)
            for move in moves
        ),
        default=-10.0,
    )

    if moves and "clear hazards" in normalized_question and my_hazard_pressure >= 1.2:
        clear_moves = [move for move in moves if _normalize_move_name(move) in HAZARD_CLEAR_KEYWORDS]
        if clear_moves:
            chosen_move = max(
                clear_moves,
                key=lambda move: (
                    _state_move_score(
                        move,
                        my_hp=hp_frac,
                        my_status=my_status,
                        opp_hp=opp_hp,
                        opp_status=opp_status,
                        total_boost=total_boost,
                        my_species=my_species,
                        opp_species=opp_species,
                        opp_observed_moves=opp_observed_moves,
                        my_observed_moves=my_observed_moves,
                        opp_ability=opp_ability,
                        opp_item=opp_item,
                        boosts=boosts,
                        active_move_info=active_move_info,
                        material_edge=material_edge,
                        best_attack_pressure=best_attack_pressure,
                        my_hazard_pressure=my_hazard_pressure,
                        opp_hazard_pressure=opp_hazard_pressure,
                        my_remaining=my_remaining,
                        opp_remaining=opp_remaining,
                    ),
                    -(int(move.get("slot") or 99)),
                ),
            )
            return {
                "type": "move",
                "best_move": chosen_move,
                "word_model_primary": primary_word,
                "prompt_tokens": list(answer.prompt_tokens),
            }

    if moves and "get hazards up" in normalized_question and opp_hazard_pressure < 1.0 and opp_remaining >= 3:
        hazard_moves = [move for move in moves if _normalize_move_name(move) in HAZARD_SET_KEYWORDS]
        if hazard_moves:
            chosen_move = max(
                hazard_moves,
                key=lambda move: (
                    _state_move_score(
                        move,
                        my_hp=hp_frac,
                        my_status=my_status,
                        opp_hp=opp_hp,
                        opp_status=opp_status,
                        total_boost=total_boost,
                        my_species=my_species,
                        opp_species=opp_species,
                        opp_observed_moves=opp_observed_moves,
                        my_observed_moves=my_observed_moves,
                        opp_ability=opp_ability,
                        opp_item=opp_item,
                        boosts=boosts,
                        active_move_info=active_move_info,
                        material_edge=material_edge,
                        best_attack_pressure=best_attack_pressure,
                        my_hazard_pressure=my_hazard_pressure,
                        opp_hazard_pressure=opp_hazard_pressure,
                        my_remaining=my_remaining,
                        opp_remaining=opp_remaining,
                    ),
                    -(int(move.get("slot") or 99)),
                ),
            )
            return {
                "type": "move",
                "best_move": chosen_move,
                "word_model_primary": primary_word,
                "prompt_tokens": list(answer.prompt_tokens),
            }

    if moves and "deny setup" in normalized_question and (opp_boost_total >= 1 or opp_role_flags["setup"]):
        denial_moves = [
            move for move in moves
            if _normalize_move_name(move) in STATUS_KEYWORDS
            or _normalize_move_name(move) in PHAZE_KEYWORDS
            or _normalize_move_name(move) in SCOUT_KEYWORDS
        ]
        if denial_moves:
            chosen_move = max(
                denial_moves,
                key=lambda move: (
                    2 if _normalize_move_name(move) in PHAZE_KEYWORDS else 1 if _normalize_move_name(move) in STATUS_KEYWORDS else 0,
                    -(int(move.get("slot") or 99)),
                ),
            )
            return {
                "type": "move",
                "best_move": chosen_move,
                "word_model_primary": primary_word,
                "prompt_tokens": list(answer.prompt_tokens),
            }

    if moves and "stop recovery" in normalized_question and opp_role_flags["recover"]:
        denial_moves = [
            move for move in moves
            if _normalize_move_name(move) in STATUS_KEYWORDS or _normalize_move_name(move) in PHAZE_KEYWORDS
        ]
        if denial_moves and opp_hp >= 0.4:
            chosen_move = max(
                denial_moves,
                key=lambda move: (
                    2 if _normalize_move_name(move) in STATUS_KEYWORDS else 1,
                    -(int(move.get("slot") or 99)),
                ),
            )
            return {
                "type": "move",
                "best_move": chosen_move,
                "word_model_primary": primary_word,
                "prompt_tokens": list(answer.prompt_tokens),
            }

    hard_switch = primary_word == "switch" and hp_frac <= config.hard_switch_hp_threshold
    soft_switch = primary_word == "switch" and hp_frac <= config.soft_switch_hp_threshold
    if (hard_switch or soft_switch) and switches:
        scored_switches = [
            (
                _bench_switch_score(
                    item,
                    battle_state=battle_state,
                    perspective_player=perspective_player,
                    opp_species=opp_species,
                    opp_ability=opp_ability,
                    opp_observed_moves=opp_observed_moves,
                    material_edge=material_edge,
                ),
                item,
            )
            for item in switches
        ]
        chosen = max(
            scored_switches,
            key=lambda entry: (entry[0], -(int(entry[1].get("slot") or 99))),
        )
        chosen_switch = chosen[1]
        return {
            "type": "switch",
            "best_switch": chosen_switch,
            "slot": chosen_switch.get("slot"),
            "word_model_primary": primary_word,
            "prompt_tokens": list(answer.prompt_tokens),
        }

    if switches and moves:
        scored_switches = [
            (
                _bench_switch_score(
                    item,
                    battle_state=battle_state,
                    perspective_player=perspective_player,
                    opp_species=opp_species,
                    opp_ability=opp_ability,
                    opp_observed_moves=opp_observed_moves,
                    material_edge=material_edge,
                ),
                item,
            )
            for item in switches
        ]
        best_switch_score, best_switch = max(scored_switches, key=lambda entry: (entry[0], -(int(entry[1].get("slot") or 99))))
        threatened = hp_frac <= config.preserve_switch_hp_threshold and opp_hp > config.preserve_switch_opp_hp_threshold and bool(opp_observed_moves)
        preserve_mode = primary_word in {"switch", "preserve", "wall", "stabilize"}
        sack_value_present = (
            hp_frac <= config.sack_value_active_hp_threshold
            and opp_hp <= config.sack_value_opp_hp_threshold
            and len(switches) <= 1
        )
        if preserve_mode and threatened and not sack_value_present and best_switch_score >= config.preserve_switch_score_threshold:
            return {
                "type": "switch",
                "best_switch": best_switch,
                "slot": best_switch.get("slot"),
                "word_model_primary": primary_word,
                "prompt_tokens": list(answer.prompt_tokens),
            }

    if moves:
        scored_moves = _scored_moves_for_predictions_with_state(
            answer.predictions,
            moves,
            my_hp=hp_frac,
            my_status=my_status,
            opp_hp=opp_hp,
            opp_status=opp_status,
            total_boost=total_boost,
            my_species=my_species,
            opp_species=opp_species,
            opp_observed_moves=opp_observed_moves,
            my_observed_moves=my_observed_moves,
            opp_ability=opp_ability,
            opp_item=opp_item,
            boosts=boosts,
            active_move_info=active_move_info,
            material_edge=material_edge,
            my_hazard_pressure=my_hazard_pressure,
            opp_hazard_pressure=opp_hazard_pressure,
            my_remaining=my_remaining,
            opp_remaining=opp_remaining,
            best_attack_pressure=best_attack_pressure,
            retrieval_bias=retrieval_bias,
            retrieval_weight=retrieval_weight,
        )
        if switches:
            scored_switches = [
                (
                    _bench_switch_score(
                        item,
                        battle_state=battle_state,
                        perspective_player=perspective_player,
                        opp_species=opp_species,
                        opp_ability=opp_ability,
                        opp_observed_moves=opp_observed_moves,
                        material_edge=material_edge,
                    ),
                    item,
                )
                for item in switches
            ]
            best_switch_score, best_switch = max(
                scored_switches,
                key=lambda entry: (entry[0], -(int(entry[1].get("slot") or 99))),
            )
            best_move_score = float(scored_moves[0]["total_score"]) if scored_moves else -10.0
            opponent_farming = opp_role_flags["setup"] or opp_role_flags["recover"]
            has_progress_tool = (
                any(name in HAZARD_SET_KEYWORDS for name in move_names)
                or any(name in PHAZE_KEYWORDS for name in move_names)
                or any(name in STATUS_KEYWORDS for name in move_names)
                or any(name in ITEM_PROGRESS_KEYWORDS for name in move_names)
            )
            real_attack_count = _real_attack_option_count(
                moves,
                my_species=my_species,
                opp_species=opp_species,
                opp_ability=opp_ability,
            )
            if (
                opponent_farming
                and primary_word != "finish"
                and not has_progress_tool
                and real_attack_count <= 1
                and best_switch_score >= config.emergency_switch_min_score
                and (
                    (
                        best_move_score <= config.emergency_switch_move_score_threshold
                        and best_switch_score >= best_move_score + config.emergency_switch_score_margin
                    )
                    or best_attack_pressure <= config.emergency_switch_low_pressure_threshold
                )
            ):
                return {
                    "type": "switch",
                    "best_switch": best_switch,
                    "slot": best_switch.get("slot"),
                    "word_model_primary": primary_word,
                    "prompt_tokens": list(answer.prompt_tokens),
                }
        chosen_move = scored_moves[0]["move"]
        return {
            "type": "move",
            "best_move": chosen_move,
            "word_model_primary": primary_word,
            "prompt_tokens": list(answer.prompt_tokens),
        }

    if switches:
        chosen = switches[0]
        return {
            "type": "switch",
            "best_switch": chosen,
            "slot": chosen.get("slot"),
            "word_model_primary": primary_word,
            "prompt_tokens": list(answer.prompt_tokens),
        }

    return {
        "type": "none",
        "best_action": None,
        "word_model_primary": primary_word,
        "prompt_tokens": list(answer.prompt_tokens),
    }


def choose_action_from_inquiry(
    question: str,
    battle_state: Dict[str, Any],
    *,
    perspective_player: str = "p1",
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
    allow_voluntary_switches: bool = True,
    top_k: int = 3,
    model: WordEmbeddingModel | None = None,
    active_payload: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    answer = answer_battle_inquiry(
        question,
        battle_state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches if allow_voluntary_switches else [],
        top_k=top_k,
        model=model,
    )
    return _choose_action_from_answer(
        answer,
        question,
        battle_state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        allow_voluntary_switches=allow_voluntary_switches,
        active_payload=active_payload,
    )


def choose_actions_from_inquiry_batch(
    inquiries: Sequence[Dict[str, Any]],
    *,
    model: WordEmbeddingModel,
    top_k: int = 3,
) -> list[Dict[str, Any]]:
    if not inquiries:
        return []

    prepared: list[dict[str, Any]] = []
    prompt_batches: list[list[str]] = []
    for inquiry in inquiries:
        question = str(inquiry["question"])
        battle_state = inquiry["battle_state"]
        perspective_player = str(inquiry.get("perspective_player") or "p1")
        legal_moves = list(inquiry.get("legal_moves") or [])
        allow_voluntary_switches = bool(inquiry.get("allow_voluntary_switches", True))
        legal_switches = list(inquiry.get("legal_switches") or []) if allow_voluntary_switches else []
        active_payload = list(inquiry.get("active") or [])
        prompt_tokens = build_battle_prompt_tokens(
            question,
            battle_state,
            perspective_player=perspective_player,
            legal_moves=legal_moves,
            legal_switches=legal_switches,
        )
        prepared.append(
            {
                "question": question,
                "battle_state": battle_state,
                "perspective_player": perspective_player,
                "legal_moves": legal_moves,
                "legal_switches": legal_switches,
                "allow_voluntary_switches": allow_voluntary_switches,
                "active_payload": active_payload,
                "prompt_tokens": prompt_tokens,
            }
        )
        prompt_batches.append(prompt_tokens)

    prediction_batches = model.predict_batch(prompt_batches, top_k=top_k)
    results: list[Dict[str, Any]] = []
    for item, predictions in zip(prepared, prediction_batches):
        answer = InquiryAnswer(
            prompt_tokens=tuple(item["prompt_tokens"]),
            predictions=tuple(predictions),
            model_source="provided",
        )
        results.append(
            _choose_action_from_answer(
                answer,
                item["question"],
                item["battle_state"],
                perspective_player=item["perspective_player"],
                legal_moves=item["legal_moves"],
                legal_switches=item["legal_switches"],
                allow_voluntary_switches=item["allow_voluntary_switches"],
                active_payload=item["active_payload"],
            )
        )
    return results


def debug_choose_action_from_inquiry(
    question: str,
    battle_state: Dict[str, Any],
    *,
    perspective_player: str = "p1",
    legal_moves: Sequence[Dict[str, Any]] | None = None,
    legal_switches: Sequence[Dict[str, Any]] | None = None,
    allow_voluntary_switches: bool = True,
    top_k: int = 3,
    model: WordEmbeddingModel | None = None,
    active_payload: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    answer = answer_battle_inquiry(
        question,
        battle_state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches if allow_voluntary_switches else [],
        top_k=top_k,
        model=model,
    )
    action = _choose_action_from_answer(
        answer,
        question,
        battle_state,
        perspective_player=perspective_player,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        allow_voluntary_switches=allow_voluntary_switches,
        active_payload=active_payload,
    )

    moves = list(legal_moves or [])
    my_active = _active_mon(battle_state, perspective_player)
    opp_active = _opponent_mon(battle_state, perspective_player)
    my_species = str(my_active.get("species") or "")
    my_observed_moves = list(my_active.get("observed_moves") or [])
    opp_species = str(opp_active.get("species") or "")
    opp_ability = str(opp_active.get("ability") or "")
    opp_item = str(opp_active.get("item") or "")
    hp_frac = float(my_active.get("hp_frac") or 0.0)
    my_status = str(my_active.get("status") or "")
    opp_hp = float(opp_active.get("hp_frac") or 0.0)
    opp_status = str(opp_active.get("status") or "")
    boosts = my_active.get("boosts") or {}
    active_move_info = _active_runtime_move_info(active_payload)
    total_boost = sum(int(boosts.get(stat, 0) or 0) for stat in ("atk", "spa", "spe"))
    material_edge = _material_edge(battle_state, perspective_player)
    opp_observed_moves = list(opp_active.get("observed_moves") or [])
    other_player = "p2" if perspective_player == "p1" else "p1"
    my_hazard_pressure = _hazard_pressure(_side_conditions(battle_state, perspective_player))
    opp_hazard_pressure = _hazard_pressure(_side_conditions(battle_state, other_player))
    my_remaining, _ = _healthy_team_state(battle_state, perspective_player)
    opp_remaining, _ = _healthy_team_state(battle_state, other_player)
    best_attack_pressure = max(
        (
            _attack_pressure_score(move, my_species=my_species, opp_species=opp_species)
            for move in moves
        ),
        default=-10.0,
    )
    scored_moves = _scored_moves_for_predictions_with_state(
        answer.predictions,
        moves,
        my_hp=hp_frac,
        my_status=my_status,
        opp_hp=opp_hp,
        opp_status=opp_status,
        total_boost=total_boost,
        my_species=my_species,
        opp_species=opp_species,
        opp_observed_moves=opp_observed_moves,
        my_observed_moves=my_observed_moves,
        opp_ability=opp_ability,
        opp_item=opp_item,
        boosts=boosts,
        active_move_info=active_move_info,
        material_edge=material_edge,
        my_hazard_pressure=my_hazard_pressure,
        opp_hazard_pressure=opp_hazard_pressure,
        my_remaining=my_remaining,
        opp_remaining=opp_remaining,
        best_attack_pressure=best_attack_pressure,
    )
    return {
        "action": action,
        "prompt_tokens": list(answer.prompt_tokens),
        "predictions": [{"word": item.word, "score": item.score} for item in answer.predictions],
        "attention_report": answer.attention_report,
        "scored_moves": scored_moves,
    }
