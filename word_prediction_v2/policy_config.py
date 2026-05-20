from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyConfig:
    recover_block_pressure: float = 3.6
    recover_block_opp_hp: float = 0.55
    recover_low_hp_bonus: float = 3.2
    recover_critical_hp_threshold: float = 0.18
    recover_danger_hp_threshold: float = 0.30
    recover_mid_hp_threshold: float = 0.42
    recover_danger_base: float = 2.0
    recover_mid_base: float = 0.6
    recover_default_penalty: float = -1.0

    status_block_pressure: float = 3.2
    status_block_opp_hp: float = 0.70
    status_default_penalty: float = -1.0

    setup_repeat_penalty: float = -3.0
    setup_attack_ready_penalty: float = -2.7
    setup_attack_ready_pressure: float = 2.3
    setup_block_pressure: float = 3.0
    setup_default_penalty: float = -1.6
    setup_opening_window_bonus: float = 0.2

    scout_safe_window_bonus: float = 0.6
    scout_default_penalty: float = -0.2

    hazard_set_large_bonus: float = 2.8
    hazard_set_medium_bonus: float = 1.5
    hazard_set_default_penalty: float = -0.8
    hazard_clear_high_bonus: float = 4.4
    hazard_clear_mid_bonus: float = 2.8
    hazard_clear_low_bonus: float = 1.5
    hazard_clear_default_penalty: float = -0.7

    attack_pressure_weight: float = 0.9
    attack_power_divisor: float = 60.0
    attack_power_cap: float = 2.2
    attack_accuracy_divisor: float = 50.0
    attack_perfect_accuracy_bonus: float = 0.45
    attack_pressure_close_margin: float = 0.45
    attack_pressure_gap_penalty: float = -0.7
    attack_pressure_gap_threshold: float = 1.0
    attack_low_power_threshold: float = 50.0
    attack_priority_bonus: float = 0.3
    attack_priority_cleanup_bonus: float = 1.2
    attack_priority_finish_bonus: float = 1.4
    attack_priority_critical_finish_bonus: float = 1.2
    attack_priority_overuse_penalty: float = -0.8
    attack_priority_overuse_gap_threshold: float = 0.9
    attack_skip_priority_closeout_penalty: float = -1.2
    attack_skip_priority_risky_penalty: float = -0.9
    attack_skip_priority_opp_hp_threshold: float = 0.35
    attack_skip_priority_critical_hp_threshold: float = 0.18
    attack_skip_priority_critical_penalty: float = -1.4
    attack_low_hp_accuracy_bonus: float = 0.8
    attack_low_hp_accuracy_penalty: float = -1.2
    attack_low_hp_imperfect_penalty: float = -0.8
    attack_self_switch_penalty: float = -0.4
    attack_self_switch_closeout_penalty: float = -0.5
    attack_low_opp_hp_bonus: float = 2.6
    attack_low_opp_hp_threshold: float = 0.4
    attack_low_opp_hp_perfect_accuracy_bonus: float = 0.6
    attack_desperation_finish_bonus: float = 0.7
    attack_desperation_opp_hp_threshold: float = 0.18
    attack_critical_self_hp_bonus: float = 1.0
    attack_default_hp_bonus: float = 1.5
    threat_low_hp_attack_penalty: float = -0.8
    threat_general_penalty: float = -0.4

    type_zero_penalty: float = -3.0
    type_double_super_bonus: float = 3.2
    type_super_bonus: float = 1.8
    type_resist_penalty: float = -0.9
    stab_bonus: float = 0.4
    setup_priority_response_bonus: float = 0.4
    recover_attack_bonus: float = 0.3
    recover_stable_attack_bonus: float = 0.5
    status_clean_attack_bonus: float = 0.4
    self_status_attack_bonus: float = 0.5
    self_status_priority_bonus: float = 0.4
    knock_off_item_bonus: float = 1.2
    knock_off_sustain_item_bonus: float = 1.8
    low_hp_super_effective_bonus: float = 0.6
    material_edge_accuracy_bonus: float = 0.5
    material_edge_risky_accuracy_penalty: float = -0.8
    losing_scout_bonus: float = 0.4

    hard_switch_hp_threshold: float = 0.08
    soft_switch_hp_threshold: float = 0.05
    preserve_switch_hp_threshold: float = 0.4
    preserve_switch_opp_hp_threshold: float = 0.35
    preserve_switch_score_threshold: float = 1.2
    sack_value_active_hp_threshold: float = 0.35
    sack_value_opp_hp_threshold: float = 0.5
    emergency_switch_move_score_threshold: float = 0.35
    emergency_switch_score_margin: float = 0.9
    emergency_switch_min_score: float = 1.5
    emergency_switch_low_pressure_threshold: float = 2.4

    move_repeat_penalty: float = -0.2
    move_novelty_bonus: float = 0.15
    move_repeat_resist_penalty: float = -0.8
    move_known_immunity_penalty: float = -4.0
    move_repeat_known_immunity_penalty: float = -4.0
    move_overused_pp_fraction: float = 0.4
    move_overused_penalty: float = -0.5
    move_sticky_repeat_pp_fraction: float = 0.2
    move_sticky_repeat_penalty: float = -1.0
    move_sticky_repeat_gap_threshold: float = 0.25
    move_knock_off_no_item_repeat_penalty: float = -0.8
    self_drop_repeat_penalty: float = -2.2
    defense_drop_repeat_penalty: float = -1.0

    # Semantic retrieval bias (word_policy_v2_semantic).
    # When semantic_retrieval_weight is 0 the module is a no-op and the decoder
    # reduces exactly to the threat-aware baseline. The semantic module ALSO
    # short-circuits to no-op when artifacts/game-embeddings.npy is missing.
    semantic_retrieval_weight: float = 0.0
    semantic_retrieval_top_k: int = 8
    semantic_retrieval_similarity_floor: float = 0.30
    semantic_retrieval_word_boost: float = 0.6
    semantic_retrieval_word_penalty: float = -0.2

    # Action-level retrieval bias (word_policy_v3_actions). When the weight is 0
    # the module is a no-op and the decoder reduces exactly to threat-aware
    # baseline. The semantic_actions module also short-circuits to no-op when
    # artifacts/game-embeddings.npy or the action loader is unavailable.
    action_retrieval_weight: float = 0.0
    action_retrieval_top_k: int = 8
    action_retrieval_similarity_floor: float = 0.30
    action_retrieval_boost: float = 0.25
    action_retrieval_penalty: float = -0.1


def _coerce_field_value(field_name: str, raw_value: Any) -> Any:
    field_map = {field.name: field for field in fields(PolicyConfig)}
    field = field_map[field_name]
    if field.type is float:
        return float(raw_value)
    if field.type is int:
        return int(raw_value)
    if field.type is bool:
        if isinstance(raw_value, bool):
            return raw_value
        text = str(raw_value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean value for {field_name}: {raw_value!r}")
    return raw_value


def _apply_overrides(base: PolicyConfig, overrides: dict[str, Any]) -> PolicyConfig:
    allowed = {field.name for field in fields(PolicyConfig)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown policy config keys: {', '.join(unknown)}")
    coerced = {key: _coerce_field_value(key, value) for key, value in overrides.items()}
    return replace(base, **coerced)


def _load_profile_overrides(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy profile file must contain a JSON object")
    return payload


PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "aggressive_closeout": {
        "attack_low_opp_hp_bonus": 3.0,
        "attack_priority_cleanup_bonus": 1.4,
        "attack_low_hp_accuracy_penalty": -1.4,
        "material_edge_accuracy_bonus": 0.65,
    },
    "anti_loop_light": {
        "recover_default_penalty": -1.15,
        "setup_default_penalty": -1.8,
        "scout_default_penalty": -0.3,
        "threat_general_penalty": -0.5,
    },
    "safe_conversion": {
        "attack_perfect_accuracy_bonus": 0.55,
        "attack_low_hp_accuracy_bonus": 1.0,
        "attack_low_hp_imperfect_penalty": -1.0,
        "material_edge_risky_accuracy_penalty": -1.0,
    },
    "baseline_plus_finish": {
        "attack_low_opp_hp_bonus": 2.85,
        "attack_priority_cleanup_bonus": 1.3,
        "attack_priority_finish_bonus": 1.55,
        "attack_low_hp_accuracy_bonus": 0.9,
    },
    "baseline_plus_accuracy": {
        "attack_perfect_accuracy_bonus": 0.6,
        "attack_low_hp_accuracy_bonus": 0.95,
        "material_edge_accuracy_bonus": 0.65,
        "material_edge_risky_accuracy_penalty": -0.95,
    },
    "baseline_less_setup": {
        "setup_attack_ready_penalty": -2.95,
        "setup_block_pressure": 2.8,
        "setup_default_penalty": -1.85,
    },
    "baseline_less_recover": {
        "recover_default_penalty": -1.2,
        "recover_mid_base": 0.45,
        "recover_block_pressure": 3.4,
    },
    "baseline_more_recover": {
        "recover_default_penalty": -0.75,
        "recover_mid_base": 0.8,
        "recover_danger_base": 2.2,
    },
    "baseline_less_scout": {
        "scout_default_penalty": -0.4,
        "scout_safe_window_bonus": 0.4,
        "attack_self_switch_penalty": -0.6,
    },
    "baseline_more_hazard_clear": {
        "hazard_clear_high_bonus": 4.8,
        "hazard_clear_mid_bonus": 3.1,
        "hazard_clear_low_bonus": 1.8,
    },
    "baseline_more_hazard_set": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
    },
    "baseline_preserve_tighter": {
        "preserve_switch_hp_threshold": 0.34,
        "preserve_switch_score_threshold": 1.45,
        "hard_switch_hp_threshold": 0.06,
    },
    "baseline_preserve_looser": {
        "preserve_switch_hp_threshold": 0.46,
        "preserve_switch_score_threshold": 1.0,
        "hard_switch_hp_threshold": 0.1,
    },
    "hazard_set_light": {
        "hazard_set_large_bonus": 2.95,
        "hazard_set_medium_bonus": 1.65,
        "hazard_set_default_penalty": -0.65,
    },
    "hazard_set_heavy": {
        "hazard_set_large_bonus": 3.35,
        "hazard_set_medium_bonus": 2.05,
        "hazard_set_default_penalty": -0.35,
    },
    "hazard_set_less_scout": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
        "scout_default_penalty": -0.35,
        "scout_safe_window_bonus": 0.45,
    },
    "hazard_set_preserve_tighter": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
        "preserve_switch_hp_threshold": 0.34,
        "preserve_switch_score_threshold": 1.4,
    },
    "hazard_set_plus_finish": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
        "attack_low_opp_hp_bonus": 2.85,
        "attack_priority_cleanup_bonus": 1.3,
    },
    "hazard_set_less_recover": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
        "recover_default_penalty": -1.15,
        "recover_mid_base": 0.5,
    },
    "hazard_set_less_setup": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
        "setup_attack_ready_penalty": -2.95,
        "setup_default_penalty": -1.8,
    },
    "hazard_set_plus_accuracy": {
        "hazard_set_large_bonus": 3.1,
        "hazard_set_medium_bonus": 1.8,
        "hazard_set_default_penalty": -0.5,
        "attack_perfect_accuracy_bonus": 0.55,
        "material_edge_accuracy_bonus": 0.65,
    },
    "less_scout_light": {
        "scout_default_penalty": -0.3,
        "scout_safe_window_bonus": 0.5,
        "attack_self_switch_penalty": -0.5,
    },
    "less_scout_heavy": {
        "scout_default_penalty": -0.5,
        "scout_safe_window_bonus": 0.3,
        "attack_self_switch_penalty": -0.7,
    },
    "less_scout_preserve_tighter": {
        "scout_default_penalty": -0.35,
        "scout_safe_window_bonus": 0.45,
        "attack_self_switch_penalty": -0.55,
        "preserve_switch_hp_threshold": 0.34,
        "preserve_switch_score_threshold": 1.4,
    },
    "less_scout_plus_finish": {
        "scout_default_penalty": -0.35,
        "scout_safe_window_bonus": 0.45,
        "attack_self_switch_penalty": -0.55,
        "attack_low_opp_hp_bonus": 2.85,
        "attack_priority_cleanup_bonus": 1.3,
    },
    # word_policy_v2_semantic — semantic-retrieval bias on top of threat-aware.
    # Side-by-side profile. Activates when artifacts/game-embeddings.npy exists;
    # falls back to baseline-equivalent behavior when it does not.
    "word_policy_v2_semantic": {
        "semantic_retrieval_weight": 1.0,
        "semantic_retrieval_top_k": 8,
        "semantic_retrieval_similarity_floor": 0.30,
        "semantic_retrieval_word_boost": 0.6,
        "semantic_retrieval_word_penalty": -0.2,
    },
    # word_policy_v3_actions — action-level retrieval bias on top of threat-aware.
    # Side-by-side profile. Mirrors v2 but biases concrete legal moves instead
    # of semantic words. Activates when artifacts/game-embeddings.npy exists
    # AND at least one indexed source file has semantic action_token strings
    # (e.g. "move:dynamaxcannon"). Falls back to baseline-equivalent behavior
    # when either is missing.
    "word_policy_v3_actions": {
        "action_retrieval_weight": 1.0,
        "action_retrieval_top_k": 8,
        "action_retrieval_similarity_floor": 0.30,
        "action_retrieval_boost": 0.25,
        "action_retrieval_penalty": -0.1,
    },
}


def policy_profile_names() -> list[str]:
    return sorted(PROFILE_OVERRIDES)


def policy_config_from_profile_name(profile_name: str) -> PolicyConfig:
    name = profile_name.strip() or "baseline"
    try:
        overrides = PROFILE_OVERRIDES[name]
    except KeyError as exc:
        raise ValueError(f"unknown policy profile: {name}") from exc
    return _apply_overrides(PolicyConfig(), overrides)


@lru_cache(maxsize=1)
def get_active_policy_config() -> PolicyConfig:
    profile_name = os.environ.get("WORD_POLICY_PROFILE_NAME", "baseline").strip() or "baseline"
    config = policy_config_from_profile_name(profile_name)
    profile_path = os.environ.get("WORD_POLICY_PROFILE_PATH", "").strip()
    if profile_path:
        config = _apply_overrides(config, _load_profile_overrides(Path(profile_path)))
    return config


def clear_policy_config_cache() -> None:
    get_active_policy_config.cache_clear()


def active_policy_config_payload() -> dict[str, Any]:
    config = get_active_policy_config()
    payload = asdict(config)
    payload["profile_name"] = os.environ.get("WORD_POLICY_PROFILE_NAME", "baseline").strip() or "baseline"
    payload["profile_path"] = os.environ.get("WORD_POLICY_PROFILE_PATH", "").strip() or None
    return payload
