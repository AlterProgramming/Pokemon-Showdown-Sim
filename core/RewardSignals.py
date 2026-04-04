from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRACE_SCHEMA_VERSION = "1.1"
EPSILON = 1e-8
MAX_BOOST_STAGE = 6


@dataclass(frozen=True)
class RewardConfig:
    hp_weight: float = 0.25
    ko_weight: float = 0.50
    wasted_move_penalty: float = 0.10
    redundant_setup_penalty: float = 0.25
    wasted_setup_penalty: float = 1.0
    terminal_weight: float = 1.0
    return_discount: float = 1.0
    offensive_move_min_uses: int = 20
    offensive_move_min_damage_rate: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "RewardConfig":
        if not payload:
            return cls()
        return cls(
            hp_weight=float(payload.get("hp_weight", cls.hp_weight)),
            ko_weight=float(payload.get("ko_weight", cls.ko_weight)),
            wasted_move_penalty=float(payload.get("wasted_move_penalty", cls.wasted_move_penalty)),
            redundant_setup_penalty=float(payload.get("redundant_setup_penalty", cls.redundant_setup_penalty)),
            wasted_setup_penalty=float(payload.get("wasted_setup_penalty", cls.wasted_setup_penalty)),
            terminal_weight=float(payload.get("terminal_weight", cls.terminal_weight)),
            return_discount=float(payload.get("return_discount", cls.return_discount)),
            offensive_move_min_uses=int(payload.get("offensive_move_min_uses", cls.offensive_move_min_uses)),
            offensive_move_min_damage_rate=float(
                payload.get("offensive_move_min_damage_rate", cls.offensive_move_min_damage_rate)
            ),
        )


def other_player(player: str) -> str:
    if player == "p1":
        return "p2"
    if player == "p2":
        return "p1"
    raise ValueError("player must be 'p1' or 'p2'")


def battle_winner(battle: Dict[str, Any]) -> Optional[str]:
    outcome = (battle.get("metadata", {}) or {}).get("outcome", {}) or {}
    winner = outcome.get("winner")
    if winner in ("p1", "p2"):
        return str(winner)
    return None


def terminal_result_for_player(battle: Dict[str, Any], player: str) -> Optional[float]:
    winner = battle_winner(battle)
    if winner is not None:
        return 1.0 if winner == player else 0.0

    outcome = (battle.get("metadata", {}) or {}).get("outcome", {}) or {}
    result = str(outcome.get("result", "")).strip().lower()
    if result in {"draw", "tie"}:
        return 0.5
    return None


def terminal_reward_from_result(terminal_result: Optional[float]) -> float:
    if terminal_result is None:
        return 0.0
    if terminal_result > 0.5:
        return 1.0
    if terminal_result < 0.5:
        return -1.0
    return 0.0


def mon_visible_to_player(mon: Optional[Dict[str, Any]], perspective_player: str) -> bool:
    if mon is None:
        return False
    return mon.get("player") == perspective_player or bool(mon.get("public_revealed"))


def get_active_mon(state: Dict[str, Any], player: str) -> Optional[Dict[str, Any]]:
    active_uid = (state.get(player, {}) or {}).get("active_uid")
    if active_uid is None:
        return None
    return (state.get("mons", {}) or {}).get(active_uid)


def positive_self_boost_deltas(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    *,
    player: str,
) -> Dict[str, int]:
    before_mon = get_active_mon(state_before, player)
    after_mon = get_active_mon(state_after, player)
    if before_mon is None or after_mon is None:
        return {}

    before_boosts = before_mon.get("boosts", {}) or {}
    after_boosts = after_mon.get("boosts", {}) or {}
    deltas: Dict[str, int] = {}
    for stat in set(before_boosts) | set(after_boosts):
        gain = int(after_boosts.get(stat, 0)) - int(before_boosts.get(stat, 0))
        if gain > 0:
            deltas[str(stat)] = gain
    return deltas


def tracked_hp_loss(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    *,
    side_player: str,
    perspective_player: str,
) -> float:
    loss = 0.0
    slot_uids = state_before.get(side_player, {}).get("slots", []) or []
    for uid in slot_uids:
        if uid is None:
            continue
        before_mon = state_before.get("mons", {}).get(uid)
        after_mon = state_after.get("mons", {}).get(uid)
        if before_mon is None or after_mon is None:
            continue

        before_visible = mon_visible_to_player(before_mon, perspective_player)
        after_visible = mon_visible_to_player(after_mon, perspective_player)
        before_hp = before_mon.get("hp_frac")
        after_hp = after_mon.get("hp_frac")
        if before_visible and after_visible and before_hp is not None and after_hp is not None:
            loss += float(before_hp) - float(after_hp)
    return loss


def newly_fainted_count(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    *,
    side_player: str,
    perspective_player: str,
) -> int:
    count = 0
    slot_uids = state_before.get(side_player, {}).get("slots", []) or []
    for uid in slot_uids:
        if uid is None:
            continue
        before_mon = state_before.get("mons", {}).get(uid)
        after_mon = state_after.get("mons", {}).get(uid)
        if before_mon is None or after_mon is None:
            continue
        if not (mon_visible_to_player(before_mon, perspective_player) or mon_visible_to_player(after_mon, perspective_player)):
            continue
        if not before_mon.get("fainted") and after_mon.get("fainted"):
            count += 1
    return count


def build_move_reward_profile(
    examples: Iterable[Dict[str, Any]],
    reward_config: RewardConfig,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, float]] = {}
    setup_stats_by_move: Dict[str, set[str]] = {}
    for ex in examples:
        action = ex.get("action")
        if not action or action[0] != "move":
            continue
        move_id = str(action[1])
        player = str(ex.get("player"))
        opp = other_player(player)
        opp_hp_loss = tracked_hp_loss(
            ex["state"],
            ex["next_state"],
            side_player=opp,
            perspective_player=player,
        )
        opp_new_faints = newly_fainted_count(
            ex["state"],
            ex["next_state"],
            side_player=opp,
            perspective_player=player,
        )

        entry = stats.setdefault(move_id, {"uses": 0.0, "damage_turns": 0.0})
        entry["uses"] += 1.0
        if opp_hp_loss > EPSILON or opp_new_faints > 0:
            entry["damage_turns"] += 1.0

        boost_deltas = positive_self_boost_deltas(
            ex["state"],
            ex["next_state"],
            player=player,
        )
        if boost_deltas:
            entry["self_boost_turns"] = entry.get("self_boost_turns", 0.0) + 1.0
        if boost_deltas and opp_hp_loss <= EPSILON and opp_new_faints == 0:
            entry["setup_turns"] = entry.get("setup_turns", 0.0) + 1.0
            setup_stats_by_move.setdefault(move_id, set()).update(boost_deltas)

    profile: Dict[str, Dict[str, Any]] = {}
    for move_id, entry in sorted(stats.items()):
        uses = int(entry["uses"])
        damage_turns = int(entry["damage_turns"])
        damage_turn_rate = (damage_turns / uses) if uses else 0.0
        self_boost_turns = int(entry.get("self_boost_turns", 0.0))
        setup_turns = int(entry.get("setup_turns", 0.0))
        is_offensive = (
            uses >= reward_config.offensive_move_min_uses
            and damage_turn_rate >= reward_config.offensive_move_min_damage_rate
        )
        profile[move_id] = {
            "uses": uses,
            "damage_turns": damage_turns,
            "damage_turn_rate": damage_turn_rate,
            "self_boost_turns": self_boost_turns,
            "setup_turns": setup_turns,
            "setup_stats": sorted(setup_stats_by_move.get(move_id, set())),
            "is_offensive": is_offensive,
            "is_setup": setup_turns > 0,
        }
    return profile


def is_redundant_setup_turn(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    action: Tuple[str, str],
    player: str,
    move_reward_profile: Dict[str, Dict[str, Any]],
) -> bool:
    if action[0] != "move":
        return False

    move_profile = move_reward_profile.get(str(action[1]), {})
    if not bool(move_profile.get("is_setup")):
        return False

    before_mon = get_active_mon(state_before, player)
    if before_mon is None:
        return False

    boosted_stats = [str(stat) for stat in move_profile.get("setup_stats", [])]
    if boosted_stats:
        boosts = before_mon.get("boosts", {}) or {}
        return all(int(boosts.get(stat, 0)) >= MAX_BOOST_STAGE for stat in boosted_stats)

    return not positive_self_boost_deltas(state_before, state_after, player=player)


def compute_reward_components(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    action: Tuple[str, str],
    player: str,
    move_reward_profile: Dict[str, Dict[str, Any]],
    *,
    is_terminal_example: bool = False,
    terminal_result: Optional[float] = None,
) -> Dict[str, float]:
    opp = other_player(player)
    opp_hp_loss = tracked_hp_loss(
        state_before,
        state_after,
        side_player=opp,
        perspective_player=player,
    )
    own_hp_loss = tracked_hp_loss(
        state_before,
        state_after,
        side_player=player,
        perspective_player=player,
    )
    opp_new_faints = newly_fainted_count(
        state_before,
        state_after,
        side_player=opp,
        perspective_player=player,
    )
    own_new_faints = newly_fainted_count(
        state_before,
        state_after,
        side_player=player,
        perspective_player=player,
    )

    wasted_offensive_move = 0.0
    redundant_setup = 0.0
    if action[0] == "move":
        move_profile = move_reward_profile.get(str(action[1]), {})
        if (
            bool(move_profile.get("is_offensive"))
            and opp_hp_loss <= EPSILON
            and opp_new_faints == 0
        ):
            wasted_offensive_move = 1.0
        if is_redundant_setup_turn(state_before, state_after, action, player, move_reward_profile):
            redundant_setup = 1.0

    return {
        "hp_swing": (opp_hp_loss - own_hp_loss) / 6.0,
        "ko_swing": float(opp_new_faints - own_new_faints),
        "wasted_offensive_move": wasted_offensive_move,
        "redundant_setup": redundant_setup,
        "wasted_setup_chain": 0.0,
        "terminal": terminal_reward_from_result(terminal_result) if is_terminal_example else 0.0,
    }


def apply_wasted_setup_chain_penalties(
    examples: List[Dict[str, Any]],
    reward_components: List[Dict[str, float]],
    move_reward_profile: Dict[str, Dict[str, Any]],
) -> None:
    pending_indices: List[int] = []
    pending_uid: Optional[str] = None

    def penalize_pending() -> None:
        nonlocal pending_indices, pending_uid
        if not pending_indices:
            pending_uid = None
            return
        for pending_idx in pending_indices:
            reward_components[pending_idx]["wasted_setup_chain"] = (
                float(reward_components[pending_idx].get("wasted_setup_chain", 0.0)) + 1.0
            )
        pending_indices = []
        pending_uid = None

    for idx, ex in enumerate(examples):
        player = str(ex["player"])
        state_before = ex["state"]
        state_after = ex["next_state"]
        action = ex["action"]

        current_uid = (state_before.get(player, {}) or {}).get("active_uid")
        if pending_indices and pending_uid is not None and current_uid != pending_uid:
            penalize_pending()

        move_profile = move_reward_profile.get(str(action[1]), {}) if action[0] == "move" else {}
        is_setup = action[0] == "move" and bool(move_profile.get("is_setup"))
        if is_setup and current_uid is not None:
            pending_uid = current_uid
            pending_indices.append(idx)

        active_after_uid = (state_after.get(player, {}) or {}).get("active_uid")
        active_after = (state_after.get("mons", {}) or {}).get(current_uid) if current_uid is not None else None
        active_left_or_fainted = (
            current_uid is not None
            and (active_after_uid != current_uid or bool(active_after and active_after.get("fainted")))
        )
        if pending_indices and active_left_or_fainted:
            penalize_pending()
            continue

        if pending_indices and pending_uid is not None and current_uid == pending_uid:
            opp = other_player(player)
            opp_hp_loss = tracked_hp_loss(
                state_before,
                state_after,
                side_player=opp,
                perspective_player=player,
            )
            opp_new_faints = newly_fainted_count(
                state_before,
                state_after,
                side_player=opp,
                perspective_player=player,
            )
            if opp_hp_loss > EPSILON or opp_new_faints > 0:
                pending_indices = []
                pending_uid = None

    if pending_indices and examples:
        terminal_result = examples[-1].get("terminal_result")
        if terminal_result is not None and float(terminal_result) <= 0.5:
            penalize_pending()


def discounted_return_to_go(rewards: List[float], discount: float) -> List[float]:
    running_total = 0.0
    returns = [0.0] * len(rewards)
    for idx in range(len(rewards) - 1, -1, -1):
        running_total = float(rewards[idx]) + float(discount) * running_total
        returns[idx] = running_total
    return returns


def compute_reward_targets(
    examples: List[Dict[str, Any]],
    reward_config: RewardConfig,
    move_reward_profile: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, float]], List[float], List[float]]:
    components: List[Dict[str, float]] = []
    for idx, ex in enumerate(examples):
        components.append(
            compute_reward_components(
                ex["state"],
                ex["next_state"],
                ex["action"],
                str(ex["player"]),
                move_reward_profile,
                is_terminal_example=(idx == len(examples) - 1),
                terminal_result=ex.get("terminal_result"),
            )
        )

    apply_wasted_setup_chain_penalties(examples, components, move_reward_profile)
    rewards = [reward_total_from_components(component, reward_config) for component in components]
    returns = discounted_return_to_go(rewards, reward_config.return_discount)
    return components, rewards, returns


def attach_reward_targets(
    examples: List[Dict[str, Any]],
    reward_config: RewardConfig,
    move_reward_profile: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not examples:
        return examples

    grouped_examples: Dict[Tuple[Optional[str], str], List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, ex in enumerate(examples):
        key = (ex.get("battle_id"), str(ex.get("player")))
        grouped_examples.setdefault(key, []).append((idx, ex))

    for grouped in grouped_examples.values():
        grouped.sort(key=lambda item: int(item[1].get("turn_number", 0)))
        ex_items = [ex for _, ex in grouped]
        components, rewards, returns = compute_reward_targets(ex_items, reward_config, move_reward_profile)
        for (_, ex), component, reward_total, return_to_go in zip(grouped, components, rewards, returns):
            ex["reward_components"] = component
            ex["reward_total"] = float(reward_total)
            ex["return_to_go"] = float(return_to_go)

    return examples


def reward_total_from_components(
    components: Dict[str, float],
    reward_config: RewardConfig,
) -> float:
    return (
        reward_config.hp_weight * float(components.get("hp_swing", 0.0))
        + reward_config.ko_weight * float(components.get("ko_swing", 0.0))
        - reward_config.wasted_move_penalty * float(components.get("wasted_offensive_move", 0.0))
        - reward_config.redundant_setup_penalty * float(components.get("redundant_setup", 0.0))
        - reward_config.wasted_setup_penalty * float(components.get("wasted_setup_chain", 0.0))
        + reward_config.terminal_weight * float(components.get("terminal", 0.0))
    )
