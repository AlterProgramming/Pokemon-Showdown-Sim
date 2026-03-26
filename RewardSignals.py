from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRACE_SCHEMA_VERSION = "1.0"
EPSILON = 1e-8


@dataclass(frozen=True)
class RewardConfig:
    hp_weight: float = 0.25
    ko_weight: float = 0.50
    wasted_move_penalty: float = 0.10
    terminal_weight: float = 1.0
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
            terminal_weight=float(payload.get("terminal_weight", cls.terminal_weight)),
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

    profile: Dict[str, Dict[str, Any]] = {}
    for move_id, entry in sorted(stats.items()):
        uses = int(entry["uses"])
        damage_turns = int(entry["damage_turns"])
        damage_turn_rate = (damage_turns / uses) if uses else 0.0
        is_offensive = (
            uses >= reward_config.offensive_move_min_uses
            and damage_turn_rate >= reward_config.offensive_move_min_damage_rate
        )
        profile[move_id] = {
            "uses": uses,
            "damage_turns": damage_turns,
            "damage_turn_rate": damage_turn_rate,
            "is_offensive": is_offensive,
        }
    return profile


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
    if action[0] == "move":
        move_profile = move_reward_profile.get(str(action[1]), {})
        if (
            bool(move_profile.get("is_offensive"))
            and opp_hp_loss <= EPSILON
            and opp_new_faints == 0
        ):
            wasted_offensive_move = 1.0

    return {
        "hp_swing": (opp_hp_loss - own_hp_loss) / 6.0,
        "ko_swing": float(opp_new_faints - own_new_faints),
        "wasted_offensive_move": wasted_offensive_move,
        "terminal": terminal_reward_from_result(terminal_result) if is_terminal_example else 0.0,
    }


def reward_total_from_components(
    components: Dict[str, float],
    reward_config: RewardConfig,
) -> float:
    return (
        reward_config.hp_weight * float(components.get("hp_swing", 0.0))
        + reward_config.ko_weight * float(components.get("ko_swing", 0.0))
        - reward_config.wasted_move_penalty * float(components.get("wasted_offensive_move", 0.0))
        + reward_config.terminal_weight * float(components.get("terminal", 0.0))
    )
