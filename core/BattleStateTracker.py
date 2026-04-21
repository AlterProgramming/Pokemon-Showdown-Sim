from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Set, Tuple

from .RewardSignals import battle_winner, terminal_result_for_player
from .TurnEventV1 import (
    TurnEventV1, hp_delta_to_bin,
    EVENT_MOVE, EVENT_SWITCH, EVENT_DAMAGE, EVENT_HEAL,
    EVENT_STATUS_START, EVENT_STATUS_END, EVENT_BOOST, EVENT_UNBOOST,
    EVENT_FAINT, EVENT_WEATHER, EVENT_FIELD, EVENT_SIDE_CONDITION,
    EVENT_FORME_CHANGE, EVENT_TURN_END,
)


# Shared ordering for boosts. Vectorization should import this from here
# to prevent drift between modules.
STAT_ORDER: List[str] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

LAYERED_SIDE_CONDITION_CAPS: Dict[str, int] = {
    "spikes": 3,
    "toxicspikes": 2,
}


def normalize_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return "".join(ch for ch in text if ch.isalnum())


def strip_effect_prefix(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if ":" in text:
        prefix, rest = text.split(":", 1)
        if prefix.strip().lower() in {"move", "ability", "item"}:
            text = rest.strip()

    return text or None


@dataclass
class MonState:
    uid: str
    player: str  # "p1" or "p2"
    species: Optional[str] = None
    hp: Optional[int] = None
    max_hp: Optional[int] = None
    fainted: bool = False
    status: Optional[str] = None  # "brn", "par", "psn", "tox", "slp", "frz"
    ability: Optional[str] = None
    item: Optional[str] = None
    tera_type: Optional[str] = None
    terastallized: bool = False
    public_revealed: bool = False
    boosts: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in STAT_ORDER})

    def hp_frac(self) -> Optional[float]:
        if self.hp is None or self.max_hp in (None, 0):
            return None
        val = self.hp / self.max_hp
        if val < 0:
            return 0.0
        if val > 1:
            return 1.0
        return float(val)


@dataclass
class SideState:
    player: str
    active_uid: Optional[str] = None
    slot_uids: List[Optional[str]] = field(default_factory=lambda: [None] * 6)
    side_conditions: Dict[str, int] = field(default_factory=dict)


class BattleStateTracker:
    """
    Tracker for Gen9 RandBats JSON schema.

    Contract:
      - snapshot() returns a stable dict shape consumed by vectorization code.
      - iter_turn_examples() yields supervised examples (state_before, action).

    This module intentionally contains NO model / vocab logic.
    """

    def __init__(self, form_change_species: Optional[Set[str]] = None, history_turns: int = 0):
        # Species that can appear under multiple battle UIDs but should share a slot.
        self.form_change_species: Set[str] = form_change_species or {"Palafin"}
        self._history_turns: int = history_turns
        self._past_turn_events: deque = deque(maxlen=history_turns) if history_turns > 0 else deque(maxlen=0)
        self._current_turn_events: list = []
        self.reset()

    def reset(self) -> None:
        self.battle_id: Optional[str] = None
        self.turn_index: int = 0
        self.mons: Dict[str, MonState] = {}
        self.sides: Dict[str, SideState] = {"p1": SideState("p1"), "p2": SideState("p2")}
        self.species_slot: Dict[Tuple[str, str], int] = {}
        self.uid_slot: Dict[str, int] = {}
        self.roster_info: Dict[str, Dict[str, Any]] = {}
        self.observed_moves: DefaultDict[str, Set[str]] = defaultdict(set)
        self.weather: Optional[str] = None
        self.global_conditions: Set[str] = set()
        self._past_turn_events.clear()

    # ---------- Loading ----------
    def load_battle(self, battle: Dict[str, Any]) -> None:
        self.reset()
        self.battle_id = battle.get("battle_id")

        team_rev = battle.get("team_revelation", {}) or {}
        teams = team_rev.get("teams", {}) or {}

        for player in ("p1", "p2"):
            roster = teams.get(player, []) or []
            slot = 0
            for entry in roster:
                uid = entry.get("pokemon_uid")
                species = entry.get("species")
                if not uid:
                    continue

                self.roster_info[uid] = dict(entry)
                ms = self._ensure_mon(uid, player)
                ms.species = species or ms.species

                base_stats = entry.get("base_stats", {}) or {}
                base_hp = base_stats.get("hp")
                if isinstance(base_hp, int):
                    ms.max_hp = base_hp

                if slot < 6 and self.sides[player].slot_uids[slot] is None:
                    self.sides[player].slot_uids[slot] = uid
                    self.uid_slot[uid] = slot
                    if species and (player, species) not in self.species_slot:
                        self.species_slot[(player, species)] = slot
                    slot += 1

        self.turn_index = 0

    def _ensure_mon(self, uid: str, player: str) -> MonState:
        ms = self.mons.get(uid)
        if ms is not None:
            return ms
        ms = MonState(uid=uid, player=player)
        self._fill_mon_from_roster(ms)
        self.mons[uid] = ms
        return ms

    def _fill_mon_from_roster(self, ms: MonState) -> None:
        entry = self.roster_info.get(ms.uid)
        if not entry:
            return

        species = entry.get("species")
        if species and ms.species is None:
            ms.species = species

        base_stats = entry.get("base_stats", {}) or {}
        base_hp = base_stats.get("hp")
        if isinstance(base_hp, int) and ms.max_hp is None:
            ms.max_hp = base_hp

    @staticmethod
    def _player_from_uid(uid: str) -> str:
        return "p2" if str(uid).startswith("p2") else "p1"

    @staticmethod
    def _player_from_ref(ref: Optional[str]) -> Optional[str]:
        if not isinstance(ref, str):
            return None
        ref = ref.strip().lower()
        if ref.startswith("p1"):
            return "p1"
        if ref.startswith("p2"):
            return "p2"
        return None

    def _resolve_uid_from_ref(self, ref: Optional[str]) -> Optional[str]:
        player = self._player_from_ref(ref)
        if player is None:
            return None

        active_uid = self.sides[player].active_uid
        if active_uid is not None:
            return active_uid

        if not isinstance(ref, str) or ":" not in ref:
            return None

        species_hint = ref.split(":", 1)[1].strip()
        hint_id = normalize_id(species_hint)
        if not hint_id:
            return None

        candidates = [
            uid
            for uid, ms in self.mons.items()
            if ms.player == player and normalize_id(ms.species) == hint_id
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _observe_mon(
        self,
        uid: str,
        *,
        public: bool = True,
        assume_active: bool = False,
    ) -> MonState:
        player = self._player_from_uid(uid)
        ms = self._ensure_mon(uid, player)
        self._fill_mon_from_roster(ms)
        if public:
            ms.public_revealed = True
        if assume_active:
            self._set_active_uid(player, uid)
        return ms

    def _set_active_uid(self, player: str, uid: str) -> None:
        self._register_uid_slot(player, uid)
        self.sides[player].active_uid = uid

    def _register_uid_slot(self, player: str, uid: str) -> None:
        ms = self._ensure_mon(uid, player)
        self._fill_mon_from_roster(ms)

        species = ms.species
        if species and species in self.form_change_species:
            slot = self.species_slot.get((player, species))
            if slot is not None:
                self.sides[player].slot_uids[slot] = uid
                self.uid_slot[uid] = slot
                return

        if uid in self.uid_slot:
            slot = self.uid_slot[uid]
            if 0 <= slot < len(self.sides[player].slot_uids):
                self.sides[player].slot_uids[slot] = uid
            return

        for idx in range(6):
            if self.sides[player].slot_uids[idx] is None:
                self.sides[player].slot_uids[idx] = uid
                self.uid_slot[uid] = idx
                if species and (player, species) not in self.species_slot:
                    self.species_slot[(player, species)] = idx
                return

    @staticmethod
    def _normalize_effect_name(raw: Optional[str]) -> Optional[str]:
        return normalize_id(strip_effect_prefix(raw))

    # ---------- Snapshot ----------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "field": {
                "weather": self.weather,
                "global_conditions": sorted(self.global_conditions),
            },
            "p1": {
                "active_uid": self.sides["p1"].active_uid,
                "slots": list(self.sides["p1"].slot_uids),
                "side_conditions": dict(self.sides["p1"].side_conditions),
            },
            "p2": {
                "active_uid": self.sides["p2"].active_uid,
                "slots": list(self.sides["p2"].slot_uids),
                "side_conditions": dict(self.sides["p2"].side_conditions),
            },
            "mons": {
                uid: {
                    "uid": ms.uid,
                    "player": ms.player,
                    "species": ms.species,
                    "hp": ms.hp,
                    "max_hp": ms.max_hp,
                    "hp_frac": ms.hp_frac(),
                    "status": ms.status,
                    "ability": ms.ability,
                    "item": ms.item,
                    "tera_type": ms.tera_type,
                    "terastallized": ms.terastallized,
                    "public_revealed": ms.public_revealed,
                    "fainted": ms.fainted,
                    "boosts": dict(ms.boosts),
                    "observed_moves": sorted(self.observed_moves.get(uid, set())),
                }
                for uid, ms in self.mons.items()
            },
        }

    # ---------- Action extraction ----------
    @staticmethod
    def extract_action_for_player(
        turn_events: List[Dict[str, Any]],
        player: str,
    ) -> Optional[Tuple[str, str]]:
        """
        Returns ("move", move_id) or ("switch", into_uid) for the first decision event
        by that player in this turn.
        """
        for ev in turn_events:
            if ev.get("player") != player:
                continue
            et = ev.get("type")
            if et == "move":
                move_id = ev.get("move_id")
                if move_id:
                    return ("move", move_id)
            elif et == "switch":
                into_uid = ev.get("into_uid")
                if into_uid:
                    return ("switch", into_uid)
        return None

    def _resolve_switch_slot(self, player: str, into_uid: str) -> Optional[int]:
        slot = self.uid_slot.get(into_uid)
        if slot is not None:
            return slot

        ms = self.mons.get(into_uid)
        if ms is None:
            ms = self._ensure_mon(into_uid, player)
        self._fill_mon_from_roster(ms)

        if ms.species:
            return self.species_slot.get((player, ms.species))
        return None

    def action_token_for_player(
        self,
        player: str,
        action: Tuple[str, str],
    ) -> Optional[str]:
        kind, value = action
        if kind == "move":
            return f"move:{value}"
        if kind == "switch":
            slot = self._resolve_switch_slot(player, value)
            if slot is None:
                return None
            return f"switch:{slot + 1}"
        return None

    # ---------- Event application ----------
    def _side_from_uid(self, uid: str) -> str:
        """Extract player side ('p1' or 'p2') from a UID string.

        Returns '' for None, empty, or unrecognised UIDs (e.g. moves with no
        target such as Encore whose target_uid is null in the battle log).
        """
        if not uid:
            return ""
        if uid.startswith("p1"):
            return "p1"
        if uid.startswith("p2"):
            return "p2"
        return ""

    def apply_turn(self, turn_obj: Dict[str, Any]) -> None:
        self._current_turn_events = []
        self.turn_index = int(turn_obj.get("turn_number", self.turn_index + 1))
        for ev in (turn_obj.get("events", []) or []):
            et = ev.get("type")
            if et == "switch":
                self._apply_switch(ev)
            elif et in {"damage", "heal"}:
                self._apply_hp_change(ev)
            elif et == "status_start":
                self._apply_status_start(ev)
            elif et == "status_end":
                self._apply_status_end(ev)
            elif et == "stat_change":
                self._apply_stat_change(ev)
            elif et == "faint":
                self._apply_faint(ev)
            elif et == "move":
                self._apply_move(ev)
            elif et == "form_change":
                self._apply_form_change(ev)
            elif et == "effect":
                self._apply_effect(ev)
        self._current_turn_events.append(TurnEventV1(event_type=EVENT_TURN_END))
        if self._history_turns > 0:
            self._past_turn_events.append([ev.to_dict() for ev in self._current_turn_events])

    def _apply_move(self, ev: Dict[str, Any]) -> None:
        uid = ev.get("pokemon_uid")
        move_id = ev.get("move_id")
        if not uid or not move_id:
            return

        self._observe_mon(uid, public=True, assume_active=True)
        self.observed_moves[uid].add(move_id)

        target_uid = ev.get("target_uid")
        if target_uid:
            target_player = self._player_from_uid(target_uid)
            if self.sides[target_player].active_uid is None:
                self._observe_mon(target_uid, public=True, assume_active=True)

        actor = self._side_from_uid(ev.get("pokemon_uid", ""))
        target = self._side_from_uid(ev.get("target_uid", ""))
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_MOVE,
            actor_side=actor,
            target_side=target,
            move_id=ev.get("move_id", ""),
        ))

    def _apply_switch(self, ev: Dict[str, Any]) -> None:
        player = ev.get("player")
        uid = ev.get("into_uid") or ev.get("pokemon_uid")
        if player not in ("p1", "p2") or not uid:
            return

        self._observe_mon(uid, public=True, assume_active=True)

        species = ""
        ms = self.mons.get(uid)
        if ms:
            species = ms.species or ""
        slot = 0
        side = self.sides.get(player)
        if side and uid in side.slot_uids:
            slot = side.slot_uids.index(uid) + 1
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_SWITCH,
            actor_side=player,
            species_id=species,
            slot_index=slot,
        ))

    def _apply_hp_change(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        if not target:
            return

        ms = self._observe_mon(target, public=True)

        # Capture HP fraction BEFORE mutation
        before_frac = (ms.hp / ms.max_hp) if (ms.hp is not None and ms.max_hp and ms.max_hp > 0) else None

        hp_after = ev.get("hp_after")
        max_hp = ev.get("max_hp")

        if hp_after is not None:
            ms.hp = int(hp_after)
        if max_hp is not None:
            ms.max_hp = int(max_hp)

        if self.sides[ms.player].active_uid is None:
            self._set_active_uid(ms.player, target)

        # Capture HP fraction AFTER mutation
        after_frac = (ms.hp / ms.max_hp) if (ms.hp is not None and ms.max_hp and ms.max_hp > 0) else None
        event_type = EVENT_HEAL if (after_frac is not None and before_frac is not None and after_frac > before_frac) else EVENT_DAMAGE
        target_side = self._side_from_uid(ev.get("target_uid", ""))
        self._current_turn_events.append(TurnEventV1(
            event_type=event_type,
            target_side=target_side,
            hp_delta_bin=hp_delta_to_bin(before_frac, after_frac),
        ))

    def _apply_status_start(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        status = ev.get("status")
        if not target or not status:
            return

        ms = self._observe_mon(target, public=True)
        ms.status = status

        if self.sides[ms.player].active_uid is None:
            self._set_active_uid(ms.player, target)

        target_side = self._side_from_uid(ev.get("target_uid", ""))
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_STATUS_START,
            target_side=target_side,
            status=ev.get("status", ""),
        ))

    def _apply_status_end(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        status = ev.get("status")
        if not target:
            return

        ms = self._observe_mon(target, public=True)
        if status is None or ms.status == status:
            ms.status = None

        target_side = self._side_from_uid(ev.get("target_uid", ""))
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_STATUS_END,
            target_side=target_side,
            status=ev.get("status", ""),
        ))

    def _apply_stat_change(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        stat = ev.get("stat")
        amount = ev.get("amount")
        if not target or not stat or amount is None:
            return

        ms = self._observe_mon(target, public=True)

        if stat in ms.boosts:
            ms.boosts[stat] = max(-6, min(6, ms.boosts[stat] + int(amount)))

        if self.sides[ms.player].active_uid is None:
            self._set_active_uid(ms.player, target)

        target_side = self._side_from_uid(ev.get("target_uid", ""))
        int_amount = ev.get("amount", 0)
        event_type = EVENT_BOOST if int_amount > 0 else EVENT_UNBOOST
        self._current_turn_events.append(TurnEventV1(
            event_type=event_type,
            target_side=target_side,
            boost_stat=ev.get("stat", ""),
            boost_delta=int_amount,
        ))

    def _apply_faint(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        if not target:
            return

        ms = self._observe_mon(target, public=True)
        ms.fainted = True
        ms.hp = 0

        if self.sides[ms.player].active_uid == target:
            self.sides[ms.player].active_uid = None

        target_side = self._side_from_uid(ev.get("target_uid", ""))
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_FAINT,
            target_side=target_side,
        ))

    def _apply_form_change(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        if not target:
            return

        ms = self._observe_mon(target, public=True)
        ms.terastallized = True

        tera_type = strip_effect_prefix(ev.get("tera_type"))
        if tera_type:
            ms.tera_type = tera_type

        if self.sides[ms.player].active_uid is None:
            self._set_active_uid(ms.player, target)

        target_side = self._side_from_uid(ev.get("target_uid", ""))
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_FORME_CHANGE,
            target_side=target_side,
            forme_change_kind="tera",
            status=tera_type or "",
        ))

    def _apply_effect(self, ev: Dict[str, Any]) -> None:
        effect_type = ev.get("effect_type")
        raw_parts = ev.get("raw_parts", []) or []

        if effect_type == "weather":
            self._apply_weather_effect(raw_parts)
        elif effect_type == "sidestart":
            self._apply_side_condition(raw_parts, add=True)
        elif effect_type == "sideend":
            self._apply_side_condition(raw_parts, add=False)
        elif effect_type == "fieldstart":
            self._apply_field_condition(raw_parts, add=True)
        elif effect_type == "fieldend":
            self._apply_field_condition(raw_parts, add=False)
        elif effect_type == "ability":
            self._apply_revealed_ability(raw_parts)
        elif effect_type == "item":
            self._apply_revealed_item(raw_parts)
        elif effect_type == "enditem":
            self._apply_revealed_item(raw_parts, consumed=True)
        elif effect_type == "formechange":
            self._apply_formechange_effect(raw_parts)

    def _apply_weather_effect(self, raw_parts: List[Any]) -> None:
        if len(raw_parts) < 2:
            return

        weather = self._normalize_effect_name(raw_parts[1])
        if weather is None or str(raw_parts[1]).startswith("["):
            return
        if weather == "none":
            self.weather = None
            self._current_turn_events.append(TurnEventV1(
                event_type=EVENT_WEATHER,
                weather="none",
            ))
            return
        self.weather = weather
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_WEATHER,
            weather=weather,
        ))

    def _apply_side_condition(self, raw_parts: List[Any], *, add: bool) -> None:
        if len(raw_parts) < 3:
            return

        player = self._player_from_ref(raw_parts[1])
        condition = self._normalize_effect_name(raw_parts[2])
        if player is None or condition is None:
            return

        if add:
            cap = LAYERED_SIDE_CONDITION_CAPS.get(condition)
            next_value = self.sides[player].side_conditions.get(condition, 0) + 1
            if cap is not None:
                next_value = min(next_value, cap)
            self.sides[player].side_conditions[condition] = next_value
        else:
            self.sides[player].side_conditions.pop(condition, None)

        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_SIDE_CONDITION,
            actor_side=player,
            side_condition=condition,
            is_removal=not add,
        ))

    def _apply_field_condition(self, raw_parts: List[Any], *, add: bool) -> None:
        if len(raw_parts) < 2:
            return

        condition = self._normalize_effect_name(raw_parts[1])
        if condition is None:
            return

        if add:
            self.global_conditions.add(condition)
        else:
            self.global_conditions.discard(condition)

        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_FIELD,
            terrain=condition,
            is_removal=not add,
        ))

    def _apply_revealed_ability(self, raw_parts: List[Any]) -> None:
        if len(raw_parts) < 3:
            return

        uid = self._resolve_uid_from_ref(raw_parts[1])
        ability = strip_effect_prefix(raw_parts[2])
        if uid is None or ability is None:
            return

        ms = self._observe_mon(uid, public=True)
        ms.ability = ability

    def _apply_revealed_item(self, raw_parts: List[Any], *, consumed: bool = False) -> None:
        if len(raw_parts) < 3:
            return

        uid = self._resolve_uid_from_ref(raw_parts[1])
        if uid is None:
            return

        ms = self._observe_mon(uid, public=True)
        if consumed:
            ms.item = None
            return

        item = strip_effect_prefix(raw_parts[2])
        if item is not None:
            ms.item = item

    def _apply_formechange_effect(self, raw_parts: List[Any]) -> None:
        if len(raw_parts) < 3:
            return

        uid = self._resolve_uid_from_ref(raw_parts[1])
        new_species = strip_effect_prefix(raw_parts[2])
        if uid is None or new_species is None:
            return

        ms = self._observe_mon(uid, public=True)
        ms.species = new_species

        target_side = self._side_from_uid(uid)
        self._current_turn_events.append(TurnEventV1(
            event_type=EVENT_FORME_CHANGE,
            target_side=target_side,
            forme_change_kind="species",
            species_id=new_species,
        ))

    def _backfill_visible_actives(self, turn_obj: Dict[str, Any]) -> None:
        events = turn_obj.get("events", []) or []

        # Turn 1 lead switches are public state, not current-turn choices.
        if self.turn_index == 0:
            for ev in events:
                if ev.get("type") != "switch":
                    continue
                player = ev.get("player")
                uid = ev.get("into_uid") or ev.get("pokemon_uid")
                if player in ("p1", "p2") and uid and self.sides[player].active_uid is None:
                    self._observe_mon(uid, public=True, assume_active=True)

        for ev in events:
            if ev.get("type") != "move":
                continue

            actor_uid = ev.get("pokemon_uid")
            if actor_uid:
                actor_player = self._player_from_uid(actor_uid)
                if self.sides[actor_player].active_uid is None:
                    self._observe_mon(actor_uid, public=True, assume_active=True)

            target_uid = ev.get("target_uid")
            if target_uid:
                target_player = self._player_from_uid(target_uid)
                actor_player = self._player_from_uid(actor_uid) if actor_uid else None
                if target_player != actor_player and self.sides[target_player].active_uid is None:
                    self._observe_mon(target_uid, public=True, assume_active=True)

            if self.sides["p1"].active_uid is not None and self.sides["p2"].active_uid is not None:
                return

    # ---------- Dataset generation ----------
    def iter_turn_examples(
        self,
        battle: Dict[str, Any],
        player: str = "p1",
        include_switches: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yields per-turn examples:
          {
            "battle_id": str,
            "turn_number": int,
            "state": snapshot_before,
            "action": ("move", move_id) or ("switch", into_uid),
          }
        """
        self.load_battle(battle)
        winner = battle_winner(battle)
        terminal_result = terminal_result_for_player(battle, player)

        for turn in (battle.get("turns", []) or []):
            events = turn.get("events", []) or []

            self._backfill_visible_actives(turn)
            state_before = self.snapshot()

            action = self.extract_action_for_player(events, player)
            other = "p2" if player == "p1" else "p1"
            opponent_action = self.extract_action_for_player(events, other)
            opponent_action_token = None
            if opponent_action is not None:
                opponent_action_token = self.action_token_for_player(other, opponent_action)

            past_events_snapshot = list(self._past_turn_events)
            self.apply_turn(turn)
            state_after = self.snapshot()

            if action is None:
                continue
            if (not include_switches) and action[0] != "move":
                continue

            action_token = self.action_token_for_player(player, action)
            if action_token is None:
                continue

            yield {
                "battle_id": self.battle_id,
                "turn_number": int(turn.get("turn_number")),
                "player": player,
                "state": state_before,
                "next_state": state_after,
                "action": action,
                "action_token": action_token,
                "opponent_action": opponent_action,
                "opponent_action_token": opponent_action_token,
                "winner": winner,
                "terminal_result": terminal_result,
                "turn_events_v1": [ev.to_dict() for ev in self._current_turn_events],
                "past_turn_events": past_events_snapshot,
            }
