from __future__ import annotations

from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Any, Optional, List, Tuple, Iterator, Set
from collections import defaultdict


# Shared ordering for boosts. Vectorization should import this from here
# to prevent drift between modules.
STAT_ORDER: List[str] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


@dataclass
class MonState:
    uid: str
    player: str  # "p1" or "p2"
    species: Optional[str] = None
    hp: Optional[int] = None
    max_hp: Optional[int] = None
    fainted: bool = False
    status: Optional[str] = None  # "brn", "par", "psn", "tox", "slp", "frz"
    boosts: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in STAT_ORDER})

    def hp_frac(self) -> Optional[float]:
        if self.hp is None or self.max_hp in (None, 0):
            return None
        # clamp to [0, 1]
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


class BattleStateTracker:
    """
    Tracker for Gen9 RandBats JSON schema (as in your Kaggle dataset):

      - battle["team_revelation"]["teams"]["p1"/"p2"] roster entries
      - battle["turns"][] each with:
          - "turn_number"
          - "events"[] containing types like:
              switch, move, damage, heal, status_start, stat_change, faint
          - events include *_uid fields (pokemon_uid, into_uid, target_uid)

    Contract:
      - snapshot() returns a stable dict shape consumed by vectorization code.
      - iter_turn_examples() yields supervised examples (state_before, action).

    This module intentionally contains NO vectorization / vocab logic.
    """

    def __init__(self, form_change_species: Optional[Set[str]] = None):
        # Some randbats mons can appear under multiple uids due to form changes.
        self.form_change_species: Set[str] = form_change_species or {"Palafin"}
        self.reset()

    def reset(self) -> None:
        self.battle_id: Optional[str] = None
        self.turn_index: int = 0
        self.mons: Dict[str, MonState] = {}
        self.sides: Dict[str, SideState] = {"p1": SideState("p1"), "p2": SideState("p2")}
        # (player, species) -> canonical slot index (0..5)
        self.species_slot: Dict[Tuple[str, str], int] = {}
        self.observed_moves: DefaultDict[str, Set[str]] = defaultdict(set)  # uid -> {move_id}


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

                ms = self._ensure_mon(uid, player)
                ms.species = species

                base_stats = entry.get("base_stats", {}) or {}
                base_hp = base_stats.get("hp")
                if isinstance(base_hp, int):
                    ms.max_hp = base_hp

                # Fill stable slots 0..5 based on roster order
                if slot < 6 and self.sides[player].slot_uids[slot] is None:
                    self.sides[player].slot_uids[slot] = uid
                    if species and (player, species) not in self.species_slot:
                        self.species_slot[(player, species)] = slot
                    slot += 1

        self.turn_index = 0

    def _ensure_mon(self, uid: str, player: str) -> MonState:
        ms = self.mons.get(uid)
        if ms is not None:
            return ms
        ms = MonState(uid=uid, player=player)
        self.mons[uid] = ms
        return ms

    @staticmethod
    def _player_from_uid(uid: str) -> str:
        # Dataset convention: p2* is opponent, else treat as p1
        return "p2" if str(uid).startswith("p2") else "p1"

    # ---------- Snapshot ----------
    def snapshot(self) -> Dict[str, Any]:
        # Do not change keys casually: StateVectorization depends on this shape.
        return {
            "turn_index": self.turn_index,
            "p1": {
                "active_uid": self.sides["p1"].active_uid,
                "slots": list(self.sides["p1"].slot_uids),
            },
            "p2": {
                "active_uid": self.sides["p2"].active_uid,
                "slots": list(self.sides["p2"].slot_uids),
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

    # ---------- Event application ----------
    def apply_turn(self, turn_obj: Dict[str, Any]) -> None:
        self.turn_index = int(turn_obj.get("turn_number", self.turn_index + 1))
        for ev in (turn_obj.get("events", []) or []):
            et = ev.get("type")
            if et == "switch":
                self._apply_switch(ev)
            elif et == "damage":
                self._apply_hp_change(ev)
            elif et == "heal":
                self._apply_hp_change(ev)
            elif et == "status_start":
                self._apply_status_start(ev)
            elif et == "stat_change":
                self._apply_stat_change(ev)
            elif et == "faint":
                self._apply_faint(ev)
            elif et == "move":
                self._apply_move(ev)
    
    def _apply_move(self, ev: Dict[str, Any]) -> None:
        uid = ev.get("pokemon_uid")
        move_id = ev.get("move_id")
        if not uid or not move_id:
            return
        self.observed_moves[uid].add(move_id)


    def _apply_switch(self, ev: Dict[str, Any]) -> None:
        player = ev.get("player")
        uid = ev.get("into_uid") or ev.get("pokemon_uid")
        if player not in ("p1", "p2") or not uid:
            return

        ms = self._ensure_mon(uid, player)

        # If form-change species has a canonical slot, remap that slot to new uid.
        if ms.species and ms.species in self.form_change_species:
            slot = self.species_slot.get((player, ms.species))
            if slot is not None:
                self.sides[player].slot_uids[slot] = uid
        else:
            # If uid not already in slots, place into first available slot (rare).
            if uid not in self.sides[player].slot_uids:
                for i in range(6):
                    if self.sides[player].slot_uids[i] is None:
                        self.sides[player].slot_uids[i] = uid
                        if ms.species and (player, ms.species) not in self.species_slot:
                            self.species_slot[(player, ms.species)] = i
                        break

        self.sides[player].active_uid = uid

    def _apply_hp_change(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        if not target:
            return

        ms = self.mons.get(target)
        if ms is None:
            ms = self._ensure_mon(target, self._player_from_uid(target))

        hp_after = ev.get("hp_after")
        max_hp = ev.get("max_hp")

        if hp_after is not None:
            ms.hp = int(hp_after)
        if max_hp is not None:
            ms.max_hp = int(max_hp)

    def _apply_status_start(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        status = ev.get("status")
        if not target or not status:
            return

        ms = self.mons.get(target)
        if ms is None:
            ms = self._ensure_mon(target, self._player_from_uid(target))
        ms.status = status

    def _apply_stat_change(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        stat = ev.get("stat")
        amount = ev.get("amount")
        if not target or not stat or amount is None:
            return

        ms = self.mons.get(target)
        if ms is None:
            ms = self._ensure_mon(target, self._player_from_uid(target))

        if stat in ms.boosts:
            ms.boosts[stat] = max(-6, min(6, ms.boosts[stat] + int(amount)))

    def _apply_faint(self, ev: Dict[str, Any]) -> None:
        target = ev.get("target_uid")
        if not target:
            return

        ms = self.mons.get(target)
        if ms is None:
            ms = self._ensure_mon(target, self._player_from_uid(target))

        ms.fainted = True
        ms.hp = 0

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

        for turn in (battle.get("turns", []) or []):
            events = turn.get("events", []) or []
            state_before = self.snapshot()

            action = self.extract_action_for_player(events, player)

            # advance state
            self.apply_turn(turn)

            if action is None:
                continue
            if (not include_switches) and action[0] != "move":
                continue

            yield {
                "battle_id": self.battle_id,
                "turn_number": int(turn.get("turn_number")),
                "state": state_before,
                "action": action,
            }
