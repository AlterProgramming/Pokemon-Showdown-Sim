"""Structured event schema for within-turn battle events (version 1).

Used by BattleStateTracker to emit ordered event sequences alongside
existing state_before / action / state_after training examples.  Each
event captures one atomic change that happened during a turn, with
fields populated according to the event family.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EVENT_MOVE: str = "move"
EVENT_SWITCH: str = "switch"
EVENT_DAMAGE: str = "damage"
EVENT_HEAL: str = "heal"
EVENT_STATUS_START: str = "status_start"
EVENT_STATUS_END: str = "status_end"
EVENT_BOOST: str = "boost"
EVENT_UNBOOST: str = "unboost"
EVENT_FAINT: str = "faint"
EVENT_WEATHER: str = "weather"
EVENT_FIELD: str = "field"
EVENT_SIDE_CONDITION: str = "side_condition"
EVENT_FORME_CHANGE: str = "forme_change"
EVENT_TURN_END: str = "turn_end"

# Ordered list of all event type strings (canonical ordering).
ALL_EVENT_TYPES: List[str] = [
    EVENT_MOVE,
    EVENT_SWITCH,
    EVENT_DAMAGE,
    EVENT_HEAL,
    EVENT_STATUS_START,
    EVENT_STATUS_END,
    EVENT_BOOST,
    EVENT_UNBOOST,
    EVENT_FAINT,
    EVENT_WEATHER,
    EVENT_FIELD,
    EVENT_SIDE_CONDITION,
    EVENT_FORME_CHANGE,
    EVENT_TURN_END,
]

# Mapping from event type string to a unique 0-indexed integer ID.
EVENT_TYPE_VOCAB: Dict[str, int] = {name: idx for idx, name in enumerate(ALL_EVENT_TYPES)}

# ---------------------------------------------------------------------------
# Utility: HP delta quantization
# ---------------------------------------------------------------------------

def hp_delta_to_bin(before_frac: float, after_frac: float) -> int:
    """Quantize an HP fraction change into a 5 % bin.

    Parameters
    ----------
    before_frac : float or None
        HP fraction before the event (0.0 -- 1.0).
    after_frac : float or None
        HP fraction after the event (0.0 -- 1.0).

    Returns
    -------
    int
        Integer bin in the range [-20, 20] representing the delta
        rounded to the nearest 5 % step.  Returns 0 when inputs are
        missing or indeterminate.
    """
    # Handle missing / indeterminate inputs.
    if before_frac is None and after_frac is None:
        return 0
    if before_frac is None or before_frac == 0:
        if after_frac is None:
            return 0
    if after_frac is None:
        return 0

    # Treat None-that-slipped-through as 0.
    bf = float(before_frac) if before_frac is not None else 0.0
    af = float(after_frac) if after_frac is not None else 0.0

    delta = af - bf
    raw_bin = round(delta / 0.05)
    return max(-20, min(20, raw_bin))

# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class TurnEventV1:
    """A single within-turn battle event.

    Fields are populated selectively depending on ``event_type``:

    * **move** -- ``actor_side``, ``target_side``, ``move_id``
    * **switch** -- ``actor_side``, ``species_id``, ``slot_index``
    * **damage / heal** -- ``target_side``, ``hp_delta_bin``
    * **boost / unboost** -- ``target_side``, ``boost_stat``, ``boost_delta``
    * **status_start / status_end** -- ``target_side``, ``status``
    * **faint** -- ``target_side``
    * **weather** -- ``weather``
    * **field** -- ``terrain``
    * **side_condition** -- ``actor_side``, ``side_condition``
    * **forme_change** -- ``target_side``, ``species_id``
    * **turn_end** -- (no extra fields)
    """

    event_type: str          # One of the EVENT_* constants
    actor_side: str = ""     # "p1" or "p2" (who performed the action)
    target_side: str = ""    # "p1" or "p2" (who was affected)
    move_id: str = ""        # Move identifier (for MOVE events)
    species_id: str = ""     # Species (for SWITCH, FORME_CHANGE)
    hp_delta_bin: int = 0    # 5% bin for DAMAGE/HEAL
    boost_stat: str = ""     # Stat name for BOOST/UNBOOST
    boost_delta: int = 0     # Exact integer stages for BOOST/UNBOOST
    status: str = ""         # Status ID for STATUS_START/END
    weather: str = ""        # Weather name for WEATHER
    terrain: str = ""        # Field condition for FIELD
    side_condition: str = "" # Side condition for SIDE_CONDITION
    slot_index: int = 0      # Slot index for SWITCH (1-6)

    # ---- Serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        """Return a dict containing only non-default fields.

        ``event_type`` is always included.  Empty strings and zero-valued
        integers are omitted for all other fields to keep the
        representation compact.
        """
        result: dict = {"event_type": self.event_type}
        for f in fields(self):
            if f.name == "event_type":
                continue
            value = getattr(self, f.name)
            # Skip default-like values.
            if isinstance(value, str) and value == "":
                continue
            if isinstance(value, int) and value == 0:
                continue
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "TurnEventV1":
        """Reconstruct a ``TurnEventV1`` from a dict (e.g. from ``to_dict``).

        Unknown keys in *d* are silently ignored so that forward
        compatibility is preserved when new optional fields are added.
        """
        valid_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_names}
        return cls(**filtered)
