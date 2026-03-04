"""Symbolic-power curriculum scheduling utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional


SP_BAND_DESCRIPTIONS = {
    1: "damage trades only",
    2: "status infliction",
    3: "setup moves",
    4: "hazard control",
    5: "win condition construction",
}


@dataclass
class CurriculumConfig:
    start_band: int = 1
    max_band: int = 5
    promotion_winrate: float = 0.6
    promotion_games: int = 100


class SymbolicCurriculumScheduler:
    """Promotes training task scope based on per-band performance."""

    def __init__(self, config: Optional[CurriculumConfig] = None) -> None:
        self.config = config or CurriculumConfig()
        self.current_SP_band = self.config.start_band
        self.performance_history: Dict[int, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.config.promotion_games)
        )

    def record_result(self, won: bool, band: Optional[int] = None) -> None:
        b = band or self.current_SP_band
        self.performance_history[b].append(1 if won else 0)

    def current_winrate(self, band: Optional[int] = None) -> float:
        b = band or self.current_SP_band
        hist = self.performance_history[b]
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def maybe_promote(self) -> bool:
        """Return True when curriculum advances to next SP band."""
        band = self.current_SP_band
        hist = self.performance_history[band]
        if len(hist) < self.config.promotion_games:
            return False
        if self.current_winrate(band) < self.config.promotion_winrate:
            return False
        if band >= self.config.max_band:
            return False
        self.current_SP_band += 1
        return True

    def allowed_bands(self) -> Iterable[int]:
        return range(1, self.current_SP_band + 1)


class SymbolicCurriculumWrapper:
    """Environment wrapper contract for SP-band filtering or biased sampling.

    The wrapped env is expected to implement `reset(sp_band=...)` or accept band
    hints via kwargs; if unsupported, wrapper gracefully falls back to plain reset.
    """

    def __init__(self, env, scheduler: SymbolicCurriculumScheduler) -> None:
        self.env = env
        self.scheduler = scheduler

    def reset(self, **kwargs):
        kwargs.setdefault("sp_band", self.scheduler.current_SP_band)
        try:
            return self.env.reset(**kwargs)
        except TypeError:
            kwargs.pop("sp_band", None)
            return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, item):
        return getattr(self.env, item)
