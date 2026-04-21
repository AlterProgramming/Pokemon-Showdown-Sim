"""
Structured logger for the Pokemon-Showdown-Sim ML backend.

Usage
-----
from core.log import log, trace, bind_battle, clear_battle

# Module-level log object (Rust macro equivalent: log::debug!)
log.d("action_selected", action=action, battle_id=bid)
log.i("turn_complete", turn=t, winner=None)
log.w("illegal_move_filtered", move=m)
log.e("model_timeout", model_id=mid)

# Bind context once per battle — all downstream log calls inherit it
tokens = bind_battle(battle_id="b-001", turn=3, model_id="model4", run_id="run-42")
...
clear_battle(tokens)

# Decorator: logs entry, exit, args count, and duration
@trace
def select_action(state, model_id): ...

# Output: JSON to stderr by default (stdout is reserved for IPC in policy workers).
# Set LOG_FORMAT=pretty for human-readable dev output.
# Set LOG_LEVEL=DEBUG|INFO|WARNING|ERROR  (default: INFO)
"""

import functools
import logging
import os
import sys
import time

import structlog
import structlog.contextvars

# ── Configuration ─────────────────────────────────────────────────────────────

def _configure() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get("LOG_FORMAT", "json").lower()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "pretty":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

_configure()

# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)


# Convenience aliases mirroring Rust's log::debug! / log::info! / etc.
# Import as: `from core.log import log`
# then: log.d("event", key=val)
log: structlog.BoundLogger = structlog.get_logger(__name__)
log.d = log.debug    # type: ignore[attr-defined]
log.i = log.info     # type: ignore[attr-defined]
log.w = log.warning  # type: ignore[attr-defined]
log.e = log.error    # type: ignore[attr-defined]


# ── Battle context binding ────────────────────────────────────────────────────

def bind_battle(
    battle_id: str = "",
    turn: int = 0,
    model_id: str = "",
    run_id: str = "",
) -> list:
    """Bind per-battle context variables. Returns tokens for later clear_battle()."""
    structlog.contextvars.clear_contextvars()
    kw: dict = {}
    if battle_id: kw["battle_id"] = battle_id
    if turn:      kw["turn"] = turn
    if model_id:  kw["model_id"] = model_id
    if run_id:    kw["run_id"] = run_id
    if kw:
        structlog.contextvars.bind_contextvars(**kw)
    return list(kw.keys())


def update_turn(turn: int) -> None:
    """Update the turn counter in the active battle context."""
    structlog.contextvars.bind_contextvars(turn=turn)


def clear_battle(_keys: list | None = None) -> None:
    """Clear all bound battle context."""
    structlog.contextvars.clear_contextvars()


# ── @trace decorator ──────────────────────────────────────────────────────────

def trace(_fn=None, *, level: str = "debug", logger: structlog.BoundLogger | None = None):
    """
    Log function entry, exit, and wall-clock duration.

    @trace
    def select_action(state, model_id): ...

    @trace(level="info")
    def run_inference(batch): ...
    """
    def decorator(fn):
        _log = logger or structlog.get_logger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            emit = getattr(_log, level)
            emit("enter", fn=fn.__name__, nargs=len(args))
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                emit("exit", fn=fn.__name__, duration_ms=round((time.perf_counter() - t0) * 1000, 2))
                return result
            except Exception as exc:
                _log.error("error", fn=fn.__name__, exc=str(exc), exc_type=type(exc).__name__)
                raise

        return wrapper

    return decorator(_fn) if _fn is not None else decorator
