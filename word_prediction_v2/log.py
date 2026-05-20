"""
Structured logger for word_prediction_model.

Mirrors core/log.py ergonomics but is self-contained (no cross-submodule import).

IMPORTANT — ipc_policy_worker.py owns stdout for IPC.
This module always writes to stderr. Never change the logger_factory sink to stdout.

Usage
-----
from word_prediction_model.log import log, trace, bind_battle, clear_battle

log.d("word_ranked", word=w, score=score)
log.i("policy_decision", action=a, battle_id=bid)

bind_battle(battle_id="b-001", turn=3, model_id="word-v1")
...
clear_battle()

@trace
def rank_words(prompt, lexicon): ...
"""

import functools
import logging
import os
import sys
import time

import structlog
import structlog.contextvars


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

    renderer = (
        structlog.dev.ConsoleRenderer(colors=True)
        if fmt == "pretty"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_configure()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)


log: structlog.BoundLogger = structlog.get_logger(__name__)
log.d = log.debug    # type: ignore[attr-defined]
log.i = log.info     # type: ignore[attr-defined]
log.w = log.warning  # type: ignore[attr-defined]
log.e = log.error    # type: ignore[attr-defined]


def bind_battle(
    battle_id: str = "",
    turn: int = 0,
    model_id: str = "",
    run_id: str = "",
) -> list:
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
    structlog.contextvars.bind_contextvars(turn=turn)


def clear_battle(_keys: list | None = None) -> None:
    structlog.contextvars.clear_contextvars()


def trace(_fn=None, *, level: str = "debug", logger: structlog.BoundLogger | None = None):
    """Decorator: logs entry, exit, and duration for any function."""
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
