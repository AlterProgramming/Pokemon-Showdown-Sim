from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from threading import Condition, Lock, Thread
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from word_prediction_model.battle_inquiry import get_battle_model
from word_prediction_model.battle_questions import _allow_voluntary_switches, default_question_for_state
from word_prediction_model.policy_adapter import choose_action_from_inquiry, choose_actions_from_inquiry_batch


IPC_BATCH_WINDOW_MS = float(os.environ.get("RL_MODEL_IPC_BATCH_WINDOW_MS", "1.5"))
IPC_BATCH_MAX_SIZE = int(os.environ.get("RL_MODEL_IPC_BATCH_MAX_SIZE", "32"))
_WRITE_LOCK = Lock()
_MODEL_LOCK = Lock()
_CACHED_MODEL = None


def _get_server_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        with _MODEL_LOCK:
            if _CACHED_MODEL is None:
                _CACHED_MODEL, _ = get_battle_model()
    return _CACHED_MODEL


def _parse_predict_request(data: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    battle_state = data.get("battle_state")
    if not isinstance(battle_state, dict):
        return None, "missing battle_state"
    perspective_player = data.get("perspective_player")
    if perspective_player not in {"p1", "p2"}:
        perspective_player = "p1"
    legal_switches = data.get("legal_switches") or []
    side_payload = data.get("side") or {}
    side_pokemon = side_payload.get("pokemon") if isinstance(side_payload, dict) else None
    if isinstance(side_pokemon, list) and legal_switches:
        enriched_switches = []
        by_request_slot = {
            index + 1: mon
            for index, mon in enumerate(side_pokemon)
            if isinstance(mon, dict)
        }
        for switch in legal_switches:
            if not isinstance(switch, dict):
                enriched_switches.append(switch)
                continue
            try:
                request_slot = int(switch.get("request_slot") or switch.get("slot") or 0)
            except (TypeError, ValueError):
                request_slot = 0
            mon = by_request_slot.get(request_slot) or {}
            moves = mon.get("moves") or []
            enriched = dict(switch)
            if mon:
                enriched.setdefault("species", str(mon.get("details") or mon.get("ident") or "").split(",")[0].replace("p1: ", "").replace("p2: ", ""))
                enriched.setdefault("condition", mon.get("condition"))
                enriched.setdefault("ability", mon.get("ability") or mon.get("baseAbility"))
                enriched.setdefault("item", mon.get("item"))
                if moves:
                    enriched.setdefault("moves", [{"move": str(move)} for move in moves])
                    enriched.setdefault("observed_moves", [str(move) for move in moves])
            enriched_switches.append(enriched)
        legal_switches = enriched_switches
    return (
        {
            "battle_state": battle_state,
            "perspective_player": perspective_player,
            "question": str(data.get("question") or default_question_for_state(data)),
            "legal_moves": data.get("legal_moves") or [],
            "legal_switches": legal_switches,
            "allow_voluntary_switches": _allow_voluntary_switches(data),
            "active": data.get("active") or [],
        },
        None,
    )


def _write_message(payload: dict[str, Any]) -> None:
    with _WRITE_LOCK:
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        sys.stdout.flush()


class _PredictBatcher:
    def __init__(self, *, batch_window_ms: float, max_batch_size: int) -> None:
        self.batch_window_s = max(0.0, batch_window_ms) / 1000.0
        self.max_batch_size = max(1, max_batch_size)
        self._cv = Condition()
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._worker = Thread(target=self._run, name="word-policy-ipc-batcher", daemon=True)
        self._worker.start()

    def submit(self, message_id: str, parsed_request: dict[str, Any]) -> None:
        with self._cv:
            self._pending.append((message_id, parsed_request))
            self._cv.notify()

    def _collect_batch(self) -> list[tuple[str, dict[str, Any]]]:
        with self._cv:
            while not self._pending:
                self._cv.wait()
            if self.batch_window_s > 0.0 and len(self._pending) < self.max_batch_size:
                deadline = time.perf_counter() + self.batch_window_s
                while len(self._pending) < self.max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    self._cv.wait(timeout=remaining)
                    if not self._pending:
                        break
            batch = self._pending[: self.max_batch_size]
            del self._pending[: len(batch)]
            return batch

    def _run(self) -> None:
        while True:
            batch = self._collect_batch()
            batch_ids = [item[0] for item in batch]
            requests_only = [item[1] for item in batch]
            try:
                if len(requests_only) == 1:
                    request_data = requests_only[0]
                    results = [
                        choose_action_from_inquiry(
                            request_data["question"],
                            request_data["battle_state"],
                            perspective_player=request_data["perspective_player"],
                            legal_moves=request_data["legal_moves"],
                            legal_switches=request_data["legal_switches"],
                            allow_voluntary_switches=request_data["allow_voluntary_switches"],
                            active_payload=request_data["active"],
                            model=_get_server_model(),
                        )
                    ]
                else:
                    results = choose_actions_from_inquiry_batch(
                        requests_only,
                        model=_get_server_model(),
                    )
                for message_id, result in zip(batch_ids, results):
                    _write_message({"id": message_id, "ok": True, "result": result})
            except Exception as error:  # pragma: no cover - defensive worker path
                for message_id in batch_ids:
                    _write_message({"id": message_id, "ok": False, "error": str(error)})


_PREDICT_BATCHER = _PredictBatcher(
    batch_window_ms=IPC_BATCH_WINDOW_MS,
    max_batch_size=IPC_BATCH_MAX_SIZE,
)


def _handle_message(message: dict[str, Any]) -> None:
    message_id = str(message.get("id") or "")
    message_type = str(message.get("type") or "")

    if message_type == "health":
        _write_message(
            {
                "id": message_id,
                "ok": True,
                "result": {
                    "service": "word_policy_ipc_worker",
                    "pid": os.getpid(),
                    "status": "ok",
                    "batch_window_ms": IPC_BATCH_WINDOW_MS,
                    "batch_max_size": IPC_BATCH_MAX_SIZE,
                },
            }
        )
        return

    if message_type != "predict":
        _write_message(
            {
                "id": message_id,
                "ok": False,
                "error": f"unsupported message type: {message_type or '<empty>'}",
            }
        )
        return

    payload = message.get("payload")
    if not isinstance(payload, dict):
        _write_message({"id": message_id, "ok": False, "error": "missing payload"})
        return

    parsed_request, error = _parse_predict_request(payload)
    if error is not None:
        _write_message(
            {
                "id": message_id,
                "ok": False,
                "error": error,
            }
        )
        return

    _PREDICT_BATCHER.submit(message_id, parsed_request)


def main() -> None:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write_message({"id": "", "ok": False, "error": "invalid JSON"})
            continue
        if not isinstance(message, dict):
            _write_message({"id": "", "ok": False, "error": "message must be an object"})
            continue
        try:
            _handle_message(message)
        except Exception as error:  # pragma: no cover - defensive worker path
            _write_message(
                {
                    "id": str(message.get("id") or ""),
                    "ok": False,
                    "error": str(error),
                }
            )


if __name__ == "__main__":
    main()
