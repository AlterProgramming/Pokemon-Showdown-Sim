from __future__ import annotations

"""Shared action-selection helpers for policy servers.

This module keeps the move/switch token handling and the switch-bias calibration
in one place so the entity server, benchmark server, and legacy Flask server all
make the same decision about voluntary switches.
"""

from typing import Any, Mapping, Sequence

import numpy as np


FORCED_SWITCH_REASONS = {
    "force_switch",
    "trapped",
}


def vocab_uses_action_tokens(action_vocab: Mapping[str, int]) -> bool:
    return any(
        token.startswith(("move:", "switch:"))
        for token in action_vocab
        if token != "<UNK>"
    )


def normalize_move_name(move_name: str) -> str:
    return str(move_name).lower().replace(" ", "")


def build_move_tokens(action_vocab: Mapping[str, int], move_name: str) -> list[str]:
    normalized = normalize_move_name(move_name)
    if vocab_uses_action_tokens(action_vocab):
        return [f"move:{normalized}"]
    return [normalized]


def build_switch_tokens(action_vocab: Mapping[str, int], slot: int) -> list[str]:
    if vocab_uses_action_tokens(action_vocab):
        return [f"switch:{slot}"]
    return []


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float32)
    shifted = shifted - np.max(shifted)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(shifted)
    return exp / denom


def resolve_switch_logit_bias(metadata: Mapping[str, Any] | None, *, default: float = 0.0) -> float:
    payload = metadata or {}
    action_selection = payload.get("action_selection")
    if isinstance(action_selection, Mapping):
        candidate = action_selection.get("switch_logit_bias")
        if candidate is not None:
            return float(candidate)

    for key in ("switch_logit_bias", "policy_switch_logit_bias"):
        candidate = payload.get(key)
        if candidate is not None:
            return float(candidate)

    return float(default)


def adjust_logits_for_switch_bias(
    logits: np.ndarray,
    action_vocab: Mapping[str, int],
    *,
    legal_moves: Sequence[dict[str, Any]] | None = None,
    legal_switches: Sequence[dict[str, Any]] | None = None,
    switch_reason: str | None = None,
    switch_logit_bias: float = 0.0,
) -> np.ndarray:
    """Lower voluntary switch logits while leaving forced switch turns alone."""

    if switch_logit_bias <= 0:
        return np.asarray(logits, dtype=np.float32)
    if switch_reason in FORCED_SWITCH_REASONS:
        return np.asarray(logits, dtype=np.float32)
    if legal_moves is not None and len(legal_moves) == 0:
        return np.asarray(logits, dtype=np.float32)
    if legal_switches is None or len(legal_switches) == 0:
        return np.asarray(logits, dtype=np.float32)

    adjusted = np.asarray(logits, dtype=np.float32).copy()
    switch_indices = [
        idx
        for token, idx in action_vocab.items()
        if token.startswith("switch:")
    ]
    if not switch_indices:
        return adjusted

    for idx in switch_indices:
        if 0 <= idx < adjusted.shape[0]:
            adjusted[idx] -= float(switch_logit_bias)
    return adjusted


def pick_best_slot_target(
    action_vocab: Mapping[str, int],
    logits: np.ndarray,
    slot_targets: Sequence[dict[str, Any]],
    target_type: str,
) -> tuple[dict[str, Any] | None, float | None]:
    probs = softmax(logits)
    best_target = None
    best_prob = -1.0

    for target in slot_targets:
        slot = target.get("slot")
        if slot is None:
            continue
        for token in build_switch_tokens(action_vocab, int(slot)):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            probability = float(probs[idx])
            if probability > best_prob:
                best_prob = probability
                best_target = {"type": target_type, "payload": target, "token": token}

    if best_target is not None:
        return best_target, best_prob

    if not slot_targets:
        return None, None

    return {"type": target_type, "payload": slot_targets[0], "token": None}, None


def pick_best_action(
    action_vocab: Mapping[str, int],
    logits: np.ndarray,
    legal_moves: Sequence[dict[str, Any]],
    legal_switches: Sequence[dict[str, Any]],
    *,
    switch_reason: str | None = None,
    switch_logit_bias: float = 0.0,
) -> tuple[dict[str, Any] | None, float]:
    adjusted_logits = adjust_logits_for_switch_bias(
        logits,
        action_vocab,
        legal_moves=legal_moves,
        legal_switches=legal_switches,
        switch_reason=switch_reason,
        switch_logit_bias=switch_logit_bias,
    )
    probs = softmax(adjusted_logits)
    best_action = None
    best_prob = -1.0

    for move in legal_moves:
        move_name = str(move.get("move") or move.get("id") or "")
        for token in build_move_tokens(action_vocab, move_name):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            probability = float(probs[idx])
            if probability > best_prob:
                best_prob = probability
                best_action = {"type": "move", "payload": move, "token": token}

    for switch in legal_switches:
        slot = switch.get("slot")
        if slot is None:
            continue
        for token in build_switch_tokens(action_vocab, int(slot)):
            idx = action_vocab.get(token)
            if idx is None:
                continue
            probability = float(probs[idx])
            if probability > best_prob:
                best_prob = probability
                best_action = {"type": "switch", "payload": switch, "token": token}

    return best_action, best_prob
