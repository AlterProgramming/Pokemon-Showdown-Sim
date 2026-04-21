from __future__ import annotations

"""Helpers for turning predicted turn-event tokens into lightweight action scores.

These utilities are intentionally heuristic. They let us reuse the existing
sequence auxiliary head at inference time without retraining a new planner.
The resulting score is not meant to be a full simulator replacement; it is a
cheap estimate of whether a predicted public turn trace looks favorable from
the acting player's perspective.
"""

from typing import Any, Mapping, Sequence


def _extract_signed_int(raw_value: str) -> int:
    try:
        return int(str(raw_value))
    except (TypeError, ValueError):
        return 0


def decode_greedy_sequence_tokens(
    sequence_probs: Sequence[Sequence[float]] | None,
    reverse_vocab: Mapping[int, str],
) -> list[str]:
    """Decode one predicted token per time step until EOS/PAD is reached."""

    if sequence_probs is None or len(sequence_probs) == 0:
        return []

    decoded: list[str] = []
    for step_probs in sequence_probs:
        if not step_probs:
            break
        best_index = max(range(len(step_probs)), key=lambda index: float(step_probs[index]))
        token = str(reverse_vocab.get(int(best_index), "<UNK>"))
        if token in {"<PAD>", "<EOS>"}:
            break
        if token == "<BOS>":
            continue
        decoded.append(token)
    return decoded


def score_sequence_tokens(
    tokens: Sequence[str],
    *,
    perspective_player: str,
) -> float:
    """Score a decoded event sequence from the acting player's perspective."""

    if perspective_player not in {"p1", "p2"}:
        raise ValueError("perspective_player must be 'p1' or 'p2'")

    opponent_player = "p2" if perspective_player == "p1" else "p1"
    score = 0.0

    for token in tokens:
        parts = str(token).split(":")
        event_type = parts[0] if parts else ""

        if event_type == "damage" and len(parts) >= 3:
            target_side = parts[1]
            magnitude = abs(_extract_signed_int(parts[2].replace("hp_bin_", "")))
            weight = 0.20 * magnitude
            if target_side == opponent_player:
                score += weight
            elif target_side == perspective_player:
                score -= weight
            continue

        if event_type == "heal" and len(parts) >= 3:
            target_side = parts[1]
            magnitude = abs(_extract_signed_int(parts[2].replace("hp_bin_", "")))
            weight = 0.15 * magnitude
            if target_side == perspective_player:
                score += weight
            elif target_side == opponent_player:
                score -= weight
            continue

        if event_type == "status_start" and len(parts) >= 3:
            target_side = parts[1]
            if target_side == opponent_player:
                score += 0.75
            elif target_side == perspective_player:
                score -= 0.75
            continue

        if event_type == "status_end" and len(parts) >= 3:
            target_side = parts[1]
            if target_side == perspective_player:
                score += 0.35
            elif target_side == opponent_player:
                score -= 0.35
            continue

        if event_type == "boost" and len(parts) >= 4:
            target_side = parts[1]
            magnitude = abs(_extract_signed_int(parts[3]))
            weight = 0.30 * magnitude
            if target_side == perspective_player:
                score += weight
            elif target_side == opponent_player:
                score -= weight
            continue

        if event_type == "unboost" and len(parts) >= 4:
            target_side = parts[1]
            magnitude = abs(_extract_signed_int(parts[3]))
            weight = 0.30 * magnitude
            if target_side == opponent_player:
                score += weight
            elif target_side == perspective_player:
                score -= weight
            continue

        if event_type == "faint" and len(parts) >= 2:
            target_side = parts[1]
            if target_side == opponent_player:
                score += 2.0
            elif target_side == perspective_player:
                score -= 2.0
            continue

    return float(score)


def combine_policy_and_auxiliary_scores(
    *,
    policy_logit: float,
    value_prediction: float | None = None,
    sequence_score: float | None = None,
    auxiliary_scale: float = 1.0,
    value_scale: float = 1.0,
    sequence_scale: float = 1.0,
) -> float:
    """Blend the base policy score with optional auxiliary signals."""

    combined = float(policy_logit)
    if value_prediction is not None:
        combined += float(auxiliary_scale) * float(value_scale) * (float(value_prediction) - 0.5)
    if sequence_score is not None:
        combined += float(auxiliary_scale) * float(sequence_scale) * float(sequence_score)
    return combined


def summarize_auxiliary_prediction(
    *,
    sequence_tokens: Sequence[str] | None,
    sequence_score: float | None,
    value_prediction: float | None,
    used_opponent_action_token: str | None,
) -> dict[str, Any]:
    """Build a compact JSON-friendly auxiliary summary."""

    return {
        "used_opponent_action_token": used_opponent_action_token,
        "value_prediction": None if value_prediction is None else float(value_prediction),
        "sequence_score": None if sequence_score is None else float(sequence_score),
        "sequence_tokens": list(sequence_tokens or []),
    }
