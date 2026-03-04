"""Auxiliary and composite losses for symbolic curriculum RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 0.5
    temporal: float = 0.2
    contrastive: float = 0.2
    irreversibility: float = 0.1
    next_state: float = 0.05
    entropy: float = 0.01


def temporal_consistency_loss(
    z_t: Tensor,
    z_tp1: Tensor,
    similarity: Tensor,
    hp_drop: Tensor,
    faint_event: Tensor,
    hp_drop_threshold: float = 0.3,
) -> Tensor:
    """MSE between adjacent symbolic latents with event-aware weighting."""
    base = F.mse_loss(z_t, z_tp1, reduction="none").mean(dim=-1)
    transition_penalty = torch.where(
        (hp_drop > hp_drop_threshold) | faint_event.bool(),
        torch.full_like(similarity, 0.25),
        torch.ones_like(similarity),
    )
    weights = similarity.clamp(0.0, 1.0) * transition_penalty
    return (base * weights).mean()


def contrastive_abstraction_loss(
    z: Tensor,
    battle_ids: Tensor,
    turn_ids: Tensor,
    temperature: float = 0.1,
    max_turn_gap: int = 3,
) -> Tensor:
    """InfoNCE: positive pairs are nearby turns in same battle, negatives otherwise."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature

    same_battle = battle_ids[:, None] == battle_ids[None, :]
    turn_gap = (turn_ids[:, None] - turn_ids[None, :]).abs()
    positive = same_battle & (turn_gap > 0) & (turn_gap <= max_turn_gap)
    eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)

    losses = []
    for i in range(z.size(0)):
        pos_idx = positive[i].nonzero(as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0:
            continue
        logits = sim[i].masked_fill(eye[i], float("-inf"))
        log_denom = torch.logsumexp(logits, dim=0)
        pos_log_prob = logits[pos_idx] - log_denom
        losses.append(-pos_log_prob.mean())

    if not losses:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(losses).mean()


def irreversibility_targets_from_heuristics(
    faint_event: Tensor,
    status_inflicted: Tensor,
    setup_boost_stages: Tensor,
    hazard_set: Tensor,
) -> Tensor:
    """Weak-supervision target for irreversible state transition candidates."""
    setup_event = setup_boost_stages >= 2
    y = faint_event.bool() | status_inflicted.bool() | setup_event.bool() | hazard_set.bool()
    return y.float()


def irreversibility_signal_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def compute_total_loss(
    *,
    policy_loss: Tensor,
    value_loss: Tensor,
    entropy: Tensor,
    temporal_loss: Tensor,
    contrastive_loss: Tensor,
    irreversibility_loss: Tensor,
    next_state_loss: Tensor,
    weights: Optional[LossWeights] = None,
    discover_mode: bool = False,
) -> Dict[str, Tensor]:
    """Compose PPO + abstraction losses, with optional discovery mode."""
    w = weights or LossWeights()

    if discover_mode:
        irr_component = torch.tensor(0.0, device=policy_loss.device)
    else:
        irr_component = w.irreversibility * irreversibility_loss

    total = (
        w.policy * policy_loss
        + w.value * value_loss
        - w.entropy * entropy
        + w.temporal * temporal_loss
        + w.contrastive * contrastive_loss
        + irr_component
        + w.next_state * next_state_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy,
        "temporal": temporal_loss,
        "contrastive": contrastive_loss,
        "irreversibility": irreversibility_loss,
        "next_state": next_state_loss,
    }
