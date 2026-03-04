"""Model components for symbolic curriculum RL.

The network is intentionally modular: a shared encoder produces a latent that feeds
multi-scale policy/value heads and an unsupervised symbolic projection. The symbolic
projection is never directly supervised as a class label; it is shaped by auxiliary
structure-preserving losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn


class ResidualMLPBlock(nn.Module):
    """A simple residual MLP block for stable latent abstraction learning."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_norm(x + self.net(x))


@dataclass
class PolicyOutput:
    """Typed forward output for policy/value + symbolic objectives."""

    final_logits: Tensor
    tactical_logits: Tensor
    strategic_logits: Tensor
    tactical_value: Tensor
    strategic_value: Tensor
    value: Tensor
    symbolic_latent: Tensor
    irreversibility_logit: Tensor
    next_state_pred: Tensor


class SymbolicPolicyNetwork(nn.Module):
    """Policy/value model with emergent symbolic abstraction latent.

    Notes:
    - Tactical and strategic heads share an encoder, but strategic processing has a
      deeper projector and should usually train with lower LR (see
      `optimizer_param_groups`).
    - Final policy is a learnable combination of tactical and strategic logits.
    - Symbolic latent is unsupervised and shaped by temporal/contrastive losses.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        encoder_dim: int = 512,
        symbolic_dim: int = 64,
        encoder_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_layers < 2:
            raise ValueError("encoder_layers must be >= 2")

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
        )
        self.encoder = nn.ModuleList(
            [ResidualMLPBlock(encoder_dim, dropout=dropout) for _ in range(encoder_layers)]
        )

        # Relational bottleneck branch (pairwise interactions in latent chunks).
        # This creates a weak structural prior without hand-crafted symbols.
        self.relational_gate = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.Sigmoid(),
        )

        self.tactical_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
        )
        self.tactical_policy = nn.Linear(encoder_dim // 2, action_dim)
        self.tactical_value = nn.Linear(encoder_dim // 2, 1)

        self.strategic_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
        )
        self.strategic_policy = nn.Linear(encoder_dim // 2, action_dim)
        self.strategic_value = nn.Linear(encoder_dim // 2, 1)

        self.symbolic_projection = nn.Linear(encoder_dim, symbolic_dim)
        self.irreversibility_head = nn.Linear(symbolic_dim, 1)

        # Auxiliary next-state prediction to enrich abstraction quality.
        self.next_state_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, input_dim),
        )

        self.alpha_logit = nn.Parameter(torch.tensor(0.7).logit())
        self.beta_logit = nn.Parameter(torch.tensor(0.3).logit())

    @property
    def alpha_beta(self) -> Tensor:
        """Return convex tactical/strategic blend weights."""
        return torch.softmax(torch.stack([self.alpha_logit, self.beta_logit]), dim=0)

    def encode(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for block in self.encoder:
            h = block(h)
        gate = self.relational_gate(h)
        return h * gate

    def forward(self, x: Tensor, legal_action_mask: Optional[Tensor] = None) -> PolicyOutput:
        h = self.encode(x)

        tactical_h = self.tactical_head(h)
        strategic_h = self.strategic_head(h)

        tactical_logits = self.tactical_policy(tactical_h)
        strategic_logits = self.strategic_policy(strategic_h)

        alpha, beta = self.alpha_beta
        final_logits = alpha * tactical_logits + beta * strategic_logits

        if legal_action_mask is not None:
            final_logits = final_logits.masked_fill(~legal_action_mask.bool(), -1e9)

        tactical_value = self.tactical_value(tactical_h).squeeze(-1)
        strategic_value = self.strategic_value(strategic_h).squeeze(-1)
        value = alpha * tactical_value + beta * strategic_value

        symbolic_latent = self.symbolic_projection(h)
        irreversibility_logit = self.irreversibility_head(symbolic_latent).squeeze(-1)
        next_state_pred = self.next_state_head(h)

        return PolicyOutput(
            final_logits=final_logits,
            tactical_logits=tactical_logits,
            strategic_logits=strategic_logits,
            tactical_value=tactical_value,
            strategic_value=strategic_value,
            value=value,
            symbolic_latent=symbolic_latent,
            irreversibility_logit=irreversibility_logit,
            next_state_pred=next_state_pred,
        )

    def optimizer_param_groups(
        self,
        base_lr: float,
        strategic_lr_scale: float = 0.5,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, object]]:
        """Convenience parameter groups with lower strategic learning rate."""
        strategic_modules: Iterable[nn.Module] = [
            self.strategic_head,
            self.strategic_policy,
            self.strategic_value,
        ]
        strategic_params = []
        for m in strategic_modules:
            strategic_params.extend(list(m.parameters()))

        strategic_ids = {id(p) for p in strategic_params}
        rest_params = [p for p in self.parameters() if id(p) not in strategic_ids]

        return [
            {"params": rest_params, "lr": base_lr, "weight_decay": weight_decay},
            {
                "params": strategic_params,
                "lr": base_lr * strategic_lr_scale,
                "weight_decay": weight_decay,
            },
        ]
