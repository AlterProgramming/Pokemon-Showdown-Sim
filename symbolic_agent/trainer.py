"""Minimal PPO-style trainer with symbolic auxiliary objectives."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .curriculum import SymbolicCurriculumScheduler
from .evaluator import cluster_separation_score, effective_rank
from .losses import (
    LossWeights,
    compute_total_loss,
    contrastive_abstraction_loss,
    irreversibility_signal_loss,
    irreversibility_targets_from_heuristics,
    temporal_consistency_loss,
)
from .model import SymbolicPolicyNetwork


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 256
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    base_lr: float = 3e-4
    strategic_lr_scale: float = 0.5


class VectorizedBattleRunner:
    """Collects async rollouts across many environment instances."""

    def __init__(self, env_fns, num_envs: int, seed: int = 0) -> None:
        self.envs = [env_fns[i]() for i in range(num_envs)]
        self.num_envs = num_envs
        for i, env in enumerate(self.envs):
            if hasattr(env, "seed"):
                env.seed(seed + i)
            random.seed(seed + i)
            np.random.seed(seed + i)

    async def _step_env(self, env, action):
        if asyncio.iscoroutinefunction(env.step):
            return await env.step(action)
        return env.step(action)

    async def step(self, actions):
        tasks = [self._step_env(env, act) for env, act in zip(self.envs, actions)]
        return await asyncio.gather(*tasks)

    def reset(self):
        obs = []
        infos = []
        for env in self.envs:
            out = env.reset()
            if isinstance(out, tuple):
                o, info = out
            else:
                o, info = out, {}
            obs.append(o)
            infos.append(info)
        return np.asarray(obs), infos


class SymbolicPPOTrainer:
    """PPO trainer with symbolic curriculum and representation logging."""

    def __init__(
        self,
        model: SymbolicPolicyNetwork,
        runner: VectorizedBattleRunner,
        scheduler: SymbolicCurriculumScheduler,
        config: Optional[PPOConfig] = None,
        loss_weights: Optional[LossWeights] = None,
        discover_mode: bool = False,
        device: str = "cpu",
        checkpoint_path: str = "symbolic_agent_checkpoint.pt",
    ) -> None:
        self.model = model.to(device)
        self.runner = runner
        self.scheduler = scheduler
        self.cfg = config or PPOConfig()
        self.weights = loss_weights or LossWeights(entropy=self.cfg.entropy_coef)
        self.discover_mode = discover_mode
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.optimizer = torch.optim.Adam(
            self.model.optimizer_param_groups(
                base_lr=self.cfg.base_lr,
                strategic_lr_scale=self.cfg.strategic_lr_scale,
            )
        )
        self.total_games = 0
        self.total_wins = 0

    @staticmethod
    def compute_gae(rewards, values, dones, gamma, lam):
        adv = torch.zeros_like(rewards)
        last_adv = 0.0
        for t in reversed(range(rewards.shape[0])):
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1] if t + 1 < values.shape[0] else 0.0
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_adv = delta + gamma * lam * next_nonterminal * last_adv
            adv[t] = last_adv
        returns = adv + values
        return adv, returns

    async def collect_rollout(self) -> Dict[str, Tensor]:
        obs_np, _ = self.runner.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        storage: Dict[str, List[Tensor]] = {k: [] for k in [
            "obs", "actions", "logp", "rewards", "dones", "values", "battle_ids", "turn_ids",
            "hp_drop", "faint_event", "status_inflicted", "setup_boost", "hazard_set", "legal_mask",
            "next_obs"
        ]}

        for _ in range(self.cfg.rollout_steps):
            legal_mask = torch.ones((obs.shape[0], self.model.tactical_policy.out_features), device=self.device, dtype=torch.bool)
            out = self.model(obs, legal_action_mask=legal_mask)
            dist = Categorical(logits=out.final_logits)
            action = dist.sample()

            step_outs = await self.runner.step(action.detach().cpu().numpy())
            next_obs, reward, done, info = [], [], [], []
            for r in step_outs:
                if len(r) == 5:
                    o, rew, terminated, truncated, inf = r
                    dn = terminated or truncated
                else:
                    o, rew, dn, inf = r
                next_obs.append(o)
                reward.append(rew)
                done.append(dn)
                info.append(inf)

            storage["obs"].append(obs)
            storage["actions"].append(action)
            storage["logp"].append(dist.log_prob(action))
            storage["rewards"].append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            storage["dones"].append(torch.tensor(done, dtype=torch.float32, device=self.device))
            storage["values"].append(out.value.detach())
            storage["battle_ids"].append(torch.tensor([i.get("battle_id", idx) for idx, i in enumerate(info)], device=self.device))
            storage["turn_ids"].append(torch.tensor([i.get("turn", 0) for i in info], device=self.device))
            storage["hp_drop"].append(torch.tensor([i.get("hp_drop", 0.0) for i in info], dtype=torch.float32, device=self.device))
            storage["faint_event"].append(torch.tensor([i.get("faint", 0) for i in info], dtype=torch.float32, device=self.device))
            storage["status_inflicted"].append(torch.tensor([i.get("status", 0) for i in info], dtype=torch.float32, device=self.device))
            storage["setup_boost"].append(torch.tensor([i.get("setup_boost", 0) for i in info], dtype=torch.float32, device=self.device))
            storage["hazard_set"].append(torch.tensor([i.get("hazard_set", 0) for i in info], dtype=torch.float32, device=self.device))
            storage["legal_mask"].append(legal_mask)

            next_obs_t = torch.tensor(np.asarray(next_obs), dtype=torch.float32, device=self.device)
            storage["next_obs"].append(next_obs_t)
            obs = next_obs_t

            for d, inf in zip(done, info):
                if d:
                    self.total_games += 1
                    won = bool(inf.get("won", False))
                    self.total_wins += int(won)
                    self.scheduler.record_result(won)

        batch = {k: torch.cat(v, dim=0) for k, v in storage.items()}
        adv, ret = self.compute_gae(
            batch["rewards"], batch["values"], batch["dones"], self.cfg.gamma, self.cfg.lam
        )
        batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["returns"] = ret
        return batch

    def ppo_update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        n = batch["obs"].shape[0]
        idxs = np.arange(n)
        logs: Dict[str, float] = {}

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idxs[start : start + self.cfg.minibatch_size]
                obs = batch["obs"][mb]
                out = self.model(obs, legal_action_mask=batch["legal_mask"][mb])
                dist = Categorical(logits=out.final_logits)

                new_logp = dist.log_prob(batch["actions"][mb])
                ratio = torch.exp(new_logp - batch["logp"][mb])
                surr1 = ratio * batch["advantages"][mb]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * batch["advantages"][mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(out.value, batch["returns"][mb])
                entropy = dist.entropy().mean()

                with torch.no_grad():
                    out_next = self.model(batch["next_obs"][mb])
                    sim = F.cosine_similarity(obs, batch["next_obs"][mb], dim=-1).clamp(min=0.0)

                temp_loss = temporal_consistency_loss(
                    out.symbolic_latent,
                    out_next.symbolic_latent,
                    similarity=sim,
                    hp_drop=batch["hp_drop"][mb],
                    faint_event=batch["faint_event"][mb],
                )
                cont_loss = contrastive_abstraction_loss(
                    out.symbolic_latent,
                    battle_ids=batch["battle_ids"][mb],
                    turn_ids=batch["turn_ids"][mb],
                )

                irr_targets = irreversibility_targets_from_heuristics(
                    faint_event=batch["faint_event"][mb],
                    status_inflicted=batch["status_inflicted"][mb],
                    setup_boost_stages=batch["setup_boost"][mb],
                    hazard_set=batch["hazard_set"][mb],
                )
                irr_loss = irreversibility_signal_loss(out.irreversibility_logit, irr_targets)
                next_state_loss = F.mse_loss(out.next_state_pred, batch["next_obs"][mb])

                loss_dict = compute_total_loss(
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    temporal_loss=temp_loss,
                    contrastive_loss=cont_loss,
                    irreversibility_loss=irr_loss,
                    next_state_loss=next_state_loss,
                    weights=self.weights,
                    discover_mode=self.discover_mode,
                )

                self.optimizer.zero_grad()
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                for k, v in loss_dict.items():
                    logs[k] = float(v.detach().item())

        return logs

    def representation_metrics(self, symbolic_latents: Tensor) -> Dict[str, float]:
        z = symbolic_latents.detach().cpu().numpy()
        sample = z[: min(5000, len(z))]
        metrics = {}
        if len(sample) >= 10:
            try:
                cluster = cluster_separation_score(sample)
                metrics["silhouette"] = cluster.silhouette
                metrics["davies_bouldin"] = cluster.davies_bouldin
            except Exception:
                metrics["silhouette"] = float("nan")
                metrics["davies_bouldin"] = float("nan")
        metrics["embedding_variance"] = float(np.var(sample))
        metrics["effective_rank"] = effective_rank(torch.tensor(sample, dtype=torch.float32))
        return metrics

    def save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_games": self.total_games,
                "total_wins": self.total_wins,
                "sp_band": self.scheduler.current_SP_band,
            },
            self.checkpoint_path,
        )

    async def train(self, updates: int) -> None:
        try:
            for _ in range(updates):
                batch = await self.collect_rollout()
                logs = self.ppo_update(batch)
                metrics = self.representation_metrics(self.model(batch["obs"]).symbolic_latent)
                self.scheduler.maybe_promote()
                _ = {**logs, **metrics, "winrate": self.winrate, "sp_band": self.scheduler.current_SP_band}
        except KeyboardInterrupt:
            metrics = self.representation_metrics(self.model(batch["obs"]).symbolic_latent) if 'batch' in locals() else {}
            print("\n[Interrupt Summary]")
            print(f"Total games: {self.total_games}")
            print(f"Winrate: {self.winrate:.3f}")
            print(f"SP band: {self.scheduler.current_SP_band}")
            print(f"Avg embedding norm: {metrics.get('embedding_variance', float('nan')):.6f}")
            print(f"Cluster separation score: {metrics.get('silhouette', float('nan')):.6f}")
            self.save_checkpoint()
            raise

    @property
    def winrate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games
