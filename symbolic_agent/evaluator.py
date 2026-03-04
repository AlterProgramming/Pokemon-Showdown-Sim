"""Representation diagnostics for symbolic abstraction emergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import Tensor


@dataclass
class ClusterMetrics:
    silhouette: float
    davies_bouldin: float


def cluster_separation_score(symbolic_latents: np.ndarray, n_clusters: int = 8) -> ClusterMetrics:
    """KMeans + silhouette + Davies-Bouldin index on symbolic projections."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = km.fit_predict(symbolic_latents)
    sil = silhouette_score(symbolic_latents, labels)
    dbi = davies_bouldin_score(symbolic_latents, labels)
    return ClusterMetrics(silhouette=float(sil), davies_bouldin=float(dbi))


def effective_rank(symbolic_latents: Tensor, eps: float = 1e-8) -> float:
    """Entropy-based effective rank of covariance matrix."""
    z = symbolic_latents - symbolic_latents.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)
    p = eigvals / eigvals.sum()
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


def linear_probe_score(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-2,
) -> float:
    """Simple offline linear probe with PyTorch logistic regression."""
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    clf = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    for _ in range(epochs):
        logits = clf(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = (torch.sigmoid(clf(x).squeeze(-1)) > 0.5).float()
        acc = (pred == y).float().mean().item()
    return float(acc)


def evaluate_representation_suite(
    symbolic_latents: np.ndarray,
    faint_next: np.ndarray,
    setup_used: np.ndarray,
    outcome: np.ndarray,
) -> Dict[str, float]:
    """Full representation stratification summary."""
    cluster = cluster_separation_score(symbolic_latents)
    zt = torch.tensor(symbolic_latents, dtype=torch.float32)
    return {
        "silhouette": cluster.silhouette,
        "davies_bouldin": cluster.davies_bouldin,
        "effective_rank": effective_rank(zt),
        "probe_faint_next": linear_probe_score(symbolic_latents, faint_next),
        "probe_setup_used": linear_probe_score(symbolic_latents, setup_used),
        "probe_outcome": linear_probe_score(symbolic_latents, outcome),
    }
