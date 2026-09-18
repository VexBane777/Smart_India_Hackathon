"""One-Class Softmax (Zhang, Yamagishi & Todisco, 2021 - "One-Class
Learning Towards Synthetic Voice Spoofing Detection"), as Idea 6's
primary lever (docs/superpowers/specs/2026-09-17-vaani-model-architecture-
brainstorm.md). Bounds a single learned target direction w0 representing
"genuine human speech"; genuine embeddings are pulled inside a tight
margin around w0, spoof embeddings pushed outside a looser one. Unlike
two-class BCE, nothing here is free to pick style as the separating axis
by construction -- style variance within genuine speech still has to fit
inside the SAME bound, whatever axis the trunk ends up using.
"""
from __future__ import annotations

import torch
from torch import nn


class OCSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim: int, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(embedding_dim))
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha

    def forward(self, embedding: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w0 = nn.functional.normalize(self.w0, dim=0)
        emb = nn.functional.normalize(embedding, dim=-1)
        cos_sim = emb @ w0  # (B,)
        margin = torch.where(y == 0, torch.full_like(cos_sim, self.m_real), torch.full_like(cos_sim, self.m_fake))
        sign = torch.where(y == 0, torch.ones_like(cos_sim), -torch.ones_like(cos_sim))
        return nn.functional.softplus(self.alpha * sign * (margin - cos_sim)).mean()


def oc_softmax_score(embedding: torch.Tensor, w0: torch.Tensor) -> torch.Tensor:
    """Cosine similarity to w0, clamped to [-1, 1] (float32 normalize can
    overshoot by ~1e-7 on exact alignment/opposition). Higher = more
    human-like. Callers rescale to a [0,1] fake-probability via
    (1 - score) / 2 for the confound gate."""
    sim = nn.functional.normalize(embedding, dim=-1) @ nn.functional.normalize(w0, dim=0)
    return sim.clamp(-1.0, 1.0)
