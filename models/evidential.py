#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Map features -> Dirichlet concentration parameters alpha (>0).
    """
    def __init__(self, in_dim: int, num_classes: int, *, evidence_fn: str = "softplus"):
        super().__init__()
        self.fc = nn.Linear(int(in_dim), int(num_classes))
        self.evidence_fn = str(evidence_fn).lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        if self.evidence_fn in ("relu",):
            evidence = F.relu(logits)
        else:
            evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return alpha


def uncertainty_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet uncertainty proxy: U = K / sum(alpha)
    """
    K = alpha.shape[-1]
    s = alpha.sum(dim=-1)
    return (float(K) / (s + 1e-12)).to(alpha.dtype)


def evidential_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    return uncertainty_from_alpha(alpha)


def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).to(dtype=torch.float32)


def _kl_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) )
    """
    K = alpha.shape[-1]
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_beta = beta.sum(dim=-1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    lnB_beta = torch.lgamma(sum_beta) - torch.lgamma(beta).sum(dim=-1, keepdim=True)

    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    kl = (alpha - beta) * (digamma_alpha - digamma_sum)
    kl = kl.sum(dim=-1, keepdim=True) + lnB_alpha - lnB_beta
    return kl.squeeze(-1)


def edl_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    *,
    epoch: int = 0,
    anneal_epochs: int = 130,
    coeff: float = 1.0,
) -> torch.Tensor:
    """
    Evidence-based loss (MSE on expected probabilities + annealed KL regularizer).

    - alpha: (B, K) Dirichlet params
    - y: (B,) class indices [0..K-1]
    """
    K = alpha.shape[-1]
    y_oh = _one_hot(y, K).to(alpha.device)
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / (S + 1e-12)

    mse = (y_oh - p).pow(2).sum(dim=-1)

    # encourage low evidence when wrong
    var = (alpha * (S - alpha)) / (S * S * (S + 1.0) + 1e-12)
    var_term = var.sum(dim=-1)

    # KL to uniform prior
    kl = _kl_dirichlet(alpha)

    t = min(1.0, float(epoch) / float(max(1, anneal_epochs)))
    loss = mse + var_term + (float(coeff) * t) * kl
    return loss.mean()
