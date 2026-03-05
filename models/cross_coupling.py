#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import torch
import torch.nn as nn


class FiLMCrossCoupling(nn.Module):
    """
    FiLM modulation:
      - spatial: (B, C, H, W)
      - spectral: (B, L, D)  (L = bands)
    Returns:
      spa_mod: (B, C, H, W)
      spec_mod: (B, L, D)
    """
    def __init__(self, d_spa: int = 128, d_spec: int = 96, **kwargs):
        # allow alias names
        if "d_spatial" in kwargs:
            d_spa = int(kwargs.pop("d_spatial"))
        if "spa_channels" in kwargs:
            d_spa = int(kwargs.pop("spa_channels"))
        if "spec_channels" in kwargs:
            d_spec = int(kwargs.pop("spec_channels"))
        super().__init__()
        self.d_spa = int(d_spa)
        self.d_spec = int(d_spec)

        self.spa_gamma = nn.Linear(self.d_spec, self.d_spa)
        self.spa_beta = nn.Linear(self.d_spec, self.d_spa)

        self.spec_gamma = nn.Linear(self.d_spa, self.d_spec)
        self.spec_beta = nn.Linear(self.d_spa, self.d_spec)

    def forward(self, spa_feat: torch.Tensor, spec_feat: torch.Tensor):
        # accept tokens (B, N, C) for spatial
        if spa_feat.dim() == 3:
            B, N, C = spa_feat.shape
            s = int(round(math.sqrt(float(N))))
            if s * s != N:
                raise RuntimeError(f"[FiLMCrossCoupling] spatial tokens N={N} not square")
            spa_feat = spa_feat.transpose(1, 2).reshape(B, C, s, s)

        B, C, H, W = spa_feat.shape
        B2, L, D = spec_feat.shape
        if B2 != B:
            raise RuntimeError("[FiLMCrossCoupling] batch mismatch")
        if C != self.d_spa:
            raise RuntimeError(f"[FiLMCrossCoupling] expected spa C={self.d_spa}, got {C}")
        if D != self.d_spec:
            raise RuntimeError(f"[FiLMCrossCoupling] expected spec D={self.d_spec}, got {D}")

        spec_pool = spec_feat.mean(dim=1)                 # (B, D)
        gamma = self.spa_gamma(spec_pool).view(B, C, 1, 1)
        beta = self.spa_beta(spec_pool).view(B, C, 1, 1)
        spa_mod = spa_feat * (1.0 + gamma) + beta

        spa_pool = spa_mod.mean(dim=(2, 3))               # (B, C)
        gamma_s = self.spec_gamma(spa_pool).unsqueeze(1)  # (B, 1, D)
        beta_s = self.spec_beta(spa_pool).unsqueeze(1)    # (B, 1, D)
        spec_mod = spec_feat * (1.0 + gamma_s) + beta_s

        return spa_mod, spec_mod
