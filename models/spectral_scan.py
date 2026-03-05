#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _best_group_divisor(channels: int, groups: int) -> int:
    g = int(max(1, groups))
    c = int(channels)
    if c % g == 0:
        return g
    # find a divisor <= groups
    for gg in range(min(g, c), 0, -1):
        if c % gg == 0:
            return gg
    return 1


class _SpectralConvBlock(nn.Module):
    def __init__(self, d_model: int, groups: int, dropout: float, kernel_size: int, dilation: int):
        super().__init__()
        self.d_model = int(d_model)
        g = _best_group_divisor(self.d_model, int(groups))

        k = int(kernel_size)
        d = int(dilation)
        pad = (k // 2) * d

        self.ln = nn.LayerNorm(self.d_model)
        self.dw = nn.Conv1d(self.d_model, self.d_model, kernel_size=k, padding=pad, dilation=d, groups=self.d_model, bias=False)
        self.pw1 = nn.Conv1d(self.d_model, self.d_model * 2, kernel_size=1, groups=g, bias=True)
        self.pw2 = nn.Conv1d(self.d_model * 2, self.d_model, kernel_size=1, groups=g, bias=True)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = self.ln(x)
        y = y.transpose(1, 2)                 # (B, C, L)
        y = self.dw(y)
        y = self.pw2(F.gelu(self.pw1(y)))
        y = y.transpose(1, 2)                 # (B, L, C)
        y = self.drop(y)
        return x + y


class GroupedBiSpectralScan(nn.Module):
    """Band-axis mixer that actually mixes along the spectral dimension.

    Input:
      x: (B, bands) or (B, bands, d_model)
    Output:
      y: (B, bands, d_model)
    """

    def __init__(
        self,
        bands: int = 274,
        d_model: int = 96,
        num_layers: int = 4,
        groups: int = 8,
        dropout: float = 0.0,
        kernel_size: int = 7,
        dilation_cycle: tuple[int, ...] = (1, 2, 4, 1),
        **kwargs,
    ):
        if "d_in" in kwargs:
            bands = int(kwargs.pop("d_in")) if isinstance(kwargs["d_in"], int) else bands
        if "d_hidden" in kwargs:
            d_model = int(kwargs.pop("d_hidden"))
        super().__init__()

        self.bands = int(bands)
        self.d_model = int(d_model)
        self.groups = int(max(1, groups))
        self.num_layers = int(max(1, num_layers))

        self.in_proj = nn.Linear(1, self.d_model, bias=True)
        self.pos = nn.Parameter(torch.zeros(self.bands, self.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        dil = tuple(int(x) for x in (dilation_cycle if len(dilation_cycle) > 0 else (1,)))
        blocks = []
        for i in range(self.num_layers):
            blocks.append(
                _SpectralConvBlock(
                    d_model=self.d_model,
                    groups=self.groups,
                    dropout=float(dropout),
                    kernel_size=int(kernel_size),
                    dilation=int(dil[i % len(dil)]),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_ln = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, bands, 1)
            x = self.in_proj(x)
        elif x.dim() == 3:
            if x.shape[-1] != self.d_model:
                raise RuntimeError(f"[GroupedBiSpectralScan] expected last dim={self.d_model}, got {x.shape[-1]}")
        else:
            raise RuntimeError(f"[GroupedBiSpectralScan] expected 2D/3D input, got {tuple(x.shape)}")

        if x.shape[1] != self.bands:
            raise RuntimeError(f"[GroupedBiSpectralScan] expected bands={self.bands}, got {x.shape[1]}")

        x = x + self.pos.unsqueeze(0)

        for blk in self.blocks:
            x = blk(x)

        x = self.out_ln(x)
        return x
