#!/usr/bin/env python3

from __future__ import annotations
"""Spectral selective scan blocks used by TriScanMamba."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _best_group_divisor(channels: int, groups: int) -> int:
    g = int(max(1, groups))
    c = int(channels)
    if c % g == 0:
        return g
    for gg in range(min(g, c), 0, -1):
        if c % gg == 0:
            return gg
    return 1


class _GroupedSpectralSSMBlock(nn.Module):
    def __init__(self, d_model: int, groups: int, dropout: float, kernel_size: int, expand: int):
        super().__init__()
        self.d_model = int(d_model)
        g_in = _best_group_divisor(self.d_model, int(groups))
        hidden = int(max(1, expand) * self.d_model)
        g_hid = _best_group_divisor(hidden, int(groups))

        k = int(max(1, kernel_size))
        pad = k // 2

        self.ln = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Conv1d(self.d_model, hidden * 2, kernel_size=1, groups=g_in, bias=True)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=k, padding=pad, groups=hidden, bias=False)
        self.mix = nn.Conv1d(hidden, hidden, kernel_size=1, groups=g_hid, bias=True)
        self.out_proj = nn.Conv1d(hidden, self.d_model, kernel_size=1, bias=True)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x).transpose(1, 2)
        t = self.in_proj(y)
        u, gate = t.chunk(2, dim=1)
        u = F.silu(u)
        u_fw = self.dw(u)
        u_bw = torch.flip(self.dw(torch.flip(u, dims=[-1])), dims=[-1])
        u = 0.5 * (u_fw + u_bw)
        u = self.mix(u)
        y = self.out_proj(u * torch.sigmoid(gate)).transpose(1, 2).contiguous()
        y = self.drop(y)
        return x + y


class GroupedBiSpectralScan(nn.Module):
    """Grouped bidirectional spectral scan."""

    def __init__(
        self,
        bands: int = 274,
        d_model: int = 96,
        num_layers: int = 4,
        groups: int = 8,
        dropout: float = 0.0,
        kernel_size: int = 7,
        expand: int = 2,
        dilation_cycle: tuple[int, ...] = (1,),
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

        self.blocks = nn.ModuleList(
            [
                _GroupedSpectralSSMBlock(
                    d_model=self.d_model,
                    groups=self.groups,
                    dropout=float(dropout),
                    kernel_size=int(kernel_size),
                    expand=int(expand),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
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
