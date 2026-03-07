#!/usr/bin/env python3
"""Spatial selective scan blocks used by TriScanMamba."""

from __future__ import annotations

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


class SelectiveScan1D(nn.Module):
    def __init__(
        self,
        dim: int,
        expand: int = 2,
        groups: int = 4,
        dropout: float = 0.0,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(max(1, expand) * self.dim)
        g_in = _best_group_divisor(self.dim, int(groups))
        g_hid = _best_group_divisor(self.hidden, int(groups))
        k = int(max(1, kernel_size))
        pad = k // 2

        self.in_proj = nn.Conv1d(self.dim, self.hidden * 2, kernel_size=1, groups=g_in, bias=True)
        self.dw = nn.Conv1d(self.hidden, self.hidden, kernel_size=k, padding=pad, groups=self.hidden, bias=False)
        self.mix = nn.Conv1d(self.hidden, self.hidden, kernel_size=1, groups=g_hid, bias=True)
        self.out_proj = nn.Conv1d(self.hidden, self.dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.in_proj(x)
        u, gate = t.chunk(2, dim=1)
        u = F.silu(u)
        # Symmetric forward/backward depthwise mixing over the sequence.
        u_fw = self.dw(u)
        u_bw = torch.flip(self.dw(torch.flip(u, dims=[-1])), dims=[-1])
        u = 0.5 * (u_fw + u_bw)
        u = self.mix(u)
        y = self.out_proj(u * torch.sigmoid(gate))
        return self.drop(y)


class SpatialSelectiveScan2D(nn.Module):
    """Two-dimensional selective scan with four directional routes."""
    def __init__(self, dim: int = 128, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.scan = SelectiveScan1D(self.dim, expand=int(expand), groups=4, dropout=float(dropout))
        self.fuse = nn.Conv2d(self.dim * 4, self.dim, kernel_size=1, bias=True)
        self.out_drop = nn.Dropout2d(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq_h = x.reshape(B, C, H * W)
        xt = x.transpose(2, 3).contiguous()
        seq_v = xt.reshape(B, C, H * W)

        seq_all = torch.cat(
            [
                seq_h,
                torch.flip(seq_h, dims=[-1]),
                seq_v,
                torch.flip(seq_v, dims=[-1]),
            ],
            dim=0,
        )
        y_all = self.scan(seq_all)
        y_h_fwd, y_h_rev, y_v_fwd, y_v_rev = y_all.chunk(4, dim=0)

        h_fwd = y_h_fwd.reshape(B, C, H, W)
        h_rev = torch.flip(y_h_rev, dims=[-1]).reshape(B, C, H, W)
        v_fwd = y_v_fwd.reshape(B, C, W, H).transpose(2, 3).contiguous()
        v_rev = torch.flip(y_v_rev, dims=[-1]).reshape(B, C, W, H).transpose(2, 3).contiguous()

        y = torch.cat([h_fwd, h_rev, v_fwd, v_rev], dim=1)
        y = self.fuse(y)
        y = self.out_drop(y)
        return y


def _ln2d(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1).contiguous()
    x = ln(x)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


class VSSBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.15, ssm_expand: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(int(dim))
        self.ssm = SpatialSelectiveScan2D(int(dim), expand=int(ssm_expand), dropout=float(drop))
        self.drop1 = nn.Dropout2d(float(drop))

        self.ln2 = nn.LayerNorm(int(dim))
        self.mlp = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(float(drop)),
            nn.Conv2d(int(dim) * 4, int(dim), kernel_size=1),
            nn.Dropout2d(float(drop)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _ln2d(x, self.ln1)
        x = x + self.drop1(self.ssm(h))
        h2 = _ln2d(x, self.ln2)
        x = x + self.mlp(h2)
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(int(in_dim), int(out_dim), kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


PatchMerging = Downsample
SS2DLike = SpatialSelectiveScan2D
