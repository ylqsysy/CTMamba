#!/usr/bin/env python3
"""Spatial selective scan blocks used by CenterTargetMamba."""

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


def _diag_selective_state_update(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Diagonal selective state recursion in closed form:
      h_t = b_t * x_t + a_t * h_{t-1}, h_{-1}=0
    """
    in_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if a.dtype in (torch.float16, torch.bfloat16):
        a = a.float()
    if b.dtype in (torch.float16, torch.bfloat16):
        b = b.float()

    a = a.clamp(min=1.0e-4, max=0.9999)
    prefix = torch.cumsum(torch.log(a), dim=-1)
    v = (b * x) * torch.exp(-prefix)
    h = torch.exp(prefix) * torch.cumsum(v, dim=-1)
    return h.to(dtype=in_dtype)


class SelectiveScan1D(nn.Module):
    def __init__(
        self,
        dim: int,
        expand: int = 2,
        groups: int = 4,
        dropout: float = 0.0,
        kernel_size: int = 5,
        mamba_ratio: float = 0.125,
        mamba_decay_min: float = 0.90,
        mamba_decay_max: float = 0.999,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(max(1, expand) * self.dim)
        self.mamba_ratio = float(max(0.0, mamba_ratio))
        self.mamba_hidden = int(self.hidden * self.mamba_ratio)
        if self.mamba_ratio > 0.0 and self.mamba_hidden < 1:
            self.mamba_hidden = 1
        self.decay_min = float(mamba_decay_min)
        self.decay_max = float(mamba_decay_max)
        g_in = _best_group_divisor(self.dim, int(groups))
        g_hid = _best_group_divisor(self.hidden, int(groups))
        k = int(max(1, kernel_size))
        pad = k // 2

        self.in_proj = nn.Conv1d(self.dim, self.hidden * 2, kernel_size=1, groups=g_in, bias=True)
        self.dw = nn.Conv1d(self.hidden, self.hidden, kernel_size=k, padding=pad, groups=self.hidden, bias=False)

        if self.mamba_hidden > 0:
            self.ssm_in = nn.Conv1d(self.hidden, self.mamba_hidden, kernel_size=1, bias=False)
            self.ssm_delta = nn.Conv1d(self.hidden, self.mamba_hidden, kernel_size=1, bias=True)
            self.ssm_b = nn.Conv1d(self.hidden, self.mamba_hidden, kernel_size=1, bias=True)
            self.ssm_c = nn.Conv1d(self.hidden, self.mamba_hidden, kernel_size=1, bias=True)
            self.ssm_d = nn.Parameter(torch.ones(1, self.mamba_hidden, 1))
            self.ssm_out = nn.Conv1d(self.mamba_hidden, self.hidden, kernel_size=1, bias=False)
        else:
            self.ssm_in = None
            self.ssm_delta = None
            self.ssm_b = None
            self.ssm_c = None
            self.ssm_d = None
            self.ssm_out = None

        self.mix = nn.Conv1d(self.hidden, self.hidden, kernel_size=1, groups=g_hid, bias=True)
        self.out_proj = nn.Conv1d(self.hidden, self.dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.in_proj(x)
        u, gate = t.chunk(2, dim=1)
        u = F.silu(u)
        u = self.dw(u)

        if self.mamba_hidden > 0:
            s = self.ssm_in(u)
            a = torch.sigmoid(self.ssm_delta(u))
            a = self.decay_min + (self.decay_max - self.decay_min) * a
            b = torch.tanh(self.ssm_b(u))
            c = torch.tanh(self.ssm_c(u))
            h = _diag_selective_state_update(s, a, b)
            mamba_y = c * h + self.ssm_d * s
            u = u + self.ssm_out(mamba_y)

        u = self.mix(u * torch.sigmoid(gate))
        y = self.out_proj(u)
        return self.drop(y)


class SpatialSelectiveScan2D(nn.Module):
    """Baseline 2D selective scan (fixed raster route)."""

    def __init__(
        self,
        dim: int = 128,
        expand: int = 2,
        dropout: float = 0.0,
        mamba_ratio: float = 0.125,
        mamba_decay_min: float = 0.90,
        mamba_decay_max: float = 0.999,
        route_mode: str = "raster",
    ):
        super().__init__()
        self.dim = int(dim)
        self.route_mode = str(route_mode).strip().lower()
        if self.route_mode != "raster":
            raise ValueError(
                f"CenterTargetMamba only supports scan_route='raster', got '{self.route_mode}'"
            )
        self.scan = SelectiveScan1D(
            self.dim,
            expand=int(expand),
            groups=4,
            dropout=float(dropout),
            mamba_ratio=float(mamba_ratio),
            mamba_decay_min=float(mamba_decay_min),
            mamba_decay_max=float(mamba_decay_max),
        )
        self.fuse = nn.Conv2d(self.dim * 4, self.dim, kernel_size=1, bias=True)
        self.out_drop = nn.Dropout2d(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xt = x.transpose(2, 3).contiguous()
        seq_h = x.reshape(b, c, h * w)
        seq_v = xt.reshape(b, c, h * w)

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

        h_fwd = y_h_fwd.reshape(b, c, h, w)
        h_rev = torch.flip(y_h_rev, dims=[-1]).reshape(b, c, h, w)
        v_fwd = y_v_fwd.reshape(b, c, w, h).transpose(2, 3).contiguous()
        v_rev = torch.flip(y_v_rev, dims=[-1]).reshape(b, c, w, h).transpose(2, 3).contiguous()

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
    def __init__(
        self,
        dim: int,
        drop: float = 0.15,
        ssm_expand: int = 2,
        mamba_ratio: float = 0.125,
        mamba_decay_min: float = 0.90,
        mamba_decay_max: float = 0.999,
        scan_route: str = "raster",
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        dim_i = int(dim)
        hidden = int(max(dim_i, round(dim_i * float(mlp_ratio))))
        self.ln1 = nn.LayerNorm(dim_i)
        self.ssm = SpatialSelectiveScan2D(
            dim_i,
            expand=int(ssm_expand),
            dropout=float(drop),
            mamba_ratio=float(mamba_ratio),
            mamba_decay_min=float(mamba_decay_min),
            mamba_decay_max=float(mamba_decay_max),
            route_mode=str(scan_route),
        )
        self.drop1 = nn.Dropout2d(float(drop))

        self.ln2 = nn.LayerNorm(dim_i)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim_i, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(float(drop)),
            nn.Conv2d(hidden, dim_i, kernel_size=1),
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
