#!/usr/bin/env python3
"""Spatial selective scan blocks used by CenterTargetMamba."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_SCAN_ROUTES = {"raster", "serpentine", "zigzag", "spiral"}


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


def _route_coords(h: int, w: int, mode: str) -> list[tuple[int, int]]:
    h_i = int(h)
    w_i = int(w)
    mode_s = str(mode).strip().lower()
    if mode_s not in _VALID_SCAN_ROUTES:
        raise ValueError(f"unsupported scan_route='{mode_s}', expected one of {sorted(_VALID_SCAN_ROUTES)}")

    if mode_s == "raster":
        return [(r, c) for r in range(h_i) for c in range(w_i)]

    if mode_s == "serpentine":
        out: list[tuple[int, int]] = []
        for r in range(h_i):
            cols = range(w_i) if (r % 2 == 0) else range(w_i - 1, -1, -1)
            out.extend((r, c) for c in cols)
        return out

    if mode_s == "zigzag":
        out = []
        for s in range(h_i + w_i - 1):
            band: list[tuple[int, int]] = []
            r0 = max(0, s - (w_i - 1))
            r1 = min(h_i - 1, s)
            for r in range(r0, r1 + 1):
                c = s - r
                band.append((r, c))
            if s % 2 == 0:
                band.reverse()
            out.extend(band)
        return out

    # "spiral": walk outward from the center and keep valid coordinates.
    out = []
    seen: set[tuple[int, int]] = set()
    r = (h_i - 1) // 2
    c = (w_i - 1) // 2

    def _push(rr: int, cc: int) -> None:
        if 0 <= rr < h_i and 0 <= cc < w_i and (rr, cc) not in seen:
            seen.add((rr, cc))
            out.append((rr, cc))

    _push(r, c)
    step = 1
    while len(out) < h_i * w_i:
        for _ in range(step):
            c += 1
            _push(r, c)
        for _ in range(step):
            r += 1
            _push(r, c)
        step += 1
        for _ in range(step):
            c -= 1
            _push(r, c)
        for _ in range(step):
            r -= 1
            _push(r, c)
        step += 1
    return out[: h_i * w_i]


def _route_indices(
    h: int,
    w: int,
    mode: str,
    *,
    vertical: bool,
    device: torch.device,
) -> torch.Tensor:
    if not vertical:
        coords = _route_coords(h, w, mode)
        idx = [r * int(w) + c for r, c in coords]
    else:
        coords_t = _route_coords(w, h, mode)
        idx = [c_t * int(w) + r_t for r_t, c_t in coords_t]
    return torch.as_tensor(idx, dtype=torch.long, device=device)


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
    """2D selective scan with configurable flattening routes."""

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
        if self.route_mode not in _VALID_SCAN_ROUTES:
            raise ValueError(
                f"unsupported scan_route='{self.route_mode}', expected one of {sorted(_VALID_SCAN_ROUTES)}"
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
        self._route_cache: dict[tuple[int, int, str, bool, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_route_and_inverse(
        self,
        h: int,
        w: int,
        *,
        vertical: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_key = str(device)
        key = (int(h), int(w), str(self.route_mode), bool(vertical), device_key)
        if key in self._route_cache:
            return self._route_cache[key]
        order = _route_indices(h, w, self.route_mode, vertical=vertical, device=device)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=device, dtype=order.dtype)
        self._route_cache[key] = (order, inv)
        return order, inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        order_h, inv_h = self._get_route_and_inverse(h, w, vertical=False, device=x.device)
        order_v, inv_v = self._get_route_and_inverse(h, w, vertical=True, device=x.device)
        seq_h = flat.index_select(-1, order_h)
        seq_v = flat.index_select(-1, order_v)

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

        h_fwd = y_h_fwd.index_select(-1, inv_h).reshape(b, c, h, w)
        h_rev = torch.flip(y_h_rev, dims=[-1]).index_select(-1, inv_h).reshape(b, c, h, w)
        v_fwd = y_v_fwd.index_select(-1, inv_v).reshape(b, c, h, w)
        v_rev = torch.flip(y_v_rev, dims=[-1]).index_select(-1, inv_v).reshape(b, c, h, w)

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
