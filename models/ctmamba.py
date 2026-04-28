#!/usr/bin/env python3
"""CenterTargetMamba model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from models.spatial_scan import Downsample, VSSBlock


class _Conv1x1Stem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _normalized_radius(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy = torch.arange(int(h), device=device, dtype=dtype)
    xx = torch.arange(int(w), device=device, dtype=dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    cy, cx = (float(h) - 1.0) * 0.5, (float(w) - 1.0) * 0.5
    d = torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
    d = d / (d.max() + 1.0e-6)
    return d.reshape(-1)


class _BoundaryContrastAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 1.0,
        dropout: float = 0.0,
        core_radius: float = 0.35,
        boundary_radius: float = 0.70,
        mode: str = "default",
    ):
        super().__init__()
        d = int(dim)
        hidden = int(max(32, round(d * float(hidden_ratio))))
        self.core_radius = float(max(0.0, min(1.0, core_radius)))
        self.boundary_radius = float(max(0.0, min(1.0, boundary_radius)))
        self.mode = str(mode).strip().lower()
        if self.core_radius >= self.boundary_radius:
            self.boundary_radius = min(1.0, self.core_radius + 0.15)

        self.mlp = nn.Sequential(
            nn.Linear(d * 5, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, d),
        )

    def _masked_mean(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask[None, :, None].to(dtype=tok.dtype)
        denom = w.sum(dim=1).clamp_min(1.0e-6)
        return (tok * w).sum(dim=1) / denom

    def forward(
        self,
        tok: torch.Tensor,
        h: int,
        w: int,
        pooled: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        dist = _normalized_radius(h, w, device=tok.device, dtype=torch.float32)
        core_mask = (dist <= self.core_radius).to(dtype=torch.float32)

        if float(core_mask.sum().item()) < 1.0:
            core = center
        else:
            core = self._masked_mean(tok, core_mask)

        if self.mode == "global_context":
            boundary = tok.mean(dim=1)
            inter = core - boundary
        else:
            boundary_mask = (dist >= self.boundary_radius).to(dtype=torch.float32)
            if float(boundary_mask.sum().item()) < 1.0:
                boundary = tok.mean(dim=1)
            else:
                boundary = self._masked_mean(tok, boundary_mask)
            inter = core * boundary

        out_in = torch.cat([pooled, center, core, boundary, inter], dim=1)
        return self.mlp(out_in)


class _RawSpectralAdapter(nn.Module):
    def __init__(
        self,
        in_bands: int,
        out_dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.0,
        mode: str = "default",
    ):
        super().__init__()
        in_dim = int(max(1, in_bands)) * 3
        out_dim = int(max(1, out_dim))
        hidden = int(max(out_dim, round(out_dim * float(hidden_ratio))))
        self.mode = str(mode).strip().lower()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        cy, cx = h // 2, w // 2
        center = x[:, :, cy, cx]
        mean = x.mean(dim=(2, 3))
        if self.mode == "center_delta":
            third = center - mean
        else:
            third = x.flatten(2).std(dim=2, unbiased=False)
        inp = torch.cat([center, mean, third], dim=1)
        return self.mlp(inp)


@dataclass
class CTMambaConfig:
    num_classes: int
    raw_bands: int

    patch_size: int = 15
    dropout: float = 0.10
    scan_route: str = "raster"

    stages: Tuple[int, int, int] = (2, 2, 5)
    stage_dims: Tuple[int, int, int] = (96, 128, 160)
    ssm_expand: int = 1
    mamba_ratio: float = 0.125
    mamba_decay_min: float = 0.90
    mamba_decay_max: float = 0.999
    block_mlp_ratio: float = 4.0

    pool: str = "mean"
    pool_sigma: float = 1.2
    back_end_hidden_ratio: float = 2.0
    back_end_dropout: float = 0.10

    cca_hidden_ratio: float = 1.0
    cca_dropout: float = 0.0
    cca_weight: float = 0.50
    cca_mode: str = "default"

    cpa_hidden_ratio: float = 1.0
    cpa_dropout: float = 0.0
    cpa_weight: float = 0.50
    cpa_inner_radius: int = 1
    cpa_outer_radius: int = 2
    cpa_mode: str = "default"

    bca_hidden_ratio: float = 1.0
    bca_dropout: float = 0.0
    bca_weight: float = 0.50
    bca_core_radius: float = 0.35
    bca_boundary_radius: float = 0.70
    bca_mode: str = "default"

    rsa_hidden_ratio: float = 2.0
    rsa_dropout: float = 0.0
    rsa_weight: float = 0.50
    rsa_mode: str = "default"


class CenterTargetMamba(nn.Module):
    def __init__(self, cfg: CTMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)
        self.raw_bands = int(cfg.raw_bands)

        s0, s1, s2 = (int(cfg.stage_dims[0]), int(cfg.stage_dims[1]), int(cfg.stage_dims[2]))

        if float(cfg.mamba_ratio) < 0.0:
            raise ValueError(f"mamba_ratio must be >= 0, got {cfg.mamba_ratio}")
        if float(cfg.mamba_decay_min) <= 0.0 or float(cfg.mamba_decay_max) <= 0.0:
            raise ValueError("mamba_decay_min/max must be > 0")
        if float(cfg.mamba_decay_min) >= float(cfg.mamba_decay_max):
            raise ValueError("mamba_decay_min must be < mamba_decay_max")
        if float(cfg.block_mlp_ratio) <= 0.0:
            raise ValueError("block_mlp_ratio must be > 0")

        self.in_proj = _Conv1x1Stem(self.raw_bands, s0)

        self.stage0 = nn.Sequential(
            *[
                VSSBlock(
                    s0,
                    drop=float(cfg.dropout),
                    ssm_expand=int(cfg.ssm_expand),
                    mamba_ratio=float(cfg.mamba_ratio),
                    mamba_decay_min=float(cfg.mamba_decay_min),
                    mamba_decay_max=float(cfg.mamba_decay_max),
                    scan_route=str(cfg.scan_route),
                    mlp_ratio=float(cfg.block_mlp_ratio),
                )
                for _ in range(int(cfg.stages[0]))
            ]
        )
        self.down01 = Downsample(s0, s1)
        self.stage1 = nn.Sequential(
            *[
                VSSBlock(
                    s1,
                    drop=float(cfg.dropout),
                    ssm_expand=int(cfg.ssm_expand),
                    mamba_ratio=float(cfg.mamba_ratio),
                    mamba_decay_min=float(cfg.mamba_decay_min),
                    mamba_decay_max=float(cfg.mamba_decay_max),
                    scan_route=str(cfg.scan_route),
                    mlp_ratio=float(cfg.block_mlp_ratio),
                )
                for _ in range(int(cfg.stages[1]))
            ]
        )
        self.down12 = Downsample(s1, s2)
        self.stage2 = nn.Sequential(
            *[
                VSSBlock(
                    s2,
                    drop=float(cfg.dropout),
                    ssm_expand=int(cfg.ssm_expand),
                    mamba_ratio=float(cfg.mamba_ratio),
                    mamba_decay_min=float(cfg.mamba_decay_min),
                    mamba_decay_max=float(cfg.mamba_decay_max),
                    scan_route=str(cfg.scan_route),
                    mlp_ratio=float(cfg.block_mlp_ratio),
                )
                for _ in range(int(cfg.stages[2]))
            ]
        )

        self.ln_head = nn.LayerNorm(s2)
        be_hidden = int(max(s2, round(s2 * float(cfg.back_end_hidden_ratio))))
        self.back_mlp = nn.Sequential(
            nn.Linear(s2, be_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.back_end_dropout)),
        )

        self.cca_mode = str(cfg.cca_mode).strip().lower()
        self.cpa_mode = str(cfg.cpa_mode).strip().lower()
        self.bca_mode = str(cfg.bca_mode).strip().lower()
        self.rsa_mode = str(cfg.rsa_mode).strip().lower()
        if self.cca_mode not in {"default", "product_gate"}:
            raise ValueError(f"unsupported cca_mode: {cfg.cca_mode}")
        if self.cpa_mode not in {"default", "single_scale"}:
            raise ValueError(f"unsupported cpa_mode: {cfg.cpa_mode}")
        if self.bca_mode not in {"default", "global_context"}:
            raise ValueError(f"unsupported bca_mode: {cfg.bca_mode}")
        if self.rsa_mode not in {"default", "center_delta"}:
            raise ValueError(f"unsupported rsa_mode: {cfg.rsa_mode}")

        cca_hidden = int(max(32, round(s2 * float(cfg.cca_hidden_ratio))))
        self.cca_weight = float(cfg.cca_weight)
        self.center_context_mlp = nn.Sequential(
            nn.Linear(s2 * 3, cca_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.cca_dropout)),
            nn.Linear(cca_hidden, s2),
        )

        cpa_hidden = int(max(32, round(s2 * float(cfg.cpa_hidden_ratio))))
        self.cpa_weight = float(cfg.cpa_weight)
        self.cpa_inner_radius = int(max(1, cfg.cpa_inner_radius))
        self.cpa_outer_radius = int(max(self.cpa_inner_radius + 1, cfg.cpa_outer_radius))
        self.center_pyramid_mlp = nn.Sequential(
            nn.Linear(s2 * 5, cpa_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.cpa_dropout)),
            nn.Linear(cpa_hidden, s2),
        )

        self.bca_weight = float(cfg.bca_weight)
        self.boundary_contrast_adapter = _BoundaryContrastAdapter(
            dim=s2,
            hidden_ratio=float(cfg.bca_hidden_ratio),
            dropout=float(cfg.bca_dropout),
            core_radius=float(cfg.bca_core_radius),
            boundary_radius=float(cfg.bca_boundary_radius),
            mode=self.bca_mode,
        )

        self.rsa_weight = float(cfg.rsa_weight)
        self.raw_spectral_adapter = _RawSpectralAdapter(
            in_bands=self.raw_bands,
            out_dim=s2,
            hidden_ratio=float(cfg.rsa_hidden_ratio),
            dropout=float(cfg.rsa_dropout),
            mode=self.rsa_mode,
        )

        self.head = nn.Linear(be_hidden, self.num_classes)

    def _pool_tokens_mode(self, tok: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
        mode = str(mode).strip().lower()
        if mode == "mean":
            return tok.mean(dim=1)
        if mode == "center":
            idx = (h // 2) * w + (w // 2)
            return tok[:, idx, :]
        if mode == "gaussian":
            sigma = max(1e-6, float(getattr(self.cfg, "pool_sigma", 1.2)))
            yy = torch.arange(h, device=tok.device, dtype=torch.float32)
            xx = torch.arange(w, device=tok.device, dtype=torch.float32)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            d2 = (gy - cy) ** 2 + (gx - cx) ** 2
            ww = torch.exp(-0.5 * d2 / (sigma * sigma)).reshape(-1)
            ww = ww / (ww.sum() + 1e-12)
            return (tok * ww[None, :, None]).sum(dim=1)
        return tok.mean(dim=1)

    def _pool_tokens(self, tok: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return self._pool_tokens_mode(tok, h, w, str(getattr(self.cfg, "pool", "mean")))

    def _forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.dim() != 4:
            raise RuntimeError(f"[CenterTargetMamba] expected x=(B,bands,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.raw_bands:
            raise RuntimeError(f"[CenterTargetMamba] expected bands={self.raw_bands}, got {x.shape[1]}")

        spa = self.in_proj(x)
        spa = self.stage0(spa)
        spa = self.down01(spa)
        spa = self.stage1(spa)
        spa = self.down12(spa)
        spa = self.stage2(spa)

        b, c, h, w = spa.shape
        tok = spa.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        tok = self.ln_head(tok)
        return tok, h, w

    def _project_feature(self, tok: torch.Tensor, h: int, w: int, x_raw: torch.Tensor) -> torch.Tensor:
        pooled = self._pool_tokens(tok, h, w)
        cy, cx = h // 2, w // 2
        center = tok[:, cy * w + cx, :]
        context = tok.mean(dim=1)

        if self.cca_mode == "product_gate":
            cca_aux = pooled * center
        else:
            cca_aux = center - context
        cca_in = torch.cat([pooled, center, cca_aux], dim=1)
        pooled = pooled + self.cca_weight * self.center_context_mlp(cca_in)

        yy = torch.arange(h, device=tok.device)
        xx = torch.arange(w, device=tok.device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        cheb = torch.maximum((gy - cy).abs(), (gx - cx).abs())
        flat_idx = (gy * w + gx).reshape(-1)

        if self.cpa_mode == "single_scale":
            local_mask = ((cheb >= 1) & (cheb <= self.cpa_outer_radius)).reshape(-1)
            local_idx = flat_idx[local_mask]
            local = tok.index_select(1, local_idx).mean(dim=1) if local_idx.numel() > 0 else center
            cpa_in = torch.cat([pooled, center, local, center - local, pooled - local], dim=1)
        else:
            inner_mask = ((cheb >= 1) & (cheb <= self.cpa_inner_radius)).reshape(-1)
            outer_mask = ((cheb > self.cpa_inner_radius) & (cheb <= self.cpa_outer_radius)).reshape(-1)
            inner_idx = flat_idx[inner_mask]
            outer_idx = flat_idx[outer_mask]
            inner = tok.index_select(1, inner_idx).mean(dim=1) if inner_idx.numel() > 0 else center
            outer = tok.index_select(1, outer_idx).mean(dim=1) if outer_idx.numel() > 0 else inner
            cpa_in = torch.cat([pooled, center, inner, outer, center - outer], dim=1)
        pooled = pooled + self.cpa_weight * self.center_pyramid_mlp(cpa_in)

        pooled = pooled + self.bca_weight * self.boundary_contrast_adapter(tok, h, w, pooled, center)
        pooled = pooled + self.rsa_weight * self.raw_spectral_adapter(x_raw)

        return self.back_mlp(pooled)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tok, h, w = self._forward_tokens(x)
        return self._project_feature(tok, h, w, x_raw=x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok, h, w = self._forward_tokens(x)
        feat = self._project_feature(tok, h, w, x_raw=x)
        return self.head(feat)


CTMamba = CenterTargetMamba
CTMambaModel = CenterTargetMamba
