#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ss2d_like import VSSBlock, Downsample
from models.spectral_scan import GroupedBiSpectralScan
from models.cross_coupling import FiLMCrossCoupling
from models.moe import MoE3


@dataclass
class VSSM3DConfig:
    num_classes: int
    raw_bands: int

    patch_size: int = 15
    dropout: float = 0.15

    # spatial backbone
    stages: Tuple[int, int, int] = (2, 2, 4)
    stage_dims: Tuple[int, int, int] = (64, 96, 128)

    # spectral scan
    spec_groups: int = 8
    spec_layers: int = 3
    spec_hidden: int = 96

    use_spec_film: bool = True

    # pooling over spatial tokens (after last stage)
    # mean | gaussian | center | center_attn
    pool: str = "center_attn"
    pool_sigma: float = 1.2
    pool_temp: float = 0.7

    # MoE
    moe_experts: int = 0
    moe_topk: int = 1

    # classification head
    head: str = "ce"


class VSSM3DMoE(nn.Module):
    def __init__(self, cfg: VSSM3DConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)
        self.raw_bands = int(cfg.raw_bands)

        s0, s1, s2 = (int(cfg.stage_dims[0]), int(cfg.stage_dims[1]), int(cfg.stage_dims[2]))

        self.in_proj = nn.Conv2d(self.raw_bands, s0, kernel_size=1)

        self.stage0 = nn.Sequential(*[VSSBlock(s0, float(cfg.dropout)) for _ in range(int(cfg.stages[0]))])
        self.down01 = Downsample(s0, s1)
        self.stage1 = nn.Sequential(*[VSSBlock(s1, float(cfg.dropout)) for _ in range(int(cfg.stages[1]))])
        self.down12 = Downsample(s1, s2)
        self.stage2 = nn.Sequential(*[VSSBlock(s2, float(cfg.dropout)) for _ in range(int(cfg.stages[2]))])

        self.spec_scan = GroupedBiSpectralScan(
            bands=self.raw_bands,
            d_model=int(cfg.spec_hidden),
            num_layers=int(cfg.spec_layers),
            groups=int(cfg.spec_groups),
            dropout=float(cfg.dropout),
        )

        self.couple = FiLMCrossCoupling(d_spa=s2, d_spec=int(cfg.spec_hidden)) if bool(getattr(cfg, "use_spec_film", True)) else None

        moe_experts = int(getattr(cfg, "moe_experts", 0))
        self.moe = MoE3(in_dim=s2, num_experts=moe_experts, top_k=int(getattr(cfg, "moe_topk", 1))) if moe_experts > 0 else None

        self.ln_head = nn.LayerNorm(s2)

        head = str(getattr(cfg, "head", "ce")).strip().lower()
        if head not in ("ce", "softmax", "logits"):
            raise ValueError(f"Unsupported head='{head}'. This repository currently supports CE/logits head only.")
        self.head = nn.Linear(s2, self.num_classes)
        self.head_type = "ce"


    def _pool_tokens(self, tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
        mode = str(getattr(self.cfg, "pool", "mean")).strip().lower()
        B, N, C = tok.shape

        if mode == "mean":
            return tok.mean(dim=1)

        if mode == "center":
            cy, cx = H // 2, W // 2
            idx = cy * W + cx
            return tok[:, idx, :]

        if mode == "gaussian":
            sigma = float(getattr(self.cfg, "pool_sigma", 1.2))
            sigma = max(1e-6, sigma)
            device = tok.device
            yy = torch.arange(H, device=device, dtype=torch.float32)
            xx = torch.arange(W, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
            d2 = (gy - cy) ** 2 + (gx - cx) ** 2
            w = torch.exp(-0.5 * d2 / (sigma * sigma)).reshape(-1)
            w = w / (w.sum() + 1e-12)
            return (tok * w[None, :, None]).sum(dim=1)

        if mode == "center_attn":
            temp = float(getattr(self.cfg, "pool_temp", 0.7))
            temp = max(1e-6, temp)
            cy, cx = H // 2, W // 2
            idx = cy * W + cx
            center = tok[:, idx, :]
            scale = 1.0 / (float(C) ** 0.5)
            scores = (tok * center[:, None, :]).sum(dim=-1) * scale / temp
            w = torch.softmax(scores, dim=1)
            return (tok * w[:, :, None]).sum(dim=1)

        return tok.mean(dim=1)

    def forward_features(self, x: torch.Tensor, x_spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_spec is None:
            # derive center spectrum from the patch
            if x.ndim == 4 and x.shape[-1] == x.shape[-2]:
                h = x.shape[-2] // 2
                w = x.shape[-1] // 2
                x_spec = x[:, :, h, w]
            elif x.ndim == 4:
                h = x.shape[1] // 2
                w = x.shape[2] // 2
                x_spec = x[:, h, w, :]
            else:
                x_spec = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        if x.dim() != 4:
            raise RuntimeError(f"[VSSM3DMoE] expected x=(B,bands,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.raw_bands:
            raise RuntimeError(f"[VSSM3DMoE] expected bands={self.raw_bands}, got {x.shape[1]}")

        if x_spec.dim() != 2 or x_spec.shape[1] != self.raw_bands:
            raise RuntimeError(f"[VSSM3DMoE] expected x_spec=(B,bands={self.raw_bands}), got {tuple(x_spec.shape)}")

        spa = self.in_proj(x)
        spa = self.stage0(spa)
        spa = self.down01(spa)
        spa = self.stage1(spa)
        spa = self.down12(spa)
        spa = self.stage2(spa)

        spec_tokens = self.spec_scan(x_spec)

        if self.couple is not None:
            spa, _ = self.couple(spa, spec_tokens)

        B, C, H, W = spa.shape
        tok = spa.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        if self.moe is not None:
            tok = self.moe(tok)

        tok = self.ln_head(tok)
        feat = self._pool_tokens(tok, H, W)
        return feat

    def forward(self, x: torch.Tensor, x_spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.forward_features(x, x_spec)
        out = self.head(feat)
        return out
