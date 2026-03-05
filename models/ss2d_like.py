#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDirScanMix(nn.Module):
    """
    Lightweight multi-directional spatial mixing on 2D feature maps.
    Input/Output: (B, C, H, W)
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = int(dim)
        self.dw = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.pw = nn.Conv2d(self.dim, self.dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.dw(x)
        y = F.gelu(y)
        y = self.pw(y)
        return y


class SS2DLike(nn.Module):
    """
    SS2D-like mixer wrapper (kept for compatibility).
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.mix = MultiDirScanMix(int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(x)


def _ln2d(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    # (B,C,H,W) -> (B,H,W,C) LN -> (B,C,H,W)
    x = x.permute(0, 2, 3, 1).contiguous()
    x = ln(x)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


class VSSBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(int(dim))
        self.ssm = SS2DLike(int(dim))
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


# Backward-compatible name used by some older patches.
PatchMerging = Downsample
