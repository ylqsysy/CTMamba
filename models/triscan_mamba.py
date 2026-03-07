#!/usr/bin/env python3
"""TriScanMamba architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.spectral_scan import GroupedBiSpectralScan
from models.spatial_scan import Downsample, VSSBlock


def _best_conv1x1_groups(c_in: int, c_out: int, groups: int) -> int:
    g = int(max(1, groups))
    m = min(g, int(c_in), int(c_out))
    for gg in range(m, 0, -1):
        if (int(c_in) % gg == 0) and (int(c_out) % gg == 0):
            return gg
    return 1


@dataclass
class TriScanMambaConfig:
    num_classes: int
    raw_bands: int

    patch_size: int = 15
    dropout: float = 0.15

    stages: Tuple[int, int, int] = (2, 2, 4)
    stage_dims: Tuple[int, int, int] = (64, 96, 128)
    ssm_expand: int = 2

    spec_groups: int = 8
    spec_layers: int = 3
    spec_hidden: int = 96
    spec_kernel: int = 7
    spec_expand: int = 2

    use_spec_branch: bool = True

    spec_pool: str = "mean"
    use_spec_cross_attn: bool = True
    spec_attn_dim: int = 0
    spec_attn_dropout: float = 0.0
    spec_fuse_init: float = 0.0

    token_bands: int = 96
    tokenizer_groups: int = 1
    tokenizer_dropout: float = 0.0

    pool: str = "center_attn"
    pool_sigma: float = 1.2
    pool_temp: float = 0.7

    head: str = "ce"


class TriScanMamba(nn.Module):
    """TriScanMamba model with spatial scan and spectral scan fusion."""

    def __init__(self, cfg: TriScanMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)
        self.raw_bands = int(cfg.raw_bands)
        self.model_bands = int(self.raw_bands)

        s0, s1, s2 = (int(cfg.stage_dims[0]), int(cfg.stage_dims[1]), int(cfg.stage_dims[2]))
        ssm_expand = int(getattr(cfg, "ssm_expand", 2))
        drop = float(cfg.dropout)

        token_bands = int(getattr(cfg, "token_bands", 0))
        if 0 < token_bands < self.raw_bands:
            tok_groups = _best_conv1x1_groups(
                self.raw_bands,
                token_bands,
                int(getattr(cfg, "tokenizer_groups", 1)),
            )
            self.band_tokenizer = nn.Sequential(
                nn.Conv2d(self.raw_bands, token_bands, kernel_size=1, groups=tok_groups, bias=False),
                nn.GroupNorm(1, token_bands),
                nn.SiLU(),
                nn.Dropout2d(float(getattr(cfg, "tokenizer_dropout", 0.0))),
            )
            self.spec_tokenizer = nn.Sequential(
                nn.LayerNorm(self.raw_bands),
                nn.Linear(self.raw_bands, token_bands),
            )
            self.model_bands = int(token_bands)
        else:
            self.band_tokenizer = None
            self.spec_tokenizer = None

        self.in_proj = nn.Conv2d(self.model_bands, s0, kernel_size=1)

        self.stage0 = nn.Sequential(*[VSSBlock(s0, drop=drop, ssm_expand=ssm_expand) for _ in range(int(cfg.stages[0]))])
        self.down01 = Downsample(s0, s1)
        self.stage1 = nn.Sequential(*[VSSBlock(s1, drop=drop, ssm_expand=ssm_expand) for _ in range(int(cfg.stages[1]))])
        self.down12 = Downsample(s1, s2)
        self.stage2 = nn.Sequential(*[VSSBlock(s2, drop=drop, ssm_expand=ssm_expand) for _ in range(int(cfg.stages[2]))])

        use_spec_branch = bool(getattr(cfg, "use_spec_branch", True))
        if use_spec_branch:
            spec_hidden = int(cfg.spec_hidden)
            self.spec_scan = GroupedBiSpectralScan(
                bands=self.model_bands,
                d_model=spec_hidden,
                num_layers=int(cfg.spec_layers),
                groups=int(cfg.spec_groups),
                dropout=drop,
                kernel_size=int(getattr(cfg, "spec_kernel", 7)),
                expand=int(getattr(cfg, "spec_expand", 2)),
            )
            self.spec_to_spa = nn.Linear(spec_hidden, s2, bias=False)

            attn_dim_cfg = int(getattr(cfg, "spec_attn_dim", 0))
            self.spec_attn_dim = int(attn_dim_cfg if attn_dim_cfg > 0 else s2)
            self.spec_attn_q = nn.Linear(s2, self.spec_attn_dim, bias=False)
            self.spec_attn_k = nn.Linear(s2, self.spec_attn_dim, bias=False)
            self.spec_attn_v = nn.Linear(s2, self.spec_attn_dim, bias=False)
            self.spec_attn_out = nn.Linear(self.spec_attn_dim, s2, bias=False)
            self.spec_attn_drop = nn.Dropout(float(getattr(cfg, "spec_attn_dropout", 0.0)))
            self.spec_attn_scale = float(self.spec_attn_dim) ** -0.5
            self.use_spec_cross_attn = bool(getattr(cfg, "use_spec_cross_attn", True))

            fuse_init = float(getattr(cfg, "spec_fuse_init", 0.0))
            self.spec_attn_gate = nn.Parameter(torch.tensor(fuse_init, dtype=torch.float32))
            self.spec_vec_proj = nn.Sequential(
                nn.LayerNorm(s2),
                nn.Linear(s2, s2),
            )
            self.spec_vec_gate = nn.Parameter(torch.tensor(fuse_init, dtype=torch.float32))
        else:
            self.spec_scan = None
            self.spec_to_spa = None
            self.spec_attn_q = None
            self.spec_attn_k = None
            self.spec_attn_v = None
            self.spec_attn_out = None
            self.spec_attn_drop = None
            self.spec_attn_scale = None
            self.use_spec_cross_attn = False
            self.spec_attn_gate = None
            self.spec_vec_proj = None
            self.spec_vec_gate = None

        self.ln_head = nn.LayerNorm(s2)

        head = str(getattr(cfg, "head", "ce")).strip().lower()
        if head not in ("ce", "softmax", "logits"):
            raise ValueError(f"Unsupported head='{head}'. This repository currently supports CE/logits head only.")
        self.head = nn.Linear(s2, self.num_classes)
        self.head_type = "ce"

    def _pool_tokens(self, tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
        mode = str(getattr(self.cfg, "pool", "mean")).strip().lower()
        _, _, C = tok.shape

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

    def _center_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2] // 2
        w = x.shape[-1] // 2
        return x[:, :, h, w]

    def _build_spec_input(self, x: torch.Tensor, x_spec: Optional[torch.Tensor]) -> torch.Tensor:
        mode = str(getattr(self.cfg, "spec_pool", "mean")).strip().lower()
        center = self._center_spectrum(x)
        mean = x.mean(dim=(2, 3))

        if mode == "center":
            return center
        if mode == "mean":
            return mean
        if mode in ("hybrid", "center_mean"):
            return 0.5 * (center + mean)
        if mode == "max":
            return x.amax(dim=(2, 3))
        if mode == "input":
            if x_spec is None:
                return mean
            return x_spec

        if x_spec is not None:
            return x_spec
        return mean

    def forward_features(self, x: torch.Tensor, x_spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"[TriScanMamba] expected x=(B,bands,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.raw_bands:
            raise RuntimeError(f"[TriScanMamba] expected bands={self.raw_bands}, got {x.shape[1]}")

        x_model = self.band_tokenizer(x) if self.band_tokenizer is not None else x

        x_spec_model = x_spec
        if x_spec_model is not None:
            if x_spec_model.dim() != 2:
                raise RuntimeError(f"[TriScanMamba] expected x_spec=(B,bands), got {tuple(x_spec_model.shape)}")
            if self.spec_tokenizer is not None:
                if x_spec_model.shape[1] == self.raw_bands:
                    x_spec_model = self.spec_tokenizer(x_spec_model)
                elif x_spec_model.shape[1] != self.model_bands:
                    raise RuntimeError(
                        f"[TriScanMamba] x_spec bands must be {self.raw_bands} or {self.model_bands}, got {x_spec_model.shape[1]}"
                    )
            elif x_spec_model.shape[1] != self.model_bands:
                raise RuntimeError(
                    f"[TriScanMamba] x_spec bands must be {self.model_bands}, got {x_spec_model.shape[1]}"
                )

        spec_input = self._build_spec_input(x_model, x_spec_model)
        if spec_input.dim() != 2 or spec_input.shape[1] != self.model_bands:
            raise RuntimeError(
                f"[TriScanMamba] expected spec input=(B,bands={self.model_bands}), got {tuple(spec_input.shape)}"
            )

        spa = self.in_proj(x_model)
        spa = self.stage0(spa)
        spa = self.down01(spa)
        spa = self.stage1(spa)
        spa = self.down12(spa)
        spa = self.stage2(spa)

        spec_tok = None
        if self.spec_scan is not None:
            spec_tok = self.spec_scan(spec_input)
            spec_tok = self.spec_to_spa(spec_tok)

        B, C, H, W = spa.shape
        tok = spa.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        if spec_tok is not None and self.use_spec_cross_attn:
            q = self.spec_attn_q(tok)
            k = self.spec_attn_k(spec_tok)
            v = self.spec_attn_v(spec_tok)
            attn = torch.matmul(q, k.transpose(1, 2)) * float(self.spec_attn_scale)
            attn = torch.softmax(attn, dim=-1)
            attn = self.spec_attn_drop(attn)
            ctx = torch.matmul(attn, v)
            tok = tok + torch.tanh(self.spec_attn_gate) * self.spec_attn_out(ctx)

        tok = self.ln_head(tok)
        feat = self._pool_tokens(tok, H, W)

        if spec_tok is not None and self.spec_vec_proj is not None:
            spec_vec = spec_tok.mean(dim=1)
            feat = feat + torch.tanh(self.spec_vec_gate) * self.spec_vec_proj(spec_vec)
        return feat

    def forward(self, x: torch.Tensor, x_spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.forward_features(x, x_spec)
        out = self.head(feat)
        return out


VSSM3DConfig = TriScanMambaConfig
VSSM3DModel = TriScanMamba
