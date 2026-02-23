from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE3(nn.Module):
    """Lightweight token-wise MoE FFN.

    Accepts multiple constructor aliases to stay compatible with older/newer model code:
      - dim / in_dim / d_model
      - experts / num_experts
      - topk / top_k

    Works for inputs shaped (..., C) where C=dim (e.g., (B,C), (B,N,C)).
    """

    def __init__(
        self,
        dim: int | None = None,
        *,
        in_dim: int | None = None,
        d_model: int | None = None,
        experts: int = 3,
        num_experts: int | None = None,
        topk: int = 1,
        top_k: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        d = dim if dim is not None else (in_dim if in_dim is not None else d_model)
        if d is None:
            raise TypeError("MoE3 requires 'dim' (or alias in_dim/d_model)")

        e = num_experts if num_experts is not None else experts
        k = top_k if top_k is not None else topk

        self.dim = int(d)
        self.experts = max(1, int(e))
        self.topk = max(1, int(k))
        self.topk = min(self.topk, self.experts)

        self.router = nn.Linear(self.dim, self.experts)
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, self.dim * 4),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.dim * 4, self.dim),
                    nn.Dropout(float(dropout)),
                )
                for _ in range(self.experts)
            ]
        )
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        logits = self.router(x)  # (..., E)

        # Degenerate case: single expert
        if self.experts <= 1:
            out = self.ffn[0](x)
            return self.norm(x + out)

        if self.topk <= 1:
            idx = torch.argmax(logits, dim=-1)  # (...,)
            out = x.new_zeros(x.shape)
            for e in range(self.experts):
                m = idx == e
                if bool(m.any()):
                    out[m] = self.ffn[e](x[m])
            return self.norm(x + out)

        k = min(self.topk, self.experts)
        topv, topi = torch.topk(logits, k=k, dim=-1)

        # softmax in fp32 for stability (AMP-safe), then cast back
        w = F.softmax(topv.float(), dim=-1).to(dtype=x.dtype)  # (..., k)

        out = x.new_zeros(x.shape)
        for j in range(k):
            ei = topi[..., j]  # (...,)
            wj = w[..., j]     # (...,)
            for e in range(self.experts):
                m = ei == e
                if bool(m.any()):
                    out[m] = out[m] + self.ffn[e](x[m]) * wj[m].unsqueeze(-1)

        return self.norm(x + out)
