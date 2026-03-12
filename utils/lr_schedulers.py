#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class WarmupCosine:
    """Epoch-based warmup and cosine decay scheduler.

    The scheduler can wrap an optimizer or operate in scalar mode.
    """

    def __init__(
        self,
        optimizer_or_base_lr,
        max_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ) -> None:
        self.max_epochs = int(max(1, max_epochs))
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.min_lr = float(min_lr)

        self.optimizer = None
        self.base_lrs: List[float] = []

        if hasattr(optimizer_or_base_lr, "param_groups"):
            self.optimizer = optimizer_or_base_lr

            if base_lr is None:
                self.base_lrs = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]
            else:
                base_lr = float(base_lr)
                g0_lr = float(self.optimizer.param_groups[0].get("lr", base_lr))
                if abs(g0_lr) < 1e-12:
                    g0_lr = base_lr
                ratios = [float(g.get("lr", g0_lr)) / g0_lr for g in self.optimizer.param_groups]
                self.base_lrs = [base_lr * r for r in ratios]

            if len(self.base_lrs) > 0:
                base0 = float(self.base_lrs[0])
                if abs(base0) < 1e-12:
                    base0 = 1.0
                self.min_lrs = [self.min_lr * (b / base0) for b in self.base_lrs]
            else:
                self.min_lrs = []
        else:
            self.optimizer = None
            self.base_lrs = [float(optimizer_or_base_lr)]
            self.min_lrs = [float(self.min_lr)]

    def _lr_at(self, epoch: int, base_lr: float, min_lr: float) -> float:
        ep = max(0, int(epoch))
        if self.warmup_epochs > 0 and ep < self.warmup_epochs:
            return base_lr * float(ep + 1) / float(max(1, self.warmup_epochs))

        denom = max(1, self.max_epochs - self.warmup_epochs)
        t = float(ep - self.warmup_epochs) / float(denom)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(min_lr + (base_lr - min_lr) * cos)

    def get_lr(self, epoch: int) -> float:
        """Return the learning rate for a given epoch in scalar mode."""
        return self._lr_at(int(epoch), float(self.base_lrs[0]), float(self.min_lrs[0]))

    def step(self, epoch: int) -> List[float]:
        """Update optimizer learning rates for a given epoch."""
        if self.optimizer is None:
            return [self.get_lr(epoch)]

        lrs: List[float] = []
        for i, g in enumerate(self.optimizer.param_groups):
            base_lr = float(self.base_lrs[i])
            min_lr = float(self.min_lrs[i]) if i < len(self.min_lrs) else float(self.min_lr)
            lr = self._lr_at(int(epoch), base_lr, min_lr)
            g["lr"] = lr
            lrs.append(float(lr))
        return lrs

    def step_epoch(self, epoch: int) -> List[float]:
        return self.step(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_epochs": int(self.max_epochs),
            "warmup_epochs": int(self.warmup_epochs),
            "min_lr": float(self.min_lr),
            "base_lrs": [float(x) for x in self.base_lrs],
            "min_lrs": [float(x) for x in getattr(self, "min_lrs", [])],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_epochs = int(state.get("max_epochs", self.max_epochs))
        self.warmup_epochs = int(state.get("warmup_epochs", self.warmup_epochs))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.base_lrs = [float(x) for x in state.get("base_lrs", self.base_lrs)]
        self.min_lrs = [float(x) for x in state.get("min_lrs", getattr(self, "min_lrs", []))]
