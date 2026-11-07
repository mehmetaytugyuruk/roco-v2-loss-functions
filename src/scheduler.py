from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer


@dataclass
class WarmupCosineConfig:
    total_steps: int
    warmup_steps: int
    min_lr: float
    base_lr: float


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, optimizer: Optimizer, cfg: WarmupCosineConfig) -> None:
        if cfg.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if cfg.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        self.optimizer = optimizer
        self.total_steps = cfg.total_steps
        self.warmup_steps = min(cfg.warmup_steps, cfg.total_steps)
        self.min_lr = float(cfg.min_lr)
        self.base_lr = float(cfg.base_lr)
        self.step_count = 0

        self._set_lr(self._compute_lr(self.step_count))

    def _compute_lr(self, step_count: int) -> float:
        if self.warmup_steps > 0 and step_count < self.warmup_steps:
            return self.base_lr * float(step_count + 1) / float(self.warmup_steps)

        if self.total_steps == self.warmup_steps:
            return self.min_lr

        progress = (step_count - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> float:
        self.step_count += 1
        lr = self._compute_lr(self.step_count)
        self._set_lr(lr)
        return lr

    def get_last_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
