"""Learning-rate schedulers for MC-LLM.

Implements linear warmup followed by cosine decay to a minimum LR, which is
the schedule used for most large-scale pretraining runs.
"""
from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.min_lr = min_lr
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self, step: int) -> float:
        step = max(0, step)
        if step < self.warmup_steps:
            frac = step / self.warmup_steps
            return self._base_lrs[0] * frac
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self._base_lrs[0] - self.min_lr) * cosine

    def step(self, step: int) -> None:
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class ConstantScheduler:
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer

    def get_lr(self, step: int) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def step(self, step: int) -> None:
        pass


def build_scheduler(name: str, optimizer, **kwargs):
    if name in ("warmup_cosine", "cosine", "warmupcosine"):
        return WarmupCosineScheduler(optimizer, **kwargs)
    if name in ("constant", "none"):
        return ConstantScheduler(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")
