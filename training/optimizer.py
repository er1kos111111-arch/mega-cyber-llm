"""Optimizers for MC-LLM training.

Provides AdamW and Adam with full weight-decay separation, plus a
from-scratch Muon (momentum + orthogonalized updates) implementation for
the 2D weight matrices, which recent work shows accelerates transformer
training.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW with decoupled weight decay.

    Mirrors the PyTorch reference semantics but is implemented here to keep
    the training stack self-contained and auditable.
    """

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


class Adam(Optimizer):
    """Plain Adam (weight decay applied inside the update, L2-style)."""

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


class Muon(Optimizer):
    """Muon (Momentum + Orthogonalized update) for 2D parameters.

    Applies momentum then Newton-Schulz orthogonalization to the update of
    matrices (ndim == 2), falling back to AdamW-style updates for 1D params.
    See Keller Jordan's Muon for the underlying idea.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 adamw_lr: float = 3e-4, betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, adamw_lr=adamw_lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def _orthogonalize(self, g: torch.Tensor, steps: int) -> torch.Tensor:
        # Newton-Schulz iteration to project the matrix onto the orthogonal group
        g = g.float()
        a, b, c = (3.4445, -4.7750, 2.0315)
        x = g / (g.norm() + 1e-8)
        for _ in range(steps):
            a_, b_, c_ = a, b, c
            y = x @ x.mT
            z = b_ * y + c_ * (y @ y)
            x = a_ * x + z @ x
        return x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            adamw_lr = group["adamw_lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["buf"] = torch.zeros_like(p)
                    state["step"] = 0
                buf = state["buf"]
                state["step"] += 1

                if wd != 0:
                    p.mul_(1 - lr * wd)

                buf.mul_(momentum).add_(grad)
                if nesterov:
                    g = grad.add(buf, alpha=momentum)
                else:
                    g = buf

                if g.ndim == 2 and g.size(0) * g.size(1) > 0:
                    g_ortho = self._orthogonalize(g, ns_steps)
                    scale = math.sqrt(max(g.size(0), g.size(1)))
                    p.add_(g_ortho.to(p.dtype), alpha=-lr * 0.5 * scale)
                else:
                    # AdamW fallback for 1D parameters (biases, norms)
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(g, denom, value=-adamw_lr)
        return loss


def build_optimizer(params: Iterable[torch.nn.Parameter], name: str, **kwargs):
    name = name.lower()
    if name == "adamw":
        return AdamW(params, **kwargs)
    if name == "adam":
        return Adam(params, **kwargs)
    if name == "muon":
        return Muon(params, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")
