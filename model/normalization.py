"""Normalization layers for MC-LLM.

RMSNorm (Root Mean Square Layer Normalization) with an optional learned
scale.  This is the standard normalization used in modern decoder-only
LLMs and is implemented here from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization.

    ``y = x / sqrt(mean(x^2) + eps) * weight``
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Compute in float32 for numerical stability, then cast back.
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.float()).to(dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class LayerNorm(nn.Module):
    """Standard LayerNorm (kept for reference / optional use)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        mean = x_f.mean(-1, keepdim=True)
        var = x_f.var(-1, keepdim=True, unbiased=False)
        x_norm = (x_f - mean) * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight.float() + self.bias.float()).to(dtype)
