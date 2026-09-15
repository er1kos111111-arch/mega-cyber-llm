"""Transformer building blocks for MC-LLM.

* ``SwiGLUMLP`` — gated feed-forward network using the SiLU activation.
* ``TransformerBlock`` — pre-norm residual block combining attention + FFN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, KVCache
from .normalization import RMSNorm


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network.

    ``down( silu(gate(x)) * up(x) )`` with three linear projections.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, use_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, max_seq_len: int, rope_theta: float,
                 rms_norm_eps: float = 1e-5, dropout: float = 0.0,
                 use_bias: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = CausalSelfAttention(
            hidden_size, num_heads, num_kv_heads, max_seq_len, rope_theta, dropout, use_bias
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size, use_bias)

    def forward(self, x: torch.Tensor, kv_cache=None, position_ids=None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, kv_cache=kv_cache, position_ids=position_ids)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return x + residual
