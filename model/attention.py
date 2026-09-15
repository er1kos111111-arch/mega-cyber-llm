"""Attention for MC-LLM.

Implements:
  * Rotary Position Embeddings (RoPE) computed from scratch;
  * Grouped-Query Attention (GQA) with causal masking;
  * an explicit KV-cache path for autoregressive generation.

The attention math is standard scaled dot-product attention but with a
from-scratch RoPE implementation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE ``cos``/``sin`` tables of shape (max_seq_len, head_dim).

    Frequencies are duplicated to full ``head_dim`` so that
    ``apply_rope`` can use the half-split rotation scheme.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    position = torch.arange(0, max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(position, inv_freq)  # (seq, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to ``x`` of shape (..., seq, head_dim)."""
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_seq_len: int, rope_theta: float = 10000.0,
                 dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_query_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=use_bias)

        # RoPE tables (registered as buffers so they move with the module).
        cos, sin = precompute_rope(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._causal_mask: Optional[torch.Tensor] = None

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(-1) < seq_len:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
            )
            self._causal_mask = mask
        return self._causal_mask[:seq_len, :seq_len]

    def _repeat_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand KV heads to match the number of query heads (GQA)."""
        if self.num_query_groups == 1:
            return k, v
        k = k[:, :, None, :, :].expand(-1, -1, self.num_query_groups, -1, -1)
        v = v[:, :, None, :, :].expand(-1, -1, self.num_query_groups, -1, -1)
        return k.reshape(k.size(0), self.num_heads, k.size(3), k.size(4)), \
               v.reshape(v.size(0), self.num_heads, v.size(3), v.size(4))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional["KVCache"] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, seq, _ = x.shape

        q = self.q_proj(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
        cos = self.rope_cos[position_ids].unsqueeze(1)  # (b, 1, seq, head_dim/2)
        sin = self.rope_sin[position_ids].unsqueeze(1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self, k, v, position_ids)

        k, v = self._repeat_kv(k, v)

        if kv_cache is not None:
            full_len = k.size(-2)
        else:
            full_len = seq
        mask = self._build_causal_mask(full_len, x.device)
        # For cached inference the query starts at position (full_len - seq).
        query_offset = full_len - seq

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Causal mask indexed by [query_position, key_position].
        key_positions = torch.arange(full_len, device=x.device)
        q_positions = torch.arange(query_offset, query_offset + seq, device=x.device)
        attn_mask = q_positions[:, None] < key_positions[None, :]  # True == masked
        scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout, training=True)
        out = torch.matmul(attn, v)  # (b, heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, seq, -1)
        return self.o_proj(out)


class KVCache:
    """Fixed-size key/value cache supporting incremental autoregressive decoding."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int, batch_size: int = 1,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        self.max_seq_len = max_seq_len
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.filled = 0  # number of positions currently valid

    @property
    def device(self) -> torch.device:
        return self.k.device

    def update(self, layer: CausalSelfAttention, k: torch.Tensor, v: torch.Tensor,
               position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, s, _ = k.shape
        positions = position_ids[0]  # assume single batch for decode cache
        start = int(positions[0])
        end = start + s
        assert end <= self.max_seq_len, "KV cache overflow: increase max_position_embeddings"
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        self.filled = max(self.filled, end)
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self) -> None:
        self.filled = 0
