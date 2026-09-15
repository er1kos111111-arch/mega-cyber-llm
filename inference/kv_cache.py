"""KV cache helpers for MC-LLM inference."""
from __future__ import annotations

from typing import List, Optional

import torch

from model.attention import KVCache
from model.architecture import MCLLM


def make_cache(model: MCLLM, batch_size: int = 1, dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None) -> List[KVCache]:
    """Build one per-layer KV cache sized to the model's max context."""
    cfg = model.config
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype
    return [
        KVCache(
            num_layers=1,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        for _ in range(cfg.num_layers)
    ]
