"""MEGA-CYBER LLM (MC-LLM) top-level architecture.

A decoder-only, causal, pre-norm transformer assembled from the building
blocks in this package.  Everything is initialised from scratch; there is
no pretrained weight path.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .attention import KVCache
from .config import ModelConfig
from .embeddings import TokenEmbedding
from .layers import TransformerBlock
from .normalization import RMSNorm


class MCLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = TokenEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                intermediate_size=config.intermediate_size,
                max_seq_len=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                dropout=config.dropout,
                use_bias=config.use_bias,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, TokenEmbedding):
            module.reset_parameters()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ) -> torch.Tensor:
        """Return logits of shape (batch, seq, vocab_size).

        ``kv_cache`` is an optional list of per-layer :class:`KVCache`
        objects (one per transformer layer), or ``None`` for training.
        """
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, layer_cache, position_ids, use_reentrant=False)
            else:
                x = layer(x, kv_cache=layer_cache, position_ids=position_ids)
        x = self.norm(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = torch.matmul(x, self.embed_tokens.weight.t())
        return logits

    # ------------------------------------------------------------------
    # loss helpers (used by training)
    # ------------------------------------------------------------------
    def forward_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Causal language-modeling cross entropy.

        Predicts ``tokens[1:]`` from ``tokens[:-1]``; ignore index is -100.
        """
        logits = self.forward(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_p: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation (see inference/generate.py for the full engine)."""
        from inference.generate import generate as _generate
        return _generate(
            self, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p,
            stop_token_ids=stop_token_ids, seed=seed,
        )
