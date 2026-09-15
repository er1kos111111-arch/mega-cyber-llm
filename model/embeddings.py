"""Embeddings for MC-LLM.

Token embeddings initialised from scratch (normal distribution, scaled to
control initial activation magnitude).  The output head can optionally be
tied to the input embedding matrix to save parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(ids, self.weight)
