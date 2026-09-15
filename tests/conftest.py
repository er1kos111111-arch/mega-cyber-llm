"""Shared pytest fixtures: a tiny trained tokenizer and a tiny model."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.architecture import MCLLM
from tokenizer.trainer import CyberTokenizerTrainer
from tokenizer.vocab import (
    SPECIAL_TOKENS,
    NUM_BYTE_TOKENS,
    build_special_token_map,
    save_vocab,
    save_merges,
)
from tokenizer.tokenizer import CyberTokenizer

CORPUS = [
    "Привет, как дела? Это тест русского текста для токенизатора.",
    "Hello world, this is an English test for the tokenizer.",
    "Смешанный mixed text на русском and English with numbers 12345.",
    "def foo(x): return x + 1  # Python code",
    '{"key": "value", "n": 42}',
    "🔥 эмодзи 🚀 и спецсимволы © ™ ∑ π √",
    "local function f() return 1 end -- Luau",
    "Привет Hello 你好 😀 test 123 456 789",
]


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tokenizer")
    tr = CyberTokenizerTrainer(vocab_size=512, min_pair_frequency=2)
    merges = tr.train(CORPUS * 100)
    sm = build_special_token_map()
    total = NUM_BYTE_TOKENS + len(SPECIAL_TOKENS) + len(merges)
    save_vocab(str(d / "vocab.json"), sm, merges, total)
    save_merges(str(d / "merges.json"), merges)
    cfg = {
        "name": "CyberTokenizer", "version": "1.0",
        "vocab_size": total, "special_tokens": SPECIAL_TOKENS,
        "vocab_file": "vocab.json", "merges_file": "merges.json",
        "model_max_length": 512,
    }
    with open(str(d / "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return str(d)


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dir):
    return CyberTokenizer(tokenizer_dir + "/tokenizer_config.json")


@pytest.fixture(scope="session")
def tiny_config(tokenizer):
    return ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    return MCLLM(tiny_config)
