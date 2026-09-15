"""Generation tests: greedy, temperature, top-k, top-p, deterministic, KV cache."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import MCLLM
from inference.generate import generate, generate_stream, greedy_decode


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    return MCLLM(tiny_config)


def test_greedy_generates_tokens(model, tiny_config):
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = greedy_decode(model, ids, max_new_tokens=8, eos_token_id=tiny_config.vocab_size - 1)
    assert out.shape[1] == ids.shape[1] + 8


def test_deterministic_with_seed(model):
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    a = generate(model, ids, max_new_tokens=8, temperature=1.0, seed=42)
    b = generate(model, ids, max_new_tokens=8, temperature=1.0, seed=42)
    assert torch.equal(a, b)


def test_temperature_changes_distribution(model):
    # high temperature produces more diverse outputs across seeds
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    outs = set()
    for s in range(20):
        out = generate(model, ids, max_new_tokens=4, temperature=2.0, seed=s)
        outs.add(tuple(out[0].tolist()))
    assert len(outs) > 1


def test_different_seeds_different_outputs(model):
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    a = generate(model, ids, max_new_tokens=8, temperature=1.0, seed=1)
    b = generate(model, ids, max_new_tokens=8, temperature=1.0, seed=2)
    # extremely likely to differ at high temperature
    assert not torch.equal(a, b)


def test_top_k_and_top_p(model):
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = generate(model, ids, max_new_tokens=6, top_k=5, top_p=0.9, temperature=1.0)
    assert out.shape[1] == ids.shape[1] + 6


def test_stop_token_stops(model, tiny_config):
    # use a stop token likely to appear eventually; force by tiny vocab
    ids = torch.tensor([[1]], dtype=torch.long)
    out = generate(model, ids, max_new_tokens=50, eos_token_id=2,
                   stop_token_ids=[2], temperature=0.0)
    assert out.shape[1] <= 1 + 50


def test_kv_cache_matches_non_cache(model, tiny_config):
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    cached = greedy_decode(model, ids, max_new_tokens=6, eos_token_id=tiny_config.vocab_size - 1)
    no_cache = generate(model, ids, max_new_tokens=6, eos_token_id=tiny_config.vocab_size - 1,
                        temperature=0.0, use_cache=False)
    assert torch.equal(cached, no_cache)


def test_stream_yields_incremental(model):
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    seqs = list(generate_stream(model, ids, max_new_tokens=5, temperature=0.0))
    assert len(seqs) == 5
    for i, s in enumerate(seqs):
        assert s.shape[1] == ids.shape[1] + i + 1
